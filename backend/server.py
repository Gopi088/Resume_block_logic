"""B10: recruiter-facing HTTP API over the existing B0–B9 pipeline.

Thin wrapper only — no parsing logic lives here. It calls
``parse_resume.build_output`` with the trained model and returns the same
JSON contract, plus a trimmed ``review`` payload the side panel consumes
directly. The extension never talks to the parser except through this file.

Run:
    .venv/bin/python -m uvicorn backend.server:app --port 8000
"""

from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from parse_resume import build_output

MODEL_PATH = os.environ.get("RESUME_MODEL", "model_real_v2.pkl")
NOTES_PATH = os.environ.get("RESUME_NOTES", "backend/notes_store.json")

# In-memory cache of parsed resumes (sha -> {filename, lines}) backing the
# /resume-view fallback page. Bounded; the parser output stays canonical.
_DOC_CACHE: dict = {}
_DOC_CACHE_MAX = 20

app = FastAPI(title="Resume Timeline Review API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MV3 side panel origin is chrome-extension://; allowlist in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": MODEL_PATH, "model_exists": os.path.exists(MODEL_PATH)}


# ---- Shared review notes (visible to every recruiter via the backend) ----

def _read_notes() -> dict:
    import json as _json

    if not os.path.exists(NOTES_PATH):
        return {}
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_notes(store: dict) -> None:
    import json as _json

    tmp = NOTES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(store, f, ensure_ascii=False)
    os.replace(tmp, NOTES_PATH)


@app.get("/api/notes/{sha}")
def get_notes(sha: str) -> dict:
    """All shared notes for one resume, keyed by item id.

    Response: {"ok": true, "notes": {itemId: [{note, by, at}]}}.
    """
    doc_notes = _read_notes().get(sha, {})
    return {"ok": True, "notes": doc_notes if isinstance(doc_notes, dict) else {}}


@app.post("/api/notes/{sha}")
async def post_note(sha: str, payload: dict) -> dict:
    """Append one shared note. Requires non-empty note + author name.

    Body: {"item_id": "E000001", "note": "...", "by": "Recruiter name"}.
    """
    from datetime import datetime as _dt

    item_id = (payload.get("item_id") or "").strip()
    note = (payload.get("note") or "").strip()
    by = (payload.get("by") or "").strip() or "auto"
    if not item_id or not note:
        return {"ok": False, "error": "item_id and note are required"}
    store = _read_notes()
    doc_notes = store.get(sha, {})
    if not isinstance(doc_notes, dict):
        doc_notes = {}
    entries = doc_notes.get(item_id, [])
    if not isinstance(entries, list):
        entries = []
    entries.append({"note": note, "by": by, "at": _dt.now().isoformat(timespec="seconds")})
    doc_notes[item_id] = entries
    store[sha] = doc_notes
    _write_notes(store)
    return {"ok": True, "notes": {item_id: entries}}


@app.post("/api/parse")
async def parse_resume(file: UploadFile = File(...)) -> dict:
    """Accept a resume file, run B0–B9, return the full parser output.

    stdout contract of parse_resume.build_output is preserved verbatim so the
    frontend consumes real data (never mock data). A compact ``review`` key is
    added for the side panel; it only re-projects existing fields.
    """
    suffix = os.path.splitext(file.filename or "resume")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        out = build_output(
            tmp_path,
            include_eval=False,
            include_text=False,  # lines[] still carry display_text + char spans
            blocks_only=False,  # lines[] needed for exact evidence
            model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    out["review"] = build_review_projection(out)
    _DOC_CACHE[out["sha256"]] = {
        "filename": out.get("filename"),
        "lines": [
            {k: ln.get(k) for k in ("line_id", "index", "page_index", "display_text")}
            for ln in out.get("lines", [])
        ],
    }
    while len(_DOC_CACHE) > _DOC_CACHE_MAX:
        _DOC_CACHE.pop(next(iter(_DOC_CACHE)))
    return {"ok": True, **out}


@app.get("/resume-view/{sha}")
def resume_view(sha: str) -> object:
    """Readable parsed-resume page used as the View-in-Resume fallback.

    Chrome's PDF viewer cannot be scripted by any extension, so when the open
    tab can't be highlighted the panel opens this page at the exact evidence
    lines (?hl=L000001,L000002#L000001). Same source lines the parser used —
    no new facts.
    """
    from fastapi.responses import HTMLResponse

    import html as _html

    cached = _DOC_CACHE.get(sha)
    if not cached:
        return HTMLResponse(
            "<p>Parsed resume not cached (backend restarted?). Re-parse the file from the extension.</p>",
            status_code=404,
        )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{_html.escape(str(cached.get('filename') or 'Resume'))}</title>",
        "<style>body{font:13px/1.5 system-ui;margin:16px auto;max-width:720px}"
        ".rl{padding:1px 6px;overflow-wrap:anywhere}.rl.hit{background:#fff3c4;outline:2px solid #b98a1d}"
        ".pg{color:#888;font-size:12px;margin:12px 0 4px}</style></head><body>",
        f"<h2>{_html.escape(str(cached.get('filename') or 'Resume'))}</h2>",
        "<p style='color:#888'>Parsed text — same lines the timeline was extracted from.</p>",
    ]
    last_page = None
    for ln in cached["lines"]:
        if ln.get("page_index") != last_page:
            last_page = ln.get("page_index")
            parts.append(f"<div class='pg'>Page {last_page + 1}</div>")
        text = (ln.get("display_text") or "").strip()
        if not text:
            continue
        parts.append(f"<div class='rl' id='{ln['line_id']}'>{_html.escape(text)}</div>")
    parts.append(
        "<script>(function(){var q=new URLSearchParams(location.search);"
        "var ids=(q.get('hl')||'').split(',').filter(Boolean);"
        "ids.forEach(function(id){var n=document.getElementById(id);"
        "if(n)n.classList.add('hit')});"
        "var t=location.hash&&document.querySelector(location.hash);"
        "if(t)t.scrollIntoView({block:'center'})})()</script></body></html>"
    )
    return HTMLResponse("".join(parts))


def build_review_projection(out: dict) -> dict:
    """Project parser output into the recruiter-facing review model.

    Pure re-projection: candidate identity, accuracy, timeline items, gaps and
    exact evidence pointers. No new facts are invented; missing data stays
    explicit (None) so the UI can render honest empty/evidence-unavailable
    states.
    """
    lines = {ln["line_id"]: ln for ln in out.get("lines", [])}
    entries = {e["entry_id"]: e for e in out.get("entries", {}).get("entries", [])}
    entry_dates = {
        ed["entry_id"]: ed for ed in out.get("entry_dates", {}).get("entry_dates", [])
    }
    final = out.get("final_sections", {})
    sections = {s["block_id"]: s for s in final.get("sections", [])}
    timeline = out.get("timeline", {})

    candidate = _candidate_identity(out, entries)
    accuracy = _accuracy(final)

    items: list[dict] = []
    for ev in timeline.get("events", []):
        entry = entries.get(ev["entry_id"], {})
        ed = entry_dates.get(ev["entry_id"], {})
        fsec = sections.get(ev.get("block_id", ""), {})
        org = _entry_org(entry.get("text", ""))
        items.append(
            {
                "kind": "event",
                "entry_id": ev["entry_id"],
                "block_id": ev.get("block_id"),
                "section": ev.get("section"),
                "title": _entry_title(entry.get("text", ""), org, ev.get("raw_range")),
                "organization": org,
                "location": _entry_location(entry.get("text", "")),
                "start_date": ev.get("start_date"),
                "end_date": ev.get("end_date"),
                "is_ongoing": ev.get("is_ongoing", False),
                "raw_range": ev.get("raw_range"),
                "confidence": round(float(fsec.get("confidence", 0.0)), 3),
                "confidence_band": _band(float(fsec.get("confidence", 0.0))),
                "trusted": bool(ev.get("trusted", False)),
                "needs_review": not bool(ev.get("trusted", False)),
                "evidence": _evidence(entry, ed, lines, ev.get("raw_range")),
            }
        )

    gaps: list[dict] = []
    for g in timeline.get("gaps", []):
        gaps.append(
            {
                "kind": "gap",
                "gap_id": g["gap_id"],
                "start_date": g["start_date"],
                "end_date": g["end_date"],
                "gap_months_approx": g.get("gap_months_approx"),
                "before_entry_id": g.get("before_entry_id"),
                "after_entry_id": g.get("after_entry_id"),
                "needs_review": True,
                "evidence": None,  # gaps are derived absences: no source span exists
            }
        )

    undated: list[dict] = []
    for eid in timeline.get("undated_entry_ids", []):
        entry = entries.get(eid, {})
        if not entry:
            continue
        fsec = sections.get(entry.get("block_id", ""), {})
        # Contact boilerplate is identity, not review work — keep review list focused.
        if entry.get("section") == "contact":
            continue
        org = _entry_org(entry.get("text", ""))
        undated.append(
            {
                "kind": "undated",
                "entry_id": eid,
                "block_id": entry.get("block_id"),
                "section": entry.get("section"),
                "title": _entry_title(entry.get("text", ""), org, None),
                "organization": org,
                "confidence": round(float(fsec.get("confidence", 0.0)), 3),
                "confidence_band": _band(float(fsec.get("confidence", 0.0))),
                "needs_review": True,
                "evidence": _evidence(entry, entry_dates.get(eid, {}), lines, None),
            }
        )

    return {
        "candidate": candidate,
        "accuracy": accuracy,
        "events": items,
        "gaps": gaps,
        "undated": undated,
    }


def _band(conf: float) -> str:
    if conf >= 0.75:
        return "High"
    if conf >= 0.5:
        return "Medium"
    return "Low"


def _accuracy(final: dict) -> dict:
    sections = final.get("sections", [])
    trust_rate = float(final.get("trust_rate", 0.0) or 0.0)
    mean_conf = (
        sum(float(s.get("confidence", 0.0)) for s in sections) / len(sections)
        if sections
        else 0.0
    )
    # Recruiter-facing score: trust dominates, confidence informs. Never fake high.
    score = round(100 * (0.7 * trust_rate + 0.3 * mean_conf))
    label = "Accurate" if score >= 75 else ("Check carefully" if score >= 45 else "Low — verify")
    return {"percent": score, "label": label, "trust_rate": round(trust_rate, 3)}


def _candidate_identity(out: dict, entries: dict) -> dict:
    import re

    full_text_lines: list[str] = []
    for ln in out.get("lines", []):
        t = (ln.get("display_text") or "").strip()
        if t:
            full_text_lines.append(t)

    name: str | None = None
    for t in full_text_lines[:12]:
        m = re.search(r"full name\s*:\s*(.+)", t, re.I)
        if m:
            name = m.group(1).strip()
            break
    if not name:
        for t in full_text_lines[:6]:
            if "@" in t or "://" in t or re.search(r"\d{5,}", t):
                continue
            if 2 <= len(t.split()) <= 5 and len(t) <= 60 and not re.search(r"[:|]", t):
                name = t
                break
    if not name:
        name = (out.get("filename") or "Candidate").rsplit(".", 1)[0].replace("_", " ")

    exp_events = sorted(
        [e for e in (out.get("timeline", {}).get("events", []) or []) if e.get("section") == "experience"],
        key=lambda e: (e.get("start_date") or "", e.get("entry_id") or ""),
        reverse=True,
    )
    current_role = current_org = None
    if exp_events:
        recent = entries.get(exp_events[0]["entry_id"], {})
        current_org = _entry_org(recent.get("text", ""))
        current_role = _entry_title(
            recent.get("text", ""), current_org, exp_events[0].get("raw_range")
        )
    years = _years_span(out.get("timeline", {}).get("events", []))
    return {
        "name": name,
        "current_role": current_role,
        "current_org": current_org,
        "years_experience": years,
        "filename": out.get("filename"),
        "document_id": out.get("document_id"),
        "sha256": out.get("sha256"),
    }


def _years_span(events: list[dict]) -> float | None:
    from datetime import date as _d

    if not events:
        return None
    starts = [e.get("start_date") for e in events if e.get("start_date")]
    ends = [
        e.get("end_date") or e.get("effective_end_date") or str(_d.today())
        for e in events
    ]
    if not starts or not ends:
        return None
    try:
        span_days = (_d.fromisoformat(max(ends)) - _d.fromisoformat(min(starts))).days
        return round(span_days / 365.25, 1) if span_days > 0 else 0.0
    except ValueError:
        return None


def _clean_md(text: str) -> str:
    import re

    t = re.sub(r"!\[.*?\]\(.*?\)", " ", text or "")  # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # links
    t = t.replace("|", " ").replace("*", "").replace("#", "").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _entry_lines(text: str) -> list[str]:
    out: list[str] = []
    for ln in (text or "").splitlines():
        # Table rows pack several facts in one line — split cells apart.
        for cell in ln.split("|"):
            c = _clean_md(cell).strip(" \t-–—:")
            if c:
                out.append(c)
    return out


def _entry_title(text: str, org: str | None = None, raw_range: str | None = None) -> str | None:
    import re

    cleaned = _clean_md(text)
    if org:
        cleaned = cleaned.replace(org, " ")
    # The job's own segment starts right after its date range.
    seg = cleaned
    if raw_range:
        anchor = _clean_md(raw_range).strip()
        idx = cleaned.find(anchor)
        if idx >= 0:
            seg = cleaned[idx + len(anchor):]
    else:
        parts = re.split(
            r"(?:19|20)\d{2}\s*[–—\-/]\s*(?:(?:19|20)\d{2}|present|current)",
            cleaned,
            maxsplit=1,
            flags=re.I,
        )
        if len(parts) > 1:
            seg = parts[1]
    seg = re.split(
        r"\b(Technology|Client|Location|Duration|Project|Skills?)\b\s*[-:]?",
        seg,
        maxsplit=1,
        flags=re.I,
    )[0]
    seg = re.sub(r"^[^A-Za-z]+", "", seg).strip()
    seg = re.sub(r"^(company\s*-\s*)", "", seg, flags=re.I)
    seg = re.sub(r"^[^A-Za-z]+", "", seg).strip()
    seg = re.sub(r"\s+", " ", seg)
    if org and len(seg) < len(org):
        # A fragment shorter than the org line (e.g. "Bachelor of") hides the
        # real content — the org line is the more complete label.
        return _word_truncate(org, 80)
    if len(seg) >= 3:
        return _word_truncate(seg, 80)
    if org:
        return _word_truncate(org, 80)
    fallback = re.sub(r"^[^A-Za-z]+", "", cleaned).strip()
    return _word_truncate(fallback, 80) if fallback else None


def _word_truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return (text[: cut if cut > 10 else limit].rstrip() + "…")


def _entry_org(text: str) -> str | None:
    import re

    m = re.search(
        r"company\s*-\s*([A-Za-z][A-Za-z0-9 .,&()\-]{1,60})", _clean_md(text), re.I
    )
    if m:
        org = re.split(
            r"\s+(Technology|Client|Location|Duration|Project|Skills?)\b",
            m.group(1).strip(),
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        return _word_truncate(org, 48)
    meaningful = [
        ln
        for ln in _entry_lines(text)
        if "@" not in ln and "://" not in ln and len(ln) > 1
    ]
    if len(meaningful) >= 2:
        org = meaningful[1]
        if re.search(r"\d{4}|present", org, re.I) and len(meaningful) >= 3:
            org = meaningful[2]
        return org[:100]
    return None


def _entry_location(text: str) -> str | None:
    import re

    m = re.search(r"(?:location|place)\s*:\s*([A-Za-z][A-Za-z ,\-]{1,40})", text or "", re.I)
    if m:
        return m.group(1).strip()
    lines = _entry_lines(text)
    if lines:
        last = lines[-1]
        if re.match(r"^[A-Z][a-z]+(, *[A-Z][a-z]+)?$", last) and len(last) <= 30:
            return last
    return None


def _evidence(entry: dict, entry_dates: dict, lines: dict, raw_range: str | None) -> dict | None:
    if not entry:
        return None
    line_ids: list[str] = list(entry.get("line_ids", []))
    spans = [
        {
            "line_id": lid,
            "page_index": (lines.get(lid, {}) or {}).get("page_index"),
            "text": (lines.get(lid, {}) or {}).get("display_text"),
        }
        for lid in line_ids
    ]
    # Char-level provenance for the matched date range, when B8 found one.
    date_chars = None
    for r in (entry_dates or {}).get("ranges", []) or []:
        if raw_range and r.get("raw") == raw_range:
            date_chars = {"start_char": r.get("start_char"), "end_char": r.get("end_char")}
            break
    if not spans:
        return None
    pages = sorted({s["page_index"] for s in spans if s["page_index"] is not None})
    return {
        "line_ids": line_ids,
        "start_line": line_ids[0] if line_ids else None,
        "end_line": line_ids[-1] if line_ids else None,
        "pages": pages,
        "raw_range": raw_range,
        "date_chars": date_chars,
        "excerpt": (entry.get("text") or "")[:600],
        "unavailable": False,
    }
