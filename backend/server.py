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
import re
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from parse_resume import build_output

MODEL_PATH = os.environ.get("RESUME_MODEL", "model_real_v2.pkl")
NOTES_PATH = os.environ.get("RESUME_NOTES", "backend/notes_store.json")

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
        return _parse_local_path(tmp_path, file.filename or "resume")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _resolve_file_url(url: str) -> str:
    """Map a file:// URL to a local path the backend can read.

    Handles POSIX paths and Windows drive paths (C:/... or /C:/...), including
    the WSL layout where C: is mounted at /mnt/c. Raises ValueError when the
    file cannot be reached from this machine.
    """
    import re
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ValueError("not a file URL")
    path = unquote(parsed.path or "")
    candidates = [path]
    m = re.match(r"^/([A-Za-z]:/.*)$", path)  # /C:/resumes/x.pdf
    if m:
        candidates.append(m.group(1))  # C:/resumes/x.pdf (native Windows)
        candidates.append(f"/mnt/{m.group(1)[0].lower()}{m.group(1)[2:]}")
    m2 = re.match(r"^([A-Za-z]:/.*)$", path)  # C:/resumes/x.pdf
    if m2:
        candidates.append(f"/mnt/{m2.group(1)[0].lower()}{m2.group(1)[2:]}")
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise ValueError(f"file not reachable from the backend: {url}")


def _download_url(url: str) -> tuple[bytes, str]:
    """Download an http(s) URL with timeout + size cap. Returns (bytes, suffix)."""
    import os as _os
    from urllib.parse import urlparse as _up
    from urllib.request import Request as _Req
    from urllib.request import urlopen as _open

    req = _Req(url, headers={"User-Agent": "ResumeReview/1.0"})
    with _open(req, timeout=60) as resp:
        data = resp.read(30 * 1024 * 1024 + 1)
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("file larger than 30 MB")
    suffix = _os.path.splitext(_up(url).path)[1].lower() or ".pdf"
    return data, suffix


@app.post("/api/parse-url")
async def parse_url(payload: dict) -> dict:
    """Parse the resume at a URL (the open tab) without a file upload.

    file:// URLs are read from local disk (incl. Windows↔WSL drive mapping);
    http(s) URLs are downloaded server-side. Same B0–B9 output shape as
    /api/parse, so the panel consumes it identically.
    """
    from urllib.parse import unquote, urlparse

    url = ((payload or {}).get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "url is required"}
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https", "file"):
        return {"ok": False, "error": f"unsupported URL scheme: {scheme or '(none)'}"}
    suffix = os.path.splitext(urlparse(url).path)[1].lower() or ".pdf"
    if scheme == "file":
        try:
            return _parse_local_path(_resolve_file_url(url), _display_name(url))
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # parser failures stay JSON, never tracebacks
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    try:
        data, suffix = _download_url(url)
    except Exception as e:
        return {"ok": False, "error": f"download failed: {str(e)[:300]}"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return _parse_local_path(tmp_path, _display_name(url))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _display_name(url: str) -> str:
    from urllib.parse import unquote, urlparse

    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    return name or "resume"


def _parse_local_path(path: str, display_name: str) -> dict:
    out = build_output(
        path,
        include_eval=False,
        include_text=False,
        blocks_only=False,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
    )
    out["filename"] = display_name
    out["review"] = build_review_projection(out)
    return {"ok": True, **out}


def build_review_projection(out: dict) -> dict:
    """Project parser output into the recruiter-facing review model.

    Pure re-projection: candidate identity, accuracy, timeline items, gaps and
    exact evidence pointers. No new facts are invented; missing data stays
    explicit (None) so the UI can render honest empty/evidence-unavailable
    states.
    """
    lines = {ln["line_id"]: ln for ln in out.get("lines", [])}
    
    def _raw_entry_text(entry: dict) -> str:
        """Reconstruct raw text from original lines (with pipe separators intact)."""
        line_ids = entry.get("line_ids", [])
        parts = []
        for lid in line_ids:
            ln = lines.get(lid)
            if ln and ln.get("raw_text"):
                parts.append(ln["raw_text"])
        return "\n".join(parts)
    
    entries = {e["entry_id"]: e for e in out.get("entries", {}).get("entries", [])}
    ordered_entries = sorted(
        out.get("entries", {}).get("entries", []), key=lambda e: e.get("index", 0)
    )
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
        raw_text = _raw_entry_text(entry)
        if ev.get("section") in HEADER_SECTIONS:
            title, org = _resolve_header(entry, ordered_entries)
            if org is None:
                org = _entry_org(entry.get("text", ""))
            if title is None:
                # Use raw text (with pipes) for title extraction
                title = _entry_title(entry.get("text", ""), org, ev.get("raw_range"), raw_text)
        else:
            title = _section_header_title(entry.get("text", "")) or _entry_title(
                entry.get("text", ""), None, ev.get("raw_range"), raw_text)
            org = None
        items.append(
            {
                "kind": "event",
                "entry_id": ev["entry_id"],
                "block_id": ev.get("block_id"),
                "section": ev.get("section"),
                "title": title,
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
        # Table-separator artifacts carry no words — not review work.
        if not re.search(r"[A-Za-z]{3,}", entry.get("text", "")):
            continue
        fsec = sections.get(entry.get("block_id", ""), {})
        # Contact boilerplate is identity, not review work — keep review list focused.
        if entry.get("section") == "contact":
            continue
        if entry.get("section") in HEADER_SECTIONS:
            title, org = _resolve_header(entry, ordered_entries)
            if org is None:
                org = _entry_org(entry.get("text", ""))
            if title is None:
                title = _entry_title(entry.get("text", ""), org, None)
        else:
            title = _section_header_title(entry.get("text", "")) or _entry_title(
                entry.get("text", ""), None, None)
            org = None
        undated.append(
            {
                "kind": "undated",
                "entry_id": eid,
                "block_id": entry.get("block_id"),
                "section": entry.get("section"),
                "title": title,
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
        ordered = sorted(entries.values(), key=lambda e: e.get("index", -1))
        current_role, current_org = _resolve_header(recent, ordered)
        if current_org is None:
            current_org = _entry_org(recent.get("text", ""))
        if current_role is None:
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


def _match_pipe_line(head: str) -> tuple[str | None, str | None]:
    """A clean two-part 'Role | Organization' header, else (None, None."""
    import re

    head = _clean_pipe_line(head)
    if not head or "@" in head or "://" in head or head.count("|") != 1:
        return None, None
    role, _, org = head.partition("|")
    role, org = role.strip(" \t-–—:,"), org.strip(" \t-–—:,")
    if not (2 <= len(role) <= 60 and 2 <= len(org) <= 50):
        return None, None
    if not re.search(r"[a-zA-Z]{3,}", role) or not re.search(r"[a-zA-Z]{3,}", org):
        return None, None
    return role, org


def _clean_pipe_line(text: str) -> str:
    import re

    t = re.sub(r"!\[.*?\]\(.*?\)", " ", text or "")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.replace("*", "").replace("#", "").replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def _pipe_header(text: str) -> tuple[str | None, str | None]:
    """Role | Organization header lines, e.g. 'Data Management Analyst | Wells Fargo'.

    Only lines up to and including the first date-bearing line qualify: a pipe
    line after the date starts the NEXT job, not this one. Returns (role, org)
    or (None, None). Skips contact lines and multi-pipe skill lists.
    """
    import re

    for ln in (text or "").splitlines():
        c = _clean_pipe_line(ln)
        if not c or "@" in c or "://" in c:
            continue
        # Role headers often share their line with the date range
        # ("Data Management Analyst | Wells Fargo   Feb 2025 – Present"):
        # judge only the part before the first date token.
        mdate = re.search(
            r"(?:(?:19|20)\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4})",
            c,
            re.I,
        )
        head = c[: mdate.start()].strip() if mdate else c
        role, org = _match_pipe_line(head)
        if role:
            return role, org
        if mdate:
            break  # pipes past the date line belong to the next entry
    return None, None


def _last_pipe(text: str) -> tuple[str | None, str | None]:
    """Last pipe header anywhere in the text (for borrowing from the previous
    same-block entry, where the header is the closest preceding role line)."""
    found: tuple[str | None, str | None] = (None, None)
    for ln in (text or "").splitlines():
        role, org = _match_pipe_line(ln)
        if role:
            found = (role, org)
    return found


def _has_company(text: str) -> bool:
    import re

    return bool(re.search(r"company\s*-\s*[A-Za-z]", _clean_md(text), re.I))


def _match_comma_head(head: str) -> tuple[str | None, str | None]:
    """Match one pre-cleaned header candidate of the form 'Role, Organization'.
    The org must carry an organization signal or be a short multi-word
    Title-Case name, so bullet sentences with commas never match."""
    import re

    if "," not in head:
        return None, None
    role, _, org = head.partition(",")
    role, org = role.strip(" \t-–—:,"), org.strip(" \t-–—:,")
    if not (2 <= len(role) <= 50 and 2 <= len(org) <= 60):
        return None, None
    if re.search(r"\d{4}|present|current", role, re.I):
        return None, None
    org_words = org.split()
    # Single-word orgs (e.g. a bare "Allahabad") must carry an explicit
    # organization signal; otherwise they are almost always locations.
    org_ok = (
        re.search(r"\b(?:" + _ORG_SIGNAL + r")\b", org, re.I) is not None
        or (
            len(org_words) >= 2
            and len(org_words) <= 4
            and all(re.match(r"^[A-Z][A-Za-z.\-&']*$", w) for w in org_words)
        )
    )
    if not org_ok:
        return None, None
    return role, org


def _tail_comma(text: str) -> tuple[str | None, str | None]:
    """Comma header borrowed from a previous entry's TAIL only (B7 often
    splits right after the header line, leaving it at the previous entry's
    end). Restricting to the tail avoids attributing an earlier job's header
    to a later, unrelated entry."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines[-2:]:
        c = _clean_pipe_line(ln)
        if not c or "@" in c or "://" in c or "," not in c:
            continue
        import re

        mdate = re.search(
            r"(?:(?:19|20)\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4})",
            c,
            re.I,
        )
        head = c[: mdate.start()].strip() if mdate else c
        head = re.sub(r"\s+" + _DATE_TRAIL + r"\s*$", "", head, flags=re.I).strip()
        role, org = _match_comma_head(head)
        if role:
            return role, org
    return None, None


_ORG_SIGNAL = (
    r"Inc|LLC|LLP|Ltd|Pvt|Corp|GmbH|Pty|Technologies|Technology|Systems|"
    r"Solutions|Services|Group|Bank|Banks|University|College|School|Institute|"
    r"Limited|Company|Financial|Broking|Securities|Capital|Holdings|Insurance|"
    r"Analytics|Consulting|Advisory|Partners|Enterprises|Industries"
)

_DATE_TRAIL = (
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
    r"\s*[–—\-/]\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|present|current|now)"
    r"|(?:19|20)\d{2}\s*[–—\-/]\s*(?:(?:19|20)\d{2}|present|current))"
)


def _comma_header(text: str) -> tuple[str | None, str | None]:
    """Role, Company header lines, e.g. 'Senior Accounting & Reconciliation
    Analyst, Ameriprise Financial' or 'Analyst, Citicorp Services India Pvt.
    Ltd. August 2022 - July 2024' (trailing date range stripped).

    Returns (role, org) or (None, None). The org part must carry an
    organization signal or be a short Title-Case name, so bullet sentences
    with commas never match.
    """
    import re

    for ln in (text or "").splitlines():
        c = _clean_pipe_line(ln)
        if not c or "@" in c or "://" in c or "," not in c:
            continue
        # A date-bearing line still ends this entry's header zone, but judge
        # the pre-date part first (headers often share the line with dates).
        mdate = re.search(
            r"(?:(?:19|20)\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4})",
            c,
            re.I,
        )
        head = c[: mdate.start()].strip() if mdate else c
        head = re.sub(r"\s+" + _DATE_TRAIL + r"\s*$", "", head, flags=re.I).strip()
        if "," not in head:
            if mdate:
                break
            continue
        role, org = _match_comma_head(head)
        if role:
            return role, org
        if mdate:
            break
    return None, None


HEADER_SECTIONS = frozenset({"experience", "education", "projects"})


def _section_header_title(text: str) -> str | None:
    """The entry's own section header line (e.g. 'PROFESSIONAL SUMMARY',
    'CORE COMPETENCIES') as its title — the resume's words, verbatim.

    Used for non-job sections where role/org extraction is meaningless.
    """
    first = (_entry_lines(text) or [None])[0]
    if first and 3 <= len(first) <= 60:
        return first
    return None


def _resolve_header(entry: dict, ordered_entries: list) -> tuple[str | None, str | None]:
    """Best role/org header for one entry: own pre-date pipe header; else the
    closest preceding pipe header in the same B2 block (B7 often splits the
    header line into its own undated entry, or trails the next job's header
    at the previous entry's tail); else (None, None) to use the fallbacks.
    """
    text = entry.get("text", "")
    section = entry.get("section", "")
    if section not in HEADER_SECTIONS:
        return None, None
    role, org = _pipe_header(text)
    if role:
        return role, org
    role, org = _comma_header(text)
    if role:
        return role, org
    if _has_company(text):
        return None, None  # Company-style resumes resolve via their own path
    block_id = entry.get("block_id")
    idx = entry.get("index", -1)
    prevs = [e for e in ordered_entries if e.get("index", -1) < idx]
    # Comma headers are only borrowed from the immediately previous entry
    # (B7 splits right after the header). Walking further back risks
    # attributing an earlier job's header to an unrelated later entry.
    if prevs and prevs[-1].get("block_id") == block_id:
        role, org = _tail_comma(prevs[-1].get("text", ""))
        if role:
            return role, org
    for prev in reversed(prevs):
        if prev.get("block_id") != block_id:
            break  # same-block entries are contiguous; never borrow across blocks
        role, org = _last_pipe(prev.get("text", ""))
        if role:
            return role, org
    return None, None


def _entry_title(text: str, org: str | None = None, raw_range: str | None = None, raw_text: str | None = None) -> str | None:
    import re

    role, pipe_org = _pipe_header(text)
    if role:
        return role
    comma_role, _ = _comma_header(text)
    if comma_role:
        return comma_role
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
    # Also keep raw_text with pipes for pipe-part extraction
    raw_seg = raw_text
    if raw_range and raw_text:
        anchor = _clean_md(raw_range).strip()
        idx = raw_text.find(anchor)
        if idx >= 0:
            raw_seg = raw_text[idx + len(anchor):]
    elif raw_text:
        parts = re.split(
            r"(?:19|20)\d{2}\s*[–—\-/]\s*(?:(?:19|20)\d{2}|present|current)",
            raw_text,
            maxsplit=1,
            flags=re.I,
        )
        if len(parts) > 1:
            raw_seg = parts[1]

    # Table-aware extraction FIRST using raw text with pipes
    # 1. After the LAST "|" separator (common in table-based resumes: metadata | title)
    # 2. After "Company - X" / "Client - X" patterns — the next meaningful text
    # 3. First non-bullet line in raw_seg that looks like a role title
    title = None
    
    # Pattern 1: Text after a "|" separator that contains a job title (often the part before Project Description)
    # In table-based resumes: metadata | title | Project Description
    if raw_seg:
        pipe_parts = raw_seg.split("|")
        if len(pipe_parts) >= 2:
            # Check each part (from the end) for a job title pattern
            # The title is often in the last non-empty part before "Project" markers
            for part in reversed(pipe_parts):
                part = part.strip()
                if not part or len(part) < 3 or len(part) >= 120:
                    continue
                # Remove "Project Description", "Project Overview", etc.
                cleaned_part = re.sub(r"\b(?:Project|Description|Overview|Summary)\b.*$", "", part, flags=re.I).strip()
                cleaned_part = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cleaned_part)
                cleaned_part = re.sub(r"\s+", " ", cleaned_part)
                if len(cleaned_part) < 3 or len(cleaned_part) >= 120:
                    continue
                # Check if it looks like a title
                if re.search(
                    r"(?:\b(?:Project|Sr\.?|Senior|Junior|Lead|Principal|Staff)\s+(?:Planning\s+)?Consultant"
                    r"|\b(?:Sr\.?|Senior|Junior|Lead|Principal|Staff)\s+(?:Data|Business|Systems?|Software)\s+(?:Analyst|Engineer|Scientist|Developer|Architect)"
                    r"|\b(?:Data|Business|Systems?|Software)\s+(?:Analyst|Engineer|Scientist|Developer|Architect)"
                    r"|\b(?:Programmer|Developer|Engineer|Analyst|Consultant|Manager|Director|Lead|Coordinator|Specialist|Administrator|Officer|Executive|Associate|Intern|Trainee)"
                    r"|\b(?:Project|Product|Program|Technical)\s+(?:Manager|Lead|Coordinator|Owner)"
                    r"|\b(?:Assistant|Associate|Deputy|Vice)\s+(?:Manager|Director|President|Chair)"
                    r")",
                    cleaned_part,
                    re.I,
                ):
                    title = cleaned_part.strip()
                    break
    
    # Pattern 2: After "Company - X" or "Client - X" — the next role-like text
    if not title and raw_seg:
        for m in re.finditer(r"(?:company|client)\s*-\s*[^|]*?\|", raw_seg, re.I):
            after = raw_seg[m.end():]
            snippet = after[:200]
            snippet = re.sub(r"^[\s\|]*", "", snippet)
            role_match = re.match(
                r"(?:\b(?:Project|Sr\.?|Senior|Junior|Lead|Principal|Staff)\s+(?:Planning\s+)?Consultant"
                r"|\b(?:Sr\.?|Senior|Junior|Lead|Principal|Staff)\s+(?:Data|Business|Systems?|Software)\s+(?:Analyst|Engineer|Scientist|Developer|Architect)"
                r"|\b(?:Data|Business|Systems?|Software)\s+(?:Analyst|Engineer|Scientist|Developer|Architect)"
                r"|\b(?:Programmer|Developer|Engineer|Analyst|Consultant|Manager|Director|Lead|Coordinator|Specialist|Administrator|Officer|Executive|Associate|Intern|Trainee)"
                r"|\b(?:Project|Product|Program|Technical)\s+(?:Manager|Lead|Coordinator|Owner)"
                r"|\b(?:Assistant|Associate|Deputy|Vice)\s+(?:Manager|Director|President|Chair)"
                r")",
                snippet,
                re.I,
            )
            if role_match:
                title = role_match.group(0)
                break
    
    # Pattern 3: First non-bullet line in raw_seg that looks like a title
    if not title and raw_seg:
        lines = [ln.strip() for ln in raw_seg.splitlines() if ln.strip()]
        for ln in lines:
            if re.match(r"^[\s\*\-\u2022\-\u25cf\u2023\u25e6]", ln):
                continue
            if re.match(r"^(developed|managed|led|created|designed|implemented|analyzed|built|maintained|responsible|assisted|collaborated|coordinated|executed|delivered|improved|optimized|reduced|increased|achieved)\b", ln, re.I):
                continue
            if ln.count("|") > 2 or ln.count(",") > 4:
                continue
            if len(ln) > 120:
                continue
            if re.search(
                r"\b(?:Project|Sr\.?|Senior|Junior|Lead|Principal|Staff)\s+(?:Planning\s+)?Consultant"
                r"|\b(?:Sr\.?|Senior|Junior|Lead|Principal|Staff)\s+(?:Data|Business|Systems?|Software)\s+(?:Analyst|Engineer|Scientist|Developer|Architect)"
                r"|\b(?:Data|Business|Systems?|Software)\s+(?:Analyst|Engineer|Scientist|Developer|Architect)"
                r"|\b(?:Programmer|Developer|Engineer|Analyst|Consultant|Manager|Director|Lead|Coordinator|Specialist|Administrator|Officer|Executive|Associate|Intern|Trainee)"
                r"|\b(?:Project|Product|Program|Technical)\s+(?:Manager|Lead|Coordinator|Owner)"
                r"|\b(?:Assistant|Associate|Deputy|Vice)\s+(?:Manager|Director|President|Chair)",
                ln,
                re.I,
            ):
                title = ln.strip()
                break
    
    if title:
        return _word_cut(title, 140)

    # Fallback: role-pattern search over the full segment (the role often sits
    # AFTER the skills list in flattened table rows, where the keyword split
    # below would discard it). Extend a match through a trailing "(...)" group.
    _ROLE_RE = (
        r"(?:Senior|Junior|Lead|Principal|Staff|Sr\.?|Assistant|Associate|Deputy|Vice)\s+"
        r"(?:[A-Z][A-Za-z&]*\s+){0,3}(?:Manager|Analyst|Engineer|Developer|Consultant|Scientist|Architect|Specialist|Administrator|Programmer|Director|Coordinator|Executive|Lead)"
        r"|(?:Data|Business|Software|Systems?|Project|Product|Program|Technical)\s+"
        r"(?:[A-Z][A-Za-z&]*\s+){0,2}(?:Analyst|Engineer|Developer|Consultant|Scientist|Architect|Manager|Programmer|Director|Coordinator|Owner|Lead)"
        r"|\b(?:Programmer|Developer|Engineer|Analyst|Consultant|Manager|Director|Coordinator|Specialist|Administrator)\b"
    )
    _role_m = re.search(_ROLE_RE, seg)
    if _role_m:
        _role_txt = _role_m.group(0).strip()
        _tail = seg[_role_m.end():].lstrip()
        if _tail.startswith("("):
            _depth, _pos = 0, 0
            for _pos, _ch in enumerate(_tail):
                if _ch == "(":
                    _depth += 1
                elif _ch == ")":
                    _depth -= 1
                    if _depth == 0:
                        break
            if _depth == 0:
                _role_txt = (_role_txt + " " + _tail[: _pos + 1]).strip()
        if 3 <= len(_role_txt) <= 120:
            return _word_cut(_role_txt, 140)

    # Fallback: original line-by-line check on cleaned text
    # Split into candidate lines and pick the first that looks like a job title.
    # A job title: not a bullet, not a date line, not an action-verb sentence fragment.
    lines = [ln.strip() for ln in seg.splitlines() if ln.strip()]
    for ln in lines:
        # Skip bullet points
        if re.match(r"^[\s\*\-\u2022\-\u25cf\u2023\u25e6]", ln):
            continue
        # Skip common action-verb starts (resume bullet style)
        if re.match(r"^(developed|managed|led|created|designed|implemented|analyzed|built|maintained|responsible|assisted|collaborated|coordinated|executed|delivered|improved|optimized|reduced|increased|achieved)\b", ln, re.I):
            continue
        # Skip very long lines (these are descriptions)
        if len(ln) > 120:
            continue
        # Skip pure skill/keyword lists (many | or ,)
        if ln.count("|") > 2 or ln.count(",") > 4:
            continue
        # This looks like a title
        if len(ln) >= 3:
            return _word_cut(ln, 140)
    
    # Fallback: original seg logic
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
        return _word_truncate(org, 80)
    if len(seg) >= 3:
        return _word_cut(seg, 140)
    if org:
        return _word_truncate(org, 80)
    fallback = re.sub(r"^[^A-Za-z]+", "", cleaned).strip()
    return _word_cut(fallback, 140) if fallback else None
def _word_truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return (text[: cut if cut > 10 else limit].rstrip() + "…")


def _word_cut(text: str, limit: int) -> str:
    """Word-boundary cut WITHOUT an ellipsis marker, for title display.

    The timeline UI wraps titles over multiple lines and must not show "…".
    The 140-char cap only guards against table-merged junk floods; normal
    titles pass through untouched.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 10 else limit].rstrip()


def _entry_org(text: str) -> str | None:
    import re

    _, pipe_org = _pipe_header(text)
    if pipe_org:
        return pipe_org
    _, comma_org = _comma_header(text)
    if comma_org:
        return _word_truncate(comma_org, 60)
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
        # Positional fallback: only accept short, date-free lines. Long
        # bullet fragments must never become the "organization" (they would
        # then also corrupt the title via org-subtraction). Bare single
        # tokens (e.g. a standalone "Noida" location line) are rejected
        # unless they carry an explicit organization suffix.
        for cand in meaningful[1:3]:
            c = cand.strip(" \t-–—:,")
            if len(c) <= 60 and not re.search(r"\d{4}|present|current", c, re.I) \
                    and not re.match(r"^[*•\-–]", c):
                if re.match(
                    r"^(perform|managed|led|created|designed|implemented|analyzed|built|"
                    r"maintained|responsible|assisted|collaborated|coordinated|executed|"
                    r"delivered|improved|optimized|reduced|increased|achieved|resolved|"
                    r"handled|processed|conducted)\b", c, re.I,
                ):
                    continue  # action-verb bullet fragment, not an employer
                if " " not in c and not re.search(
                        r"\b(Inc|LLC|LLP|Ltd|Pvt|Corp|GmbH|Pty|Bank|Labs|Works)\b", c, re.I):
                    continue
                return c[:100]
        return None
    return None


def _entry_location(text: str) -> str | None:
    import re

    m = re.search(r"(?:location|place)\s*:\s*([A-Za-z][A-Za-z ,\-]{1,40})", text or "", re.I)
    if m:
        return m.group(1).strip()
    lines = _entry_lines(text)
    if lines:
        last = lines[-1]
        # A trailing section header ("Experience") is not a location.
        if re.match(
            r"^(experience|education|skills|projects|summary|certifications?|"
            r"awards?|publications?|languages?|volunteering|interests|references?)\s*$",
            last, re.I,
        ):
            return None
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
