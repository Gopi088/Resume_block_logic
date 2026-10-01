#!/usr/bin/env python3
"""Print a resume as clean structured JSON (Resume-Matcher schema) using THIS
pipeline only (B0->B9; optional B5 LLM verification for escalations).

Usage:
    venv/bin/python print_resume_json.py <resume-path> [--model MODEL_PATH] [--compact]

Default output: one structured resume and a concise source-grounded evaluation. Entry title/company/years splitting is a simple
deterministic heuristic over B7 entries + B8 dates (readability layer only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-re-exec inside project venv if running with system Python lacking dependencies
_venv_py = Path(__file__).resolve().parent / "venv" / "bin" / "python"
if _venv_py.exists() and sys.executable != str(_venv_py):
    try:
        import numpy
        import sklearn
    except ImportError:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

import argparse
import json
import re

from resume_parser.source_evaluation import evaluate_extraction, source_blocks, values as values_for_output, verify_extraction
from resume_parser.resolution import complete_json
from parse_resume import build_output
from resume_parser.models import LLMConfig, SectionLabel
from resume_parser.resolution import AnthropicLLMClient, NvidiaNIMLLMClient, OpenRouterLLMClient

DEFAULT_MODEL = str(Path(__file__).resolve().parent / "model_b3_real.pkl")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+|linkedin\.com[^\s]*|github\.com[^\s]*)", re.I)
LOC_RE = re.compile(r"^[A-Za-z][A-Za-z .()-]{2,60}(?:,\s*[A-Za-z][A-Za-z .()-]{1,50}){1,2}$")
NARR_RE = re.compile(
    r"^I\s+(?:am currently working|worked|work|was|have been working)\s+as\s+(.+?)\s+at\s+(.+?)$", re.I)


def narr_parts(line):
    """Split a narrative job line ('I Worked as X at Y, dates') into parts."""
    m = NARR_RE.match(clean_bullets(line))
    if not m:
        return None
    role, rest = m.group(1).strip(), m.group(2).strip()
    dm = YEARS_RE.search(rest)
    company = re.split(r"\s*[,–—-]\s*|\s+from\s+|\s+(?=(?:19|20)\d{2}\b)", rest, flags=re.I)[0].strip().rstrip(" .")
    return role[:80], company[:60], (dm.group(0).strip(' .') if dm else "")
DATE_RE = re.compile(r"(\b(?:19|20)\d{2}\b|present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_DATE_VALUE = r"(?:(?:\d{1,2}[/-])?(?:19|20)\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,/-]+(?:19|20)\d{2})"
_DATE_LINE_RE = re.compile(rf"^\s*{_DATE_VALUE}(?:\s*(?:-|–|—|to)\s*(?:{_DATE_VALUE}|present|current))?\s*$", re.I)
YEARS_RE = re.compile(r"(\b(?:19|20)\d{2}\b|\bpresent\b|\bcurrent\b)", re.I)
BULLET_RE = re.compile(r"^\s*(?:[•●▪◦·–]\s*|[-*+]\s+|\d{1,3}[.)]\s+)")
ROLE_HINT = re.compile(r"(engineer|developer|analyst|manager|consultant|architect|lead|director|specialist|administrator|designer|intern|officer|executive|scientist|owner|master|accountant|tester|associate)", re.I)
SECTION_HEAD_RE = re.compile(
    r"^(work experience|professional experience|employment|experience|education|academic background|"
    r"technical skills|technology & skills|skills|keyskills|key skills|core skills|competencies|"
    r"summary|professional summary|profile|career summary|objective|about me|"
    r"projects|personal projects|key projects|"
    r"certifications?|licenses & certifications?|courses & certifications?|"
    r"contact|contact details|contact information|personal details|personal information|social links|"
    r"languages?|languages known|"
    r"accomplishments|achievements|awards|honors|honours|honors & awards|"
    r"hobbies|interests|extra curricular(?:\s+activities)?|activities|volunteering|references)\s*:?\s*$",
    re.I
)
SPACED_HEAD_RE = re.compile(r"^(?:[A-Z]\s*){3,}$")
PROFILE_SENT_RE = re.compile(
    r"^An?\s+(?:senior|junior|lead|principal|software|data|\w+\s+){0,3}"
    r"(engineer|developer|analyst|manager|consultant|architect|designer|specialist|professional)\b"
    r".{0,60}\b(with|having|offering|bringing)\b.{0,40}\b(year|experience|expertise|skill)",
    re.I)
DEGREE_RE = re.compile(r"(b\.?tech|m\.?tech|b\.?e\b|m\.?e\b|bca|mca|mba|b\.?sc|m\.?sc|bachelor|master|ph\.?d|doctorate|diploma|b\.?com|m\.?com)\b", re.I)
COMPANY_ROW_RE = re.compile(r"compan(?:y|ies)\s*-+\s*(.+?)(?:\*\*|\||$)", re.I)


def _cell(t):
    return clean_bullets(t).replace("|", " ").replace("*", "").strip()
ORG_RE = re.compile(r"\b(ltd|limited|inc|corp|corporation|llc|llp|pvt|bank|university|college|institute|school)\b", re.I)


def lines_of(entry):
    return [ln.strip() for ln in entry["text"].split("\n")]


def clean_bullets(text):
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # embedded images carry no content
    t = BULLET_RE.sub("", t).strip()
    if t and re.fullmatch(r"[\s|:\-]+", t):
        return ""  # markdown table scaffolding, not content
    return t

def clean_description_lines(texts):
    """Clean extracted description lines without joining separate bullets."""
    return [text for raw in texts
            if (text := re.sub(r"\s+", " ", clean_bullets(raw)).strip())]


def split_title_company(line):
    t = re.sub(r"^\(?((?:19|20)\d{2}[^|]{0,24}?(?:present|current)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*,?\s*(?:19|20)\d{2})\s*[-–—]\s*[^|]{0,24}\)?\s*:?\s*",
               "", line.strip(), flags=re.I).strip()
    line = t or line.strip()
    m = re.search(r"\bat\b\s+(.+?)(?:\s*[|,;–—-]\s*|\s+(?:\b(?:19|20)\d{2}\b|\bpresent\b|\bcurrent\b)|$)",
                  line, flags=re.I)
    at_company = m.group(1).strip() if m else ""
    for sep in (" – ", " — ", " | ", " - ", " @ ", ", "):
        if sep in line:
            left, right = line.split(sep, 1)
            left, right = left.strip(" -–|,:"), right.strip(" -–|,:")
            if not left or not right:
                continue
            if ROLE_HINT.search(left) and (len(right) < 60 or ORG_RE.search(right)) \
                    and (right[:1].isupper() or ORG_RE.search(right)):
                return left, (at_company or right)
            if not ROLE_HINT.search(left) and ROLE_HINT.search(right) \
                    and (ORG_RE.search(left) or len(left) < 50):
                seg = next((s.strip() for s in right.split("|") if ROLE_HINT.search(s)), right)
                return seg[:90], left
    if at_company:
        return line[:90], at_company
    return line.strip()[:90], ""


def _is_date_line(text):
    """Accept only date-shaped lines; reject years embedded in skills like SQL 2008."""
    cleaned = clean_bullets(text).replace("*", "").strip(" |:")
    return bool(_DATE_LINE_RE.fullmatch(cleaned))


_JOB_RANGE_RE = re.compile(
    r"(?P<raw>(?:(?:0?[1-9]|1[0-2])[/-]|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,/-]+)?"
    r"(?:19|20)\d{2}\s*(?:-|–|—|to)\s*"
    r"(?:(?:(?:0?[1-9]|1[0-2])[/-]|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,/-]+)?(?:19|20)\d{2}|present|current))",
    re.I,
)
_JOB_TITLE_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?]))\s*"
    r"(?P<title>(?:(?:Sr\.?|Senior|Assistant|Project\s+Planning|Data|Software|Business|Database)\s+){0,3}"
    r"(?:Manager|Analyst|Consultant|Developer|Programmer|Engineer|Architect|Lead|Specialist|"
    r"Administrator|Designer|Officer|Executive|Scientist|Coordinator|Owner|Tester)"
    r"(?:\s*\([^)]{1,90}\))?)",
    re.I,
)


def _plain_cell(text):
    return re.sub(r"\s+", " ", clean_bullets(text).replace("**", "")).strip(" *|\t")


def _company_from_meta(meta, date_match):
    tail = meta[date_match.end():]
    # Keep only employer metadata, stopping before client, location, or skills.
    explicit = re.search(r"\bCompany\s*[-:]\s*(.+?)(?=\s+Client\s*[-:]|\s+Technology\b|$)", tail, re.I)
    if explicit:
        company = explicit.group(1)
        company = re.sub(r"\s+(?:Ahemdabad|Ahmedabad)\b.*$", "", company, flags=re.I)
        company = re.sub(r",\s*(?:Noida|New Delhi|Gujarat|India)\b.*$", "", company, flags=re.I)
    else:
        tail = re.split(r"\bTechnology\b", tail, maxsplit=1, flags=re.I)[0]
        tail = re.split(r"\bClient\s*[-:]", tail, maxsplit=1, flags=re.I)[0]
        company = re.sub(r"\*+", "", tail).strip(" ,:-")
        company = re.sub(r",\s*(?:Noida|Delhi|New Delhi|India|Gujarat)\b.*$", "", company, flags=re.I).strip(" ,:-")
    return re.sub(r"\s+", " ", company).strip(" ,:-")[:80]


def table_job_items(entries):
    """Recover jobs from MarkItDown's paired metadata/content table cells."""
    jobs = []
    for entry in entries:
        cells = [_plain_cell(cell) for cell in entry.get("text", "").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r"[-:\s]+", cell)]
        for offset in range(0, len(cells) - 1, 2):
            meta, body = cells[offset], cells[offset + 1]
            date_matches = list(_JOB_RANGE_RE.finditer(meta))
            title_matches = list(_JOB_TITLE_RE.finditer(body))
            if not date_matches or len(date_matches) != len(title_matches):
                continue
            for n, (date_match, title_match) in enumerate(zip(date_matches, title_matches)):
                segment_end = date_matches[n + 1].start() if n + 1 < len(date_matches) else len(meta)
                segment = meta[date_match.start():segment_end]
                segment_date = _JOB_RANGE_RE.search(segment)
                title = title_match.group("title").strip()
                body_end = title_matches[n + 1].start() if n + 1 < len(title_matches) else len(body)
                description_text = body[title_match.end():body_end]
                description_text = re.sub(r"\s+\*\s+", "\n", description_text)
                description = clean_description_lines(description_text.splitlines())
                # Preserve meaningful narrative even when the source has no bullet spacing.
                if not description and description_text.strip():
                    description = [description_text.strip()]
                jobs.append({
                    "id": len(jobs) + 1, "title": title, "company": _company_from_meta(segment, segment_date),
                    "location": "", "years": date_match.group("raw").strip(),
                    "description": description, "mlConfidence": None,
                    "entry_id": entry.get("entry_id"),
                    "line_ids": list(entry.get("line_ids", [])),
                })
    return jobs


def table_edu_items(entries):
    """Recover education degrees from table rows (e.g. date | degree institution)."""
    items = []
    for entry in entries:
        for line in lines_of(entry):
            if "|" not in line:
                continue
            cells = [_cell(c) for c in line.split("|")]
            cells = [c for c in cells if c and not re.fullmatch(r"[-:\s]+", c)]
            if len(cells) >= 2:
                date_m = re.search(r"\b(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b|\b(?:19|20)\d{2}\b", cells[0])
                years = date_m.group(0) if date_m else ""
                deg_inst = cells[1]
                parts = re.split(r"\s{2,}|\s{2,}-\s{2,}", deg_inst)
                if len(parts) >= 2:
                    deg, inst = parts[0].strip(), " ".join(parts[1:]).strip()
                elif DEGREE_RE.search(deg_inst):
                    deg, inst = deg_inst.strip(), ""
                else:
                    deg, inst = "", deg_inst.strip()
                if deg or inst:
                    items.append({
                        "id": len(items) + 1,
                        "institution": inst,
                        "degree": deg,
                        "years": years,
                        "description": None,
                        "mlConfidence": None,
                        "entry_id": entry.get("entry_id"),
                        "line_ids": list(entry.get("line_ids", [])),
                    })
    return items

def job_years(entries_group, date_by_entry):
    """Legacy entry-level years (kept for compatibility); prefer
    job_years_from_lines for line-grouped jobs."""
    for e in entries_group:
        pr = (date_by_entry.get(e["entry_id"], {}) or {}).get("primary_range") or {}
        if pr.get("raw"):
            return pr["raw"].strip()
    for e in entries_group:
        for ln in lines_of(e):
            if _is_date_line(ln):
                return clean_bullets(ln).strip(" *|:")
    return ""


def _is_md_heading(text):
    """Markdown heading lines are structural headings, never job content."""
    return clean_bullets(text).lstrip().startswith("#")


def group_jobs(entries, date_by_entry):
    """Flatten experience lines in doc order; a new job starts at a title line:
    short, non-bullet, non-heading, with a role word plus a company/date cue
    in the same or following two lines."""
    flat = [(ln.strip(), e["entry_id"]) for e in entries for ln in lines_of(e)]
    flat = [(t, eid) for t, eid in flat if t]
    jobs, cur = [], []
    for i, (t, eid) in enumerate(flat):
        c = clean_bullets(t)
        cell = _cell(t)
        window = " ".join(x[0] for x in flat[i:i + 3])
        is_narr = narr_parts(t) is not None
        is_company_row = COMPANY_ROW_RE.search(cell) is not None
        is_employer_date = re.match(r"^.+?\s*:\s*(?:[A-Za-z]{3,9}[- ]*)?(?:19|20)\d{2}", cell) is not None
        is_title = (
            not BULLET_RE.match(t) and len(cell) < 200
            and not _is_md_heading(t)
            and not SECTION_HEAD_RE.match(c)
            and not PROFILE_SENT_RE.match(cell)
            and (is_narr or is_company_row or (
                ROLE_HINT.search(cell)
                and not re.match(r"^(?:project|client|responsibilities)\s*[:–-]", c, re.I)
                and not re.search(r"\bmaster\s+data\b", c, re.I)
                and (any(sep in t for sep in (" – ", " — ", " | ", " @ ", " - "))
                     or ORG_RE.search(window) or YEARS_RE.search(window))))
            and not (t.lstrip().startswith(("I ", "I'", "We ")) and not is_narr)
        )
        # Date-led job entries (mirrors B7: a date range starts a new entry).
        # Roles whose header line carries no role word still begin a new job.
        is_date_boundary = (
            not BULLET_RE.match(t) and not _is_md_heading(t)
            and not SECTION_HEAD_RE.match(c)
            and _JOB_RANGE_RE.search(t) is not None
        )
        # A role/company after a date-led header completes that same job.
        # A later title after narrative still starts the next role as before.
        if is_title and cur and any(_JOB_RANGE_RE.search(row) for row, _ in cur) and not any(BULLET_RE.match(row) for row, _ in cur):
            is_title = False
        prev_is_dateless_role_title = False
        if is_date_boundary and cur:
            # Scan back to the last date/bullet: a date line belongs with a
            # preceding role title (possibly spread over title + company
            # lines) that has not yet received its date range.
            tail_role = False
            for prev_text, _ in reversed(cur):
                if _JOB_RANGE_RE.search(prev_text) or BULLET_RE.match(prev_text):
                    break
                prev_clean = clean_bullets(prev_text)
                if (not _is_md_heading(prev_text)
                        and not SECTION_HEAD_RE.match(prev_clean)
                        and ROLE_HINT.search(_cell(prev_text)) is not None):
                    tail_role = True
            prev_is_dateless_role_title = tail_role
        if (is_title or is_date_boundary or is_employer_date) and not prev_is_dateless_role_title \
                and cur and any(
                not SECTION_HEAD_RE.match(clean_bullets(t)) and not _is_md_heading(t) for t, _ in cur):
            jobs.append(cur)
            cur = [(t, eid)]
        else:
            cur.append((t, eid))
    if cur:
        jobs.append(cur)
    # drop a leading contact-preamble group (contact markers, no job content)
    if jobs and _is_contact_preamble([t for t, _ in jobs[0]]):
        jobs = jobs[1:]
    return jobs


CONTACT_LINE_RE = re.compile(r"^(full name|phone|email|linkedin|place|location|address|mobile)\s*:", re.I)


def _is_contact_preamble(texts):
    if not texts:
        return False
    hits = sum(1 for t in texts
               if EMAIL_RE.search(t) or (PHONE_RE.search(t) and not _JOB_RANGE_RE.search(t)
                                          and len(re.sub(r"\D", "", PHONE_RE.search(t).group())) >= 10) or URL_RE.search(t)
               or CONTACT_LINE_RE.match(clean_bullets(t)))
    return hits / max(1, len(texts)) > 0.4


def job_years_from_lines(lines, entry_ids, date_by_entry):
    for t in lines:
        c = clean_bullets(t)
        if SECTION_HEAD_RE.match(c) or _is_md_heading(t):
            continue
        if _is_date_line(t) and "@" not in t:
            return clean_bullets(t).strip(" *|:")
    # Inline dates (e.g. "Technical Lead | Coforge Ltd. | Nov 2025 – Present"):
    # reuse B8's range parser instead of requiring a standalone date line.
    for t in lines:
        if "@" in t or SECTION_HEAD_RE.match(clean_bullets(t)) or _is_md_heading(t):
            continue
        m = _JOB_RANGE_RE.search(t)
        if m:
            return m.group("raw").strip()
    for eid in dict.fromkeys(entry_ids):
        pr = (date_by_entry.get(eid, {}) or {}).get("primary_range") or {}
        if pr.get("raw") and "@" not in pr["raw"]:
            return pr["raw"].strip()
    return ""


def _looks_company(c):
    return bool(c) and 3 <= len(c) <= 60 and (
        ORG_RE.search(c) or (len(c) <= 40 and not YEARS_RE.search(c)))


def job_item(i, group, date_by_entry):
    texts = [t for t, _ in group]
    eids = [eid for _, eid in group]
    header = next((t for t in texts
                   if not SECTION_HEAD_RE.match(clean_bullets(t))
                   and not _is_md_heading(t)
                   and not SPACED_HEAD_RE.match(clean_bullets(t))
                   and not _is_date_line(t)
                   and not _JOB_RANGE_RE.fullmatch(clean_bullets(t).strip())), "")
    if not header:
        return None
    employer = re.match(r"^(.+?)\s*:\s*((?:[A-Za-z]{3,9}[- ]*)?(?:19|20)\d{2}.*)$", clean_bullets(header).lstrip("• "))
    if employer:
        return {"id": i, "company": employer.group(1).strip(), "years": employer.group(2).strip(),
                "description": clean_description_lines([t for t in texts if t != header and not SECTION_HEAD_RE.match(clean_bullets(t))]),
                "line_ids": [eid.rsplit(":", 1)[-1] for eid in eids]}
    np = narr_parts(header) if header else None
    if np:
        title, company, narr_years = np
    else:
        title, company = split_title_company(header)
        narr_years = ""
    if not _looks_company(company) and header:
        hsegs = [x.strip() for x in re.split(r"\s*\|\s*|\s+[–—-]\s*|,\s*|;\s*", header) if x.strip()]
        cand = next((x for x in hsegs
                     if ORG_RE.search(x) and x != title
                     and not YEARS_RE.search(x.replace("Present", "").replace("present", ""))), "")
        cand = re.sub(r"^\s*(present|current)\s*:\s*", "", cand, flags=re.I).strip()
        if cand:
            company = cand[:60]
    if not _looks_company(company):
        orgline = next((t for t in texts if t != header and ORG_RE.search(t)), "")
        if orgline:
            left = re.split(r"\s+[–—-]\s*", clean_bullets(orgline))[0]
            company = left[:60] if ORG_RE.search(left) else ""
        else:
            company = ""
    # drop role-less dateless fragments (name-only / heading-only groups)
    if not narr_parts(header) and not ROLE_HINT.search(title or "") \
            and not any(YEARS_RE.search(t) for t in texts):
        return None
    desc = clean_description_lines(
        [t for t in texts
         if t != header
         and not SECTION_HEAD_RE.match(clean_bullets(t))
         and not _is_date_line(t)
         and not _JOB_RANGE_RE.fullmatch(clean_bullets(t).strip())]
    )
    # drop a leading standalone date line (already captured in years)
    if desc and YEARS_RE.search(desc[0]) and len(desc[0]) < 60:
        desc = desc[1:]
    title = re.sub(r"\s+", " ", title).strip()
    company = re.sub(r"\s+", " ", company).strip()
    title = re.sub(r"\*+", "", title).strip(" |-")
    company = re.sub(r"\*+", "", company).strip(" |-")
    return {"id": i, "title": title, "company": company, "location": "",
            "years": narr_years or job_years_from_lines(texts, eids, date_by_entry),
            "description": desc}


def skill_items(entries, allowed_source=None):
    items = []
    for entry in entries:
        for raw in lines_of(entry):
            cells = raw.split("|") if "|" in raw else [raw]
            for cell in cells:
                text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cell)
                if re.fullmatch(r"\s*[-:\s]*\s*", text):
                    continue
                # MarkItDown leaves table bullets inline. Split those into
                # separate values before normalizing Markdown emphasis.
                pieces = re.split(r"\s+(?=[*•●▪◦·]\s+)", text)
                for piece in pieces:
                    value = clean_bullets(piece)
                    value = re.sub(r"\*{1,2}", "", value)
                    value = re.sub(r"\s+", " ", value).strip(" |:")
                    if not value or value.startswith("#") or SECTION_HEAD_RE.match(value):
                        continue
                    if (len(value) < 2 and not (value.upper() == "R" and entries is not None)) or re.fullmatch(r"[-:\s]+", value):
                        continue
                    if allowed_source is not None:
                        from collections import Counter
                        from resume_parser.source_evaluation import tokens
                        if Counter(tokens(value)) - Counter(tokens(allowed_source)):
                            continue
                    items.append(value)
    return items


def section_text(entries, skip_headings=True):
    parts = []
    for e in entries:
        for ln in lines_of(e):
            t = ln.strip()
            if not t:
                continue
            if skip_headings and (t.startswith("#") or (t.isupper() and len(t.split()) <= 4)):
                continue
            parts.append(clean_bullets(ln))
    return " ".join(re.sub(r"\s+", " ", p) for p in parts)



def semantic_entries(out, fallback_entries):
    """Project B3's ML-labeled semantic spans into the output schema.

    Explicit section headings have already shaped B3 span boundaries and
    model context. The schema layer preserves each span's classifier result
    instead of overriding it with a heading remembered from earlier text.
    """
    if out.get("lines"):
        from resume_parser.structure_projection import bounded_entries
        return bounded_entries(out)
    classifications = out.get("classification", {}).get("classifications", [])
    finals = {x["block_id"]: x for x in out.get("final_sections", {}).get("sections", [])}
    rows = []
    for classification in classifications:
        bid = classification["block_id"]
        final = finals.get(bid, {})
        spans = classification.get("semantic_spans", [])
        for n, span in enumerate(spans):
            ml_section = span.get("section", classification.get("predicted_section", "unknown"))
            section = ml_section
            section_source = "model"
            span_resolutions = final.get("span_resolutions", [])
            span_resolution = next((r for r in span_resolutions
                                    if r.get("start_line_id") == span.get("start_line_id")
                                    and r.get("end_line_id") == span.get("end_line_id")), None)
            if len(spans) == 1 and section_source == "model":
                section = final.get("final_section", section)
            span_agrees = bool(span_resolution and span_resolution.get("section") == section)
            if span_resolution and not span_agrees and section_source == "model":
                section = span.get("section", section)
            rows.append({
                "entry_id": f"{bid}:S{n:04d}", "block_id": bid,
                "section": section, "ml_section": ml_section,
                "section_source": section_source,
                "trusted": bool(final.get("trusted")) and len(spans) == 1,
                "source": final.get("source", "ml_unresolved"),
                "text": span.get("text", ""),
                "line_ids": span.get("source_line_ids", []),
                "ml_confidence": span.get("confidence") if section == ml_section else None,
                "start_line_id": span.get("start_line_id"),
                "end_line_id": span.get("end_line_id"),
            })
    return rows or fallback_entries



def _span_resolution(span, resolution):
    if not resolution:
        return None
    return next((r for r in resolution.get("span_resolutions", [])
                 if r.get("start_line_id") == span.get("start_line_id")
                 and r.get("end_line_id") == span.get("end_line_id")), None)


def _span_llm_verification(span, resolution, ml_section):
    span_resolution = _span_resolution(span, resolution)
    if span_resolution:
        agrees = span_resolution.get("section") == ml_section
        return {"used": True, "prediction": span_resolution.get("section"),
                "confidence": span_resolution.get("confidence", 0.0),
                "agreementWithML": agrees, "resolved": True,
                "reason": span_resolution.get("reason", "")}
    if resolution:
        prediction = resolution.get("resolved_section")
        return {"used": True, "prediction": prediction,
                "confidence": resolution.get("confidence", 0.0),
                "agreementWithML": prediction == ml_section,
                "resolved": bool(resolution.get("resolved")),
                "reason": resolution.get("reason", "")}
    return {"used": False}


def _span_verification_status(span, resolution, ml_section, verdict):
    if _span_resolution(span, resolution):
        return "VERIFIED_BY_LLM" if _span_resolution(span, resolution).get("section") == ml_section else "CONFLICT"
    if resolution and resolution.get("resolved"):
        return "VERIFIED_BY_LLM" if resolution.get("resolved_section") == ml_section else "CONFLICT"
    return "VERIFIED_BY_B4" if verdict.get("verdict") == "accept" else "UNRESOLVED"

def _section_meta_with_detection(base_meta, detections, boundaries, llm_statistics, line_coverage, raw_confidence_summary):
    labels_by_key = {
        "personalInfo": {"contact", "personal_info"}, "summary": {"summary", "professional_summary"},
        "workExperience": {"experience"}, "education": {"education"},
        "personalProjects": {"projects"},
        "additional": {"skills", "technical_skills", "certifications", "awards", "languages"},
        "professional_summary": {"summary"}, "other": {"other"}, "unknown": {"unknown"},
    }
    result = []
    for item in base_meta:
        labels = labels_by_key.get(item["key"], set())
        rows = [d for d in detections if d["section"] in labels]
        if item["key"] == "summary":
            rows = [d for d in rows if d.get("semanticName") != "PROFESSIONAL_SUMMARY"]
        elif item["key"] == "professional_summary":
            rows = [d for d in detections if d.get("semanticName") == "PROFESSIONAL_SUMMARY"]
        elif item["key"] == "personalInfo":
            rows = [d for d in rows if d.get("semanticName") == "PERSONAL_INFO" or d["section"] in {"contact", "personal_info"}]
        if rows:
            # This is the weakest contributing raw model probability, not an
            # average or a fabricated combined confidence.
            scored_rows = [d["mlConfidence"] for d in rows if isinstance(d.get("mlConfidence"), (int, float))]
            confidence = min(scored_rows) if scored_rows else None
            statuses = {d["verificationStatus"] for d in rows}
            status = (next(iter(statuses)) if len(statuses) == 1
                      else "MIXED_VERIFICATION_STATUS")
            relevant_boundaries = [b for b in boundaries
                                   if b["leftSection"] in labels or b["rightSection"] in labels]
            llm_rows = [d["llmVerification"] for d in rows if d.get("llmVerification", {}).get("used")]
            item["detection"] = {
                "confidence": confidence,
                "confidencePercent": round(confidence * 100) if confidence is not None else None,
                "confidenceLevel": ("HIGH" if confidence >= 0.75 else "MEDIUM" if confidence >= 0.5 else "LOW") if confidence is not None else "UNAVAILABLE",
                "confidenceMethod": "minimum_raw_ML_predict_proba_across_contributing_spans",
                "calibrationStatus": "RAW_UNCALIBRATED_NO_GOLD_VALIDATION_SET",
                "classifier": "ML",
                "verificationStatus": status,
                "spanDetections": rows,
                "boundaries": relevant_boundaries,
                "llmVerifications": llm_rows,
                "llmStatistics": llm_statistics,
                "lineCoverage": line_coverage,
                "rawConfidenceSummary": raw_confidence_summary,
            }
        result.append(item)
    return result


def _public_section_meta(section_meta):
    """Strip document internals while keeping public confidence/verification metadata."""
    output = []
    for section in section_meta:
        item = {key: value for key, value in section.items() if key != "detection"}
        detection = section.get("detection")
        if isinstance(detection, dict):
            public = {key: detection.get(key) for key in (
                "confidence", "confidencePercent", "confidenceLevel",
                "confidenceMethod", "calibrationStatus", "classifier",
                "verificationStatus", "llmStatistics", "lineCoverage",
                "rawConfidenceSummary") if key in detection}
            public["spanDetections"] = [
                {key: span.get(key) for key in (
                    "section", "classifierSection", "sectionSource", "mlConfidence",
                    "cosineSimilarity", "matchesSchema", "schemaSection",
                    "semanticName", "b4Verdict", "b4Reasons",
                    "llmVerification", "verificationStatus") if key in span}
                for span in detection.get("spanDetections", [])]
            public["boundaries"] = [
                {key: boundary.get(key) for key in (
                    "leftSection", "rightSection", "boundaryConfidence") if key in boundary}
                for boundary in detection.get("boundaries", [])]
            item["detection"] = public
        output.append(item)
    return output


def _review_status_for_lines(line_ids, detections, final_sections, fallback_labels=None):
    line_ids = set(line_ids)
    contributing = [d for d in detections if line_ids.intersection(d.get("lineIds", []))]
    if not contributing and fallback_labels:
        labels = {fallback_labels} if isinstance(fallback_labels, str) else set(fallback_labels)
        contributing = [d for d in detections if d.get("section") in labels]
    # Classification review is not content verification.
    return "NOT_VERIFIED"


def _section_confidence(detections, labels):
    values = [d["mlConfidence"] for d in detections if d["section"] in labels
              and isinstance(d.get("mlConfidence"), (int, float))]
    return round(min(values), 4) if values else None


def _attach_confidence(item, detections, labels):
    """Put the weakest contributing raw ML probability on the content block."""
    item["confidence"] = _section_confidence(detections, labels)
    return item


def structured_resume(out):
    """Existing B3/B7/B8 projection, bounded by the original resume headings."""
    from resume_parser.structure_projection import project_items
    entries = semantic_entries(out, out.get("entries", {}).get("entries", []))
    by_sec = {}
    for entry in entries:
        by_sec.setdefault(entry["section"], []).append(entry)
    dates = {d["entry_id"]: d for d in out.get("entry_dates", {}).get("entry_dates", [])}
    resume = {}
    contact = [clean_bullets(t) for e in by_sec.get("contact", []) for t in lines_of(e)]
    if contact:
        from resume_parser.section_normalization import personal_fields
        info = personal_fields(contact)
        if info:
            resume["personalInfo"] = info
    summary = section_text(by_sec.get("summary", []))
    if summary:
        resume["summary"] = {"text": summary}
    exp = by_sec.get("experience", [])
    work = table_job_items(exp)
    if not work:
        work = [job for i, group in enumerate(group_jobs(exp, dates))
                if (job := job_item(i + 1, group, dates)) is not None]
    if work:
        resume["workExperience"] = work
    projects = project_items(by_sec.get("projects", []), clean_bullets)
    if projects:
        resume["personalProjects"] = projects
    edu_entries = by_sec.get("education", [])
    education = table_edu_items(edu_entries)
    if not education:
        current = None
        for entry in edu_entries:
            for raw in lines_of(entry):
                text = clean_bullets(raw)
                if DEGREE_RE.search(text):
                    current = {"description": []}
                    education.append(current)
                    year = re.search(r"(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}", text)
                    body = text[:year.start()].strip() if year else text
                    parts = re.split(r"\s+from\s+", body, maxsplit=1, flags=re.I)
                    current["degree"] = parts[0]
                    if len(parts) > 1:
                        current["institution"] = parts[1]
                    if year:
                        current["years"] = year.group()
                elif current is not None:
                    current["description"].append(text)
                elif text:
                    current = {"description": [text]}
                    education.append(current)
    if education:
        resume["education"] = education
    additional = {}
    for label, field in (("skills", "technicalSkills"), ("certifications", "certificationsTraining"),
                         ("languages", "languages"), ("awards", "awards")):
        items = skill_items(by_sec.get(label, []))
        if items:
            additional[field] = items
    if additional:
        resume["additional"] = additional
    known = {"contact", "summary", "experience", "projects", "education", "skills", "certifications", "languages", "awards"}
    custom = {key: {"text": "\n".join(e["text"] for e in group)} for key, group in by_sec.items() if key not in known}
    if custom:
        resume["customSections"] = custom
    detections = []
    for cls in out.get("classification", {}).get("classifications", []):
        for span in cls.get("semantic_spans", []):
            detections.append({"section": span["section"], "mlConfidence": span.get("confidence"),
                               "lineIds": span.get("source_line_ids", [])})
    return resume, detections


def main(argv=None):
    ap = argparse.ArgumentParser(description="Resume -> clean structured JSON with optional B5 verification")
    ap.add_argument("resume")
    ap.add_argument("--model", default=DEFAULT_MODEL if Path(DEFAULT_MODEL).exists() else None)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--json", action="store_true", help="Compatibility flag; clean JSON is always printed")
    ap.add_argument("--llm", choices=["anthropic", "nvidia", "openrouter"], default=None,
                    help="Verify spans with Anthropic, NVIDIA NIM, or OpenRouter")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--verify-all", action="store_true", default=False,
                    help="Compatibility flag; every extracted real block is now independently reviewed")
    ap.add_argument("--escalated-only", action="store_true", default=False,
                    help="Compatibility flag; every extracted real block is independently reviewed")
    ap.add_argument("--include-meta", action="store_true", default=False,
                    help="Compatibility flag; parser metadata remains internal")
    ap.add_argument("--judge", choices=["anthropic", "nvidia", "openrouter"], default=None,
                    help="Judge every corresponding original/output section")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--llm-timeout", type=float, default=30.0,
                    help="Timeout in seconds for each focused content verification request")
    a = ap.parse_args(argv)
    if not a.llm:
        if os.environ.get("NVIDIA_API_KEY"):
            a.llm = "nvidia"
        elif os.environ.get("OPENROUTER_API_KEY"):
            a.llm = "openrouter"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            a.llm = "anthropic"

    if a.llm and not a.llm_model:
        if a.llm == "nvidia":
            a.llm_model = NvidiaNIMLLMClient.DEFAULT_MODEL
        elif a.llm == "openrouter":
            a.llm_model = OpenRouterLLMClient.DEFAULT_MODEL
        elif a.llm == "anthropic":
            a.llm_model = LLMConfig().model

    if not a.model or not Path(a.model).is_file():
        sys.stdout.write(json.dumps({
            "ok": False,
            "error": "B3 section-classification model is unavailable. Provide an existing model with --model.",
            "model": a.model,
        }, indent=2) + "\n")
        return 2
    try:
        out = build_output(a.resume, include_eval=False, include_text=True,
                           blocks_only=False, model_path=a.model,
                           llm_provider=None, llm_model=a.llm_model,
                           llm_verify_all=False)
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)[:300]}, indent=2) + "\n")
        return 1
    if out.get("classification_error"):
        sys.stdout.write(json.dumps({"ok": False, **out}, indent=2) + "\n")
        return 1

    from resume_parser.structure_projection import public_resume
    from resume_parser.source_evaluation import concise_evaluation
    resume, detections = structured_resume(out)
    from resume_parser.source_evaluation import supplement_model_probabilities
    detections = supplement_model_probabilities(out, detections, a.model)
    from resume_parser.section_normalization import normalize_resume
    resume = normalize_resume(resume)
    evaluation = evaluate_extraction(out, resume, detections)
    provider = a.judge or a.llm or "openrouter"
    try:
        model = a.judge_model if a.judge else a.llm_model
        config = LLMConfig(model=model or LLMConfig().model,
                           timeout_seconds=a.llm_timeout, max_tokens=1200,
                           transport_retries=0)
        client = {"nvidia": NvidiaNIMLLMClient, "openrouter": OpenRouterLLMClient,
                  "anthropic": AnthropicLLMClient}[provider](config)
    except Exception as exc:
        verify_extraction(evaluation, None, unavailable_reason=str(exc)[:300])
    else:
        verify_extraction(evaluation, client)
    output = {"resume": public_resume(resume), "evaluation": concise_evaluation(evaluation)}
    sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=None if a.compact else 2) + "\n")
    sys.stderr.write("Evaluation summary: " + json.dumps(output["evaluation"]["evaluationSummary"], ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
