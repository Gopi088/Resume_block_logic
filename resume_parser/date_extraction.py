"""B8: Date / Event Extraction.

B8 answers: "What dates belong to each entry?" For every B7 entry it extracts
date ranges and single dates from the entry text, in text order, with
character-level provenance into the entry text.

What B8 does (and does NOT do):
- Parses resume date formats deterministically: month-name dates
  ("Jan 2019", "March 2020", "Mar."), numeric ("04/2024", "03/2022"),
  bare years ("2018"), and ongoing markers ("Present", "Current", "Now",
  "Till date") as RANGE ENDS. A lone Present is not a date.
- Validates: month 1-12, day 1-31, year 1900-2100. Invalid matches are
  rejected, never coerced (e.g. "13/2020" is not a date).
- Ranges are claimed first; singles are found only outside range spans
  (space-masked, offsets stable), so "04/2024 - Present" yields ONE range,
  not a range plus stray singles.
- Selects primary_range: the first range (the entry-header date by B7
  construction), else the first single as a point range, else None.
- Records reference_date for B9's use with Present. It does not affect
  extraction; ongoing-ness is lexical.
- Does NOT order entries chronologically, compute durations, detect gaps,
  or associate dates across entries (B9). Does NOT use ML or LLM.

Granularity: each endpoint carries year + optional month/day. Mixed
granularity ranges ("Jan 2019 - 2021") are preserved as-is.

Shared with B7: entries.py reuses extract_text_dates for entry-boundary
detection, so a boundary fires exactly when B8 extracts a range (single
source of truth for date patterns; B7 never interprets the dates itself).
"""

from __future__ import annotations

import re
from datetime import date

from .models import (
    DateExtractionPolicy,
    DateExtractionResult,
    DateGranularity,
    DateRange,
    EntryDates,
    EntrySegmentationResult,
    IntegrityReport,
    ParsedDate,
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = r"(?:jan(?:uary)?\.?|feb(?:ruary)?\.?|mar(?:ch)?\.?|apr(?:il)?\.?|may\.?|jun(?:e)?\.?|jul(?:y)?\.?|aug(?:ust)?\.?|sep(?:t(?:ember)?)?\.?|oct(?:ober)?\.?|nov(?:ember)?\.?|dec(?:ember)?\.?)"
_YEAR_RE = r"(?:19|20)\d{2}"
_PRESENT_RE = r"(?:present|current|now|ongoing|till\s+date|till\s+now|to\s+date)"
_SEP_RE = r"(?:-|–|—|to)"

MIN_YEAR, MAX_YEAR = 1900, 2100

# Range patterns as (kind, regex); most-specific first. Endpoint spans are
# derived from the named groups (no text searching), so hyphens inside words
# ("full-time") can never corrupt the split.
_RANGE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Jan 2019 - Present | March 2020 to Dec 2021 | 12 Jan 2020 - 15 Mar 2021
    ("mname", re.compile(
        rf"(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m1>{_MONTH_RE})\s+(?P<y1>\d{{4}})"
        rf"\s*{_SEP_RE}\s*"
        rf"(?:(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m2>{_MONTH_RE})\s+(?P<y2>\d{{4}})|(?P<p>{_PRESENT_RE}))",
        re.IGNORECASE)),
    # Jan 2019 - 2021 (month start, year end)
    ("mname", re.compile(
        rf"(?P<m1>{_MONTH_RE})\s+(?P<y1>\d{{4}})"
        rf"\s*{_SEP_RE}\s*"
        rf"(?:(?P<m2>{_MONTH_RE})\s+(?P<y2>\d{{4}})|(?P<y2b>{_YEAR_RE})|(?P<p>{_PRESENT_RE}))",
        re.IGNORECASE)),
    # 04/2022 - Present | 1/2020-3/2022 | 03/2022 - 04/2024
    ("num", re.compile(
        rf"(?P<mo1>\d{{1,2}})[/-](?P<y1>\d{{4}})"
        rf"\s*{_SEP_RE}\s*"
        rf"(?:(?P<mo2>\d{{1,2}})[/-](?P<y2>\d{{4}})|(?P<p>{_PRESENT_RE}))",
        re.IGNORECASE)),
    # 2016 - 2018 | 2015-2017 | 2020 - Present | 2019 - Mar 2021
    ("year", re.compile(
        rf"(?P<y1>{_YEAR_RE})"
        rf"\s*{_SEP_RE}\s*"
        rf"(?:(?P<m2>{_MONTH_RE})\s+(?P<y2>\d{{4}})|(?P<y2b>{_YEAR_RE})|(?P<p>{_PRESENT_RE}))",
        re.IGNORECASE)),
]

# Single-date patterns (applied only outside claimed range spans).
_SINGLE_PATTERNS = [
    re.compile(
        rf"(?:(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+)?(?P<m>{_MONTH_RE})\.?\s+(?P<y>\d{{4}})",
        re.IGNORECASE),
    re.compile(rf"(?P<mo>\d{{1,2}})[/-](?P<y>\d{{4}})"),
    # Year-first numeric month: 2020-04. Must precede bare-year below.
    re.compile(rf"(?P<y>{_YEAR_RE})[/-](?P<mo>\d{{1,2}})(?!\d)"),
    re.compile(rf"(?<!\d)(?P<y>{_YEAR_RE})(?!\d)"),
]


def _month_num(name: str) -> int | None:
    return _MONTHS.get(name.lower().rstrip("."), None)


def _valid_ymd(year: int, month: int | None, day: int | None) -> bool:
    if not MIN_YEAR <= year <= MAX_YEAR:
        return False
    if month is not None and not 1 <= month <= 12:
        return False
    if day is not None and not 1 <= day <= 31:
        return False
    return True


def _granularity(month: int | None, day: int | None) -> DateGranularity:
    if day is not None:
        return DateGranularity.DAY
    if month is not None:
        return DateGranularity.MONTH
    return DateGranularity.YEAR


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    return any(s < e2 and s2 < e for s, e in [span] for s2, e2 in claimed)


def _span_of(m: re.Match, *names: str) -> tuple[int, int] | None:
    """Union span of the named groups that participated in the match.

    Group names absent from this pattern are skipped (range patterns of the
    same kind define different group sets), as are non-participating groups.
    """
    defined = m.re.groupindex
    spans = [m.span(n) for n in names if n in defined and m.group(n) is not None]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def _make_date(text: str, span: tuple[int, int], year: int,
               month: int | None, day: int | None) -> ParsedDate | None:
    if not _valid_ymd(year, month, day):
        return None
    s0, s1 = span
    return ParsedDate(raw=text[s0:s1], year=year, month=month, day=day,
                      granularity=_granularity(month, day),
                      start_char=s0, end_char=s1)


def _parse_range_match(kind: str, m: re.Match, text: str) -> DateRange | None:
    """Build a DateRange from a range-pattern match; None if invalid.

    Endpoint spans come straight from the regex groups, so separators inside
    words ("full-time") or 'to' inside text can never corrupt the split.
    """
    g = m.groupdict()
    s0, s1 = m.span()

    def grp_int(name: str | None) -> int | None:
        if name is None or g.get(name) is None:
            return None
        try:
            return int(g[name])
        except (TypeError, ValueError):
            return None

    # --- start endpoint ---
    if kind == "mname":
        month, year, day = _month_num(g["m1"]), grp_int("y1"), grp_int("d1")
        span = _span_of(m, "d1", "m1", "y1")
    elif kind == "num":
        month, year, day = grp_int("mo1"), grp_int("y1"), None
        span = _span_of(m, "mo1", "y1")
    elif kind == "year":
        month, year, day = None, grp_int("y1"), None
        span = _span_of(m, "y1")
    else:
        return None
    if month is None and kind in ("mname", "num"):
        return None
    if year is None or span is None:
        return None
    start = _make_date(text, span, year, month, day)
    if start is None:
        return None

    # --- end endpoint (absent iff ongoing) ---
    if g.get("p") is not None:
        return DateRange(raw=text[s0:s1], start=start, end=None,
                         is_ongoing=True, start_char=s0, end_char=s1)
    if kind == "mname":
        emonth = _month_num(g["m2"]) if g.get("m2") is not None else None
        espan = _span_of(m, "d2", "m2", "y2", "y2b")
        eyear = grp_int("y2") if g.get("y2") is not None else grp_int("y2b")
        eday = grp_int("d2")
    elif kind == "num":
        emonth, eyear, eday = grp_int("mo2"), grp_int("y2"), None
        espan = _span_of(m, "mo2", "y2")
    else:  # year
        if g.get("m2") is not None:
            emonth = _month_num(g["m2"])
            eyear, eday = grp_int("y2"), grp_int("d2")
            espan = _span_of(m, "d2", "m2", "y2")
        else:
            emonth, eyear, eday = None, grp_int("y2b"), None
            espan = _span_of(m, "y2b")
    # A month is required only when month groups participated (year-only ends
    # like "2021" in "Jan 2019 - 2021" are valid without one).
    if any(g.get(n) is not None for n in ("m2", "d2", "mo2")) and emonth is None:
        return None
    if eyear is None or espan is None:
        return None
    end = _make_date(text, espan, eyear, emonth, eday)
    if end is None:
        return None
    return DateRange(raw=text[s0:s1], start=start, end=end,
                     is_ongoing=False, start_char=s0, end_char=s1)


def _parse_single_match(m: re.Match, text: str) -> ParsedDate | None:
    g = m.groupdict()
    s0, s1 = m.span()
    if g.get("m") is not None:
        month = _month_num(g["m"])
        try:
            year = int(g["y"])
        except (TypeError, ValueError):
            return None
        try:
            day = int(g["d"]) if g.get("d") is not None else None
        except (TypeError, ValueError):
            return None
    elif g.get("mo") is not None:
        try:
            month, year = int(g["mo"]), int(g["y"])
        except (TypeError, ValueError):
            return None
        day = None
    else:
        try:
            year = int(g["y"])
        except (TypeError, ValueError):
            return None
        month, day = None, None
    if not _valid_ymd(year, month, day):
        return None
    return ParsedDate(raw=text[s0:s1], year=year, month=month, day=day,
                      granularity=_granularity(month, day),
                      start_char=s0, end_char=s1)


def extract_text_dates(text: str) -> tuple[list[DateRange], list[ParsedDate]]:
    """Extract (ranges, singles) from one entry text, in text order.

    Pure function. Ranges are claimed first; singles only outside claims.
    """
    ranges: list[DateRange] = []
    claimed: list[tuple[int, int]] = []
    for kind, rx in _RANGE_PATTERNS:
        for m in rx.finditer(text):
            span = (m.start(), m.end())
            if _overlaps(span, claimed):
                continue
            parsed = _parse_range_match(kind, m, text)
            if parsed is None:
                continue
            ranges.append(parsed)
            claimed.append(span)
    masked = list(text)
    for s0, s1 in claimed:
        for i in range(s0, s1):
            masked[i] = " "
    masked_text = "".join(masked)
    singles: list[ParsedDate] = []
    for rx in _SINGLE_PATTERNS:
        for m in rx.finditer(masked_text):
            # Map back to original text for the raw span.
            raw = text[m.start():m.end()]
            probe = _parse_single_match(m, masked_text)
            if probe is None:
                continue
            if _overlaps((m.start(), m.end()), claimed):
                continue
            singles.append(probe.model_copy(update={"raw": raw}))
            claimed.append((m.start(), m.end()))
    ranges.sort(key=lambda r: (r.start_char, r.end_char))
    singles.sort(key=lambda d: (d.start_char, d.end_char))
    return ranges, singles


def extract_entry_dates(
    entries: EntrySegmentationResult,
    policy: DateExtractionPolicy | None = None,
) -> "DateExtractionResult":
    """B8 entry point: dates for every B7 entry. Pure function."""
    policy = policy or DateExtractionPolicy()
    reference = policy.reference_date or date.today()

    out = []
    for entry in entries.entries:
        ranges, singles = extract_text_dates(entry.text)
        primary = None
        if ranges:
            primary = ranges[0]
        elif singles:
            s = singles[0]
            primary = DateRange(raw=s.raw, start=s, end=s, is_ongoing=False,
                                start_char=s.start_char, end_char=s.end_char)
        out.append(EntryDates(
            entry_id=entry.entry_id, block_id=entry.block_id,
            document_id=entry.document_id, section=entry.section,
            trusted=entry.trusted, ranges=ranges, single_dates=singles,
            primary_range=primary, has_dates=bool(ranges or singles),
        ))

    return DateExtractionResult(
        document_id=entries.document_id,
        entry_dates=out,
        reference_date=reference,
        n_entries=len(out),
        n_with_dates=sum(1 for e in out if e.has_dates),
        n_ranges=sum(len(e.ranges) for e in out),
        integrity=build_dates_integrity(entries, out),
    )


def build_dates_integrity(entries: EntrySegmentationResult, dated) -> IntegrityReport:
    """B8 integrity: full entry coverage, ordered non-overlapping spans,
    spans reconstruct, primary follows the selection rule."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    by_entry = {e.entry_id: e for e in entries.entries}
    ok_cover = [e.entry_id for e in dated] == [e.entry_id for e in entries.entries]
    checks["exactly_one_dates_per_entry_in_order"] = ok_cover
    if not ok_cover:
        violations.append(
            f"dates coverage mismatch: {len(entries.entries)} entries, {len(dated)} dated."
        )

    spans_ok, order_ok, primary_ok = True, True, True
    for ed in dated:
        entry = by_entry.get(ed.entry_id)
        if entry is None:
            spans_ok = False
            violations.append(f"dates for unknown entry {ed.entry_id}.")
            continue
        if ed.document_id != entries.document_id or entry.text is None:
            spans_ok = False
            violations.append(f"{ed.entry_id}: document/text mismatch.")
        text = entry.text
        seen: list[tuple[int, int]] = []
        for r in list(ed.ranges) + list(ed.single_dates):
            span = (r.start_char, r.end_char)
            if not (0 <= span[0] <= span[1] <= len(text)) or text[span[0]:span[1]] != r.raw:
                spans_ok = False
                violations.append(f"{ed.entry_id}: span [{span[0]}:{span[1]}] != {r.raw!r}.")
            if isinstance(r, DateRange):
                for ep_name, ep in (("start", r.start), ("end", r.end)):
                    if ep is None:
                        continue
                    espan = (ep.start_char, ep.end_char)
                    if not (span[0] <= espan[0] <= espan[1] <= span[1]) \
                            or text[espan[0]:espan[1]] != ep.raw:
                        spans_ok = False
                        violations.append(
                            f"{ed.entry_id}: {ep_name} endpoint span invalid for {r.raw!r}.")
                if (r.end is None) != r.is_ongoing:
                    spans_ok = False
                    violations.append(f"{ed.entry_id}: ongoing/end mismatch for {r.raw!r}.")
            if any(s < e2 and s2 < e for s, e in [span] for s2, e2 in seen):
                order_ok = False
                violations.append(f"{ed.entry_id}: overlapping date spans.")
            seen.append(span)
        ordered = [(r.start_char, r.end_char) for r in list(ed.ranges) + list(ed.single_dates)]
        if ordered != sorted(ordered):
            order_ok = False
            violations.append(f"{ed.entry_id}: dates not in text order.")
        # Primary rule: first range, else first single as a point, else None.
        expected = None
        if ed.ranges:
            expected = ("range", ed.ranges[0].start_char)
        elif ed.single_dates:
            expected = ("single", ed.single_dates[0].start_char)
        got = None
        if ed.primary_range is not None:
            if ed.ranges and ed.primary_range == ed.ranges[0]:
                got = ("range", ed.primary_range.start_char)
            elif ed.single_dates and ed.primary_range.start == ed.single_dates[0] \
                    and ed.primary_range.end == ed.single_dates[0]:
                got = ("single", ed.primary_range.start_char)
        if expected != got:
            primary_ok = False
            violations.append(f"{ed.entry_id}: primary_range breaks the selection rule.")
        if ed.has_dates != bool(ed.ranges or ed.single_dates):
            primary_ok = False
            violations.append(f"{ed.entry_id}: has_dates flag wrong.")
    checks["spans_reconstruct"] = spans_ok
    checks["dates_ordered_non_overlapping"] = order_ok
    checks["primary_selection_rule"] = primary_ok

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


__all__ = [
    "build_dates_integrity",
    "extract_entry_dates",
    "extract_text_dates",
    "has_date_range",
    "is_entry_boundary",
    "is_short_bare_year",
]
