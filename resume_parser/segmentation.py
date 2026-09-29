"""B2: candidate block segmentation + deterministic label proposals.

What B2 does (and does NOT do)
------------------------------
- Pages: a form-feed (``\\f``) inside a raw line is a page break. ``page_index``
  (reserved as None in B1) is filled here; ``\\f`` never appears in display text.
- Boilerplate: page-opening lines (first 2 nonblank display lines of a page)
  whose text repeats on 2+ pages (e.g. repeated PDF headers) are flagged
  ``boilerplate_candidate``. Flagged lines are listed separately, never deleted.
- Segmentation: contiguous spans of content lines. The leading span is
  ``contact_header`` (PoC parity); each lexicon header line opens a new block.
  Labels are PROPOSALS (``proposal_source``); ML may replace them later.
- Display cleaning (display only, raw lines untouched): strip control /
  private-use glyphs (PDF icon fonts), skip blank lines, join hyphenated
  line-breaks (``de-`` + ``ployed`` -> ``deployed``) with ``source_line_ids``
  provenance per display line.

B2 integrity: every nonblank, non-boilerplate line appears in exactly one
block exactly once; block IDs unique/contiguous/ordered; hyphen joins reference
valid ordered line IDs; ``text == "\\n".join(display line texts)`` per block.
"""

from __future__ import annotations

import unicodedata

from pydantic import BaseModel, Field

from .lexicon import TYPES, rule_header
from .models import ConversionWarningModel, IntegrityReport, LineRecord, NormalizedDocument

CONTACT = "contact_header"
SRC_LEXICON = "rule_lexicon"
SRC_POSITION = "rule_position_contact"


class DisplayLine(BaseModel):
    model_config = {"frozen": True}

    text: str
    source_line_ids: list[str]  # 1 id normally, 2 when a hyphen-break was joined
    page_index: int | None = None
    is_header: bool = False


class ContentBlock(BaseModel):
    model_config = {"frozen": True}

    block_id: str  # B000000, ...
    index: int
    label_proposal: str  # deterministic proposal; ML may replace later
    proposal_source: str  # rule_lexicon | rule_position_contact
    header_line_id: str | None = None
    line_ids: list[str]  # content line IDs in document order (excl. blanks/boilerplate)
    page_indices: list[int]
    display_lines: list[DisplayLine]
    text: str  # "\n".join of display line texts


class BoilerplateLine(BaseModel):
    model_config = {"frozen": True}

    line_ids: list[str]
    text: str
    page_indices: list[int]


class SegmentationResult(BaseModel):
    model_config = {"frozen": True}

    document_id: str
    page_count: int
    blocks: list[ContentBlock]
    boilerplate: list[BoilerplateLine]
    hyphen_joins: int
    warnings: list[ConversionWarningModel] = Field(default_factory=list)
    integrity: IntegrityReport


# ------------------------------------------------------------- display cleaning

def display_text(raw_normalized: str) -> str:
    """Display-only cleanup: drop control/format/private-use chars (PDF icon
    fonts, form-feeds). Internal spacing and punctuation are preserved."""
    out = "".join(
        ch for ch in raw_normalized
        if not (unicodedata.category(ch) in ("Cc", "Cf", "Co") and ch != "\t")
    )
    return out.rstrip(" \t")


def is_display_blank(normalized: str) -> bool:
    return display_text(normalized).strip() == ""


def should_join(prev: str, nxt: str) -> bool:
    """Hyphenated line-break: ``de-`` + ``ployed``. Conservative: trailing
    hyphen after a letter, next line starts with a lowercase letter."""
    return (
        len(prev) > 1 and prev.endswith("-") and prev[-2].isalpha()
        and bool(nxt) and nxt[0].isalpha() and nxt[0].islower()
    )


# ------------------------------------------------------------------ B2 stages

def paginate(lines: list[LineRecord]) -> dict[str, int]:
    """Assign page_index per line_id. A raw line containing \\f opens a new page."""
    pages: dict[str, int] = {}
    page = 0
    for ln in lines:
        # A raw line containing \f opens a new page (except at index 0,
        # where it belongs to the first page). The \f stays in raw_text;
        # display_text strips it.
        if "\f" in ln.raw_text and ln.index > 0:
            page += 1
        pages[ln.line_id] = page
    return pages


def find_boilerplate(
    lines: list[LineRecord], pages: dict[str, int]
) -> set[str]:
    """Flag repeated page-opening lines (first 2 nonblank display lines per page
    whose text repeats on 2+ pages). Returns flagged line_ids."""
    openings: dict[int, list[LineRecord]] = {}
    for ln in lines:
        if not is_display_blank(ln.normalized_text):
            openings.setdefault(pages[ln.line_id], []).append(ln)
    text_pages: dict[str, set[int]] = {}
    text_lines: dict[str, list[str]] = {}
    for pg, lns in openings.items():
        for ln in lns[:2]:
            t = display_text(ln.normalized_text).strip()
            text_pages.setdefault(t, set()).add(pg)
            text_lines.setdefault(t, []).append(ln.line_id)
    return {lid for t, pgs in text_pages.items() if len(pgs) >= 2 for lid in text_lines[t]}


def segment_document(doc: NormalizedDocument) -> SegmentationResult:
    lines = doc.lines
    pages = paginate(lines)
    boiler_ids = find_boilerplate(lines, pages) if lines else set()

    warnings: list[ConversionWarningModel] = []
    # --- collect content lines (nonblank display, non-boilerplate), detect headers
    content: list[tuple[LineRecord, str | None]] = []  # (line, header_label or None)
    for ln in lines:
        if ln.line_id in boiler_ids or is_display_blank(ln.normalized_text):
            continue
        content.append((ln, rule_header(display_text(ln.normalized_text))))

    # --- build blocks
    blocks: list[ContentBlock] = []
    cur_label, cur_source = CONTACT, SRC_POSITION
    cur: list[LineRecord] = []
    cur_header: str | None = None
    hyphen_joins = 0

    def flush() -> None:
        nonlocal hyphen_joins
        if not cur:
            return
        disp: list[DisplayLine] = []
        i = 0
        while i < len(cur):
            ln = cur[i]
            t = display_text(ln.normalized_text)
            ids = [ln.line_id]
            if i + 1 < len(cur):
                nxt_t = display_text(cur[i + 1].normalized_text)
                if should_join(t, nxt_t):
                    t = t[:-1] + nxt_t
                    ids.append(cur[i + 1].line_id)
                    hyphen_joins += 1
                    i += 1
            disp.append(DisplayLine(text=t, source_line_ids=ids,
                                    page_index=pages[ln.line_id],
                                    is_header=(cur_header is not None and ln.line_id == cur_header)))
            i += 1
        pgs = sorted({pages[ln.line_id] for ln in cur})
        blocks.append(ContentBlock(
            block_id=f"B{len(blocks):06d}", index=len(blocks),
            label_proposal=cur_label, proposal_source=cur_source,
            header_line_id=cur_header,
            line_ids=[ln.line_id for ln in cur],
            page_indices=pgs, display_lines=disp,
            text="\n".join(d.text for d in disp),
        ))

    for ln, hdr in content:
        if hdr and (not cur or ln.line_id != (cur_header or "")):
            flush()
            cur, cur_label, cur_source, cur_header = [ln], hdr, SRC_LEXICON, ln.line_id
        else:
            cur.append(ln)
    flush()

    # --- boilerplate listing (annotated, never deleted)
    by_text: dict[str, list[LineRecord]] = {}
    for ln in lines:
        if ln.line_id in boiler_ids:
            by_text.setdefault(display_text(ln.normalized_text).strip(), []).append(ln)
    boilerplate = [
        BoilerplateLine(line_ids=[ln.line_id for ln in lns], text=t,
                        page_indices=sorted({pages[ln.line_id] for ln in lns}))
        for t, lns in sorted(by_text.items())
    ]
    if boilerplate:
        warnings.append(ConversionWarningModel(
            code="BOILERPLATE_FLAGGED",
            message=f"{len(boiler_ids)} repeated page-opening line(s) flagged as boilerplate; kept in output, excluded from block text.",
            detail=",".join(sorted(boiler_ids)),
        ))
    if hyphen_joins:
        warnings.append(ConversionWarningModel(
            code="DISPLAY_HYPHEN_JOIN",
            message=f"{hyphen_joins} hyphenated line-break(s) joined in display text; raw lines unchanged, see source_line_ids.",
        ))

    result = SegmentationResult(
        document_id=doc.document_id,
        page_count=(max(pages.values()) + 1) if pages else 0,
        blocks=blocks, boilerplate=boilerplate,
        hyphen_joins=hyphen_joins, warnings=warnings,
        integrity=build_b2_integrity(doc, pages, boiler_ids, blocks),
    )
    return result


def build_b2_integrity(
    doc: NormalizedDocument,
    pages: dict[str, int],
    boiler_ids: set[str],
    blocks: list[ContentBlock],
) -> IntegrityReport:
    violations: list[str] = []
    checks: dict[str, bool] = {}

    expected = [f"B{i:06d}" for i in range(len(blocks))]
    ok = [b.block_id for b in blocks] == expected
    checks["block_ids_unique_contiguous_ordered"] = ok
    if not ok:
        violations.append("block_ids not unique/contiguous/ordered.")

    covered = [lid for b in blocks for lid in b.line_ids]
    should = [ln.line_id for ln in doc.lines
              if ln.line_id not in boiler_ids and not is_display_blank(ln.normalized_text)]
    ok_cover = sorted(covered) == sorted(should) and len(covered) == len(set(covered))
    checks["content_lines_covered_exactly_once"] = ok_cover
    if not ok_cover:
        violations.append(
            f"block coverage mismatch: {len(should)} content lines, {len(covered)} covered ({len(set(covered))} unique)."
        )
    # order: blocks follow document order
    order = [ln.index for b in blocks for lid in b.line_ids
             for ln in doc.lines if ln.line_id == lid]
    ok_order = order == sorted(order)
    checks["blocks_follow_document_order"] = ok_order
    if not ok_order:
        violations.append("blocks do not follow document order.")

    id_index = {ln.line_id: ln.index for ln in doc.lines}
    ok_hy = True
    for b in blocks:
        if b.text != "\n".join(d.text for d in b.display_lines):
            ok_hy = False
            violations.append(f"{b.block_id}: text != join(display lines).")
        for d in b.display_lines:
            if any(s not in id_index for s in d.source_line_ids):
                ok_hy = False
                violations.append(f"{b.block_id}: unknown source_line_id in {d.source_line_ids}.")
            idx = [id_index[s] for s in d.source_line_ids if s in id_index]
            if idx != sorted(idx) or len(set(idx)) != len(idx):
                ok_hy = False
                violations.append(f"{b.block_id}: source_line_ids not ordered/unique: {d.source_line_ids}.")
        if b.label_proposal not in TYPES:
            ok_hy = False
            violations.append(f"{b.block_id}: unknown label_proposal {b.label_proposal!r}.")
    checks["display_provenance_valid"] = ok_hy

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


__all__ = [
    "BoilerplateLine", "ContentBlock", "DisplayLine", "SegmentationResult",
    "display_text", "find_boilerplate", "is_display_blank", "paginate",
    "segment_document", "should_join",
]
