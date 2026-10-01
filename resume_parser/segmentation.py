"""B2: Candidate block segmentation — STRUCTURAL ONLY.

What B2 does (and does NOT do)
------------------------------
- Pages: a form-feed (``\\f``) inside a raw line is a page break. ``page_index``
  (reserved as None in B1) is filled here; ``\\f`` never appears in display text.
- Boilerplate: page-opening lines (first 2 nonblank display lines of a page)
  whose text repeats on 2+ pages (e.g. repeated PDF headers) are flagged
  ``boilerplate_candidate``. Flagged lines are listed separately, never deleted.
- Segmentation: contiguous spans of content lines grouped by STRUCTURAL signals
  only. NO semantic section labels are assigned. Every meaningful B1 line belongs
  to exactly one candidate block. No meaningful content is silently discarded.
- Display cleaning (display only, raw lines untouched): strip control /
  private-use glyphs (PDF icon fonts), skip blank lines, join hyphenated
  line-breaks (``de-`` + ``ployed`` -> ``deployed``) with ``source_line_ids``
  provenance per display line.
- B2 integrity: every nonblank, non-boilerplate line appears in exactly one
  block exactly once; block IDs unique/contiguous/ordered; hyphen joins reference
  valid ordered line IDs; ``text == "\\n".join(display line texts)`` per block.

Structural boundary signals used (enum BoundarySignal):
  - BLANK_LINE: one or more blank lines
  - HEADING_LIKE: line looks like a heading (ALL CAPS short, trailing colon, etc.)
  - MARKDOWN_HEADING: Markdown ``#`` heading
  - INDENTATION_CHANGE: significant indent shift
  - BULLET_TRANSITION: entering/exiting a bullet sequence
  - TABLE_BOUNDARY: entering/exiting a table
  - CODE_FENCE_BOUNDARY: entering/exiting a code fence
  - PAGE_BREAK: form-feed page break
  - FORMATTING_CHANGE: significant formatting shift
  - CONTINUATION_DETECTED: hyphen-joined continuation (handled in display)

Semantic signals NOT used in B2:
  - role/company/date vocabulary
  - section lexicon matches
  - experience/education/skills/etc. classification
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from pydantic import BaseModel, Field

from .models import (
    AssignmentStatus,
    BoundarySignal,
    BoilerplateLine,
    CandidateBlock,
    ConversionWarningModel,
    DisplayLine,
    IntegrityReport,
    LineKind,
    LineRecord,
    NormalizedDocument,
    SegmentationResult,
)

# Display cleaning --------------------------------------------------------------

# Private-use characters commonly used as bullet markers in PDF extraction
_BULLET_PUA_CHARS = {
    "\uf0b7",  # ● (common bullet)
    "\uf0a7",  # ▪
    "\uf020",  # another bullet variant
    "\u2022",  # • (standard bullet, but sometimes in PUA)
    "\u25cf",  # ●
    "\u25aa",  # ▪
    "\u25e6",  # ◦
    "\u2043",  # ⁃
}

def display_text(raw_normalized: str) -> str:
    """Display-only cleanup: drop control/format/private-use chars (PDF icon
    fonts, form-feeds). Internal spacing and punctuation are preserved."""
    out = "".join(
        ch for ch in raw_normalized
        if not (unicodedata.category(ch) in ("Cc", "Cf", "Co") and ch != "\t")
    )
    return out.rstrip(" \t")


def _is_bullet_marker_only(text: str) -> bool:
    """Check if text consists only of bullet-like private-use characters (and whitespace).
    
    These are structural bullet markers from PDF extraction, not blank lines.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # All non-whitespace chars must be known bullet PUA chars
    return all(ch in _BULLET_PUA_CHARS or ch.isspace() for ch in stripped)


def is_display_blank(normalized: str) -> bool:
    """Check if a line is blank for display purposes.
    
    Lines that are only bullet markers (private-use bullet chars) are NOT blank -
    they are structural bullet indicators.
    """
    if _is_bullet_marker_only(normalized):
        return False
    return display_text(normalized).strip() == ""


def should_join(prev: str, nxt: str) -> bool:
    """Hyphenated line-break: ``de-`` + ``ployed``. Conservative: trailing
    hyphen after a letter, next line starts with a lowercase letter."""
    return (
        len(prev) > 1 and prev.endswith("-") and prev[-2].isalpha()
        and bool(nxt) and nxt[0].isalpha() and nxt[0].islower()
    )


# Structural line analysis ------------------------------------------------------

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s")
_BULLET_RE = re.compile(r"^(?:[-*+]\s+|\d{1,3}[.)]\s+)")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_LIKE_RE = re.compile(r"^[A-Z][A-Z\s\d&/+-]{2,}:?$")  # ALL CAPS short line, optional colon


def get_line_indent(text: str) -> int:
    """Return leading whitespace length."""
    return len(text) - len(text.lstrip())


def is_markdown_heading(text: str) -> bool:
    return bool(_MARKDOWN_HEADING_RE.match(display_text(text).strip()))


def is_bullet(text: str) -> bool:
    return bool(_BULLET_RE.match(display_text(text).strip()))


def is_table_row(text: str) -> bool:
    return bool(_TABLE_ROW_RE.match(display_text(text).strip()))


def is_code_fence(text: str) -> bool:
    return bool(_CODE_FENCE_RE.match(display_text(text).strip()))


def is_heading_like(text: str) -> bool:
    """Conservative structural heading detection (formatting only).
    
    Does NOT classify semantically. Only identifies lines that LOOK like
    section headers based on formatting.
    """
    disp = display_text(text).strip()
    if not disp or len(disp) > 80:
        return False
    if is_bullet(disp) or is_table_row(disp) or is_code_fence(disp):
        return False
    # Markdown heading
    if is_markdown_heading(disp):
        return True
    # ALL CAPS short line (common for section headers in PDF extraction)
    if _HEADING_LIKE_RE.match(disp) and len(disp.split()) <= 6:
        return True
    # Trailing colon, short
    if disp.endswith(":") and len(disp.split()) <= 6:
        return True
    return False


def detect_boundary_signals(
    prev_line: Optional[LineRecord],
    curr_line: LineRecord,
    next_line: Optional[LineRecord],
    prev_display_kind: Optional[LineKind],
    curr_display_kind: LineKind,
    next_display_kind: Optional[LineKind],
    prev_indent: Optional[int],
    curr_indent: int,
) -> list[BoundarySignal]:
    """Detect structural boundary signals between prev and curr line.
    
    Returns list of signals that suggest a block boundary BEFORE curr_line.
    """
    signals: list[BoundarySignal] = []
    
    if prev_line is None:
        return signals  # first line never has a preceding boundary
    
    prev_disp = display_text(prev_line.normalized_text)
    curr_disp = display_text(curr_line.normalized_text)
    next_disp = display_text(next_line.normalized_text) if next_line else ""
    
    # 1. Blank line(s) between content
    if prev_disp.strip() == "" or curr_disp.strip() == "":
        # Check if it's a run of blanks
        if prev_disp.strip() == "" and curr_disp.strip() == "":
            signals.append(BoundarySignal.BLANK_LINE)
        elif prev_disp.strip() == "":
            signals.append(BoundarySignal.BLANK_LINE)
    
    # 2. Markdown heading
    if is_markdown_heading(curr_disp):
        signals.append(BoundarySignal.MARKDOWN_HEADING)
    
    # 3. Heading-like (ALL CAPS, trailing colon, short)
    if is_heading_like(curr_disp):
        signals.append(BoundarySignal.HEADING_LIKE)
    
    # 4. Indentation change (significant shift)
    if prev_indent is not None and abs(curr_indent - prev_indent) > 4:
        signals.append(BoundarySignal.INDENTATION_CHANGE)
    
    # 5. Bullet transition
    prev_bullet = is_bullet(prev_disp)
    curr_bullet = is_bullet(curr_disp)
    if prev_bullet != curr_bullet:
        signals.append(BoundarySignal.BULLET_TRANSITION)
    
    # 6. Table boundary
    prev_table = is_table_row(prev_disp)
    curr_table = is_table_row(curr_disp)
    if prev_table != curr_table:
        signals.append(BoundarySignal.TABLE_BOUNDARY)
    
    # 7. Code fence boundary
    prev_fence = is_code_fence(prev_disp)
    curr_fence = is_code_fence(curr_disp)
    if prev_fence != curr_fence:
        signals.append(BoundarySignal.CODE_FENCE_BOUNDARY)
    
    # 8. Page break (form feed in raw text)
    if "\f" in curr_line.raw_text and curr_line.index > 0:
        signals.append(BoundarySignal.PAGE_BREAK)
    
    # 9. Formatting change (kind change)
    if prev_display_kind != curr_display_kind and prev_display_kind != LineKind.BLANK:
        signals.append(BoundarySignal.FORMATTING_CHANGE)
    
    return signals


def is_strong_boundary(signals: list[BoundarySignal]) -> bool:
    """Determine if signals constitute a strong structural boundary.
    
    Strong boundaries always start a new block.
    Weak boundaries (single blank line) only split if reinforced.
    """
    strong_signals = {
        BoundarySignal.MARKDOWN_HEADING,
        BoundarySignal.HEADING_LIKE,
        BoundarySignal.BULLET_TRANSITION,
        BoundarySignal.TABLE_BOUNDARY,
        BoundarySignal.CODE_FENCE_BOUNDARY,
        BoundarySignal.PAGE_BREAK,
        BoundarySignal.FORMATTING_CHANGE,
    }
    if any(s in strong_signals for s in signals):
        return True
    
    # BLANK_LINE alone is weak — only strong if multiple blanks or combined with INDENTATION_CHANGE
    if BoundarySignal.BLANK_LINE in signals:
        # Multiple blanks or blank + indent change = strong
        if BoundarySignal.INDENTATION_CHANGE in signals:
            return True
        # Could check for multiple consecutive blanks here via context
        # For now, single blank line is weak
        return False
    
    return False


# Pagination --------------------------------------------------------------------

def paginate(lines: list[LineRecord]) -> dict[str, int]:
    """Assign page_index per line_id. A raw line containing \\f opens a new page."""
    pages: dict[str, int] = {}
    page = 0
    for ln in lines:
        if "\f" in ln.raw_text and ln.index > 0:
            page += 1
        pages[ln.line_id] = page
    return pages


# Boilerplate detection ---------------------------------------------------------

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


# Candidate block segmentation --------------------------------------------------

def segment_document(doc: NormalizedDocument) -> SegmentationResult:
    """Structural candidate block segmentation — NO semantic labels.
    
    Every meaningful (nonblank, non-boilerplate) B1 line is assigned to exactly
    one CandidateBlock. Blocks are formed by grouping lines until a strong
    structural boundary is detected.
    """
    lines = doc.lines
    pages = paginate(lines)
    boiler_ids = find_boilerplate(lines, pages) if lines else set()
    
    warnings: list[ConversionWarningModel] = []
    
    # Collect content lines (nonblank display, non-boilerplate) with metadata
    content_lines: list[tuple[LineRecord, LineKind, int]] = []  # (line, display_kind, indent)
    for ln in lines:
        if ln.line_id in boiler_ids or is_display_blank(ln.normalized_text):
            continue
        disp_text = display_text(ln.normalized_text).strip()
        if disp_text.startswith("```") or disp_text.startswith("~~~"):
            disp_kind = LineKind.CODE_FENCE
        elif _MARKDOWN_HEADING_RE.match(disp_text):
            disp_kind = LineKind.MARKDOWN_HEADING
        elif disp_text.startswith("|"):
            disp_kind = LineKind.TABLE_ROW
        elif _BULLET_RE.match(disp_text):
            disp_kind = LineKind.BULLET
        elif disp_text == "":
            disp_kind = LineKind.BLANK
        else:
            disp_kind = LineKind.TEXT
        
        indent = get_line_indent(ln.normalized_text)
        content_lines.append((ln, disp_kind, indent))
    
    # Build candidate blocks
    blocks: list[CandidateBlock] = []
    cur_lines: list[tuple[LineRecord, LineKind, int]] = []
    cur_boundary_signals: list[BoundarySignal] = []
    hyphen_joins = 0
    
    def flush_block() -> None:
        nonlocal hyphen_joins
        if not cur_lines:
            return
        
        # Build display lines with hyphen joins
        disp_lines: list[DisplayLine] = []
        i = 0
        while i < len(cur_lines):
            ln, disp_kind, indent = cur_lines[i]
            t = display_text(ln.normalized_text)
            ids = [ln.line_id]
            is_hdr = False
            if i + 1 < len(cur_lines):
                nxt_t = display_text(cur_lines[i + 1][0].normalized_text)
                if should_join(t, nxt_t):
                    t = t[:-1] + nxt_t
                    ids.append(cur_lines[i + 1][0].line_id)
                    hyphen_joins += 1
                    i += 1
            # Check if this line is heading-like (structural)
            if i == 0 and is_heading_like(t):
                is_hdr = True
            disp_lines.append(DisplayLine(
                text=t,
                source_line_ids=ids,
                page_index=pages[ln.line_id],
                is_header=is_hdr,
                line_kind=disp_kind,
            ))
            i += 1
        
        block_line_ids = [ln.line_id for ln, _, _ in cur_lines]
        pgs = sorted({pages[ln.line_id] for ln, _, _ in cur_lines})
        start_idx = min(ln.index for ln, _, _ in cur_lines)
        end_idx = max(ln.index for ln, _, _ in cur_lines)
        header_lid = cur_lines[0][0].line_id if disp_lines and disp_lines[0].is_header else None
        
        # Determine if this block is a continuation of previous
        is_cont = len(blocks) > 0 and not cur_boundary_signals
        
        blocks.append(CandidateBlock(
            block_id=f"B{len(blocks):06d}",
            index=len(blocks),
            line_ids=block_line_ids,
            start_line_index=start_idx,
            end_line_index=end_idx,
            page_indices=pgs,
            display_lines=disp_lines,
            text="\n".join(d.text for d in disp_lines),
            header_line_id=header_lid,
            is_continuation=is_cont,
            boundary_signals=list(set(cur_boundary_signals)),  # dedupe
        ))
        cur_lines.clear()
        cur_boundary_signals.clear()
    
    # Iterate through content lines, detecting boundaries
    for i, (ln, disp_kind, indent) in enumerate(content_lines):
        prev_l = content_lines[i - 1] if i > 0 else None
        next_l = content_lines[i + 1] if i + 1 < len(content_lines) else None
        
        prev_disp_kind = prev_l[1] if prev_l else None
        next_disp_kind = next_l[1] if next_l else None
        prev_indent = prev_l[2] if prev_l else None
        
        signals = detect_boundary_signals(
            prev_l[0] if prev_l else None,
            ln,
            next_l[0] if next_l else None,
            prev_disp_kind,
            disp_kind,
            next_disp_kind,
            prev_indent,
            indent,
        )
        
        # Check for strong boundary BEFORE this line
        if cur_lines and is_strong_boundary(signals):
            flush_block()
        
        cur_lines.append((ln, disp_kind, indent))
        cur_boundary_signals.extend(signals)
    
    flush_block()
    
    # Boilerplate listing (annotated, never deleted)
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
        blocks=blocks,
        boilerplate=boilerplate,
        hyphen_joins=hyphen_joins,
        warnings=warnings,
        integrity=build_b2_integrity(doc, pages, boiler_ids, blocks),
    )
    return result


# B2 Integrity ------------------------------------------------------------------

def build_b2_integrity(
    doc: NormalizedDocument,
    pages: dict[str, int],
    boiler_ids: set[str],
    blocks: list[CandidateBlock],
) -> IntegrityReport:
    violations: list[str] = []
    checks: dict[str, bool] = {}
    
    # Block IDs unique, contiguous, ordered
    expected = [f"B{i:06d}" for i in range(len(blocks))]
    ok = [b.block_id for b in blocks] == expected
    checks["block_ids_unique_contiguous_ordered"] = ok
    if not ok:
        violations.append("block_ids not unique/contiguous/ordered.")
    
    # Every meaningful line covered exactly once
    covered = [lid for b in blocks for lid in b.line_ids]
    should = [ln.line_id for ln in doc.lines
              if ln.line_id not in boiler_ids and not is_display_blank(ln.normalized_text)]
    ok_cover = sorted(covered) == sorted(should) and len(covered) == len(set(covered))
    checks["content_lines_covered_exactly_once"] = ok_cover
    if not ok_cover:
        violations.append(
            f"block coverage mismatch: {len(should)} content lines, {len(covered)} covered ({len(set(covered))} unique)."
        )
    
    # Block ordering follows document order
    order = [ln.index for b in blocks for lid in b.line_ids
             for ln in doc.lines if ln.line_id == lid]
    ok_order = order == sorted(order)
    checks["blocks_follow_document_order"] = ok_order
    if not ok_order:
        violations.append("blocks do not follow document order.")
    
    # Display provenance valid
    id_index = {ln.line_id: ln.index for ln in doc.lines}
    ok_prov = True
    for b in blocks:
        if b.text != "\n".join(d.text for d in b.display_lines):
            ok_prov = False
            violations.append(f"{b.block_id}: text != join(display lines).")
        for d in b.display_lines:
            if any(s not in id_index for s in d.source_line_ids):
                ok_prov = False
                violations.append(f"{b.block_id}: unknown source_line_id in {d.source_line_ids}.")
            idx = [id_index[s] for s in d.source_line_ids if s in id_index]
            if idx != sorted(idx) or len(set(idx)) != len(idx):
                ok_prov = False
                violations.append(f"{b.block_id}: source_line_ids not ordered/unique: {d.source_line_ids}.")
        if b.header_line_id and b.header_line_id not in b.line_ids:
            ok_prov = False
            violations.append(f"{b.block_id}: header_line_id {b.header_line_id} not in block line_ids.")
        if b.start_line_index > b.end_line_index:
            ok_prov = False
            violations.append(f"{b.block_id}: start_line_index > end_line_index.")
    checks["display_provenance_valid"] = ok_prov
    
    # Line IDs inside blocks are ordered
    ok_line_order = True
    for b in blocks:
        indices = [id_index[lid] for lid in b.line_ids if lid in id_index]
        if indices != sorted(indices):
            ok_line_order = False
            violations.append(f"{b.block_id}: line_ids not in document order.")
    checks["line_ids_in_blocks_ordered"] = ok_line_order
    
    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


__all__ = [
    "BoilerplateLine",
    "BoundarySignal",
    "CandidateBlock",
    "DisplayLine",
    "SegmentationResult",
    "display_text",
    "find_boilerplate",
    "is_display_blank",
    "paginate",
    "segment_document",
    "should_join",
]