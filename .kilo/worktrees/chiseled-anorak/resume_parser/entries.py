"""B7: Entry Segmentation.

B7 answers: "Which lines belong to the same logical entry?" It splits section
blocks into entries (jobs inside EXPERIENCE, degrees inside EDUCATION,
projects inside PROJECTS) using DATE-RANGE boundary signals. It does NOT
parse dates, associate events, or order anything chronologically (B8/B9).

Boundary policy (structural evidence only):
- EXPERIENCE / PROJECTS: a display line containing a DATE RANGE starts a new
  entry (e.g. "04/2024 - Present", "Jan 2019 - Mar 2021", "2016 - 2018").
  Bare years do NOT split (bullets often mention years).
- EDUCATION: date ranges too, plus short bare-year lines ("2018"), since
  education entries commonly lead with a lone year.
- All other sections: one entry per block (explicit pass-through).
- Leading lines before the first boundary belong to the first entry; the
  block header therefore stays with entry #1. No content is ever dropped.

Range detection is B8's parser (single source of truth — see date_extraction:
a boundary fires exactly when B8 would extract a range, so split entries
always carry parseable dates). B7 never interprets the dates themselves.
"""

from __future__ import annotations

import re

from .date_extraction import extract_text_dates
from .models import (
    BlockEntry,
    EntrySegmentationResult,
    FinalSectionOutput,
    IntegrityReport,
    SectionLabel,
    SegmentationResult,
)

_YEAR = r"(?:19|20)\d{2}"
_BARE_YEAR_RE = re.compile(rf"^\s*{_YEAR}\s*$")

DATE_SECTIONS = frozenset({SectionLabel.EXPERIENCE, SectionLabel.EDUCATION, SectionLabel.PROJECTS})

REASON_SINGLE = "block_start_single"
REASON_FIRST = "block_start_first"
REASON_DATE = "date_boundary"


def has_date_range(text: str) -> bool:
    """True if the line contains a date RANGE (not a bare year).

    Delegates to B8's parser: a boundary fires exactly when B8 extracts a
    range here, so the resulting entry always has a parseable header date.
    """
    ranges, _ = extract_text_dates(text)
    return bool(ranges)


def is_short_bare_year(text: str) -> bool:
    """True for lone-year lines like '2018' (education entry headers)."""
    return bool(_BARE_YEAR_RE.match(text)) and len(text.split()) <= 3


def is_entry_boundary(text: str, section: SectionLabel) -> bool:
    """Boundary decision for one display line within a section block."""
    if section not in DATE_SECTIONS:
        return False
    if has_date_range(text):
        return True
    return section == SectionLabel.EDUCATION and is_short_bare_year(text)


def parse_line_index(line_id: str) -> int:
    """B1 line IDs encode their index (L000123 -> 123). Strict: fail loudly."""
    if not re.fullmatch(r"L\d+", line_id):
        raise ValueError(f"malformed B1 line_id (expected L+digits): {line_id!r}")
    return int(line_id[1:])


def build_entries(
    seg: SegmentationResult,
    final: FinalSectionOutput,
) -> EntrySegmentationResult:
    """Split every final section block into entries. Pure function."""
    by_block = {b.block_id: b for b in seg.blocks}
    by_final = {s.block_id: s for s in final.sections}

    entries: list[BlockEntry] = []
    for block in seg.blocks:
        fsec = by_final.get(block.block_id)
        if fsec is None:
            # No final assignment: keep lines explicitly untrusted under OTHER
            # rather than dropping the block. (Integrity below fails loudly.)
            entries.extend(_unassigned_block_entries(seg.document_id, block))
            continue
        entries.extend(_block_entries(seg.document_id, block, fsec.final_section,
                                      fsec.trusted, fsec.source.value))

    by_section: dict[str, int] = {}
    for e in entries:
        by_section[e.section.value] = by_section.get(e.section.value, 0) + 1

    # Entry IDs are document-global: stamp after all blocks are processed.
    entries = _assign_entry_ids(entries)

    return EntrySegmentationResult(
        document_id=seg.document_id,
        entries=entries,
        n_entries=len(entries),
        n_blocks=len(seg.blocks),
        by_section=by_section,
        integrity=build_entry_integrity(seg, final, entries),
    )


def _block_entries(
    document_id: str,
    block,
    section: SectionLabel,
    trusted: bool,
    source: str,
) -> list[BlockEntry]:
    """Split one block's display lines into entries."""
    disp = [(d.text, list(d.source_line_ids)) for d in block.display_lines]
    if not disp:
        # Defensive: B2 never emits content blocks without display lines.
        return [BlockEntry(
            entry_id="", index=-1, block_id=block.block_id, document_id=document_id,
            section=section, trusted=trusted, source=source,
            line_ids=list(block.line_ids),
            start_line_index=min(parse_line_index(l) for l in block.line_ids) if block.line_ids else -1,
            end_line_index=max(parse_line_index(l) for l in block.line_ids) if block.line_ids else -1,
            text="", split_reason=REASON_SINGLE,
        )]

    # Cut before every boundary line except the first line.
    cuts = [0]
    if section in DATE_SECTIONS:
        for i in range(1, len(disp)):
            if is_entry_boundary(disp[i][0], section):
                cuts.append(i)
    cuts.append(len(disp))
    multi = len(cuts) > 2

    out: list[BlockEntry] = []
    for k in range(len(cuts) - 1):
        chunk = disp[cuts[k]:cuts[k + 1]]
        lids = [lid for _, ids in chunk for lid in ids]
        idx = [parse_line_index(l) for l in lids]
        out.append(BlockEntry(
            entry_id="", index=-1, block_id=block.block_id, document_id=document_id,
            section=section, trusted=trusted, source=source,
            line_ids=lids,
            start_line_index=min(idx), end_line_index=max(idx),
            text="\n".join(t for t, _ in chunk),
            split_reason=(REASON_DATE if k > 0 else (REASON_FIRST if multi else REASON_SINGLE)),
        ))
    return out


def _unassigned_block_entries(document_id: str, block) -> list[BlockEntry]:
    """Fallback for blocks missing a B6 assignment: single untrusted entry."""
    lids = list(block.line_ids)
    idx = [parse_line_index(l) for l in lids] if lids else [-1]
    return [BlockEntry(
        entry_id="", index=-1, block_id=block.block_id, document_id=document_id,
        section=SectionLabel.OTHER, trusted=False, source="ml_unresolved",
        line_ids=lids, start_line_index=min(idx), end_line_index=max(idx),
        text=block.text, split_reason=REASON_SINGLE,
    )]


def _assign_entry_ids(entries: list[BlockEntry]) -> list[BlockEntry]:
    """Stamp document-order entry IDs (called by build_entries once all
    blocks are processed, since IDs are document-global)."""
    return [
        e.model_copy(update={"entry_id": f"E{i:06d}", "index": i})
        for i, e in enumerate(entries)
    ]


def build_entry_integrity(
    seg: SegmentationResult,
    final: FinalSectionOutput,
    entries: list[BlockEntry],
) -> IntegrityReport:
    """B7 integrity: exact coverage, order, no cross-block entries, verified text."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    expected_ids = [f"E{i:06d}" for i in range(len(entries))]
    ok_ids = [e.entry_id for e in entries] == expected_ids
    checks["entry_ids_unique_contiguous_ordered"] = ok_ids
    if not ok_ids:
        violations.append("entry_ids not unique/contiguous/ordered.")

    # Every block yields >= 1 entry; entries of one block are contiguous.
    by_block: dict[str, list[BlockEntry]] = {}
    for e in entries:
        by_block.setdefault(e.block_id, []).append(e)
    ok_blocks = all(len(by_block.get(b.block_id, [])) >= 1 for b in seg.blocks)
    checks["every_block_has_entry"] = ok_blocks
    if not ok_blocks:
        missing = [b.block_id for b in seg.blocks if not by_block.get(b.block_id)]
        violations.append(f"blocks without entries: {missing}.")

    # Coverage: exactly the B2 content lines, once each, in document order.
    expected_lines = [lid for b in seg.blocks for lid in b.line_ids]
    covered = [lid for e in entries for lid in e.line_ids]
    ok_cover = sorted(covered) == sorted(expected_lines) \
        and len(covered) == len(set(covered)) == len(expected_lines)
    checks["content_lines_covered_exactly_once"] = ok_cover
    if not ok_cover:
        violations.append(
            f"entry coverage mismatch: {len(expected_lines)} block lines, "
            f"{len(covered)} covered ({len(set(covered))} unique)."
        )

    # No entry crosses block boundaries; order within entry follows document.
    block_lines = {b.block_id: set(b.line_ids) for b in seg.blocks}
    order_ok, text_ok, idx_ok, echo_ok = True, True, True, True
    by_final = {s.block_id: s for s in final.sections}
    disp_map = {b.block_id: [(d.text, list(d.source_line_ids)) for d in b.display_lines]
                for b in seg.blocks}
    for e in entries:
        allowed = block_lines.get(e.block_id, set())
        if any(lid not in allowed for lid in e.line_ids):
            order_ok = False
            violations.append(f"{e.entry_id}: lines outside block {e.block_id}.")
        idx = [parse_line_index(l) for l in e.line_ids] if e.line_ids else []
        if idx != sorted(idx) or (idx and (e.start_line_index != min(idx) or e.end_line_index != max(idx))):
            idx_ok = False
            violations.append(f"{e.entry_id}: line order/index mismatch.")
        # Text must equal the joined display texts of its source lines.
        if e.block_id in disp_map:
            want: list[str] = []
            remaining = list(e.line_ids)
            for t, ids in disp_map[e.block_id]:
                if remaining[:len(ids)] == ids:
                    want.append(t)
                    remaining = remaining[len(ids):]
            if remaining or e.text != "\n".join(want):
                text_ok = False
                violations.append(f"{e.entry_id}: text != joined display lines.")
        fsec = by_final.get(e.block_id)
        if fsec is None or e.section != fsec.final_section \
                or e.trusted != fsec.trusted or e.source != fsec.source.value:
            echo_ok = False
            violations.append(f"{e.entry_id}: section/trust echo mismatch with B6.")
    checks["entries_within_single_block"] = order_ok
    checks["entry_text_matches_display_lines"] = text_ok
    checks["entry_indices_valid"] = idx_ok
    checks["section_trust_echo_b6"] = echo_ok

    # Final/output document agreement.
    if final.document_id != seg.document_id:
        violations.append(f"final document_id {final.document_id} != {seg.document_id}.")
    checks["document_ids_match"] = final.document_id == seg.document_id

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)
