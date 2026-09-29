"""B6: Final Section Output.

B6 answers: "What is the final trusted sectioned representation?" It merges
the three upstream outcomes into exactly one assignment per B2 block:

  B4 ACCEPT   -> final = ML label,              trusted (source ml_accepted)
  B5 resolved -> final = LLM label,             trusted (source llm_resolved)
  otherwise   -> final = ML label as best-effort signal, UNTRUSTED
                 (source ml_unresolved; llm_reason says why: B5 absent,
                 call failed, invalid/silent answer, or budget cap)

B6 never invents labels, never drops blocks, and never re-runs ML/LLM. It is a
pure merge with verification: the merge is CHECKED (accepted finals equal the
ML labels, resolved finals equal the LLM labels), not assumed. B5 input is
optional — without it every escalated block becomes ml_unresolved.

This output is the contract B7 (entry segmentation) will consume. No B7 logic
lives here: no entry splitting, no date handling, no timelines.
"""

from __future__ import annotations

from .models import (
    ClassificationResult,
    FinalBlockSection,
    FinalSectionOutput,
    FinalSource,
    IntegrityReport,
    LLMResolutionResult,
    SectionLabel,
    SegmentationResult,
    ValidationResult,
)

REASON_LLM_NOT_RUN = "llm_not_run"


def build_final_sections(
    seg: SegmentationResult,
    cls_result: ClassificationResult,
    val_result: ValidationResult,
    llm_result: LLMResolutionResult | None = None,
) -> FinalSectionOutput:
    """Merge B4/B5 outcomes into the final per-block section assignment."""
    by_cls = {c.block_id: c for c in cls_result.classifications}
    by_verdict = {v.block_id: v for v in val_result.verdicts}
    by_resolution = {r.block_id: r for r in (llm_result.resolutions if llm_result else [])}

    sections: list[FinalBlockSection] = []
    for block in seg.blocks:
        cls = by_cls.get(block.block_id)
        verdict = by_verdict.get(block.block_id)
        if cls is None or verdict is None:
            # Cannot verify provenance: keep the block explicitly untrusted
            # rather than dropping it. (Integrity below fails loudly.)
            missing_label = cls.predicted_section if cls else SectionLabel.UNKNOWN
            missing_conf = cls.confidence if cls else 0.0
            sections.append(FinalBlockSection(
                block_id=block.block_id, document_id=seg.document_id,
                final_section=missing_label,
                confidence=missing_conf,
                source=FinalSource.ML_UNRESOLVED, trusted=False,
                ml_section=missing_label,
                ml_confidence=missing_conf,
                alternatives=list(cls.alternatives) if cls else [],
                b4_reasons=list(verdict.reasons) if verdict else ["missing_verdict"],
                llm_reason="missing_classification_or_verdict",
                source_line_ids=list(block.line_ids),
                start_line_index=block.start_line_index,
                end_line_index=block.end_line_index,
                text=block.text,
            ))
            continue

        resolution = by_resolution.get(block.block_id)
        if not verdict.needs_llm:
            sections.append(FinalBlockSection(
                block_id=block.block_id, document_id=seg.document_id,
                final_section=cls.predicted_section, confidence=cls.confidence,
                source=FinalSource.ML_ACCEPTED, trusted=True,
                ml_section=cls.predicted_section, ml_confidence=cls.confidence,
                alternatives=list(cls.alternatives),
                b4_reasons=list(verdict.reasons),
                llm_reason="",
                source_line_ids=list(cls.source_line_ids),
                start_line_index=cls.start_line_index,
                end_line_index=cls.end_line_index,
                text=block.text,
            ))
        elif resolution is not None and resolution.resolved:
            sections.append(FinalBlockSection(
                block_id=block.block_id, document_id=seg.document_id,
                final_section=resolution.resolved_section,
                confidence=resolution.confidence,
                source=FinalSource.LLM_RESOLVED, trusted=True,
                ml_section=cls.predicted_section, ml_confidence=cls.confidence,
                alternatives=list(cls.alternatives),
                b4_reasons=list(verdict.reasons),
                llm_reason=resolution.reason,
                source_line_ids=list(cls.source_line_ids),
                start_line_index=cls.start_line_index,
                end_line_index=cls.end_line_index,
                text=block.text,
            ))
        else:
            llm_reason = resolution.reason if resolution is not None else REASON_LLM_NOT_RUN
            sections.append(FinalBlockSection(
                block_id=block.block_id, document_id=seg.document_id,
                final_section=cls.predicted_section, confidence=cls.confidence,
                source=FinalSource.ML_UNRESOLVED, trusted=False,
                ml_section=cls.predicted_section, ml_confidence=cls.confidence,
                alternatives=list(cls.alternatives),
                b4_reasons=list(verdict.reasons),
                llm_reason=llm_reason,
                source_line_ids=list(cls.source_line_ids),
                start_line_index=cls.start_line_index,
                end_line_index=cls.end_line_index,
                text=block.text,
            ))

    n_unresolved = sum(1 for s in sections if not s.trusted)
    n_trusted = len(sections) - n_unresolved
    by_section: dict[str, int] = {}
    for s in sections:
        by_section[s.final_section.value] = by_section.get(s.final_section.value, 0) + 1

    return FinalSectionOutput(
        document_id=seg.document_id,
        sections=sections,
        n_trusted=n_trusted,
        n_unresolved=n_unresolved,
        trust_rate=(n_trusted / len(sections)) if sections else 1.0,
        by_section=by_section,
        integrity=build_final_integrity(seg, cls_result, val_result, llm_result, sections),
    )


def build_final_integrity(
    seg: SegmentationResult,
    cls_result: ClassificationResult,
    val_result: ValidationResult,
    llm_result: LLMResolutionResult | None,
    sections: list[FinalBlockSection],
) -> IntegrityReport:
    """B6 integrity: verified merge — finals match their claimed sources."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    block_ids = [b.block_id for b in seg.blocks]
    ok_cover = [s.block_id for s in sections] == block_ids
    checks["exactly_one_final_per_block_in_order"] = ok_cover
    if not ok_cover:
        violations.append(
            f"final coverage mismatch: {len(block_ids)} blocks, {len(sections)} finals."
        )

    inputs_ok = True
    for id_name, result, attr in (
        ("classification", cls_result, "classifications"),
        ("validation", val_result, "verdicts"),
    ):
        ids = [getattr(x, "block_id") for x in getattr(result, attr)]
        if result.document_id != seg.document_id:
            inputs_ok = False
            violations.append(f"{id_name} document_id {result.document_id} != {seg.document_id}.")
        if sorted(ids) != sorted(block_ids) or len(set(ids)) != len(ids):
            inputs_ok = False
            violations.append(f"{id_name} does not cover exactly the B2 blocks.")
    checks["inputs_cover_b2_blocks"] = inputs_ok

    llm_ok = True
    if llm_result is not None:
        if llm_result.document_id != seg.document_id:
            llm_ok = False
            violations.append(f"llm result document_id {llm_result.document_id} != {seg.document_id}.")
        accepted_ids = {v.block_id for v in val_result.verdicts if not v.needs_llm}
        leaked = sorted({r.block_id for r in llm_result.resolutions} & accepted_ids)
        if leaked:
            llm_ok = False
            violations.append(f"llm resolutions touch accepted blocks: {leaked}.")
    checks["llm_result_consistent"] = llm_ok

    # The merge itself is verified, not assumed.
    by_cls = {c.block_id: c for c in cls_result.classifications}
    by_res = {r.block_id: r for r in (llm_result.resolutions if llm_result else [])}
    merge_ok = True
    for s in sections:
        cls = by_cls.get(s.block_id)
        res = by_res.get(s.block_id)
        if s.source == FinalSource.ML_ACCEPTED:
            if cls is None or s.final_section != cls.predicted_section \
                    or s.confidence != cls.confidence:
                merge_ok = False
                violations.append(f"{s.block_id}: ml_accepted final != ML label.")
        elif s.source == FinalSource.LLM_RESOLVED:
            if res is None or not res.resolved \
                    or s.final_section != res.resolved_section \
                    or s.confidence != res.confidence:
                merge_ok = False
                violations.append(f"{s.block_id}: llm_resolved final != LLM answer.")
        elif s.source == FinalSource.ML_UNRESOLVED:
            if cls is None:
                merge_ok = False
                violations.append(f"{s.block_id}: ml_unresolved with no ML fallback.")
            elif s.final_section != cls.predicted_section:
                merge_ok = False
                violations.append(f"{s.block_id}: ml_unresolved final != ML label.")
        if s.trusted == (s.source == FinalSource.ML_UNRESOLVED):
            merge_ok = False
            violations.append(f"{s.block_id}: trusted flag inconsistent with source.")
    checks["merge_matches_sources"] = merge_ok

    seen: set[str] = set()
    dupes: set[str] = set()
    for s in sections:
        for lid in s.source_line_ids:
            if lid in seen:
                dupes.add(lid)
            seen.add(lid)
    checks["no_overlapping_source_lines"] = not dupes
    if dupes:
        violations.append(f"source lines covered twice: {sorted(dupes)}")

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


__all__ = [
    "REASON_LLM_NOT_RUN",
    "build_final_integrity",
    "build_final_sections",
]
