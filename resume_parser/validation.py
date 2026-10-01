"""B4: Deterministic Validation / Guardrails.

B4 answers: "Does the ML interpretation make structural sense, or should this
block be escalated?" It NEVER relabels (that would make deterministic logic
the classifier) and NEVER calls an LLM (B5, not implemented).

Inputs: B2 SegmentationResult (structure) + B3 ClassificationResult (ML labels).
Output: ValidationResult with one verdict per B2 block:
  ACCEPT   -> confident and structurally consistent; B6 may use the ML label.
  ESCALATE -> ambiguous or conflicting; needs_llm=True with machine-readable
              reasons for B5. Content is never dropped or rewritten.

Guardrail checks (all structural/model-internal, NO semantic lexicons):
  1. confidence_floor      — confidence < policy.confidence_threshold
  2. narrow_margin          — top1 - runner_up < policy.margin_threshold
  3. b3_low_confidence      — B3 classification_status == "low_confidence"
  4. unknown_section        — predicted UNKNOWN (model explicitly uncertain)
  5. continuation_mismatch  — B2 is_continuation but label differs from the
                              previous block's predicted label
  6. duplicate_singleton    — a second+ block claiming a singleton section
                              (CONTACT/SUMMARY by default); first is kept,
                              later ones escalate

Document-level integrity: exactly one verdict per B2 block (a missing
classification escalates with reason "missing_classification" instead of being
skipped); verdict ids match block ids; no source line is covered twice.
"""

from __future__ import annotations

from .models import (
    BlockClassification,
    BlockVerdict,
    CandidateBlock,
    ClassificationResult,
    IntegrityReport,
    SectionLabel,
    SegmentationResult,
    ValidationPolicy,
    ValidationResult,
    Verdict,
)

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_NARROW_MARGIN = "narrow_margin"
REASON_B3_STATUS = "b3_low_confidence_status"
REASON_UNKNOWN = "unknown_section"
REASON_CONTINUATION = "continuation_label_mismatch"
REASON_DUPLICATE = "duplicate_singleton_section"
REASON_MISSING = "missing_classification"
REASON_MIXED_SPANS = "mixed_semantic_spans"
REASON_SPAN_LOW_CONFIDENCE = "low_confidence_semantic_span"


def _margin(classification: BlockClassification) -> tuple[float, SectionLabel | None, float]:
    """Top1 minus runner-up margin from the model's own distribution."""
    runner_up: SectionLabel | None = None
    runner_conf = 0.0
    for alt in classification.alternatives:
        if alt.confidence > runner_conf:
            runner_up, runner_conf = alt.section, alt.confidence
    return classification.confidence - runner_conf, runner_up, runner_conf


def _verdict_for_block(
    block: CandidateBlock,
    classification: BlockClassification | None,
    previous_label: SectionLabel | None,
    seen_singletons: set[SectionLabel],
    policy: ValidationPolicy,
    document_id: str,
) -> BlockVerdict:
    """Pure per-block guardrail evaluation. Never relabels."""
    if classification is None:
        # Cannot verify what was never classified: escalate loudly, don't skip.
        return BlockVerdict(
            block_id=block.block_id,
            document_id=document_id,
            predicted_section=SectionLabel.UNKNOWN,
            confidence=0.0,
            alternatives=[],
            classification_status="missing",
            source_line_ids=list(block.line_ids),
            verdict=Verdict.ESCALATE,
            needs_llm=True,
            reasons=[REASON_MISSING],
            evidence={"previous_block_label": previous_label.value if previous_label else None},
        )

    reasons: list[str] = []
    margin, runner_up, runner_conf = _margin(classification)
    if policy.llm_verify_all:
        reasons.append("full_llm_verification")

    if classification.confidence < policy.confidence_threshold:
        reasons.append(REASON_LOW_CONFIDENCE)
    if margin < policy.margin_threshold:
        reasons.append(REASON_NARROW_MARGIN)
    if classification.classification_status == "low_confidence":
        reasons.append(REASON_B3_STATUS)
    span_labels = {span.section for span in classification.semantic_spans}
    if len(span_labels) > 1:
        reasons.append(REASON_MIXED_SPANS)
    if any(span.confidence < policy.confidence_threshold for span in classification.semantic_spans):
        reasons.append(REASON_SPAN_LOW_CONFIDENCE)
    if classification.predicted_section == SectionLabel.UNKNOWN:
        reasons.append(REASON_UNKNOWN)
    if block.is_continuation and previous_label is not None \
            and classification.predicted_section != previous_label:
        reasons.append(REASON_CONTINUATION)
    if classification.predicted_section in policy.singleton_sections \
            and classification.predicted_section in seen_singletons:
        reasons.append(REASON_DUPLICATE)

    verdict = Verdict.ESCALATE if reasons else Verdict.ACCEPT
    return BlockVerdict(
        block_id=block.block_id,
        document_id=document_id,
        predicted_section=classification.predicted_section,
        confidence=classification.confidence,
        alternatives=list(classification.alternatives),
        classification_status=classification.classification_status,
        source_line_ids=list(classification.source_line_ids),
        verdict=verdict,
        needs_llm=verdict == Verdict.ESCALATE,
        reasons=reasons,
        evidence={
            "confidence": classification.confidence,
            "margin": margin,
            "runner_up": runner_up.value if runner_up else None,
            "runner_up_confidence": runner_conf,
            "confidence_threshold": policy.confidence_threshold,
            "margin_threshold": policy.margin_threshold,
            "is_continuation": block.is_continuation,
            "previous_block_label": previous_label.value if previous_label else None,
            "semantic_spans": [
                {"section": span.section.value, "ml_confidence": span.confidence,
                 "start_line_id": span.start_line_id, "end_line_id": span.end_line_id,
                 "text": span.text, "feature_context": list(span.feature_context)}
                for span in classification.semantic_spans
            ],
        },
    )


def build_validation_integrity(
    seg: SegmentationResult,
    cls_result: ClassificationResult,
    verdicts: list[BlockVerdict],
) -> IntegrityReport:
    """Document-level B4 integrity: full coverage, no duplication, id match."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    block_ids = [b.block_id for b in seg.blocks]
    verdict_ids = [v.block_id for v in verdicts]
    classified_ids = {c.block_id for c in cls_result.classifications}

    ok_cover = verdict_ids == block_ids
    checks["exactly_one_verdict_per_block_in_order"] = ok_cover
    if not ok_cover:
        violations.append(
            f"verdict coverage mismatch: {len(block_ids)} blocks, {len(verdict_ids)} verdicts."
        )

    missing_cls = [b for b in block_ids if b not in classified_ids]
    checks["all_blocks_classified"] = not missing_cls
    if missing_cls:
        violations.append(f"blocks without B3 classification (escalated): {missing_cls}")

    extra_cls = sorted(classified_ids - set(block_ids))
    checks["no_unknown_classifications"] = not extra_cls
    if extra_cls:
        violations.append(f"classifications for unknown blocks: {extra_cls}")

    seen: set[str] = set()
    dupes: set[str] = set()
    for v in verdicts:
        for lid in v.source_line_ids:
            if lid in seen:
                dupes.add(lid)
            seen.add(lid)
    checks["no_overlapping_source_lines"] = not dupes
    if dupes:
        violations.append(f"source lines covered by multiple verdicts: {sorted(dupes)}")

    ok_flag = all(v.needs_llm == (v.verdict == Verdict.ESCALATE) for v in verdicts)
    checks["needs_llm_matches_verdict"] = ok_flag
    if not ok_flag:
        violations.append("needs_llm flag inconsistent with verdict.")

    ok_reasons = all(bool(v.reasons) == (v.verdict == Verdict.ESCALATE) for v in verdicts)
    checks["reasons_present_iff_escalated"] = ok_reasons
    if not ok_reasons:
        violations.append("reasons must be non-empty exactly for escalated verdicts.")

    return IntegrityReport(passed=not violations, violations=violations, checks=checks)


def validate_classification(
    seg: SegmentationResult,
    cls_result: ClassificationResult,
    policy: ValidationPolicy | None = None,
) -> ValidationResult:
    """Main B4 entry point: pure function of (structure, ML labels, policy)."""
    policy = policy or ValidationPolicy()
    by_block_id = {c.block_id: c for c in cls_result.classifications}

    verdicts: list[BlockVerdict] = []
    seen_singletons: set[SectionLabel] = set()
    previous_label: SectionLabel | None = None
    for block in seg.blocks:
        classification = by_block_id.get(block.block_id)
        verdicts.append(_verdict_for_block(
            block, classification, previous_label,
            seen_singletons, policy, seg.document_id,
        ))
        if classification is not None:
            if classification.predicted_section in policy.singleton_sections:
                seen_singletons.add(classification.predicted_section)
            previous_label = classification.predicted_section

    n_escalate = sum(1 for v in verdicts if v.verdict == Verdict.ESCALATE)
    n_accept = len(verdicts) - n_escalate
    reason_counts: dict[str, int] = {}
    for v in verdicts:
        for r in v.reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    return ValidationResult(
        document_id=seg.document_id,
        verdicts=verdicts,
        policy=policy,
        n_accept=n_accept,
        n_escalate=n_escalate,
        escalation_rate=(n_escalate / len(verdicts)) if verdicts else 0.0,
        reason_counts=reason_counts,
        integrity=build_validation_integrity(seg, cls_result, verdicts),
    )


__all__ = [
    "REASON_B3_STATUS",
    "REASON_CONTINUATION",
    "REASON_DUPLICATE",
    "REASON_LOW_CONFIDENCE",
    "REASON_MISSING",
    "REASON_NARROW_MARGIN",
    "REASON_UNKNOWN",
    "build_validation_integrity",
    "validate_classification",
]
