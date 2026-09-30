"""B4 deterministic validation / guardrail tests.

B4 verifies ML labels against structural/model-internal evidence and decides
ACCEPT vs ESCALATE (route to B5). It must never relabel, never call an LLM,
never use semantic lexicons, and never silently skip a block.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from resume_parser.classification import (
    LabelledBlock,
    LabelledDocument,
    SectionClassifier,
    TrainingDataset,
    classify_blocks,
)
from resume_parser.conversion import convert_bytes
from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    ClassificationResult,
    IntegrityReport,
    SectionLabel,
    ValidationPolicy,
    Verdict,
)
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import segment_document
from resume_parser.validation import (
    REASON_CONTINUATION,
    REASON_DUPLICATE,
    REASON_LOW_CONFIDENCE,
    REASON_MISSING,
    REASON_NARROW_MARGIN,
    validate_classification,
)


# ---------------------------------------------------------------- helpers

def _chain(text: str):
    conv = convert_bytes(text.encode("utf-8"), filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
    doc = normalize_document(conv)
    return doc, segment_document(doc)


def _spec(section, conf, alts=(), status=None):
    return (section, conf, list(alts), status)


def _fake_result(seg, specs):
    """Build a ClassificationResult from (section, conf, [(sec, conf)], status)."""
    clss = []
    for b, (sec, conf, alts, st) in zip(seg.blocks, specs):
        clss.append(BlockClassification(
            block_id=b.block_id, document_id=seg.document_id,
            predicted_section=sec, confidence=conf,
            alternatives=[AlternativePrediction(section=s, confidence=c) for s, c in alts],
            source_line_ids=list(b.line_ids),
            start_line_index=b.start_line_index, end_line_index=b.end_line_index,
            classification_status=st or ("classified" if conf >= 0.5 else "low_confidence"),
        ))
    return ClassificationResult(
        document_id=seg.document_id, classifications=clss, model_metadata={},
        integrity=IntegrityReport(passed=True, violations=[], checks={}))


def _labels(seg, section=SectionLabel.EXPERIENCE, conf=0.9, alt_conf=0.05,
            alt_section=SectionLabel.SKILLS, overrides=None):
    specs = []
    for i in range(len(seg.blocks)):
        if overrides and i in overrides:
            specs.append(overrides[i])
        else:
            specs.append(_spec(section, conf, [(alt_section, alt_conf)]))
    return _fake_result(seg, specs)


MULTI = "John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n2022 - 2024\n\nSKILLS\nPython SQL\n"


def test_accept_confident_consistent_blocks():
    _, seg = _chain(MULTI)
    assert len(seg.blocks) >= 2
    res = validate_classification(seg, _labels(seg))
    assert res.integrity.passed, res.integrity.violations
    assert res.n_escalate == 0 and res.n_accept == len(seg.blocks)
    assert res.escalation_rate == 0.0 and res.reason_counts == {}
    for v in res.verdicts:
        assert v.verdict == Verdict.ACCEPT and not v.needs_llm and v.reasons == []


def test_escalate_low_confidence():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={0: _spec(SectionLabel.EXPERIENCE, 0.3, [(SectionLabel.SKILLS, 0.05)])}))
    v = res.verdicts[0]
    assert v.verdict == Verdict.ESCALATE and v.needs_llm
    assert REASON_LOW_CONFIDENCE in v.reasons
    assert res.verdicts[1].verdict == Verdict.ACCEPT


def test_escalate_narrow_margin():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={1: _spec(SectionLabel.SKILLS, 0.8, [(SectionLabel.PROJECTS, 0.75)])}))
    v = res.verdicts[1]
    assert v.verdict == Verdict.ESCALATE and REASON_NARROW_MARGIN in v.reasons
    assert v.evidence["margin"] == pytest.approx(0.05)


def test_margin_boundary_accepts():
    _, seg = _chain(MULTI)
    policy = ValidationPolicy(margin_threshold=0.25)
    ok = validate_classification(seg, _labels(seg, conf=0.6, alt_conf=0.35), policy)
    assert ok.verdicts[0].verdict == Verdict.ACCEPT  # margin == threshold, strict <
    bad = validate_classification(seg, _labels(seg, conf=0.6, alt_conf=0.36), policy)
    assert bad.verdicts[0].verdict == Verdict.ESCALATE


def test_b3_low_confidence_status_escalates_despite_floor():
    _, seg = _chain(MULTI)
    policy = ValidationPolicy(confidence_threshold=0.2)
    res = validate_classification(
        seg, _labels(seg, overrides={0: _spec(SectionLabel.EXPERIENCE, 0.3, [(SectionLabel.SKILLS, 0.05)], "low_confidence")}),
        policy)
    v = res.verdicts[0]
    assert v.verdict == Verdict.ESCALATE and "b3_low_confidence_status" in v.reasons
    assert REASON_LOW_CONFIDENCE not in v.reasons  # floor itself passed


def test_unknown_section_always_escalates():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={0: _spec(SectionLabel.UNKNOWN, 0.9, [(SectionLabel.OTHER, 0.05)])}))
    v = res.verdicts[0]
    assert v.verdict == Verdict.ESCALATE and "unknown_section" in v.reasons


def test_other_confident_accepts():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={0: _spec(SectionLabel.OTHER, 0.8, [(SectionLabel.SKILLS, 0.1)])}))
    assert res.verdicts[0].verdict == Verdict.ACCEPT


def test_continuation_mismatch_escalates():
    _, seg = _chain(MULTI)
    assert len(seg.blocks) >= 2
    blocks = [seg.blocks[0], seg.blocks[1].model_copy(update={"is_continuation": True})] + list(seg.blocks[2:])
    seg2 = seg.model_copy(update={"blocks": blocks})
    res = validate_classification(seg2, _labels(seg2, overrides={
        0: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]),
        1: _spec(SectionLabel.SKILLS, 0.9, [(SectionLabel.PROJECTS, 0.05)]),
    }))
    assert res.verdicts[1].verdict == Verdict.ESCALATE
    assert REASON_CONTINUATION in res.verdicts[1].reasons


def test_continuation_match_accepts():
    _, seg = _chain(MULTI)
    blocks = [seg.blocks[0], seg.blocks[1].model_copy(update={"is_continuation": True})] + list(seg.blocks[2:])
    seg2 = seg.model_copy(update={"blocks": blocks})
    res = validate_classification(seg2, _labels(seg2, overrides={
        0: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]),
        1: _spec(SectionLabel.EXPERIENCE, 0.85, [(SectionLabel.SKILLS, 0.05)]),
    }))
    assert res.verdicts[1].verdict == Verdict.ACCEPT


def test_duplicate_singleton_first_kept_rest_escalated():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={
        0: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SUMMARY, 0.05)]),
        1: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SUMMARY, 0.05)]),
    }))
    assert res.verdicts[0].verdict == Verdict.ACCEPT
    assert res.verdicts[1].verdict == Verdict.ESCALATE
    assert REASON_DUPLICATE in res.verdicts[1].reasons
    # predicted label echoed, NOT relabelled
    assert res.verdicts[1].predicted_section == SectionLabel.CONTACT


def test_duplicate_repeatable_sections_accepted():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={
        0: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]),
        1: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]),
    }))
    assert all(v.verdict == Verdict.ACCEPT for v in res.verdicts)


def test_custom_policy_thresholds_and_singletons():
    _, seg = _chain(MULTI)
    strict = ValidationPolicy(confidence_threshold=0.95)
    res = validate_classification(seg, _labels(seg), strict)
    assert all(v.verdict == Verdict.ESCALATE for v in res.verdicts)
    no_singletons = ValidationPolicy(singleton_sections=())
    res2 = validate_classification(seg, _labels(seg, overrides={
        0: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SUMMARY, 0.05)]),
        1: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SUMMARY, 0.05)]),
    }), no_singletons)
    assert all(v.verdict == Verdict.ACCEPT for v in res2.verdicts)


def test_policy_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ValidationPolicy(confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        ValidationPolicy(margin_threshold=-0.1)


def test_policy_defaults_and_echo():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg))
    assert res.policy.confidence_threshold == 0.5
    assert res.policy.margin_threshold == 0.15
    assert set(res.policy.singleton_sections) == {SectionLabel.CONTACT, SectionLabel.SUMMARY}


def test_needs_llm_and_reasons_invariant():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={
        0: _spec(SectionLabel.EXPERIENCE, 0.3, [(SectionLabel.SKILLS, 0.25)]),
    }))
    for v in res.verdicts:
        assert v.needs_llm == (v.verdict == Verdict.ESCALATE)
        assert bool(v.reasons) == (v.verdict == Verdict.ESCALATE)
    assert res.integrity.passed


def test_alternatives_carried_for_b5():
    _, seg = _chain(MULTI)
    alts = [(SectionLabel.PROJECTS, 0.2), (SectionLabel.EDUCATION, 0.1)]
    res = validate_classification(seg, _labels(seg, overrides={0: _spec(SectionLabel.SKILLS, 0.6, alts)}))
    v = res.verdicts[0]
    assert [(a.section, a.confidence) for a in v.alternatives] == alts
    assert v.confidence == 0.6 and v.classification_status == "classified"


def test_b4_never_relabels():
    _, seg = _chain(MULTI)
    cls = _labels(seg, overrides={
        0: _spec(SectionLabel.VOLUNTEERING, 0.2, [(SectionLabel.SKILLS, 0.19)]),
        1: _spec(SectionLabel.UNKNOWN, 0.9, [(SectionLabel.OTHER, 0.05)]),
    })
    res = validate_classification(seg, cls)
    for v, c in zip(res.verdicts, cls.classifications):
        assert v.predicted_section == c.predicted_section
        assert v.confidence == c.confidence


def test_provenance_preservation():
    doc, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg))
    assert res.document_id == seg.document_id == doc.document_id
    assert [v.block_id for v in res.verdicts] == [b.block_id for b in seg.blocks]
    for v, b in zip(res.verdicts, seg.blocks):
        assert v.source_line_ids == b.line_ids
    assert res.integrity.checks["exactly_one_verdict_per_block_in_order"]


def test_integrity_missing_classification_fails_loudly():
    _, seg = _chain(MULTI)
    full = _labels(seg)
    dropped = [c for c in full.classifications if c.block_id != seg.blocks[0].block_id]
    partial = ClassificationResult(document_id=seg.document_id, classifications=dropped,
                                   model_metadata={},
                                   integrity=IntegrityReport(passed=True, violations=[], checks={}))
    res = validate_classification(seg, partial)
    assert not res.integrity.passed
    assert any(seg.blocks[0].block_id in v for v in res.integrity.violations)
    # ...yet the block still gets an escalate verdict: no silent skip
    assert len(res.verdicts) == len(seg.blocks)
    v0 = res.get_verdict(seg.blocks[0].block_id)
    assert v0.verdict == Verdict.ESCALATE and REASON_MISSING in v0.reasons


def test_integrity_extra_classification_fails():
    _, seg = _chain(MULTI)
    cls = _labels(seg)
    bogus = cls.classifications[0].model_copy(update={"block_id": "B999999"})
    extra = ClassificationResult(document_id=seg.document_id,
                                 classifications=[*cls.classifications, bogus],
                                 model_metadata={},
                                 integrity=IntegrityReport(passed=True, violations=[], checks={}))
    res = validate_classification(seg, extra)
    assert not res.integrity.passed
    assert any("B999999" in v for v in res.integrity.violations)


def test_empty_document():
    _, seg = _chain("   \n  \n")
    assert seg.blocks == []
    cls = ClassificationResult(document_id=seg.document_id, classifications=[],
                               model_metadata={},
                               integrity=IntegrityReport(passed=True, violations=[], checks={}))
    res = validate_classification(seg, cls)
    assert res.verdicts == [] and res.n_accept == 0 and res.n_escalate == 0
    assert res.escalation_rate == 0.0 and res.integrity.passed


def test_summary_stats_exact():
    _, seg = _chain(MULTI)
    assert len(seg.blocks) >= 3
    res = validate_classification(seg, _labels(seg, overrides={
        0: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SUMMARY, 0.05)]),
        1: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SUMMARY, 0.05)]),
        2: _spec(SectionLabel.EXPERIENCE, 0.3, [(SectionLabel.SKILLS, 0.05)]),
    }))
    assert res.n_accept == len(seg.blocks) - 2 and res.n_escalate == 2
    assert res.escalation_rate == pytest.approx(2 / len(seg.blocks))
    assert res.reason_counts[REASON_DUPLICATE] == 1
    assert res.reason_counts[REASON_LOW_CONFIDENCE] == 1
    assert res.get_verdict(seg.blocks[0].block_id).verdict == Verdict.ACCEPT


def test_evidence_contents():
    _, seg = _chain(MULTI)
    res = validate_classification(seg, _labels(seg, overrides={
        0: _spec(SectionLabel.SKILLS, 0.7, [(SectionLabel.PROJECTS, 0.2)]),
    }))
    ev = res.verdicts[0].evidence
    assert ev["confidence"] == 0.7 and ev["margin"] == pytest.approx(0.5)
    assert ev["runner_up"] == "projects" and ev["runner_up_confidence"] == 0.2
    assert ev["confidence_threshold"] == 0.5 and ev["margin_threshold"] == 0.15
    assert ev["is_continuation"] is False and ev["previous_block_label"] is None
    assert res.verdicts[1].evidence["previous_block_label"] == "skills"


def test_determinism_no_shared_state():
    _, seg = _chain(MULTI)
    policy = ValidationPolicy()
    cls = _labels(seg)
    a = validate_classification(seg, cls, policy)
    b = validate_classification(seg, cls, policy)
    assert a.model_dump() == b.model_dump()
    assert policy.confidence_threshold == 0.5  # policy untouched


def test_b3_model_to_b4_wiring():
    """Real ML model -> classify_blocks -> validate: full B3+B4 path."""
    docs = []
    for i in range(2):
        docs.append(LabelledDocument(
            document_id=f"w_exp_{i}", filename=f"e{i}.txt",
            blocks=[LabelledBlock(document_id=f"w_exp_{i}", block_id="B000000",
                                  line_ids=["L000000"], start_line_index=0, end_line_index=0,
                                  text="Senior Software Engineer at Infosys 2020 Present built systems",
                                  gold_section=SectionLabel.EXPERIENCE)]))
        docs.append(LabelledDocument(
            document_id=f"w_ski_{i}", filename=f"s{i}.txt",
            blocks=[LabelledBlock(document_id=f"w_ski_{i}", block_id="B000000",
                                  line_ids=["L000000"], start_line_index=0, end_line_index=0,
                                  text="Python SQL Docker AWS Git Linux",
                                  gold_section=SectionLabel.SKILLS)]))
    clf = SectionClassifier()
    texts, labels, _ = TrainingDataset(docs).get_texts_and_labels()
    clf.train(texts, labels)
    _, seg = _chain(MULTI)
    res = validate_classification(seg, classify_blocks(seg, clf))
    assert res.integrity.passed
    assert len(res.verdicts) == len(seg.blocks)
    assert res.n_accept + res.n_escalate == len(seg.blocks)


def test_boilerplate_lines_in_no_verdict():
    _, seg = _chain("Softwa\nSoftwa\nWORK EXPERIENCE\nAcme\n\fSoftwa\nSoftwa\nSKILLS\nPython\n")
    assert seg.boilerplate
    flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
    res = validate_classification(seg, _labels(seg))
    assert res.integrity.passed
    for v in res.verdicts:
        assert not (set(v.source_line_ids) & flagged)


def test_parse_resume_cli_wiring(tmp_path):
    """parse_resume --model runs B3+B4 and exposes validation + integrity_b4."""
    import json
    from parse_resume import build_output
    clf = SectionClassifier()
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"cli_{sec.value}_{i}"
            docs.append(LabelledDocument(
                document_id=did, filename=f"{did}.txt",
                blocks=[LabelledBlock(document_id=did, block_id="B000000",
                                      line_ids=["L000000"], start_line_index=0,
                                      end_line_index=0, text=text, gold_section=sec)]))
    texts, labels, _ = TrainingDataset(docs).get_texts_and_labels()
    clf.train(texts, labels)
    model_path = tmp_path / "m.pkl"
    clf.save(model_path)
    resume = tmp_path / "r.txt"
    resume.write_text("John Doe\njohn@x.com\n\nSKILLS\nPython SQL Docker\n")
    out = build_output(str(resume), include_eval=False, include_text=False,
                       blocks_only=True, model_path=str(model_path),
                       min_confidence=0.99, min_margin=0.0)
    assert out["integrity_b4"]["passed"] is True
    assert len(out["validation"]["verdicts"]) == out["counts"]["candidate_blocks"]
    assert out["validation"]["policy"]["confidence_threshold"] == 0.99
    json.dumps(out)  # pipe-safe JSON
