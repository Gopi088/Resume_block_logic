"""B6 final section output tests.

B6 merges B4-accepted (ML label, trusted), B5-resolved (LLM label, trusted),
and the rest (ML label as best-effort, UNTRUSTED) into exactly one verified
assignment per B2 block. It must never invent labels, drop blocks, or leak
B7 (entry/date/timeline) concerns.
"""

from __future__ import annotations

import pytest

from resume_parser.conversion import convert_bytes
from resume_parser.final_sections import (
    REASON_LLM_NOT_RUN,
    build_final_integrity,
    build_final_sections,
)
from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    ClassificationResult,
    FinalSource,
    IntegrityReport,
    LLMConfig,
    SectionLabel,
    Verdict,
)
from resume_parser.normalization import normalize_document
from resume_parser.resolution import ScriptedLLMClient, resolve_escalated
from resume_parser.segmentation import segment_document
from resume_parser.validation import validate_classification


# ---------------------------------------------------------------- helpers

MULTI = "John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n2022 - 2024\n\nSKILLS\nPython SQL\n"


def _chain(text: str):
    conv = convert_bytes(text.encode("utf-8"), filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
    doc = normalize_document(conv)
    return doc, segment_document(doc)


def _spec(section, conf, alts=(), status=None):
    return (section, conf, list(alts), status)


def _fake_cls(seg, specs):
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


def _vlabels(seg, overrides=None):
    specs = []
    for i in range(len(seg.blocks)):
        if overrides and i in overrides:
            specs.append(overrides[i])
        else:
            specs.append(_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]))
    return validate_classification(seg, _fake_cls(seg, specs))


def _resolve(seg, val, mapping):
    return resolve_escalated(seg, val, ScriptedLLMClient(mapping), LLMConfig())


# ---------------------------------------------------------------- merge behavior

def test_accepted_blocks_trusted_with_ml_label():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          for _ in seg.blocks])
    val = validate_classification(seg, cls)
    assert all(v.verdict == Verdict.ACCEPT for v in val.verdicts)
    final = build_final_sections(seg, cls, val, None)
    assert final.integrity.passed, final.integrity.violations
    for s, c in zip(final.sections, cls.classifications):
        assert s.source == FinalSource.ML_ACCEPTED and s.trusted
        assert s.final_section == c.predicted_section and s.confidence == c.confidence
    assert final.n_trusted == len(seg.blocks) and final.n_unresolved == 0
    assert final.trust_rate == 1.0


def test_resolved_blocks_take_llm_label():
    _, seg = _chain(MULTI)
    assert len(seg.blocks) >= 2
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          if i != 1 else
                          _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)])
                          for i in range(len(seg.blocks))])
    val = validate_classification(seg, cls)
    bid = seg.blocks[1].block_id
    assert val.get_verdict(bid).needs_llm
    llm = _resolve(seg, val, {bid: {"section": "projects", "confidence": 0.9,
                                    "reason": "built things"}})
    final = build_final_sections(seg, cls, val, llm)
    assert final.integrity.passed, final.integrity.violations
    s = final.get_section(bid)
    assert s.source == FinalSource.LLM_RESOLVED and s.trusted
    assert s.final_section == SectionLabel.PROJECTS and s.confidence == 0.9
    assert s.ml_section == SectionLabel.SKILLS  # ML signal preserved alongside
    assert s.llm_reason == "built things"
    assert final.get_section(seg.blocks[0].block_id).source == FinalSource.ML_ACCEPTED


def test_unresolved_without_llm_keeps_ml_untrusted():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          if i != 1 else
                          _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)])
                          for i in range(len(seg.blocks))])
    val = validate_classification(seg, cls)
    final = build_final_sections(seg, cls, val, None)  # B5 never ran
    assert final.integrity.passed
    s = final.get_section(seg.blocks[1].block_id)
    assert s.source == FinalSource.ML_UNRESOLVED and not s.trusted
    assert s.final_section == SectionLabel.SKILLS  # best-effort ML label kept
    assert s.llm_reason == REASON_LLM_NOT_RUN
    assert final.n_unresolved == 1 and final.trust_rate < 1.0


def test_unresolved_failed_llm_keeps_ml_untrusted():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          if i != 1 else
                          _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)])
                          for i in range(len(seg.blocks))])
    val = validate_classification(seg, cls)
    llm = _resolve(seg, val, {})  # LLM silent on the escalated block
    final = build_final_sections(seg, cls, val, llm)
    s = final.get_section(seg.blocks[1].block_id)
    assert not s.trusted and s.final_section == SectionLabel.SKILLS
    assert s.llm_reason == "llm_no_answer_for_block"


def test_by_section_counts_and_rate():
    _, seg = _chain(MULTI)
    n = len(seg.blocks)
    cls = _fake_cls(seg, [_spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)])
                          for _ in seg.blocks])
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    assert final.n_trusted == 0 and final.n_unresolved == n
    assert final.trust_rate == 0.0
    assert final.by_section == {"skills": n}
    assert sum(final.by_section.values()) == n


def test_order_and_get_section():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          for _ in seg.blocks])
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    assert [s.block_id for s in final.sections] == [b.block_id for b in seg.blocks]
    assert final.get_section("B999999") is None
    assert final.get_section(seg.blocks[0].block_id).document_id == seg.document_id


def test_provenance_and_decision_trail():
    doc, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          if i != 1 else
                          _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.2)])
                          for i in range(len(seg.blocks))])
    val = validate_classification(seg, cls)
    bid = seg.blocks[1].block_id
    llm = _resolve(seg, val, {bid: {"section": "projects", "confidence": 0.8, "reason": "r"}})
    final = build_final_sections(seg, cls, val, llm)
    s = final.get_section(bid)
    assert s.source_line_ids == seg.blocks[1].line_ids
    assert (s.start_line_index, s.end_line_index) == (
        seg.blocks[1].start_line_index, seg.blocks[1].end_line_index)
    assert s.text == seg.blocks[1].text
    assert [(a.section, a.confidence) for a in s.alternatives] == [
        (SectionLabel.PROJECTS, 0.2)]
    assert s.b4_reasons == val.get_verdict(bid).reasons
    assert s.document_id == doc.document_id


# ---------------------------------------------------------------- integrity

def test_integrity_doc_mismatch_fails():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          for _ in seg.blocks])
    bad_cls = ClassificationResult(document_id="doc_other", classifications=cls.classifications,
                                   model_metadata={},
                                   integrity=IntegrityReport(passed=True, violations=[], checks={}))
    final = build_final_sections(seg, bad_cls, validate_classification(seg, cls), None)
    assert not final.integrity.passed
    assert any("classification document_id" in v for v in final.integrity.violations)


def test_integrity_missing_block_fails_but_covers():
    _, seg = _chain(MULTI)
    full = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                           for _ in seg.blocks])
    dropped = [c for c in full.classifications if c.block_id != seg.blocks[0].block_id]
    partial = ClassificationResult(document_id=seg.document_id, classifications=dropped,
                                   model_metadata={},
                                   integrity=IntegrityReport(passed=True, violations=[], checks={}))
    val = validate_classification(seg, partial)
    final = build_final_sections(seg, partial, val, None)
    assert not final.integrity.passed  # missing classification is loud...
    assert len(final.sections) == len(seg.blocks)  # ...but no block is dropped
    assert final.get_section(seg.blocks[0].block_id).llm_reason == "missing_classification_or_verdict"


def test_merge_tamper_detected():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          if i != 1 else
                          _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)])
                          for i in range(len(seg.blocks))])
    val = validate_classification(seg, cls)
    bid = seg.blocks[1].block_id
    llm = _resolve(seg, val, {bid: {"section": "projects", "confidence": 0.8, "reason": "r"}})
    good = build_final_sections(seg, cls, val, llm)
    assert good.integrity.passed
    tampered = [s.model_copy(update={"confidence": s.confidence + 0.01}) if s.block_id == seg.blocks[0].block_id else s
                for s in good.sections]
    rep = build_final_integrity(seg, cls, val, llm, tampered)
    assert not rep.passed and any("ml_accepted final" in v for v in rep.violations)
    tampered2 = [s.model_copy(update={"final_section": SectionLabel.OTHER}) if s.block_id == bid else s
                 for s in good.sections]
    rep2 = build_final_integrity(seg, cls, val, llm, tampered2)
    assert not rep2.passed and any("llm_resolved final" in v for v in rep2.violations)


def test_trusted_flag_consistency():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          if i != 1 else
                          _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)])
                          for i in range(len(seg.blocks))])
    val = validate_classification(seg, cls)
    for llm in (None, _resolve(seg, val, {})):
        final = build_final_sections(seg, cls, val, llm)
        from resume_parser.models import FinalSource as FS
        for s in final.sections:
            assert s.trusted == (s.source != FS.ML_UNRESOLVED)


def test_empty_document():
    _, seg = _chain("   \n  \n")
    assert seg.blocks == []
    cls = ClassificationResult(document_id=seg.document_id, classifications=[],
                               model_metadata={},
                               integrity=IntegrityReport(passed=True, violations=[], checks={}))
    from resume_parser.validation import validate_classification as vc
    final = build_final_sections(seg, cls, vc(seg, cls), None)
    assert final.sections == [] and final.integrity.passed
    assert final.trust_rate == 1.0  # vacuously: nothing untrusted


def test_boilerplate_lines_in_no_final():
    _, seg = _chain("Softwa\nSoftwa\nWORK EXPERIENCE\nAcme\n\fSoftwa\nSoftwa\nSKILLS\nPython\n")
    assert seg.boilerplate
    flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          for _ in seg.blocks])
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    assert final.integrity.passed
    for s in final.sections:
        assert not (set(s.source_line_ids) & flagged)


def test_determinism():
    _, seg = _chain(MULTI)
    cls = _fake_cls(seg, [_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                          for _ in seg.blocks])
    val = validate_classification(seg, cls)
    a = build_final_sections(seg, cls, val, None)
    b = build_final_sections(seg, cls, val, None)
    assert a.model_dump() == b.model_dump()


def test_no_b7_fields_leak():
    """B6 is the section layer: no entry/date/timeline/job concepts allowed."""
    from resume_parser.models import FinalBlockSection, FinalSectionOutput
    block_fields = set(FinalBlockSection.model_fields)
    output_fields = set(FinalSectionOutput.model_fields)
    banned = {"entries", "jobs", "dates", "events", "timeline", "gaps", "entries_by_section"}
    assert not (block_fields & banned) and not (output_fields & banned)
    assert block_fields >= {"block_id", "final_section", "confidence", "source", "trusted",
                            "source_line_ids", "text"}


def test_real_model_chain_to_final():
    """Tiny trained ML model -> B3 -> B4 -> scripted B5 -> B6, integrity green."""
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset, classify_blocks)
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b6_{sec.value}_{i}"
            docs.append(LabelledDocument(
                document_id=did, filename=f"{did}.txt",
                blocks=[LabelledBlock(document_id=did, block_id="B000000",
                                      line_ids=["L000000"], start_line_index=0,
                                      end_line_index=0, text=text, gold_section=sec)]))
    texts, labels, _ = TrainingDataset(docs).get_texts_and_labels()
    clf = SectionClassifier()
    clf.train(texts, labels)
    _, seg = _chain(MULTI)
    cls_result = classify_blocks(seg, clf)
    val = validate_classification(seg, cls_result)
    mapping = {v.block_id: {"section": "experience", "confidence": 0.9, "reason": "t"}
               for v in val.verdicts if v.needs_llm}
    llm = _resolve(seg, val, mapping)
    final = build_final_sections(seg, cls_result, val, llm)
    assert final.integrity.passed
    assert len(final.sections) == len(seg.blocks)
    assert final.n_trusted + final.n_unresolved == len(seg.blocks)


def test_parse_resume_cli_final_sections(tmp_path):
    """build_output --model exposes final_sections + integrity_b6 (no LLM)."""
    import json
    from parse_resume import build_output
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset)
    clf = SectionClassifier()
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b6cli_{sec.value}_{i}"
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
                       blocks_only=True, model_path=str(model_path))
    assert out["integrity_b6"]["passed"] is True
    assert len(out["final_sections"]["sections"]) == out["counts"]["candidate_blocks"]
    assert "llm_resolution" not in out  # B5 not requested
    assert out["final_sections"]["sections"][0]["source"] in (
        "ml_accepted", "ml_unresolved")
    json.dumps(out)
