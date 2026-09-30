"""B5 LLM fallback resolution tests.

B5 resolves ONLY B4-escalated blocks via one batched LLM call. Accepted blocks
are never sent or modified; every failure mode is explicitly unresolved.
All LLM behavior is exercised through ScriptedLLMClient (deterministic);
no test touches the network.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from resume_parser.conversion import convert_bytes
from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    ClassificationResult,
    IntegrityReport,
    LLMConfig,
    SectionLabel,
)
from resume_parser.normalization import normalize_document
from resume_parser.resolution import (
    AnthropicLLMClient,
    ScriptedLLMClient,
    build_resolution_prompt,
    parse_resolution_response,
    resolve_escalated,
)
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


def _escalating_val(seg):
    """Block0 accepts; blocks 1-2 escalate (low confidence)."""
    assert len(seg.blocks) >= 3
    return _vlabels(seg, overrides={
        0: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]),
        1: _spec(SectionLabel.SKILLS, 0.3, [(SectionLabel.PROJECTS, 0.1)]),
        2: _spec(SectionLabel.EDUCATION, 0.35, [(SectionLabel.SKILLS, 0.1)]),
    })


def _mapping(seg, sec1="skills", sec2="education"):
    b1, b2 = seg.blocks[1].block_id, seg.blocks[2].block_id
    return {b1: {"section": sec1, "confidence": 0.92, "reason": "skill list"},
            b2: {"section": sec2, "confidence": 0.88, "reason": "degree lines"}}


# ---------------------------------------------------------------- routing discipline

def test_only_escalated_blocks_sent():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    client = ScriptedLLMClient(_mapping(seg))
    res = resolve_escalated(seg, val, client, LLMConfig())
    assert res.integrity.passed, res.integrity.violations
    assert res.n_sent == 2
    prompt = client.prompts[0]
    assert seg.blocks[1].block_id in prompt and seg.blocks[2].block_id in prompt
    assert seg.blocks[0].block_id not in prompt
    assert "john@x.com" not in prompt  # accepted block's text never sent
    assert res.get_resolution(seg.blocks[0].block_id) is None


def test_single_batched_call():
    _, seg = _chain(MULTI)
    client = ScriptedLLMClient(_mapping(seg))
    resolve_escalated(seg, _escalating_val(seg), client, LLMConfig())
    assert len(client.prompts) == 1


def test_no_escalations_no_call():
    _, seg = _chain(MULTI)
    val = _vlabels(seg)  # all accept
    assert all(not v.needs_llm for v in val.verdicts)
    client = ScriptedLLMClient({})
    res = resolve_escalated(seg, val, client, LLMConfig())
    assert client.prompts == [] and res.n_sent == 0
    assert res.resolutions == [] and res.integrity.passed


# ---------------------------------------------------------------- prompt + parsing units

def test_prompt_contains_limited_context():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    client = ScriptedLLMClient(_mapping(seg))
    resolve_escalated(seg, val, client, LLMConfig())
    items = json.loads(client.prompts[0].split("Items:\n", 1)[1])
    assert len(items) == 2
    first = items[0]
    assert first["id"] == seg.blocks[1].block_id
    assert "WORK EXPERIENCE" in first["text"]
    assert first["ml_prediction"] == "skills"
    assert first["escalation_reasons"]  # B4 reasons included
    assert first["alternatives"][0]["section"] == "projects"
    assert first["previous_block_label"] == "experience"  # neighbour context
    assert items[1]["id"] == seg.blocks[2].block_id
    assert "SKILLS" in items[1]["text"]
    assert "Reply with ONLY a JSON object" in client.prompts[0]


def test_parse_valid_and_fences():
    entries, invalid, err = parse_resolution_response(
        '```json\n{"B000001": {"section": "skills", "confidence": 0.9, "reason": "r"}}\n```')
    assert err == "" and invalid == []
    assert entries["B000001"]["section"] == SectionLabel.SKILLS
    assert entries["B000001"]["confidence"] == 0.9


def test_parse_rejects_bad_entries():
    entries, invalid, err = parse_resolution_response(json.dumps({
        "B000001": {"section": "ceo", "confidence": 0.9},          # bad section
        "B000002": {"section": "skills", "confidence": 1.5},       # out of range
        "B000003": {"section": "skills", "confidence": "high"},    # wrong type
        "B000004": {"section": "skills", "confidence": True},      # bool, not number
        "B000005": "skills",                                        # not an object
        "B000006": {"section": "skills", "confidence": 0.7},       # valid
    }))
    assert err == ""
    assert list(entries) == ["B000006"]
    assert sorted(invalid) == ["B000001", "B000002", "B000003", "B000004", "B000005"]


def test_parse_unparseable_and_non_object():
    _, _, err = parse_resolution_response("sorry, no JSON here")
    assert "unparseable" in err
    _, _, err2 = parse_resolution_response("[1, 2, 3]")
    assert "not a JSON object" in err2
    _, _, err3 = parse_resolution_response("")
    assert err3 != ""


# ---------------------------------------------------------------- resolution outcomes

def test_resolved_fields_complete():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    res = resolve_escalated(seg, val, ScriptedLLMClient(_mapping(seg)), LLMConfig())
    assert res.n_resolved == 2 and res.n_unresolved == 0 and res.error == ""
    r = res.get_resolution(seg.blocks[1].block_id)
    assert r.resolved and r.source == "llm"
    assert r.resolved_section == SectionLabel.SKILLS and r.confidence == 0.92
    assert r.reason == "skill list" and r.model == LLMConfig().model
    assert r.prompt_version == "1.0" and r.source_line_ids == seg.blocks[1].line_ids
    assert r.b4_reasons == val.get_verdict(seg.blocks[1].block_id).reasons
    assert r.document_id == seg.document_id


def test_missing_answer_unresolved():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    mapping = _mapping(seg)
    del mapping[seg.blocks[2].block_id]
    res = resolve_escalated(seg, val, ScriptedLLMClient(mapping), LLMConfig())
    assert res.n_resolved == 1 and res.n_unresolved == 1
    r = res.get_resolution(seg.blocks[2].block_id)
    assert not r.resolved and r.resolved_section is None and r.source == "unresolved"
    assert r.reason == "llm_no_answer_for_block" and r.confidence == 0.0
    assert res.integrity.passed  # explicit unknown-state, not a failure


def test_invalid_answer_unresolved():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    mapping = _mapping(seg)
    mapping[seg.blocks[1].block_id] = {"section": "ceo", "confidence": 0.9}
    res = resolve_escalated(seg, val, ScriptedLLMClient(mapping), LLMConfig())
    r = res.get_resolution(seg.blocks[1].block_id)
    assert not r.resolved and r.reason == "llm_invalid_answer"


def test_nan_confidence_unresolved():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    raw = '{"%s": {"section": "skills", "confidence": NaN, "reason": "x"}}' % seg.blocks[1].block_id
    res = resolve_escalated(seg, val, ScriptedLLMClient(raw), LLMConfig())
    assert res.get_resolution(seg.blocks[1].block_id).reason == "llm_invalid_answer"


def test_unparseable_response_all_unresolved_with_error():
    _, seg = _chain(MULTI)
    res = resolve_escalated(seg, _escalating_val(seg), ScriptedLLMClient("not json"), LLMConfig())
    assert res.n_resolved == 0 and res.n_unresolved == 2
    assert "unparseable" in res.error
    assert all((not r.resolved and r.reason == res.error) for r in res.resolutions)
    assert res.integrity.passed


def test_transport_failure_all_unresolved_with_error():
    from resume_parser.resolution import LLMClient

    class Boom(LLMClient):
        def complete(self, prompt: str) -> str:
            raise ConnectionError("dns down")

    _, seg = _chain(MULTI)
    res = resolve_escalated(seg, _escalating_val(seg), Boom(), LLMConfig())
    assert res.n_unresolved == 2 and res.error.startswith("llm_call_failed")
    assert "ConnectionError" in res.error
    assert res.integrity.passed


def test_unknown_response_block_fails_integrity():
    _, seg = _chain(MULTI)
    mapping = _mapping(seg)
    mapping["B999999"] = {"section": "skills", "confidence": 0.9}
    res = resolve_escalated(seg, _escalating_val(seg), ScriptedLLMClient(mapping), LLMConfig())
    assert not res.integrity.passed
    assert any("B999999" in v for v in res.integrity.violations)
    assert res.get_resolution("B999999") is None  # ignored, not smuggled in


def test_budget_cap_excess_unresolved_unsent():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    cfg = LLMConfig(max_escalated_blocks=1)
    res = resolve_escalated(seg, val, ScriptedLLMClient(_mapping(seg)), cfg)
    assert res.n_sent == 1
    excess = res.get_resolution(seg.blocks[2].block_id)
    assert not excess.resolved and excess.reason == "llm_budget_exceeded"
    assert excess.model == ""  # never sent
    assert res.integrity.passed


def test_config_validation():
    assert LLMConfig().max_tokens == 2000
    with pytest.raises(ValidationError):
        LLMConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        LLMConfig(timeout_seconds=-1.0)
    with pytest.raises(ValidationError):
        LLMConfig(max_escalated_blocks=0)


def test_anthropic_client_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMClient(LLMConfig())


def test_unknown_provider_rejected():
    from parse_resume import _run_llm_fallback
    _, seg = _chain(MULTI)
    with pytest.raises(RuntimeError, match="unknown B5 provider"):
        _run_llm_fallback(seg, _escalating_val(seg), "openai", None)


def test_determinism():
    _, seg = _chain(MULTI)
    val = _escalating_val(seg)
    a = resolve_escalated(seg, val, ScriptedLLMClient(_mapping(seg)), LLMConfig())
    b = resolve_escalated(seg, val, ScriptedLLMClient(_mapping(seg)), LLMConfig())
    assert a.model_dump() == b.model_dump()


def test_provenance_and_counts():
    doc, seg = _chain(MULTI)
    val = _escalating_val(seg)
    res = resolve_escalated(seg, val, ScriptedLLMClient(_mapping(seg)), LLMConfig())
    assert res.document_id == seg.document_id == doc.document_id
    assert [r.block_id for r in res.resolutions] == [seg.blocks[1].block_id, seg.blocks[2].block_id]
    for r, b in zip(res.resolutions, [seg.blocks[1], seg.blocks[2]]):
        assert r.source_line_ids == b.line_ids
    assert res.n_resolved + res.n_unresolved == len(res.resolutions) == 2
    assert res.provider == "anthropic" and res.prompt_version == "1.0"


def test_cli_wiring_llm_error_without_key(tmp_path, monkeypatch):
    """Full B0->B5 CLI path: no key -> llm_error key, still valid JSON, no traceback."""
    import json
    from parse_resume import build_output
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    clf = SectionClassifier()
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b5cli_{sec.value}_{i}"
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
                       llm_provider="anthropic", llm_model=None)
    assert "llm_error" in out and "ANTHROPIC_API_KEY" in out["llm_error"]["message"]
    assert "classification" in out and "validation" in out  # B3+B4 intact
    json.dumps(out)
