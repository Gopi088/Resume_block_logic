"""B7 entry segmentation tests.

B7 splits section blocks into entries on date-range boundaries (jobs in
EXPERIENCE, degrees in EDUCATION, projects in PROJECTS). It never merges
across blocks, never parses dates, and carries section/trust through.
"""

from __future__ import annotations

import pytest

from resume_parser.conversion import convert_bytes
from resume_parser.entries import (
    build_entries,
    build_entry_integrity,
    has_date_range,
    is_entry_boundary,
    is_short_bare_year,
    parse_line_index,
)
from resume_parser.final_sections import build_final_sections
from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    ClassificationResult,
    FinalSource,
    IntegrityReport,
    SectionLabel,
)
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import segment_document
from resume_parser.validation import validate_classification


# ---------------------------------------------------------------- helpers

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


def _final(seg, overrides=None):
    specs = []
    for i in range(len(seg.blocks)):
        if overrides and i in overrides:
            specs.append(overrides[i])
        else:
            specs.append(_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]))
    cls = _fake_cls(seg, specs)
    return build_final_sections(seg, cls, validate_classification(seg, cls), None)


JOBS = ("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer | Acme\n04/2024 - Present\n"
        "Built APIs\n\nJava Developer | Beta\n03/2022 - 04/2024\nFixed bugs\n")


# ---------------------------------------------------------------- boundary unit tests

def test_has_date_range_formats():
    for line in ["04/2024 - Present", "03/2022 - 04/2024", "Jan 2019 - Present",
                 "March 2020 to Dec 2021", "2016 - 2018", "2015-2017",
                 "Jun 2021\u2013Mar 2023", "2020 - current", "1/2020-3/2022"]:
        assert has_date_range(line), line


def test_has_date_range_rejects_bare_years_and_mentions():
    assert not has_date_range("2018")
    assert not has_date_range("cut cost by 30% in 2021")
    assert not has_date_range("Senior Engineer")
    assert not has_date_range("")


def test_bare_year_rule():
    assert is_short_bare_year("2018")
    assert not is_short_bare_year("Graduated in 2018 with honors")
    assert not is_short_bare_year("04/2024 - Present")


def test_boundary_section_policy():
    assert is_entry_boundary("04/2024 - Present", SectionLabel.EXPERIENCE)
    assert is_entry_boundary("2016 - 2018", SectionLabel.PROJECTS)
    assert is_entry_boundary("2018", SectionLabel.EDUCATION)
    assert not is_entry_boundary("2018", SectionLabel.EXPERIENCE)  # bare year, no split
    assert not is_entry_boundary("04/2024 - Present", SectionLabel.SKILLS)  # not a date section
    assert not is_entry_boundary("Python SQL", SectionLabel.EXPERIENCE)


def test_parse_line_index_strict():
    assert parse_line_index("L000123") == 123
    assert parse_line_index("L7") == 7
    with pytest.raises(ValueError):
        parse_line_index("B000001")
    with pytest.raises(ValueError):
        parse_line_index("hello")


# ---------------------------------------------------------------- splitting behavior

def test_experience_multi_job_split():
    _, seg = _chain(JOBS)
    exp_blocks = [b for b in seg.blocks if "WORK EXPERIENCE" in b.text or "Acme" in b.text or "Beta" in b.text]
    assert exp_blocks, "expected an experience block"
    final = _final(seg, overrides={i: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                                   for i in range(len(seg.blocks))})
    ent = build_entries(seg, final)
    assert ent.integrity.passed, ent.integrity.violations
    exp_entries = [e for e in ent.entries if e.section == SectionLabel.EXPERIENCE]
    assert len(exp_entries) >= 2  # at least two date-delimited jobs
    assert any("04/2024 - Present" in e.text for e in exp_entries)
    assert any("03/2022 - 04/2024" in e.text for e in exp_entries)
    reasons = {e.split_reason for e in exp_entries}
    assert "date_boundary" in reasons


def test_header_stays_with_first_entry():
    _, seg = _chain(JOBS)
    final = _final(seg, overrides={0: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SKILLS, 0.05)]),
                                   **{i: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                                      for i in range(1, len(seg.blocks))}})
    ent = build_entries(seg, final)
    first_exp = next(e for e in ent.entries if e.section == SectionLabel.EXPERIENCE)
    assert "WORK EXPERIENCE" in first_exp.text
    assert first_exp.split_reason in ("block_start_first", "block_start_single")


def test_education_bare_year_split():
    _, seg = _chain("EDUCATION\n2018\nB.Tech\n\n2020\nM.Tech\n")
    final = _final(seg, overrides={i: _spec(SectionLabel.EDUCATION, 0.9, [(SectionLabel.SKILLS, 0.05)])
                                   for i in range(len(seg.blocks))})
    ent = build_entries(seg, final)
    assert ent.integrity.passed
    edu = [e for e in ent.entries if e.section == SectionLabel.EDUCATION]
    assert len(edu) >= 2
    assert any("B.Tech" in e.text for e in edu) and any("M.Tech" in e.text for e in edu)


def test_skills_single_entry_passthrough():
    _, seg = _chain("Jane\nmail\n\nSKILLS\nPython SQL 2020 - 2024\nDocker\n")
    final = _final(seg, overrides={
        0: _spec(SectionLabel.CONTACT, 0.9, [(SectionLabel.SKILLS, 0.05)]),
        **{i: _spec(SectionLabel.SKILLS, 0.9, [(SectionLabel.PROJECTS, 0.05)])
           for i in range(1, len(seg.blocks))}})
    ent = build_entries(seg, final)
    assert ent.integrity.passed
    assert len(ent.entries) == len(seg.blocks)  # no splitting outside date sections
    assert all(e.split_reason == "block_start_single" for e in ent.entries)


def test_no_date_block_single_entry():
    _, seg = _chain("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\nBuilt things\n")
    final = _final(seg)
    ent = build_entries(seg, final)
    exp = [e for e in ent.entries if e.section == SectionLabel.EXPERIENCE]
    # one entry per block, no date splits anywhere
    assert len(exp) == len(seg.blocks)
    assert all(e.split_reason == "block_start_single" for e in exp)


def test_bullet_year_mention_does_not_split():
    _, seg = _chain("04/2024 - Present\nSenior Engineer\ncut cost 30% in 2021\nMore work\n")
    final = _final(seg, overrides={i: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                                   for i in range(len(seg.blocks))})
    ent = build_entries(seg, final)
    exp = [e for e in ent.entries if e.section == SectionLabel.EXPERIENCE]
    assert len(exp) == 1  # range on line 0 starts the entry; the year mention splits nothing


def test_trust_and_source_propagate():
    _, seg = _chain(JOBS)
    final = _final(seg, overrides={
        i: (_spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
            if i != 1 else _spec(SectionLabel.SKILLS, 0.2, [(SectionLabel.PROJECTS, 0.15)]))
        for i in range(len(seg.blocks))})
    ent = build_entries(seg, final)
    for e in ent.entries:
        fsec = final.get_section(e.block_id)
        assert e.section == fsec.final_section and e.trusted == fsec.trusted
        assert e.source == fsec.source.value
    assert any(not e.trusted for e in ent.entries)  # the low-conf block flows through


# ---------------------------------------------------------------- integrity

def test_ids_unique_contiguous_ordered():
    _, seg = _chain(JOBS)
    ent = build_entries(seg, _final(seg))
    assert [e.entry_id for e in ent.entries] == [f"E{i:06d}" for i in range(len(ent.entries))]
    assert [e.index for e in ent.entries] == list(range(len(ent.entries)))
    assert ent.n_entries == len(ent.entries)


def test_coverage_exactly_once_and_ordered():
    _, seg = _chain(JOBS)
    ent = build_entries(seg, _final(seg))
    expected = [lid for b in seg.blocks for lid in b.line_ids]
    covered = [lid for e in ent.entries for lid in e.line_ids]
    assert sorted(covered) == sorted(expected) and len(set(covered)) == len(covered)
    assert ent.integrity.passed


def test_no_cross_block_entries():
    _, seg = _chain(JOBS)
    ent = build_entries(seg, _final(seg))
    block_lines = {b.block_id: set(b.line_ids) for b in seg.blocks}
    for e in ent.entries:
        assert set(e.line_ids) <= block_lines[e.block_id]
    assert ent.integrity.checks["entries_within_single_block"]


def test_indices_recomputed():
    _, seg = _chain(JOBS)
    ent = build_entries(seg, _final(seg))
    for e in ent.entries:
        idx = [int(lid[1:]) for lid in e.line_ids]
        assert e.start_line_index == min(idx) and e.end_line_index == max(idx)
        assert e.start_line_index <= e.end_line_index


def test_tamper_overlap_detected():
    _, seg = _chain(JOBS)
    ent = build_entries(seg, _final(seg))
    assert len(ent.entries) >= 2
    bad = [e.model_copy(update={"line_ids": ent.entries[0].line_ids}) if e.entry_id == ent.entries[1].entry_id else e
           for e in ent.entries]
    rep = build_entry_integrity(seg, _final(seg), bad)
    assert not rep.passed


def test_empty_document():
    _, seg = _chain("   \n  \n")
    assert seg.blocks == []
    final = _final(seg)
    ent = build_entries(seg, final)
    assert ent.entries == [] and ent.n_entries == 0 and ent.integrity.passed
    assert ent.by_section == {}


def test_boilerplate_lines_in_no_entry():
    _, seg = _chain("Softwa\nSoftwa\nWORK EXPERIENCE\nAcme\n04/2024 - Present\n\fSoftwa\nSoftwa\nSKILLS\nPython\n")
    assert seg.boilerplate
    flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
    ent = build_entries(seg, _final(seg))
    assert ent.integrity.passed
    for e in ent.entries:
        assert not (set(e.line_ids) & flagged)


def test_determinism():
    _, seg = _chain(JOBS)
    final = _final(seg)
    a = build_entries(seg, final)
    b = build_entries(seg, final)
    assert a.model_dump() == b.model_dump()


def test_no_b8_fields_leak():
    """B7 is structure only: no parsed dates, chronology, or gaps allowed."""
    from resume_parser.models import BlockEntry, EntrySegmentationResult
    entry_fields = set(BlockEntry.model_fields)
    result_fields = set(EntrySegmentationResult.model_fields)
    banned = {"dates", "start_date", "end_date", "events", "timeline",
              "chronology", "gaps", "duration", "parsed_date"}
    assert not (entry_fields & banned) and not (result_fields & banned)
    assert entry_fields >= {"entry_id", "block_id", "section", "line_ids", "text",
                            "split_reason", "trusted"}


def test_real_model_chain_to_entries():
    """Tiny trained ML model -> B3 -> B4 -> B6 -> B7, integrity green."""
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset, classify_blocks)
    from resume_parser.validation import validate_classification as vc
    from resume_parser.final_sections import build_final_sections as bfs
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b7_{sec.value}_{i}"
            docs.append(LabelledDocument(
                document_id=did, filename=f"{did}.txt",
                blocks=[LabelledBlock(document_id=did, block_id="B000000",
                                      line_ids=["L000000"], start_line_index=0,
                                      end_line_index=0, text=text, gold_section=sec)]))
    texts, labels, _ = TrainingDataset(docs).get_texts_and_labels()
    clf = SectionClassifier()
    clf.train(texts, labels)
    _, seg = _chain(JOBS)
    final = bfs(seg, classify_blocks(seg, clf), vc(seg, classify_blocks(seg, clf)), None)
    ent = build_entries(seg, final)
    assert ent.integrity.passed
    assert ent.n_entries >= len(seg.blocks) and ent.n_blocks == len(seg.blocks)
    assert sum(ent.by_section.values()) == ent.n_entries


def test_parse_resume_cli_entries(tmp_path):
    """build_output --model exposes entries + integrity_b7."""
    import json
    from parse_resume import build_output
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset)
    clf = SectionClassifier()
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b7cli_{sec.value}_{i}"
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
    resume.write_text("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWorked\n")
    out = build_output(str(resume), include_eval=False, include_text=False,
                       blocks_only=True, model_path=str(model_path))
    assert out["integrity_b7"]["passed"] is True
    assert out["counts"]["entries"] == len(out["entries"]["entries"])
    assert all("entry_id" in e and "split_reason" in e for e in out["entries"]["entries"])
    json.dumps(out)
