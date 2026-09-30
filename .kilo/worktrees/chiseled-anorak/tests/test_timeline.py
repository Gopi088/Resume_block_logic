"""B9 timeline + gap detection tests.

B9 sorts dated entries chronologically and measures employment gaps over
EXPERIENCE events. It never forecasts, scores, or invents dates.
"""

from __future__ import annotations

from datetime import date

from resume_parser.conversion import convert_bytes
from resume_parser.date_extraction import extract_entry_dates
from resume_parser.entries import build_entries
from resume_parser.final_sections import build_final_sections
from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    ClassificationResult,
    IntegrityReport,
    SectionLabel,
    TimelinePolicy,
)
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import segment_document
from resume_parser.timeline import build_timeline, build_timeline_integrity
from resume_parser.validation import validate_classification


# ---------------------------------------------------------------- helpers

REF = date(2026, 9, 29)


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


def _labels_for(seg, mapping):
    """mapping: block_index -> SectionLabel (default EXPERIENCE)."""
    return _fake_cls(seg, [_spec(mapping.get(i, SectionLabel.EXPERIENCE), 0.9,
                                 [(SectionLabel.SKILLS, 0.05)])
                           for i in range(len(seg.blocks))])


def _timeline(text: str, mapping=None, policy=None):
    _, seg = _chain(text)
    cls = _labels_for(seg, mapping or {})
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    ent = build_entries(seg, final)
    dated = extract_entry_dates(ent)
    return build_timeline(dated, ent, policy or TimelinePolicy(reference_date=REF))


JOBS_GAP = ("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer | Acme\n04/2024 - Present\n"
            "Built APIs\n\nJava Developer | Beta\n01/2020 - 03/2022\nFixed bugs\n")


# ---------------------------------------------------------------- chronology

def test_events_chronological_with_ongoing():
    tl = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    assert tl.integrity.passed, tl.integrity.violations
    starts = [e.start_date for e in tl.events]
    assert starts == sorted(starts)
    ongoing = [e for e in tl.events if e.is_ongoing]
    assert len(ongoing) == 1
    assert ongoing[0].end_date is None and ongoing[0].effective_end_date == REF
    assert tl.reference_date == REF


def test_undated_entries_explicit_not_lost():
    tl = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    assert tl.undated_entry_ids  # contact lines have no dates
    assert tl.n_undated == len(tl.undated_entry_ids)
    assert not (set(tl.undated_entry_ids) & {e.entry_id for e in tl.events})


def test_untrusted_entries_on_timeline():
    tl = _timeline("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n")
    assert tl.n_events >= 1 and tl.integrity.passed


def test_reference_inherited_from_b8():
    from resume_parser.date_extraction import extract_entry_dates as ex
    from resume_parser.entries import build_entries as be
    from resume_parser.models import DateExtractionPolicy as DPol
    _, seg = _chain(JOBS_GAP)
    cls = _labels_for(seg, {0: SectionLabel.CONTACT})
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    ent = be(seg, final)
    dated = ex(ent, DPol(reference_date=date(2025, 1, 15)))
    tl = build_timeline(dated, ent, TimelinePolicy())  # no pin: inherits B8
    assert tl.reference_date == date(2025, 1, 15)
    tl2 = build_timeline(dated, ent, TimelinePolicy(reference_date=date(2024, 6, 1)))
    assert tl2.reference_date == date(2024, 6, 1)


# ---------------------------------------------------------------- gaps

def test_gap_detected_exact_days():
    tl = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    assert tl.n_gaps == 1
    g = tl.gaps[0]
    assert (g.start_date, g.end_date) == (date(2022, 4, 1), date(2024, 3, 31))
    assert g.gap_days == 731 and g.gap_months_approx == 24.0
    assert g.gap_id == "G000000"
    assert tl.total_gap_days == 731 and tl.longest_gap_days == 731


def test_no_gap_when_continuous():
    tl = _timeline("WORK EXPERIENCE\nSenior Engineer\n04/2022 - Present\nWork\n\nJunior Engineer\n04/2020 - 03/2022\nStuff\n")
    assert tl.n_gaps == 0 and tl.gaps == [] and tl.integrity.passed


def test_threshold_filters_short_breaks():
    text = "WORK EXPERIENCE\nSenior Engineer\n06/2022 - Present\nWork\n\nJunior\n01/2020 - 04/2022\nStuff\n"
    tl90 = _timeline(text, policy=TimelinePolicy(reference_date=REF, min_gap_days=90))
    assert tl90.n_gaps == 0  # May 2022 = 31 days < 90
    tl30 = _timeline(text, policy=TimelinePolicy(reference_date=REF, min_gap_days=30))
    assert tl30.n_gaps == 1
    assert tl30.gaps[0].gap_days == 31  # May 1-31


def test_overlapping_jobs_no_gap():
    text = ("WORK EXPERIENCE\nSenior Engineer\n01/2021 - Present\nWork\n\n"
            "Consultant\n06/2020 - 06/2022\nStuff\n")
    tl = _timeline(text)
    assert tl.n_gaps == 0 and tl.integrity.passed


def test_education_does_not_bridge_gap():
    text = ("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n\n"
            "Junior\n01/2020 - 03/2022\nStuff\n\nEDUCATION\n2022 - 2023\nMasters\n")
    _, seg = _chain(text)
    # label education block properly: find it by content
    mapping = {}
    for i, b in enumerate(seg.blocks):
        mapping[i] = SectionLabel.EDUCATION if ("EDUCATION" in b.text or "Masters" in b.text) else SectionLabel.EXPERIENCE
    tl = _timeline(text, mapping)
    assert tl.n_gaps == 1  # 2022-04..2024-03 gap stands despite 2022-2023 education
    assert any(e.section == SectionLabel.EDUCATION for e in tl.events)  # still on timeline


def test_gap_endpoints_reference_covering_events():
    tl = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    g = tl.gaps[0]
    by_id = {e.entry_id: e for e in tl.events}
    assert by_id[g.before_entry_id].effective_end_date < g.start_date
    assert g.end_date < by_id[g.after_entry_id].start_date


def test_single_job_no_gaps():
    tl = _timeline("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n")
    assert tl.n_gaps == 0 and tl.n_events >= 1 and tl.integrity.passed


def test_mixed_granularity_comparison():
    tl = _timeline("WORK EXPERIENCE\nSenior Engineer\n2020 - Present\nWork\n\nJunior\n2015 - 2019\nStuff\n")
    assert tl.n_gaps == 0  # 2019-12-31 -> 2020-01-01 adjacent
    assert tl.integrity.passed


def test_policy_validation():
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        TimelinePolicy(min_gap_days=0)


# ---------------------------------------------------------------- integrity

def test_partition_and_helpers():
    tl = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    assert tl.n_events + tl.n_undated == tl.n_events + len(tl.undated_entry_ids)
    assert tl.integrity.checks["dated_undated_partition_complete"]
    assert tl.integrity.checks["events_chronological"]


def test_tampered_gap_detected():
    tl = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    bad = [g.model_copy(update={"gap_days": g.gap_days + 5}) if i == 0 else g
           for i, g in enumerate(tl.gaps)]
    _, seg = _chain(JOBS_GAP)
    cls = _labels_for(seg, {0: SectionLabel.CONTACT})
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    from resume_parser.entries import build_entries as be
    from resume_parser.date_extraction import extract_entry_dates as ex
    ent = be(seg, final)
    dated = ex(ent)
    from resume_parser.timeline import build_timeline_integrity
    rep = build_timeline_integrity(dated, ent, tl.events, tl.undated_entry_ids, bad, REF,
                                   TimelinePolicy(reference_date=REF))
    assert not rep.passed


def test_empty_timeline():
    _, seg = _chain("   \n  \n")
    cls = _labels_for(seg, {})
    final = build_final_sections(seg, cls, validate_classification(seg, cls), None)
    from resume_parser.entries import build_entries as be
    from resume_parser.date_extraction import extract_entry_dates as ex
    tl = build_timeline(ex(be(seg, final)), be(seg, final),
                        TimelinePolicy(reference_date=REF))
    assert tl.events == [] and tl.gaps == [] and tl.undated_entry_ids == []
    assert tl.integrity.passed and tl.total_gap_days == 0


def test_determinism():
    a = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    b = _timeline(JOBS_GAP, {0: SectionLabel.CONTACT})
    assert a.model_dump() == b.model_dump()


def test_no_forecast_fields_leak():
    """B9 measures the stated past/present: no scores, predictions, or advice."""
    from resume_parser.models import CareerGap, TimelineEvent, TimelineResult
    fields = set(TimelineEvent.model_fields) | set(CareerGap.model_fields) | set(TimelineResult.model_fields)
    banned = {"score", "rating", "recommendation", "prediction", "forecast", "employable",
              "employability", "advice", "suggestion", "total_experience", "duration_years"}
    assert not (fields & banned)
    assert {"events", "gaps", "undated_entry_ids", "reference_date"} <= fields


def test_real_model_chain_to_timeline():
    """Tiny trained ML model -> B3 -> B4 -> B6 -> B7 -> B8 -> B9, green."""
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset, classify_blocks)
    from resume_parser.validation import validate_classification as vc
    from resume_parser.final_sections import build_final_sections as bfs
    from resume_parser.entries import build_entries as be
    from resume_parser.date_extraction import extract_entry_dates as ex
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b9_{sec.value}_{i}"
            docs.append(LabelledDocument(
                document_id=did, filename=f"{did}.txt",
                blocks=[LabelledBlock(document_id=did, block_id="B000000",
                                      line_ids=["L000000"], start_line_index=0,
                                      end_line_index=0, text=text, gold_section=sec)]))
    texts, labels, _ = TrainingDataset(docs).get_texts_and_labels()
    clf = SectionClassifier()
    clf.train(texts, labels)
    _, seg = _chain(JOBS_GAP)
    cls_result = classify_blocks(seg, clf)
    final = bfs(seg, cls_result, vc(seg, cls_result), None)
    ent = be(seg, final)
    tl = build_timeline(ex(ent), ent, TimelinePolicy(reference_date=REF))
    assert tl.integrity.passed
    assert tl.n_events + tl.n_undated == len(ent.entries)


def test_parse_resume_cli_timeline(tmp_path):
    """build_output --model exposes timeline + integrity_b9."""
    import json
    from parse_resume import build_output
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset)
    clf = SectionClassifier()
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b9cli_{sec.value}_{i}"
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
    resume.write_text("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\n"
                      "Work\n\nJunior\n01/2020 - 03/2022\nStuff\n")
    out = build_output(str(resume), include_eval=False, include_text=False,
                       blocks_only=True, model_path=str(model_path))
    assert out["integrity_b9"]["passed"] is True
    assert out["counts"]["timeline_events"] >= 1
    assert "reference_date" in out["timeline"]
    json.dumps(out)
