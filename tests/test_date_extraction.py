"""B8 date/event extraction tests.

B8 parses resume date formats deterministically with char-level provenance,
selects each entry's primary range, and never orders, times, or gaps anything
(B9). No ML, no LLM, no network.
"""

from __future__ import annotations

from datetime import date

from resume_parser.conversion import convert_bytes
from resume_parser.date_extraction import (
    extract_entry_dates,
    extract_text_dates,
)
from resume_parser.entries import build_entries
from resume_parser.final_sections import build_final_sections
from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    ClassificationResult,
    DateExtractionPolicy,
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


def _dated(text: str, overrides=None):
    _, seg = _chain(text)
    final = _final(seg, overrides=overrides)
    return extract_entry_dates(build_entries(seg, final))


def _first_range(text: str):
    rgs, _ = extract_text_dates(text)
    assert len(rgs) == 1, text
    return rgs[0]


# ---------------------------------------------------------------- format parsing

def test_numeric_range_present():
    r = _first_range("04/2024 - Present")
    assert (r.start.year, r.start.month) == (2024, 4) and r.end is None and r.is_ongoing
    assert r.raw == "04/2024 - Present" and (r.start_char, r.end_char) == (0, 17)


def test_numeric_range_closed():
    r = _first_range("03/2022 - 04/2024")
    assert (r.start.year, r.start.month) == (2022, 3)
    assert (r.end.year, r.end.month) == (2024, 4) and not r.is_ongoing


def test_month_name_ranges():
    r = _first_range("Jan 2019 - Mar 2021")
    assert (r.start.year, r.start.month) == (2019, 1)
    assert (r.end.year, r.end.month) == (2021, 3)
    r2 = _first_range("March 2020 to Dec 2021")
    assert (r2.start.month, r2.end.month) == (3, 12)
    r3 = _first_range("Jun 2021\u2013Mar 2023")  # en dash
    assert (r3.start.year, r3.end.year) == (2021, 2023)


def test_year_ranges_and_ongoing_variants():
    assert _first_range("2016 - 2018").start.year == 2016
    assert _first_range("2015-2017").end.year == 2017
    for marker in ("Present", "present", "PRESENT", "Current", "Now", "Till date"):
        r = _first_range(f"2020 - {marker}")
        assert r.is_ongoing and r.end is None, marker


def test_mixed_granularity_preserved():
    r = _first_range("Jan 2019 - 2021")
    assert (r.start.year, r.start.month) == (2019, 1)
    assert (r.end.year, r.end.month) == (2021, None)
    r2 = _first_range("2019 - Mar 2021")
    assert (r2.start.year, r2.start.month) == (2019, None)
    assert (r2.end.month) == 3


def test_day_granularity():
    r = _first_range("12 Jan 2020 - 15 Mar 2021")
    assert (r.start.year, r.start.month, r.start.day) == (2020, 1, 12)
    assert (r.end.year, r.end.month, r.end.day) == (2021, 3, 15)
    assert r.start.granularity.value == "day"


def test_year_first_numeric_single():
    rgs, sgs = extract_text_dates("2020-04")
    assert rgs == [] and len(sgs) == 1
    assert (sgs[0].year, sgs[0].month) == (2020, 4)


def test_bare_year_and_month_singles():
    rgs, sgs = extract_text_dates("2018")
    assert rgs == [] and [(s.raw, s.year) for s in sgs] == [("2018", 2018)]
    _, sgs2 = extract_text_dates("Joined Mar 2020")
    assert [(s.year, s.month) for s in sgs2] == [(2020, 3)]
    _, sgs3 = extract_text_dates("Since Mar. 2020")
    assert [(sgs3[0].year, sgs3[0].month)] == [(2020, 3)]


def test_invalid_dates_rejected():
    rgs, _ = extract_text_dates("13/2020 - 14/2021")  # month 13/14 invalid
    assert rgs == []
    rgs2, _ = extract_text_dates("00/2020 - 01/2021")
    assert rgs2 == []
    assert extract_text_dates("Senior Engineer")[0] == []
    assert extract_text_dates("cut cost 30% in 2021")[0] == []  # year is a single, not a range


def test_range_masks_inner_singles():
    rgs, sgs = extract_text_dates("04/2024 - Present")
    assert len(rgs) == 1 and sgs == []
    rgs2, sgs2 = extract_text_dates("2016 - 2018")
    assert len(rgs2) == 1 and sgs2 == []


def test_multiple_dates_in_order():
    rgs, sgs = extract_text_dates("2018 graduate, worked 2020 - 2022, then Jan 2023 - Present")
    assert [r.raw for r in rgs] == ["2020 - 2022", "Jan 2023 - Present"]
    assert [s.raw for s in sgs] == ["2018"]


def test_spans_reconstruct():
    text = "Senior Engineer 04/2024 - Present at Acme since 2018"
    rgs, sgs = extract_text_dates(text)
    for r in rgs + sgs:
        assert text[r.start_char:r.end_char] == r.raw
    for r in rgs:
        assert text[r.start.start_char:r.start.end_char] == r.start.raw
        if r.end is not None:
            assert text[r.end.start_char:r.end.end_char] == r.end.raw
            assert r.start.start_char >= r.start_char and r.end.end_char <= r.end_char


def test_endpoint_spans_inside_range():
    r = _first_range("04/2024 - Present")
    assert (r.start.raw, (r.start.start_char, r.start.end_char)) == ("04/2024", (0, 7))


# ---------------------------------------------------------------- entry-level behavior

def test_entry_without_dates():
    res = _dated("John Doe\njohn@x.com\n\nSKILLS\nPython SQL Docker\n")
    assert res.integrity.passed, res.integrity.violations
    assert res.n_with_dates == 0 and res.n_ranges == 0
    for ed in res.entry_dates:
        assert ed.ranges == [] and ed.single_dates == []
        assert ed.primary_range is None and not ed.has_dates


def test_primary_prefers_range_over_single():
    # Single and range inside ONE entry: the range wins (entry-tenure rule).
    res = _dated("WORK EXPERIENCE\n04/2024 - Present\ngraduated 2018\nSenior Engineer\n",
                 {i: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                  for i in range(10)})
    dated = [ed for ed in res.entry_dates if ed.has_dates]
    assert len(dated) == 1
    assert dated[0].primary_range is not None
    assert dated[0].primary_range.raw == "04/2024 - Present"
    assert dated[0].primary_range.is_ongoing


def test_primary_point_from_single():
    res = _dated("EDUCATION\n2018\nB.Tech\n",
                 {i: _spec(SectionLabel.EDUCATION, 0.9, [(SectionLabel.SKILLS, 0.05)])
                  for i in range(10)})
    dated = [ed for ed in res.entry_dates if ed.has_dates]
    assert dated
    p = dated[0].primary_range
    assert p is not None and p.start.year == 2018 and p.end == p.start
    assert not p.is_ongoing


def test_untrusted_entries_still_parsed():
    res = _dated("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n",
                 {0: _spec(SectionLabel.SKILLS, 0.2, [(SectionLabel.PROJECTS, 0.15)])})
    assert res.integrity.passed
    assert any(ed.has_dates for ed in res.entry_dates)
    untrusted = [ed for ed in res.entry_dates if not ed.trusted]
    assert untrusted  # trust flows through; dates parsed regardless


def test_reference_date_recorded():
    _, seg = _chain("John\n\nSKILLS\nPython\n")
    ent = build_entries(seg, _final(seg))
    from resume_parser.date_extraction import extract_entry_dates as ex
    explicit = ex(ent, DateExtractionPolicy(reference_date=date(2025, 6, 30)))
    assert explicit.reference_date == date(2025, 6, 30)
    defaulted = ex(ent, DateExtractionPolicy())
    assert defaulted.reference_date == date.today()


def test_counts_and_helpers():
    res = _dated("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n\nEDUCATION\n2018\nB.Tech\n",
                 {i: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)])
                  for i in range(10)})
    assert res.n_entries == len(res.entry_dates)
    assert res.n_with_dates >= 2 and res.n_ranges >= 1
    assert res.dates_for_entry(res.entry_dates[0].entry_id) is not None
    assert res.dates_for_entry("E999999") is None


def test_tampered_span_detected():
    _, seg = _chain("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n")
    ent = build_entries(seg, _final(seg, overrides={i: _spec(SectionLabel.EXPERIENCE, 0.9, [(SectionLabel.SKILLS, 0.05)]) for i in range(10)}))
    from resume_parser.date_extraction import build_dates_integrity, extract_entry_dates as ex
    good = ex(ent)
    assert good.integrity.passed
    target = next(ed for ed in good.entry_dates if ed.ranges)
    bad_entry = target.model_copy(update={
        "ranges": [target.ranges[0].model_copy(update={"start_char": 999})]})
    bad = [bad_entry if e.entry_id == bad_entry.entry_id else e for e in good.entry_dates]
    rep = build_dates_integrity(ent, bad)
    assert not rep.passed


def test_empty_entries():
    _, seg = _chain("   \n  \n")
    ent = build_entries(seg, _final(seg))
    res = extract_entry_dates(ent)
    assert res.entry_dates == [] and res.n_entries == 0 and res.integrity.passed


def test_determinism():
    _, seg = _chain("WORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n")
    ent = build_entries(seg, _final(seg))
    from resume_parser.date_extraction import extract_entry_dates as ex
    from resume_parser.models import DateExtractionPolicy as Pol
    a = ex(ent, Pol(reference_date=date(2026, 1, 1)))
    b = ex(ent, Pol(reference_date=date(2026, 1, 1)))
    assert a.model_dump() == b.model_dump()


def test_no_b9_fields_leak():
    """B8 extracts dates only: no chronology, durations, gaps, or ordering."""
    from resume_parser.models import DateExtractionResult, EntryDates
    entry_fields = set(EntryDates.model_fields)
    result_fields = set(DateExtractionResult.model_fields)
    banned = {"timeline", "chronology", "gaps", "durations", "order", "sorted_entries",
              "total_experience", "overlaps", "conflicts"}
    assert not (entry_fields & banned) and not (result_fields & banned)
    assert entry_fields >= {"entry_id", "ranges", "single_dates", "primary_range", "has_dates"}


def test_real_model_chain_to_dates():
    """Tiny trained ML model -> B3 -> B4 -> B6 -> B7 -> B8, integrity green."""
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
            did = f"b8_{sec.value}_{i}"
            docs.append(LabelledDocument(
                document_id=did, filename=f"{did}.txt",
                blocks=[LabelledBlock(document_id=did, block_id="B000000",
                                      line_ids=["L000000"], start_line_index=0,
                                      end_line_index=0, text=text, gold_section=sec)]))
    texts, labels, _ = TrainingDataset(docs).get_texts_and_labels()
    clf = SectionClassifier()
    clf.train(texts, labels)
    _, seg = _chain("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n")
    cls_result = classify_blocks(seg, clf)
    final = bfs(seg, cls_result, vc(seg, cls_result), None)
    res = ex(be(seg, final))
    assert res.integrity.passed
    assert res.n_with_dates >= 1 and res.n_ranges >= 1


def test_parse_resume_cli_entry_dates(tmp_path):
    """build_output --model exposes entry_dates + integrity_b8."""
    import json
    from parse_resume import build_output
    from resume_parser.classification import (
        LabelledBlock, LabelledDocument, SectionClassifier, TrainingDataset)
    clf = SectionClassifier()
    docs = []
    for i in range(2):
        for sec, text in ((SectionLabel.EXPERIENCE, "Senior Engineer at Infosys 2020 built systems"),
                          (SectionLabel.SKILLS, "Python SQL Docker AWS")):
            did = f"b8cli_{sec.value}_{i}"
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
    resume.write_text("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n04/2024 - Present\nWork\n")
    out = build_output(str(resume), include_eval=False, include_text=False,
                       blocks_only=True, model_path=str(model_path))
    assert out["integrity_b8"]["passed"] is True
    assert out["counts"]["date_ranges"] >= 1
    assert any(e["has_dates"] for e in out["entry_dates"]["entry_dates"])
    json.dumps(out)
