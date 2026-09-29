"""B2 segmentation tests: headers, hyphen joins, boilerplate, integrity."""

from __future__ import annotations

from resume_parser.conversion import convert_bytes
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import (
    display_text,
    segment_document,
    should_join,
)


def _seg(text: str):
    conv = convert_bytes(text.encode("utf-8"), filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
    return segment_document(normalize_document(conv))


def test_headers_open_labelled_blocks():
    seg = _seg("Jane Doe\njane@mail.com\n\nWORK EXPERIENCE\nAcme Corp\n\nEDUCATION\nB.Tech\n")
    assert seg.integrity.passed, seg.integrity.violations
    labels = [(b.label_proposal, b.proposal_source) for b in seg.blocks]
    assert labels[0] == ("contact_header", "rule_position_contact")
    assert labels[1] == ("experience", "rule_lexicon")
    assert labels[2] == ("education", "rule_lexicon")
    assert seg.blocks[1].header_line_id is not None
    assert seg.blocks[1].display_lines[0].is_header


def test_leading_span_is_contact_header_poc_parity():
    seg = _seg("Aarav Sharma\n+91 99999\n\nSKILLS\nPython\n")
    assert seg.blocks[0].label_proposal == "contact_header"
    assert "Aarav Sharma" in seg.blocks[0].text


def test_hyphen_break_joined_with_provenance():
    seg = _seg("Maintained Java services for finan-\ncial apps on AWS.\n")
    assert seg.hyphen_joins == 1
    assert "financial apps" in seg.blocks[0].text
    assert len(seg.blocks[0].display_lines[0].source_line_ids) == 2
    # raw lines untouched
    assert seg.blocks[0].line_ids == ["L000000", "L000001"]


def test_no_join_without_hyphen_signal():
    seg = _seg("DesignedandoptimizedRESTfulAPIs\nnext line\n")
    assert seg.hyphen_joins == 0  # missing-space corruption is NOT guessed


def test_repeated_page_openings_flagged_not_deleted():
    text = "Softwa\nSoftwa\nWORK EXPERIENCE\nAcme\n\fSoftwa\nSoftwa\nSKILLS\nPython\n"
    seg = _seg(text)
    flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
    assert {"L000000", "L000001", "L000004", "L000005"} <= flagged
    assert seg.page_count == 2
    # flagged lines excluded from block text but present in boilerplate list
    assert "Softwa" not in seg.blocks[0].text
    assert any(b.text == "Softwa" for b in seg.boilerplate)
    assert seg.integrity.passed, seg.integrity.violations


def test_scattered_repeats_are_not_boilerplate():
    seg = _seg("Pune\nAcme\nPune\nBeta\nPune\n")
    assert seg.boilerplate == []
    assert "Pune" in seg.blocks[0].text


def test_display_strips_private_use_glyphs():
    assert display_text("\ue901 ashish@mail.com") == " ashish@mail.com"
    assert "ashish@mail.com" in _seg("\ue901 ashish@mail.com\n").blocks[0].text


def test_should_join_guards():
    assert should_join("de-", "ployed")
    assert not should_join("-", "foo")       # bullet, not a break
    assert not should_join("EC2-", "S3")     # next starts uppercase
    assert not should_join("word", "next")   # no trailing hyphen
    assert not should_join("end -", "next")  # hyphen after space


def test_b2_integrity_catches_coverage_tamper():
    seg = _seg("Jane\n\nSKILLS\nPython\n")
    bad = [b.model_copy(update={"line_ids": b.line_ids[:-1]}) for b in seg.blocks]
    from resume_parser.conversion import convert_bytes as cb
    from resume_parser.normalization import normalize_document as nd
    conv = cb(b"Jane\n\nSKILLS\nPython\n", filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": "Jane\n\nSKILLS\nPython\n",
                                   "extracted_char_count": len("Jane\n\nSKILLS\nPython\n")})
    doc = nd(conv)
    from resume_parser.segmentation import build_b2_integrity, paginate
    rep = build_b2_integrity(doc, paginate(doc.lines), set(), bad)
    assert not rep.passed
