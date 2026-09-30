"""Integrity tests: required fixture 7 (malformed offsets must fail) + guards."""

from __future__ import annotations

import pytest

from resume_parser.conversion import convert_bytes
from resume_parser.errors import IntegrityError
from resume_parser.integrity import build_integrity_report, verify_integrity
from resume_parser.models import AssignmentStatus, LineKind, LineRecord
from resume_parser.normalization import normalize_document


def _healthy_doc():
    conv = convert_bytes(b"a\nb\n", filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": "a\nb\n", "extracted_char_count": 4})
    return normalize_document(conv)


def _line(**kw) -> LineRecord:
    base = dict(
        line_id="L000000", index=0, document_id="doc_x", raw_text="a",
        normalized_text="a", raw_start_char=0, raw_end_char=1,
        normalized_start_char=0, normalized_end_char=1,
        line_kind=LineKind.TEXT, page_index=None,
        assignment_status=AssignmentStatus.UNASSIGNED_PENDING_SEGMENTATION,
    )
    base.update(kw)
    return LineRecord(**base)


def test_healthy_document_passes():
    doc = _healthy_doc()
    assert doc.integrity.passed
    rep = verify_integrity(
        lines=doc.lines, extracted_markdown=doc.extracted_markdown,
        normalized_text=doc.normalized_text,
    )
    assert rep.passed


# 7. Deliberate malformed offset case -> integrity MUST fail explicitly.
def test_malformed_normalized_offset_fails():
    doc = _healthy_doc()
    bad = [ln.model_copy(update={"normalized_start_char": 999, "normalized_end_char": 1000})
           for ln in doc.lines[:1]] + list(doc.lines[1:])
    rep = build_integrity_report(
        lines=bad, extracted_markdown=doc.extracted_markdown, normalized_text=doc.normalized_text
    )
    assert not rep.passed and rep.violations
    with pytest.raises(IntegrityError):
        verify_integrity(lines=bad, extracted_markdown=doc.extracted_markdown,
                         normalized_text=doc.normalized_text)


def test_malformed_raw_offset_fails():
    doc = _healthy_doc()
    bad = [doc.lines[0].model_copy(update={"raw_start_char": 0, "raw_end_char": 9999})]
    rep = build_integrity_report(
        lines=bad, extracted_markdown=doc.extracted_markdown, normalized_text=doc.normalized_text
    )
    assert not rep.passed


def test_duplicate_line_id_fails():
    a = _line(line_id="L000000", index=0)
    b = _line(line_id="L000000", index=1, raw_text="b", normalized_text="b",
              raw_start_char=2, raw_end_char=3, normalized_start_char=2, normalized_end_char=3)
    rep = build_integrity_report(lines=[a, b], extracted_markdown="a\nb", normalized_text="a\nb")
    assert not rep.passed
    assert any("line_ids" in v for v in rep.violations)


def test_reconstruction_tamper_fails():
    doc = _healthy_doc()
    rep = build_integrity_report(
        lines=doc.lines, extracted_markdown=doc.extracted_markdown,
        normalized_text=doc.normalized_text + " TAMPERED",
    )
    assert not rep.passed


def test_missing_line_detected():
    doc = _healthy_doc()
    rep = build_integrity_report(
        lines=doc.lines[:-1], extracted_markdown=doc.extracted_markdown,
        normalized_text=doc.normalized_text,
    )
    assert not rep.passed
