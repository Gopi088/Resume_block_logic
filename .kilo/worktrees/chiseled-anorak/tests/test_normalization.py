"""B1 normalization tests: fixtures 1-6 of the required set."""

from __future__ import annotations

from resume_parser.conversion import convert_bytes
from resume_parser.evaluation import evaluate_document
from resume_parser.models import AssignmentStatus, LineKind
from resume_parser.normalization import (
    classify_line_kind,
    normalize_document,
    normalize_line_text,
    split_raw_lines,
)


def _doc(text: str, filename: str = "r.txt") -> object:
    conv = convert_bytes(text.encode("utf-8"), filename=filename, content_type="text/plain")
    # Bypass MarkItDown reformatting for contract-precise fixtures: build the
    # ConvertedDocument view directly over the exact input text.
    conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
    return normalize_document(conv)


# 1. Plain text resume with blank lines + Unicode.
def test_plain_text_unicode_blank_lines():
    text = "José García\nSenior Engineer — München\n\njose@example.com • +49 170\n"
    doc = _doc(text)
    assert doc.integrity.passed, doc.integrity.violations
    assert "José García" in doc.normalized_text
    assert "München" in doc.normalized_text
    assert any(ln.line_kind == LineKind.BLANK for ln in doc.lines)
    assert all(ln.assignment_status == AssignmentStatus.UNASSIGNED_PENDING_SEGMENTATION for ln in doc.lines)
    rep = evaluate_document(doc)
    assert rep.lost_line_count == 0 and rep.reconstruction_pass


# 2. Markdown-like resume: headings, bullets, links, tables.
def test_markdown_features():
    text = (
        "# Jane Doe\n"
        "## Experience\n"
        "- Built [parser](https://example.com) for résumés\n"
        "1. Led team\n"
        "| Name | Year |\n"
        "| A | 2020 |\n"
        "```python\ncode()\n```\n"
    )
    doc = _doc(text)
    kinds = {ln.normalized_text: ln.line_kind for ln in doc.lines}
    assert kinds["# Jane Doe"] == LineKind.MARKDOWN_HEADING
    assert kinds["- Built [parser](https://example.com) for résumés"] == LineKind.BULLET
    assert kinds["1. Led team"] == LineKind.BULLET
    assert kinds["| Name | Year |"] == LineKind.TABLE_ROW
    assert kinds["```python"] == LineKind.CODE_FENCE
    assert "https://example.com" in doc.normalized_text  # URLs preserved
    assert doc.integrity.passed


# 3. Mixed CRLF + LF endings.
def test_mixed_line_endings():
    text = "line1\r\nline2\nline3\rline4"
    doc = _doc(text)
    assert doc.integrity.passed, doc.integrity.violations
    assert doc.normalized_text == "line1\nline2\nline3\nline4"
    assert [ln.raw_text for ln in doc.lines] == ["line1", "line2", "line3", "line4"]
    for ln in doc.lines:  # raw spans valid in original incl. \r\n
        assert doc.extracted_markdown[ln.raw_start_char:ln.raw_end_char] == ln.raw_text


# 4. Repeated identical lines: no dedup, order preserved, IDs unique.
def test_repeated_identical_lines():
    text = "Acme Corp\nAcme Corp\nAcme Corp"
    doc = _doc(text)
    assert [ln.normalized_text for ln in doc.lines] == ["Acme Corp"] * 3
    assert len({ln.line_id for ln in doc.lines}) == 3
    assert [ln.line_id for ln in doc.lines] == ["L000000", "L000001", "L000002"]
    assert doc.integrity.passed


# 5. Leading, internal, trailing blank lines preserved.
def test_blank_line_positions():
    text = "\n\nTitle\n\nBody\n\n"
    doc = _doc(text)
    assert doc.normalized_text == text  # exact: blanks never collapsed
    assert doc.lines[0].line_kind == LineKind.BLANK
    assert doc.lines[-1].line_kind == LineKind.BLANK
    assert "\n".join(ln.normalized_text for ln in doc.lines) == doc.normalized_text
    assert doc.integrity.passed


# 6. Tabs + trailing whitespace: raw kept, normalized stripped-right only.
def test_tabs_trailing_whitespace():
    text = "\tIndented line   \nkey:\tvalue\t \n   \n"
    doc = _doc(text)
    assert doc.lines[0].raw_text == "\tIndented line   "
    assert doc.lines[0].normalized_text == "\tIndented line"  # leading tab kept
    assert doc.lines[1].normalized_text == "key:\tvalue"  # internal tab kept
    assert doc.lines[2].raw_text == "   " and doc.lines[2].normalized_text == ""
    assert doc.integrity.passed


def test_raw_never_overwritten_and_offsets_tile():
    text = "a  \n\nb"
    doc = _doc(text)
    assert doc.lines[0].raw_text == "a  "
    assert doc.lines[0].normalized_text == "a"
    assert doc.lines[0].normalized_start_char == 0
    assert doc.lines[1].normalized_start_char == len("a") + 1


def test_split_raw_lines_separator_coverage():
    spans = split_raw_lines("a\r\nb\rc\nd")
    assert [t for t, _, _ in spans] == ["a", "b", "c", "d"]
    raw = "a\r\nb\rc\nd"
    for t, s, e in spans:
        assert raw[s:e] == t


def test_classify_kinds_unit():
    assert classify_line_kind("") == LineKind.BLANK
    assert classify_line_kind("## H") == LineKind.MARKDOWN_HEADING
    assert classify_line_kind("- x") == LineKind.BULLET
    assert classify_line_kind("| a |") == LineKind.TABLE_ROW
    assert classify_line_kind("```") == LineKind.CODE_FENCE
    assert classify_line_kind("plain") == LineKind.TEXT
    assert normalize_line_text("   ") == ""
