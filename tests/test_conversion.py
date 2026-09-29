"""B0 conversion tests: real MarkItDown paths + explicit failure states."""

from __future__ import annotations

import hashlib

import pytest

from resume_parser.conversion import convert_bytes, convert_file, make_document_id
from resume_parser.errors import ConversionError, UnsupportedFormatError


def test_convert_bytes_txt_roundtrip():
    data = "Jane Doe\nSoftware Engineer\n".encode("utf-8")
    doc = convert_bytes(data, filename="resume.txt", content_type="text/plain")
    assert doc.sha256 == hashlib.sha256(data).hexdigest()
    assert doc.document_id == make_document_id(doc.sha256)
    assert doc.extracted_char_count == len(doc.extracted_markdown)
    assert "Jane Doe" in doc.extracted_markdown
    assert doc.converter.name == "markitdown"
    assert doc.converter.version  # version string present


def test_convert_file_txt(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("John Smith\nData Analyst\n", encoding="utf-8")
    doc = convert_file(p)
    assert "John Smith" in doc.extracted_markdown
    assert doc.filename == "resume.txt"
    assert doc.document_id.startswith("doc_")


def test_convert_bytes_deterministic_id():
    data = b"same content"
    a = convert_bytes(data, filename="a.txt", content_type="text/plain")
    b = convert_bytes(data, filename="a.txt", content_type="text/plain")
    assert a.document_id == b.document_id
    assert a.sha256 == b.sha256


def test_conversion_failure_missing_file():
    with pytest.raises(ConversionError):
        convert_file("/nonexistent/path/resume.pdf")


def test_unsupported_empty_bytes():
    with pytest.raises(UnsupportedFormatError):
        convert_bytes(b"", filename="resume.txt", content_type="text/plain")


def test_unsupported_binary_garbage_extension():
    # Unknown extension + non-text bytes -> explicit typed error, never silent.
    with pytest.raises((UnsupportedFormatError, ConversionError)):
        convert_bytes(b"\x00\x01\x02\xff\xfe\x00\x99", filename="resume.zzzunknown")


def test_unknown_extension_warns_but_attempts(tmp_path):
    data = "Plain text content in odd extension".encode()
    doc = convert_bytes(data, filename="resume.odd-ext-xyz", content_type=None)
    assert "Plain text content" in doc.extracted_markdown
    assert any(w.code == "UNKNOWN_EXTENSION" for w in doc.warnings)
