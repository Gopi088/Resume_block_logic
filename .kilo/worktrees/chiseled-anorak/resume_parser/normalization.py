"""B1: Conservative normalization + line representation with full provenance.

NORMALIZATION CONTRACT (exact, testable)
----------------------------------------
Input : ``extracted_markdown`` (immutable, any of \\r\\n / \\r / \\n endings).
Step 1: Split ``extracted_markdown`` into raw lines WITHOUT losing information:
        scan for separators (``\\r\\n`` | ``\\r`` | ``\\n``). Each raw line's
        ``raw_text`` is the content span excluding the separator; ``raw_start_char``
        / ``raw_end_char`` are exact spans in ``extracted_markdown`` such that
        ``extracted_markdown[start:end] == raw_text``.
Step 2: Per line, derive ``normalized_text`` by whitespace cleanup ONLY:
          - if ``raw_text.strip() == ""`` -> ``""`` (blank fold; raw kept intact)
          - else ``raw_text.rstrip(" \\t\\f\\v")`` (strip trailing spaces/tabs
            only; leading/internal spacing, punctuation, Unicode, bullets,
            markdown markers, URLs, table pipes are all preserved).
        No lowercasing, no punctuation removal, no merge/split of nonblank lines.
Step 3: ``normalized_text`` (document) = ``"\\n".join(line.normalized_text)``.
        Line ``normalized_start/end`` tile this string exactly:
        line 0 starts at 0; each next line starts at prev_end + 1 (the \\n);
        last line ends at len(normalized_text).
Step 4: IDs ``L000000..`` assigned sequentially in document order.

RECONSTRUCTION CONTRACT
-----------------------
``"\\n".join(line.normalized_text for line in lines) == normalized_text``
exactly (verified by integrity.py). Raw text is recoverable line-by-line via
``extracted_markdown[raw_start:raw_end]``; separators are NOT stored per line
by design (they are normalized to ``\\n`` at document level).

line_kind is SYNTACTIC ONLY (no semantics, no section logic):
  blank -> normalized == ""; code_fence -> stripped starts with ``` or ~~~;
  markdown_heading -> ^#{1,6}\\s; table_row -> stripped starts with "|";
  bullet -> "- "/"* "/"+ " or "1. "/"1) " style; else text (non-blank).
"""

from __future__ import annotations

import re

from .models import (
    AssignmentStatus,
    ConvertedDocument,
    ConverterInfo,
    LineKind,
    LineRecord,
    NormalizedDocument,
)
from .integrity import build_integrity_report

_HEADING_RE = re.compile(r"^#{1,6}\s")
_BULLET_RE = re.compile(r"^(?:[-*+]\s+|\d{1,3}[.)]\s+)")
_TRAILING_WS = " \t\f\v"


def normalize_line_text(raw_text: str) -> str:
    """Whitespace cleanup affecting ONLY the normalized copy."""
    if raw_text.strip() == "":
        return ""
    return raw_text.rstrip(_TRAILING_WS)


def classify_line_kind(normalized_text: str) -> LineKind:
    if normalized_text == "":
        return LineKind.BLANK
    s = normalized_text.strip()
    if s.startswith("```") or s.startswith("~~~"):
        return LineKind.CODE_FENCE
    if _HEADING_RE.match(s):
        return LineKind.MARKDOWN_HEADING
    if s.startswith("|"):
        return LineKind.TABLE_ROW
    if _BULLET_RE.match(s):
        return LineKind.BULLET
    if s == "":
        return LineKind.BLANK
    return LineKind.TEXT


def split_raw_lines(extracted: str) -> list[tuple[str, int, int]]:
    """Split on \\r\\n | \\r | \\n; return (raw_text, start, end) per line.

    Guarantees: ``extracted[start:end] == raw_text`` for every line, and the
    concatenation of spans + separators reproduces ``extracted`` exactly.
    """
    spans: list[tuple[str, int, int]] = []
    if extracted == "":
        return spans  # empty doc -> zero lines (documented edge case)
    n = len(extracted)
    start = 0
    i = 0
    while i < n:
        ch = extracted[i]
        if ch == "\r":
            spans.append((extracted[start:i], start, i))
            if i + 1 < n and extracted[i + 1] == "\n":
                i += 2
            else:
                i += 1
            start = i
        elif ch == "\n":
            spans.append((extracted[start:i], start, i))
            i += 1
            start = i
        else:
            i += 1
    spans.append((extracted[start:n], start, n))  # trailing segment (may be "")
    return spans


def format_line_id(index: int) -> str:
    return f"L{index:06d}"


def build_lines(
    converted: ConvertedDocument, raw_spans: list[tuple[str, int, int]]
) -> tuple[list[LineRecord], str]:
    lines: list[LineRecord] = []
    norm_texts: list[str] = []
    cursor = 0
    for index, (raw_text, rs, re_) in enumerate(raw_spans):
        norm = normalize_line_text(raw_text)
        norm_texts.append(norm)
        ns, ne = cursor, cursor + len(norm)
        lines.append(
            LineRecord(
                line_id=format_line_id(index),
                index=index,
                document_id=converted.document_id,
                raw_text=raw_text,
                normalized_text=norm,
                raw_start_char=rs,
                raw_end_char=re_,
                normalized_start_char=ns,
                normalized_end_char=ne,
                line_kind=classify_line_kind(norm),
                page_index=None,
                assignment_status=AssignmentStatus.UNASSIGNED_PENDING_SEGMENTATION,
            )
        )
        cursor = ne + 1  # +1 for the "\n" separator in the join contract
    normalized_text = "\n".join(norm_texts)
    return lines, normalized_text


def normalize_document(converted: ConvertedDocument) -> NormalizedDocument:
    """Pure function: immutable B0 input -> B1 normalized representation."""
    raw_spans = split_raw_lines(converted.extracted_markdown)
    lines, normalized_text = build_lines(converted, raw_spans)
    doc = NormalizedDocument(
        document_id=converted.document_id,
        filename=converted.filename,
        mime_type=converted.mime_type,
        extension=converted.extension,
        sha256=converted.sha256,
        converter=ConverterInfo(
            name=converted.converter.name,
            version=converted.converter.version,
            title=converted.converter.title,
        ),
        extracted_markdown=converted.extracted_markdown,
        normalized_text=normalized_text,
        lines=lines,
        warnings=list(converted.warnings),
        integrity=build_integrity_report(
            lines=lines,
            extracted_markdown=converted.extracted_markdown,
            normalized_text=normalized_text,
        ),
    )
    return doc


__all__ = [
    "build_lines",
    "classify_line_kind",
    "format_line_id",
    "normalize_document",
    "normalize_line_text",
    "split_raw_lines",
]
