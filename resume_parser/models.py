"""Pydantic v2 data models for B0 (conversion) and B1 (normalization).

Design notes
------------
- ``ConvertedDocument.extracted_markdown`` is IMMUTABLE provenance.
- ``NormalizedDocument`` keeps ``raw_text`` per line untouched; only
  ``normalized_text`` carries whitespace cleanup.
- All lines are semantically unclassified at B1:
  ``assignment_status == "unassigned_pending_segmentation"``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LineKind(str, Enum):
    BLANK = "blank"
    TEXT = "text"
    MARKDOWN_HEADING = "markdown_heading"
    BULLET = "bullet"
    TABLE_ROW = "table_row"
    CODE_FENCE = "code_fence"
    UNKNOWN = "unknown"


class AssignmentStatus(str, Enum):
    UNASSIGNED_PENDING_SEGMENTATION = "unassigned_pending_segmentation"


class ConverterInfo(BaseModel):
    model_config = {"frozen": True}

    name: str = Field(default="markitdown")
    version: str = Field(default="unknown")
    title: str | None = Field(default=None)


class ConversionWarningModel(BaseModel):
    model_config = {"frozen": True}

    code: str
    message: str
    detail: str = ""


class ConvertedDocument(BaseModel):
    """Immutable B0 output. Never mutated by B1."""

    model_config = {"frozen": True}

    document_id: str
    filename: str
    mime_type: str | None = None
    extension: str | None = None
    sha256: str
    converter: ConverterInfo
    extracted_markdown: str
    extracted_char_count: int
    warnings: list[ConversionWarningModel] = Field(default_factory=list)


class LineRecord(BaseModel):
    """One canonical line with full provenance to ``extracted_markdown``."""

    model_config = {"frozen": True}

    line_id: str  # L000000, L000001, ...
    index: int
    document_id: str
    raw_text: str  # exact span of extracted_markdown, separator excluded
    normalized_text: str  # whitespace-cleaned copy; raw_text never overwritten
    raw_start_char: int  # valid span in extracted_markdown
    raw_end_char: int
    normalized_start_char: int  # valid span in normalized_text
    normalized_end_char: int
    line_kind: LineKind
    page_index: int | None = None  # reserved; MarkItDown gives no pages -> None
    assignment_status: AssignmentStatus = (
        AssignmentStatus.UNASSIGNED_PENDING_SEGMENTATION
    )


class IntegrityReport(BaseModel):
    model_config = {"frozen": True}

    passed: bool
    violations: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    """B1 output: normalized view + immutable provenance + integrity."""

    model_config = {"frozen": True}

    document_id: str
    filename: str
    mime_type: str | None = None
    extension: str | None = None
    sha256: str
    converter: ConverterInfo
    extracted_markdown: str  # immutable copy of B0 output
    normalized_text: str
    lines: list[LineRecord]
    warnings: list[ConversionWarningModel] = Field(default_factory=list)
    integrity: IntegrityReport


class EvaluationReport(BaseModel):
    """Small B0+B1 evaluation harness report."""

    model_config = {"frozen": True}

    document_id: str
    extracted_char_count: int
    normalized_char_count: int
    total_line_count: int
    nonblank_line_count: int
    duplicate_line_id_count: int
    missing_line_count: int
    reconstruction_pass: bool
    lost_line_count: int  # must be 0 on valid fixtures
    integrity_passed: bool
    warnings: list[ConversionWarningModel] = Field(default_factory=list)
    # Placeholder until B2 segmentation exists.
    boundary_accuracy: str = "unavailable_until_B2_segmentation"
