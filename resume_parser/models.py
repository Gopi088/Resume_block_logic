"""Pydantic v2 data models for B0 (conversion), B1 (normalization), and B2 (candidate block segmentation).

Design notes
------------
- ``ConvertedDocument.extracted_markdown`` is IMMUTABLE provenance.
- ``NormalizedDocument`` keeps ``raw_text`` per line untouched; only
  ``normalized_text`` carries whitespace cleanup.
- All lines are semantically unclassified at B1:
  ``assignment_status == "unassigned_pending_segmentation"``.
- B2 candidate blocks are STRUCTURAL ONLY — no semantic section labels.
  Semantic classification is deferred to B3 (ML).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# B0 / B1 MODELS (unchanged)
# ============================================================

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
    UNASSIGNED_PENDING_ML_CLASSIFICATION = "unassigned_pending_ml_classification"


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


# ============================================================
# B2 MODELS — Candidate Block Segmentation (STRUCTURAL ONLY)
# ============================================================

class BoundarySignal(str, Enum):
    """Structural boundary signals detected during segmentation.
    
    These are FORMATTING/STRUCTURAL cues only — NO semantic meaning.
    """
    BLANK_LINE = "blank_line"
    HEADING_LIKE = "heading_like"
    MARKDOWN_HEADING = "markdown_heading"
    INDENTATION_CHANGE = "indentation_change"
    BULLET_TRANSITION = "bullet_transition"
    TABLE_BOUNDARY = "table_boundary"
    CODE_FENCE_BOUNDARY = "code_fence_boundary"
    PAGE_BREAK = "page_break"
    FORMATTING_CHANGE = "formatting_change"
    CONTINUATION_DETECTED = "continuation_detected"


class DisplayLine(BaseModel):
    """Display-level line with provenance to source B1 lines."""
    model_config = {"frozen": True}

    text: str
    source_line_ids: list[str]  # 1 id normally, 2 when a hyphen-break was joined
    page_index: int | None = None
    is_header: bool = False  # STRUCTURAL ONLY: formatting looks like a heading
    line_kind: LineKind


class CandidateBlock(BaseModel):
    """Structurally coherent candidate block — NO semantic section label.
    
    B3 (ML) will assign semantic labels. B2 only determines which lines
    structurally belong together.
    """
    model_config = {"frozen": True}

    block_id: str  # B000000, B000001, ...
    index: int
    line_ids: list[str]  # LineRecord.line_id in document order (excl. blanks/boilerplate)
    start_line_index: int  # min index of lines in this block
    end_line_index: int    # max index of lines in this block
    page_indices: list[int]
    display_lines: list[DisplayLine]
    text: str  # "\n".join(d.text for d in display_lines)
    
    # Structural metadata only
    header_line_id: str | None = None  # line_id of heading-like line that starts block
    is_continuation: bool = False      # continues previous block without strong boundary
    boundary_signals: list[BoundarySignal] = Field(default_factory=list)
    assignment_status: AssignmentStatus = AssignmentStatus.UNASSIGNED_PENDING_ML_CLASSIFICATION


class BoilerplateLine(BaseModel):
    """Repeated page-opening line flagged as boilerplate. Kept, never deleted."""
    model_config = {"frozen": True}

    line_ids: list[str]
    text: str
    page_indices: list[int]


class SegmentationResult(BaseModel):
    """B2 output: candidate blocks + boilerplate + integrity."""
    model_config = {"frozen": True}

    document_id: str
    page_count: int
    blocks: list[CandidateBlock]
    boilerplate: list[BoilerplateLine]
    hyphen_joins: int
    warnings: list[ConversionWarningModel] = Field(default_factory=list)
    integrity: IntegrityReport


# ============================================================
# B3 MODELS — ML Semantic Classification
# ============================================================

class SectionLabel(str, Enum):
    """Semantic section labels for resume sections.
    
    These are the target classes for the ML classifier.
    """
    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    AWARDS = "awards"
    PUBLICATIONS = "publications"
    LANGUAGES = "languages"
    VOLUNTEERING = "volunteering"
    INTERESTS = "interests"
    REFERENCES = "references"
    OTHER = "other"
    UNKNOWN = "unknown"
    BOILERPLATE = "boilerplate"  # B2-flagged boilerplate lines


# Default section label order for consistent indexing
SECTION_LABELS = [
    SectionLabel.CONTACT,
    SectionLabel.SUMMARY,
    SectionLabel.EXPERIENCE,
    SectionLabel.EDUCATION,
    SectionLabel.SKILLS,
    SectionLabel.PROJECTS,
    SectionLabel.CERTIFICATIONS,
    SectionLabel.AWARDS,
    SectionLabel.PUBLICATIONS,
    SectionLabel.LANGUAGES,
    SectionLabel.VOLUNTEERING,
    SectionLabel.INTERESTS,
    SectionLabel.REFERENCES,
    SectionLabel.OTHER,
    SectionLabel.UNKNOWN,
    SectionLabel.BOILERPLATE,
]


class AlternativePrediction(BaseModel):
    """Alternative section prediction with confidence."""
    model_config = {"frozen": True}
    
    section: SectionLabel
    confidence: float = Field(ge=0.0, le=1.0)


class BlockClassification(BaseModel):
    """Semantic classification result for a single candidate block.
    
    Preserves full provenance from B1/B2.
    """
    model_config = {"frozen": True}
    
    block_id: str
    document_id: str
    predicted_section: SectionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[AlternativePrediction] = Field(default_factory=list)
    source_line_ids: list[str]  # B1 line IDs this classification covers
    start_line_index: int
    end_line_index: int
    # For mixed blocks: individual semantic spans (optional, for future B3 enhancement)
    semantic_spans: list["SemanticSpan"] = Field(default_factory=list)
    classification_status: str = "classified"  # classified, low_confidence, mixed, boilerplate_excluded


class SemanticSpan(BaseModel):
    """A semantic span within a candidate block (for mixed blocks).
    
    Represents a contiguous sub-range of lines that share a semantic section.
    """
    model_config = {"frozen": True}
    
    start_line_id: str
    end_line_id: str
    start_line_index: int
    end_line_index: int
    section: SectionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    text: str  # The actual text content of this span


class ClassificationResult(BaseModel):
    """B3 output: semantic classifications for all candidate blocks."""
    model_config = {"frozen": True}
    
    document_id: str
    classifications: list[BlockClassification]
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    integrity: IntegrityReport
    
    def get_classification(self, block_id: str) -> BlockClassification | None:
        """Get classification by block_id."""
        for c in self.classifications:
            if c.block_id == block_id:
                return c
        return None
