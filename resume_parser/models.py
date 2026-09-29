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

from datetime import date
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


# ============================================================
# B4 MODELS — Deterministic Validation / Guardrails
# ============================================================

class Verdict(str, Enum):
    """B4 decision per candidate block.

    ACCEPT: confident and structurally consistent -> use the ML label.
    ESCALATE: ambiguous or conflicting -> route to B5 (LLM), never relabelled here.
    """
    ACCEPT = "accept"
    ESCALATE = "escalate"


class ValidationPolicy(BaseModel):
    """Frozen, explicit guardrail thresholds. No global state; passed in."""

    model_config = {"frozen": True}

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    margin_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    # Sections expected at most once per resume (document knowledge, not a
    # classifier). Repeats are escalated, never relabelled.
    singleton_sections: tuple[SectionLabel, ...] = (
        SectionLabel.CONTACT,
        SectionLabel.SUMMARY,
    )


class BlockVerdict(BaseModel):
    """B4 verdict for one candidate block. Carries everything B5 will need."""

    model_config = {"frozen": True}

    block_id: str
    document_id: str
    predicted_section: SectionLabel  # B3 label, echoed (B4 never relabels)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[AlternativePrediction] = Field(default_factory=list)
    classification_status: str = "classified"
    source_line_ids: list[str]
    verdict: Verdict
    needs_llm: bool  # == (verdict == ESCALATE); the B5 routing flag
    reasons: list[str] = Field(default_factory=list)  # empty iff ACCEPT
    evidence: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """B4 output: one verdict per B2 block plus document-level summary."""

    model_config = {"frozen": True}

    document_id: str
    verdicts: list[BlockVerdict]
    policy: ValidationPolicy
    n_accept: int
    n_escalate: int
    escalation_rate: float
    reason_counts: dict[str, int] = Field(default_factory=dict)
    integrity: IntegrityReport

    def get_verdict(self, block_id: str) -> BlockVerdict | None:
        """Get verdict by block_id."""
        for v in self.verdicts:
            if v.block_id == block_id:
                return v
        return None


# ============================================================
# B5 MODELS — LLM Fallback Resolution
# ============================================================

class LLMConfig(BaseModel):
    """Frozen LLM call configuration. Secrets never live here (env only)."""

    model_config = {"frozen": True}

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = Field(default=2000, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    # Safety cap: at most this many escalated blocks per document per call.
    # Excess blocks stay explicitly unresolved, never silently dropped.
    max_escalated_blocks: int = Field(default=100, gt=0)
    prompt_version: str = "1.0"


class BlockResolution(BaseModel):
    """B5 outcome for ONE escalated block. Unresolved is explicit, never a guess."""

    model_config = {"frozen": True}

    block_id: str
    document_id: str
    resolved_section: SectionLabel | None  # None when unresolved
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""  # LLM's factual reason, or the unresolved explanation
    resolved: bool  # True iff the LLM returned a valid section
    source: str = "llm"  # llm | unresolved
    source_line_ids: list[str]
    b4_reasons: list[str] = Field(default_factory=list)  # why it was escalated
    model: str = ""
    prompt_version: str = "1.0"


class LLMResolutionResult(BaseModel):
    """B5 output: resolutions for escalated blocks only. Accepted blocks are
    NOT repeated here (B6 merges accepted + resolved later)."""

    model_config = {"frozen": True}

    document_id: str
    resolutions: list[BlockResolution]
    n_sent: int  # escalated blocks sent to the LLM
    n_resolved: int
    n_unresolved: int
    provider: str
    model: str
    prompt_version: str
    error: str = ""  # non-empty iff the LLM call itself failed
    integrity: IntegrityReport

    def get_resolution(self, block_id: str) -> BlockResolution | None:
        """Get resolution by block_id."""
        for r in self.resolutions:
            if r.block_id == block_id:
                return r
        return None


# ============================================================
# B6 MODELS — Final Section Output
# ============================================================

class FinalSource(str, Enum):
    """Where a final section assignment came from.

    ML_ACCEPTED:   B4 accepted the ML label (confident + consistent).
    LLM_RESOLVED:  B4 escalated and B5 returned a valid LLM label.
    ML_UNRESOLVED: escalated but no valid LLM answer (B5 absent, failed, or
                   silent). The ML label is carried as best-effort signal and
                   trusted=False — content is kept, trust is withheld.
    """
    ML_ACCEPTED = "ml_accepted"
    LLM_RESOLVED = "llm_resolved"
    ML_UNRESOLVED = "ml_unresolved"


class FinalBlockSection(BaseModel):
    """The final trusted section assignment for one candidate block.

    Self-contained for B7 (entry segmentation): text, line provenance,
    alternatives, and the full decision trail travel with the label.
    """

    model_config = {"frozen": True}

    block_id: str
    document_id: str
    final_section: SectionLabel  # always set; ML label when unresolved
    confidence: float = Field(ge=0.0, le=1.0)
    source: FinalSource
    trusted: bool  # False only for ML_UNRESOLVED
    ml_section: SectionLabel
    ml_confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[AlternativePrediction] = Field(default_factory=list)
    b4_reasons: list[str] = Field(default_factory=list)
    llm_reason: str = ""  # B5 reason, or why no LLM answer exists
    source_line_ids: list[str]
    start_line_index: int
    end_line_index: int
    text: str  # B2 block display text, verbatim


class FinalSectionOutput(BaseModel):
    """B6 output: the final trusted sectioned representation of the resume."""

    model_config = {"frozen": True}

    document_id: str
    sections: list[FinalBlockSection]
    n_trusted: int
    n_unresolved: int
    trust_rate: float
    by_section: dict[str, int] = Field(default_factory=dict)
    integrity: IntegrityReport

    def get_section(self, block_id: str) -> FinalBlockSection | None:
        """Get final section by block_id."""
        for s in self.sections:
            if s.block_id == block_id:
                return s
        return None


# ============================================================
# B7 MODELS — Entry Segmentation
# ============================================================

class BlockEntry(BaseModel):
    """One logical entry inside a section block (e.g. one job inside
    EXPERIENCE, one degree inside EDUCATION).

    B7 splits structure only: which lines belong to the same entry. It does
    NOT parse dates, associate events, or order chronology (B8/B9).
    """

    model_config = {"frozen": True}

    entry_id: str  # E000000, E000001, ... (document order)
    index: int
    block_id: str
    document_id: str
    section: SectionLabel  # copied from the final section (best-effort ok)
    trusted: bool  # copied from the final section
    source: str  # FinalSource value echoed for downstream decisions
    line_ids: list[str]  # B1 line IDs, document order, never cross blocks
    start_line_index: int
    end_line_index: int
    text: str  # display text of this entry's lines, verbatim
    split_reason: str  # date_boundary | block_start_single | block_start_first


class EntrySegmentationResult(BaseModel):
    """B7 output: entries for every B6 section block. The B8 contract."""

    model_config = {"frozen": True}

    document_id: str
    entries: list[BlockEntry]
    n_entries: int
    n_blocks: int
    by_section: dict[str, int] = Field(default_factory=dict)
    integrity: IntegrityReport

    def entries_for_block(self, block_id: str) -> list[BlockEntry]:
        """Get entries belonging to one block, in order."""
        return [e for e in self.entries if e.block_id == block_id]


# ============================================================
# B8 MODELS — Date / Event Extraction
# ============================================================

class DateGranularity(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"


class ParsedDate(BaseModel):
    """One calendar date parsed from entry text, with char provenance."""

    model_config = {"frozen": True}

    raw: str  # exact matched substring of the entry text
    year: int
    month: int | None = None
    day: int | None = None
    granularity: DateGranularity
    start_char: int  # span in the ENTRY text: entry.text[start:end] == raw
    end_char: int


class DateRange(BaseModel):
    """A dated span: employment stint, degree period, etc.

    end is None exactly when the range is ongoing (Present/Current/...).
    """

    model_config = {"frozen": True}

    raw: str  # full matched substring, endpoints included
    start: ParsedDate
    end: ParsedDate | None
    is_ongoing: bool
    start_char: int
    end_char: int


class EntryDates(BaseModel):
    """All dates belonging to one B7 entry, in text order."""

    model_config = {"frozen": True}

    entry_id: str
    block_id: str
    document_id: str
    section: SectionLabel
    trusted: bool  # echoed; dates are extracted regardless of trust
    ranges: list[DateRange] = Field(default_factory=list)
    single_dates: list[ParsedDate] = Field(default_factory=list)
    primary_range: DateRange | None = None  # first range, else first single as a point
    has_dates: bool = False


class DateExtractionPolicy(BaseModel):
    """Frozen B8 configuration. reference_date=None means 'today at run time';
    the resolved date is always recorded in the output for traceability."""

    model_config = {"frozen": True}

    reference_date: date | None = None


class DateExtractionResult(BaseModel):
    """B8 output: dates for every B7 entry. The B9 contract."""

    model_config = {"frozen": True}

    document_id: str
    entry_dates: list[EntryDates]
    reference_date: date  # resolved reference date (B9 uses this for Present)
    n_entries: int
    n_with_dates: int
    n_ranges: int
    integrity: IntegrityReport

    def dates_for_entry(self, entry_id: str) -> EntryDates | None:
        """Get dates by entry_id."""
        for e in self.entry_dates:
            if e.entry_id == entry_id:
                return e
        return None


# ============================================================
# B9 MODELS — Timeline + Gap Detection
# ============================================================

class TimelineEvent(BaseModel):
    """One dated entry placed on the career timeline.

    Granularity-aware comparison: year-only starts resolve to Jan 1 and
    year-only ends to Dec 31; ongoing ends resolve to the reference date
    (kept separately as effective_end_date so the lexical fact is preserved).
    """

    model_config = {"frozen": True}

    entry_id: str
    block_id: str
    document_id: str
    section: SectionLabel
    trusted: bool  # echoed; dates are facts regardless of trust
    source: str  # FinalSource value echoed
    start_date: date
    start_granularity: DateGranularity
    end_date: date | None  # None iff ongoing (lexical fact, not a calendar date)
    end_granularity: DateGranularity | None
    is_ongoing: bool
    effective_end_date: date  # end_date, or reference_date when ongoing
    raw_range: str  # the matched range/single text, verbatim
    text: str  # entry display text, verbatim


class CareerGap(BaseModel):
    """An employment gap between two EXPERIENCE events.

    Gaps are computed on EXPERIENCE entries only: education or other dated
    entries appear in the timeline but never bridge employment gaps (v1).
    """

    model_config = {"frozen": True}

    gap_id: str  # G000000, ...
    document_id: str
    start_date: date  # day after the previous coverage end
    end_date: date  # day before the next coverage start
    gap_days: int  # (end - start).days + 1, always >= 1
    gap_months_approx: float  # gap_days / 30.44, rounded to 1 decimal
    before_entry_id: str
    after_entry_id: str


class TimelinePolicy(BaseModel):
    """Frozen B9 configuration."""

    model_config = {"frozen": True}

    min_gap_days: int = Field(default=90, ge=1)  # shorter breaks are not gaps
    reference_date: date | None = None  # None -> inherit B8's recorded date


class TimelineResult(BaseModel):
    """B9 output: the final structured career timeline. No forecasting."""

    model_config = {"frozen": True}

    document_id: str
    reference_date: date
    events: list[TimelineEvent]  # chronological by start date
    undated_entry_ids: list[str]  # entries without dates: explicit, not lost
    gaps: list[CareerGap]
    n_events: int
    n_undated: int
    n_gaps: int
    total_gap_days: int
    longest_gap_days: int
    integrity: IntegrityReport
