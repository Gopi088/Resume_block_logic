"""Public package surface for B0+B1+B2."""

from .conversion import convert_bytes, convert_file
from .errors import (
    ConversionError,
    ConversionWarning,
    IntegrityError,
    UnsupportedFormatError,
)
from .evaluation import evaluate_document, print_report
from .integrity import build_integrity_report, verify_integrity
from .lexicon import LEXICON, REV, TYPES, rule_header
from .models import (
    AssignmentStatus,
    ConvertedDocument,
    EvaluationReport,
    IntegrityReport,
    LineKind,
    LineRecord,
    NormalizedDocument,
)
from .normalization import normalize_document
from .segmentation import (
    BoilerplateLine,
    ContentBlock,
    DisplayLine,
    SegmentationResult,
    segment_document,
)

__all__ = [
    "AssignmentStatus",
    "BoilerplateLine",
    "ConversionError",
    "ConversionWarning",
    "ConvertedDocument",
    "ContentBlock",
    "DisplayLine",
    "EvaluationReport",
    "IntegrityReport",
    "IntegrityError",
    "LEXICON",
    "LineKind",
    "LineRecord",
    "NormalizedDocument",
    "REV",
    "SegmentationResult",
    "TYPES",
    "UnsupportedFormatError",
    "build_integrity_report",
    "convert_bytes",
    "convert_file",
    "evaluate_document",
    "normalize_document",
    "print_report",
    "rule_header",
    "segment_document",
    "verify_integrity",
]
