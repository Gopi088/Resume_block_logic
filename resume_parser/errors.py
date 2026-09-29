"""Typed error hierarchy for B0/B1.

All failures are explicit typed states — never silent content loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversionWarning:
    """Non-fatal conversion/normalization notice. Content is preserved."""

    code: str
    message: str
    detail: str = ""


@dataclass(frozen=True)
class PipelineError(Exception):
    """Base class for typed pipeline failures."""

    code: str
    message: str
    detail: str = ""
    recoverable: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = f"[{self.code}] {self.message}"
        return f"{base} | {self.detail}" if self.detail else base


@dataclass(frozen=True)
class ConversionError(PipelineError):
    """MarkItDown conversion failed for a readable input."""

    code: str = "CONVERSION_FAILED"
    message: str = "Document conversion failed"
    detail: str = ""
    recoverable: bool = False


@dataclass(frozen=True)
class UnsupportedFormatError(PipelineError):
    """Input format is explicitly unsupported or undecodable."""

    code: str = "UNSUPPORTED_FORMAT"
    message: str = "Unsupported or undecodable input format"
    detail: str = ""
    recoverable: bool = False


@dataclass(frozen=True)
class IntegrityError(PipelineError):
    """A B1 structural invariant was violated. Actionable, never silent."""

    code: str = "INTEGRITY_VIOLATION"
    message: str = "Integrity check failed"
    detail: str = ""
    recoverable: bool = False
    violations: tuple[str, ...] = field(default_factory=tuple)
