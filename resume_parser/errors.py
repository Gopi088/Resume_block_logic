"""Typed error hierarchy for B0/B1.

All failures are explicit typed states — never silent content loss.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversionWarning:
    """Non-fatal conversion/normalization notice. Content is preserved."""

    code: str
    message: str
    detail: str = ""


# Exception-protocol attributes the interpreter / stdlib must be able to set
# on a live exception (traceback chaining, `raise .. from ..`, contextlib).
# Frozen dataclass __setattr__ blocks these and breaks propagation through
# generator-based context managers (contextlib._GeneratorContextManager
# assigns exc.__traceback__ on exit). They stay mutable; data fields stay frozen.
_EXCEPTION_STATE_ATTRS = frozenset({
    "__traceback__", "__cause__", "__context__", "__suppress_context__",
})


def _error_setattr(self, name: str, value) -> None:
    """Field writes allowed exactly once (during __init__); exception-protocol
    dunders (__traceback__/__cause__/__context__) always mutable; everything
    else frozen. Applied post-decoration (frozen dataclasses refuse an
    in-body __setattr__)."""
    if name in _EXCEPTION_STATE_ATTRS:
        object.__setattr__(self, name, value)
        return
    fields = getattr(type(self), "__dataclass_fields__", {})
    if name in fields and name not in self.__dict__:
        object.__setattr__(self, name, value)  # __init__ assignment only
        return
    raise dataclasses.FrozenInstanceError(f"cannot assign to field {name!r}")


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

    def __setstate__(self, state: dict) -> None:
        # Unpickling restores the full field state at once; route around the
        # frozen __setattr__ (the C pickler would otherwise setattr per key).
        self.__dict__.update(state)


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


# Rebind AFTER decoration: frozen dataclasses reject an in-body __setattr__,
# and the generated one blocks the exception protocol (traceback chaining,
# contextlib propagation). Data fields stay write-once; see _error_setattr.
PipelineError.__setattr__ = _error_setattr
ConversionError.__setattr__ = _error_setattr
UnsupportedFormatError.__setattr__ = _error_setattr
IntegrityError.__setattr__ = _error_setattr
