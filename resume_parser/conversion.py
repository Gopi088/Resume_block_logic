"""B0: Document conversion using MarkItDown only.

- File-to-Markdown/text conversion layer. No section/header understanding.
- Emits explicit warnings instead of silently discarding content.
- Raises typed errors (ConversionError / UnsupportedFormatError).
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from io import BytesIO
from pathlib import Path

from .errors import ConversionError, ConversionWarning, UnsupportedFormatError
from .models import ConvertedDocument, ConverterInfo, ConversionWarningModel

CONVERTER_NAME = "markitdown"

# Extensions MarkItDown is known to handle (incl. plain text resumes).
# Unknown extensions are NOT silently dropped: conversion is still attempted
# when bytes decode as text; otherwise UnsupportedFormatError is raised.
_KNOWN_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx", ".pptx",
    ".xlsx", ".csv", ".json", ".xml", ".epub",
}


def get_converter_version() -> str:
    try:
        return pkg_version("markitdown")
    except PackageNotFoundError:
        return "unknown"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_document_id(sha256_hex: str) -> str:
    """Stable, deterministic document ID derived from content hash."""
    return f"doc_{sha256_hex[:16]}"


def _guess_extension(filename: str, content_type: str | None) -> str | None:
    ext = os.path.splitext(filename or "")[1].lower() or None
    if ext:
        return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        return guessed.lower() if guessed else None
    return None


def _warn(code: str, message: str, detail: str = "") -> ConversionWarningModel:
    return ConversionWarningModel(code=code, message=message, detail=detail)


def _build_document(
    *,
    raw_input: bytes,
    filename: str,
    content_type: str | None,
    extracted: str,
    title: str | None,
    extra_warnings: list[ConversionWarningModel] | None = None,
) -> ConvertedDocument:
    digest = sha256_bytes(raw_input)
    ext = _guess_extension(filename, content_type)
    warnings: list[ConversionWarningModel] = list(extra_warnings or [])
    if not extracted:
        warnings.append(
            _warn("EMPTY_EXTRACTION", "MarkItDown returned empty text; content preserved as empty string.")
        )
    return ConvertedDocument(
        document_id=make_document_id(digest),
        filename=filename or "unknown",
        mime_type=content_type,
        extension=ext,
        sha256=digest,
        converter=ConverterInfo(
            name=CONVERTER_NAME, version=get_converter_version(), title=title
        ),
        extracted_markdown=extracted,
        extracted_char_count=len(extracted),
        warnings=warnings,
    )


def _run_markitdown(
    source, *, filename: str, content_type: str | None, extension: str | None
) -> tuple[str, str | None]:
    """Invoke MarkItDown; returns (markdown, title). Raises typed errors."""
    try:
        from markitdown import MarkItDown
        from markitdown._stream_info import StreamInfo
    except ImportError as exc:
        raise ConversionError(detail=f"markitdown package not installed: {exc}") from exc

    md = MarkItDown()
    kwargs: dict = {}
    # Pass explicit StreamInfo when we have filename/mimetype hints.
    if not isinstance(source, (str, Path)) or True:
        info_kwargs: dict = {}
        if content_type:
            info_kwargs["mimetype"] = content_type.split(";")[0].strip()
        if extension:
            info_kwargs["extension"] = extension
        if filename:
            info_kwargs["filename"] = filename
        if info_kwargs:
            kwargs["stream_info"] = StreamInfo(**info_kwargs)
    try:
        result = md.convert(source, **kwargs)
    except Exception as exc:
        raise ConversionError(
            detail=f"MarkItDown failed for {filename!r}: {type(exc).__name__}: {exc}"
        ) from exc
    text = result.markdown if result.markdown is not None else ""
    return text, getattr(result, "title", None)


def convert_file(local_path: str | Path) -> ConvertedDocument:
    """Convert a local file via MarkItDown. Raises typed errors on failure."""
    path = Path(local_path)
    if not path.is_file():
        raise ConversionError(detail=f"Input file not found: {path}")
    raw = path.read_bytes()  # also used for sha256; bytes never altered
    content_type, _ = mimetypes.guess_type(path.name)
    ext = _guess_extension(path.name, content_type)
    # BytesIO path keeps hashing/converting consistent with convert_bytes.
    text, title = _run_markitdown(
        BytesIO(raw), filename=path.name, content_type=content_type, extension=ext
    )
    return _build_document(
        raw_input=raw,
        filename=path.name,
        content_type=content_type,
        extracted=text,
        title=title,
    )


def convert_bytes(
    data: bytes, *, filename: str, content_type: str | None = None
) -> ConvertedDocument:
    """Convert in-memory bytes via MarkItDown.

    Never silently discards content. Undecodable/unknown inputs raise
    UnsupportedFormatError with an actionable message.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise UnsupportedFormatError(detail="Input must be bytes.")
    raw = bytes(data)
    if len(raw) == 0:
        raise UnsupportedFormatError(detail="Empty input: 0 bytes received.")
    if not filename:
        raise UnsupportedFormatError(detail="filename is required for byte inputs.")
    ext = _guess_extension(filename, content_type)

    if ext is not None and ext not in _KNOWN_TEXT_EXTS:
        # Attempt text fallback before failing: decode check only, no rewrite.
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw.decode("latin-1")
            except Exception:
                raise UnsupportedFormatError(
                    detail=f"Extension {ext!r} is not in the supported set and bytes are not decodable text."
                ) from None
            # latin-1 decodable -> still try MarkItDown below (explicit warning added)

    warnings: list[ConversionWarningModel] = []
    if ext is not None and ext not in _KNOWN_TEXT_EXTS:
        warnings.append(
            _warn(
                "UNKNOWN_EXTENSION",
                f"Extension {ext!r} not in known set; attempted conversion anyway.",
            )
        )
    text, title = _run_markitdown(
        BytesIO(raw), filename=filename, content_type=content_type, extension=ext
    )
    return _build_document(
        raw_input=raw,
        filename=filename,
        content_type=content_type,
        extracted=text,
        title=title,
        extra_warnings=warnings,
    )


__all__ = [
    "convert_bytes",
    "convert_file",
    "get_converter_version",
    "make_document_id",
    "sha256_bytes",
]
