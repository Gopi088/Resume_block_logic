"""B1 integrity checker: explicit, actionable failures; never silent."""

from __future__ import annotations

from .errors import IntegrityError
from .models import IntegrityReport


def build_integrity_report(*, lines, extracted_markdown: str, normalized_text: str) -> IntegrityReport:
    """Run all 8 invariant groups; return a report (no raise)."""
    violations: list[str] = []
    checks: dict[str, bool] = {}

    # --- Check A: line IDs unique, contiguous, ordered (covers req 2 + 4) ---
    ids = [ln.line_id for ln in lines]
    expected = [f"L{i:06d}" for i in range(len(lines))]
    ok_ids_ordered = ids == expected
    checks["line_ids_unique_contiguous_ordered"] = ok_ids_ordered
    if not ok_ids_ordered:
        violations.append(
            f"line_ids not unique/contiguous/ordered: got {ids[:5]!r}... expected {expected[:5]!r}... (n={len(lines)})"
        )
    ok_index = all(ln.index == i for i, ln in enumerate(lines))
    checks["indices_contiguous_ordered"] = ok_index
    if not ok_index:
        violations.append("line indices are not 0..N-1 in order.")

    # --- Check B: raw offsets valid + round-trip (covers req 6) ---
    ok_raw = True
    for ln in lines:
        s, e = ln.raw_start_char, ln.raw_end_char
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(extracted_markdown)):
            ok_raw = False
            violations.append(
                f"{ln.line_id}: raw span [{s}:{e}] out of bounds for extracted_markdown len={len(extracted_markdown)}."
            )
            continue
        if extracted_markdown[s:e] != ln.raw_text:
            ok_raw = False
            violations.append(
                f"{ln.line_id}: extracted_markdown[{s}:{e}] != raw_text ({ln.raw_text!r})."
            )
    checks["raw_offsets_valid"] = ok_raw

    # --- Check C: normalized offsets valid + tiling (covers req 1 + 7) ---
    ok_norm = True
    cursor = 0
    for i, ln in enumerate(lines):
        s, e = ln.normalized_start_char, ln.normalized_end_char
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(normalized_text)):
            ok_norm = False
            violations.append(
                f"{ln.line_id}: normalized span [{s}:{e}] out of bounds for normalized_text len={len(normalized_text)}."
            )
            continue
        if normalized_text[s:e] != ln.normalized_text:
            ok_norm = False
            violations.append(
                f"{ln.line_id}: normalized_text[{s}:{e}] != normalized_text field ({ln.normalized_text!r})."
            )
        if s != cursor:
            ok_norm = False
            violations.append(
                f"{ln.line_id}: normalized tiling broken: expected start={cursor}, got {s}."
            )
        cursor = e + 1  # +1 for "\n" separator
        if i < len(lines) - 1 and not (e < len(normalized_text) and normalized_text[e:e + 1] == "\n"):
            ok_norm = False
            violations.append(f"{ln.line_id}: expected '\\n' separator at {e}.")
    if lines and cursor - 1 != len(normalized_text):
        ok_norm = False
        violations.append(
            f"normalized tiling end mismatch: tiled end={cursor - 1}, len(normalized_text)={len(normalized_text)}."
        )
    if not lines and normalized_text != "":
        ok_norm = False
        violations.append("zero lines but normalized_text is non-empty.")
    checks["normalized_offsets_tiled"] = ok_norm

    # --- Check D: reconstruction (covers req 1 + 5) ---
    reconstructed = "\n".join(ln.normalized_text for ln in lines)
    # Empty-doc edge: zero lines <-> normalized_text == "" is also exact.
    ok_recon = reconstructed == normalized_text
    checks["reconstruction_exact"] = ok_recon
    if not ok_recon:
        violations.append("concatenated line records do not recreate normalized_text exactly.")

    # --- Check E: no normalized nonblank line missing (covers req 3) ---
    # Every "\n"-separated segment of normalized_text must appear exactly once
    # in order in the line records.
    segments = normalized_text.split("\n") if (lines or normalized_text) else []
    ok_cover = len(segments) == len(lines) and all(
        seg == ln.normalized_text for seg, ln in zip(segments, lines)
    )
    checks["no_missing_nonblank_line"] = ok_cover
    if not ok_cover:
        violations.append(
            f"line coverage mismatch: normalized_text has {len(segments)} segments, {len(lines)} line records."
        )

    passed = not violations
    return IntegrityReport(passed=passed, violations=violations, checks=checks)


def verify_integrity(*, lines, extracted_markdown: str, normalized_text: str) -> IntegrityReport:
    """Like build_integrity_report but raises IntegrityError on failure."""
    report = build_integrity_report(
        lines=lines, extracted_markdown=extracted_markdown, normalized_text=normalized_text
    )
    if not report.passed:
        raise IntegrityError(
            detail="; ".join(report.violations),
            violations=tuple(report.violations),
        )
    return report


__all__ = ["build_integrity_report", "verify_integrity"]
