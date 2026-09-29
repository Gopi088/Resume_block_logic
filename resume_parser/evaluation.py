"""Small evaluation harness for B0+B1."""

from __future__ import annotations

from .integrity import build_integrity_report
from .models import EvaluationReport, NormalizedDocument


def evaluate_document(doc: NormalizedDocument) -> EvaluationReport:
    ids = [ln.line_id for ln in doc.lines]
    duplicate_line_id_count = len(ids) - len(set(ids))
    # Missing lines: gaps in the 0..N-1 index sequence.
    expected_idx = set(range(len(doc.lines)))
    actual_idx = {ln.index for ln in doc.lines}
    missing_line_count = len(expected_idx - actual_idx)
    nonblank = sum(1 for ln in doc.lines if ln.normalized_text != "")
    reconstructed = "\n".join(ln.normalized_text for ln in doc.lines)
    reconstruction_pass = reconstructed == doc.normalized_text
    # Lost lines: normalized segments not represented by line records (0 when healthy).
    segments = doc.normalized_text.split("\n") if (doc.lines or doc.normalized_text) else []
    lost_line_count = 0 if (len(segments) == len(doc.lines)) else abs(len(segments) - len(doc.lines))
    if reconstruction_pass and duplicate_line_id_count == 0 and missing_line_count == 0:
        # Double-check content equality segment-wise before claiming zero loss.
        if any(seg != ln.normalized_text for seg, ln in zip(segments, doc.lines)):
            lost_line_count = sum(
                1 for seg, ln in zip(segments, doc.lines) if seg != ln.normalized_text
            )
    report = build_integrity_report(
        lines=doc.lines,
        extracted_markdown=doc.extracted_markdown,
        normalized_text=doc.normalized_text,
    )
    return EvaluationReport(
        document_id=doc.document_id,
        extracted_char_count=len(doc.extracted_markdown),
        normalized_char_count=len(doc.normalized_text),
        total_line_count=len(doc.lines),
        nonblank_line_count=nonblank,
        duplicate_line_id_count=duplicate_line_id_count,
        missing_line_count=missing_line_count,
        reconstruction_pass=reconstruction_pass,
        lost_line_count=lost_line_count,
        integrity_passed=report.passed,
        warnings=list(doc.warnings),
    )


def print_report(report: EvaluationReport) -> str:
    lines = [
        f"document_id:            {report.document_id}",
        f"extracted_char_count:   {report.extracted_char_count}",
        f"normalized_char_count:  {report.normalized_char_count}",
        f"total_line_count:       {report.total_line_count}",
        f"nonblank_line_count:    {report.nonblank_line_count}",
        f"duplicate_line_id_count:{report.duplicate_line_id_count}",
        f"missing_line_count:     {report.missing_line_count}",
        f"reconstruction_pass:    {report.reconstruction_pass}",
        f"lost_line_count:        {report.lost_line_count}",
        f"integrity_passed:       {report.integrity_passed}",
        f"warnings:               {len(report.warnings)}",
        f"boundary_accuracy:      {report.boundary_accuracy}",
    ]
    return "\n".join(lines)


__all__ = ["evaluate_document", "print_report"]
