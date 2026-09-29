#!/usr/bin/env python3
"""Parse a resume file and print the result as JSON to the terminal.

Stages: B0 conversion (MarkItDown) -> B1 normalization -> B2 candidate block segmentation
(structural only, no semantic labels) -> B3 ML semantic classification.

Usage:
    python parse_resume.py <resume-path> [--compact] [--eval] [--no-text] [--blocks-only]
                           [--model MODEL_PATH]

    <resume-path>  PDF, DOCX, TXT, HTML, ... (anything MarkItDown reads)
    --compact      single-line JSON (for piping) instead of pretty-printed
    --eval         include the B0/B1 evaluation harness summary
    --no-text      omit the full extracted/normalized text blobs
    --blocks-only  omit the per-line appendix (blocks + summary only)
    --model        Path to trained B3 classifier model (.pkl)

stdout is always pure JSON (pipe-safe). Exit code is 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from resume_parser.classification import SectionClassifier, classify_blocks
from resume_parser.conversion import convert_file
from resume_parser.errors import PipelineError
from resume_parser.evaluation import evaluate_document
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import display_text, paginate, segment_document


def build_output(path: str, include_eval: bool, include_text: bool,
                 blocks_only: bool, model_path: str | None = None) -> dict:
    converted = convert_file(path)          # B0: MarkItDown, typed errors on failure
    doc = normalize_document(converted)     # B1: conservative normalization
    seg = segment_document(doc)             # B2: structural candidate blocks
    pages = paginate(doc.lines)

    out: dict = {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "mime_type": doc.mime_type,
        "extension": doc.extension,
        "sha256": doc.sha256,
        "converter": doc.converter.model_dump(),
        "counts": {
            "extracted_chars": len(doc.extracted_markdown),
            "normalized_chars": len(doc.normalized_text),
            "pages": seg.page_count,
            "total_lines": len(doc.lines),
            "nonblank_lines": sum(1 for ln in doc.lines if ln.normalized_text != ""),
            "candidate_blocks": len(seg.blocks),
            "boilerplate_lines": sum(len(b.line_ids) for b in seg.boilerplate),
            "hyphen_joins": seg.hyphen_joins,
        },
        "warnings": [w.model_dump() for w in doc.warnings] + [w.model_dump() for w in seg.warnings],
        "integrity_b1": doc.integrity.model_dump(),
        "integrity_b2": seg.integrity.model_dump(),
        "candidate_blocks": [b.model_dump() for b in seg.blocks],
        "boilerplate": [b.model_dump() for b in seg.boilerplate],
    }
    
    # B3: ML Semantic Classification
    if model_path:
        try:
            classifier = SectionClassifier.load(model_path)
            cls_result = classify_blocks(seg, classifier)
            out["classification"] = cls_result.model_dump()
            out["integrity_b3"] = cls_result.integrity.model_dump()
        except Exception as e:
            out["classification_error"] = {"code": "CLASSIFICATION_FAILED", "message": str(e)}
    
    if include_text:
        out["extracted_markdown"] = doc.extracted_markdown
        out["normalized_text"] = doc.normalized_text
    if include_eval:
        out["evaluation"] = evaluate_document(doc).model_dump()
    if not blocks_only:
        flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
        out["lines"] = [
            {**ln.model_dump(),
             "page_index": pages[ln.line_id],
             "display_text": display_text(ln.normalized_text),
             "boilerplate_candidate": ln.line_id in flagged}
            for ln in doc.lines
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resume -> block-grouped JSON (B0+B1+B2+B3)")
    ap.add_argument("resume", help="Path to the resume file")
    ap.add_argument("--compact", action="store_true", help="Single-line JSON output")
    ap.add_argument("--eval", action="store_true", help="Include evaluation summary")
    ap.add_argument("--no-text", action="store_true", help="Omit full text blobs")
    ap.add_argument("--blocks-only", action="store_true", help="Omit per-line appendix")
    ap.add_argument("--model", type=str, help="Path to trained B3 classifier model (.pkl)")
    a = ap.parse_args(argv)
    try:
        out = build_output(a.resume, include_eval=a.eval, include_text=not a.no_text,
                           blocks_only=a.blocks_only, model_path=a.model)
    except PipelineError as e:
        sys.stdout.write(json.dumps({"ok": False, "error": {"code": e.code, "message": e.message,
                                                              "detail": e.detail}}, indent=2) + "\n")
        return 1
    except Exception as e:  # never die with a traceback when JSON was promised
        sys.stdout.write(json.dumps({"ok": False, "error": {"code": "UNEXPECTED",
                                                              "message": type(e).__name__,
                                                              "detail": str(e)[:500]}}, indent=2) + "\n")
        return 1
    out = {"ok": True, **out}
    sys.stdout.write((json.dumps(out, ensure_ascii=False) if a.compact
                      else json.dumps(out, indent=2, ensure_ascii=False)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())