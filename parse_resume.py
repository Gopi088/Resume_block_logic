#!/usr/bin/env python3
"""Parse a resume file and print the result as JSON to the terminal.

Stages: B0 conversion (MarkItDown) -> B1 normalization -> B2 candidate block segmentation
(structural only, no semantic labels) -> B3 ML semantic classification (-> B4 deterministic
validation when a model is supplied) -> B5 LLM fallback for escalated blocks (--llm)
-> B6 final trusted section output.
-> B7 entry segmentation (no dates yet).
-> B8 date/event extraction (no chronology yet).
-> B9 timeline + gap detection (final stage).

Usage:
    python parse_resume.py <resume-path> [--compact] [--eval] [--no-text] [--blocks-only]
                           [--model MODEL_PATH] [--min-confidence X] [--min-margin Y]
                           [--llm {anthropic,nvidia}] [--llm-model MODEL_ID]

Usage:
    python parse_resume.py <resume-path> [--compact] [--eval] [--no-text] [--blocks-only]
                           [--model MODEL_PATH]

    <resume-path>  PDF, DOCX, TXT, HTML, ... (anything MarkItDown reads)
    --compact      single-line JSON (for piping) instead of pretty-printed
    --eval         include the B0/B1 evaluation harness summary
    --no-text      omit the full extracted/normalized text blobs
    --blocks-only  omit the per-line appendix (blocks + summary only)
    --model        Path to trained B3 classifier model (.pkl)
    --min-confidence B4 confidence floor (default 0.5)
    --min-margin   B4 top1-top2 margin floor (default 0.15)
    --llm          B5 provider for escalated blocks: anthropic or nvidia.
                   Requires OPENROUTER_API_KEY / ANTHROPIC_API_KEY or NVIDIA_API_KEY.
    --llm-model    B5 model id (OpenRouter defaults to anthropic/claude-haiku-4.5)

stdout is always pure JSON (pipe-safe). Exit code is 0 on success, 1 on failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-re-exec inside project venv if running with system Python lacking dependencies
_venv_py = Path(__file__).resolve().parent / "venv" / "bin" / "python"
if _venv_py.exists() and sys.executable != str(_venv_py):
    try:
        import numpy
        import sklearn
    except ImportError:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

import argparse
import json

from resume_parser.classification import SectionClassifier, classify_blocks
from resume_parser.conversion import convert_file
from resume_parser.date_extraction import extract_entry_dates
from resume_parser.entries import build_entries
from resume_parser.errors import PipelineError
from resume_parser.timeline import build_timeline
from resume_parser.evaluation import evaluate_document
from resume_parser.final_sections import build_final_sections
from resume_parser.models import LLMConfig, ValidationPolicy
from resume_parser.normalization import normalize_document
from resume_parser.resolution import AnthropicLLMClient, NvidiaNIMLLMClient, OpenRouterLLMClient, resolve_escalated
from resume_parser.segmentation import display_text, paginate, segment_document
from resume_parser.validation import validate_classification


def _run_llm_fallback(seg, val_result, provider: str, model: str | None):
    """B5 provider dispatch. Unknown providers and missing credentials raise
    actionably (caught by the caller into llm_error, never a silent skip)."""
    if provider not in {"anthropic", "nvidia", "openrouter"}:
        raise RuntimeError(f"unknown B5 provider {provider!r} (supported: anthropic, nvidia, openrouter)")
    if provider == "nvidia":
        default_model = NvidiaNIMLLMClient.DEFAULT_MODEL
        cfg = LLMConfig(model=model or default_model, provider=provider)
        client = NvidiaNIMLLMClient(cfg)
    elif provider == "openrouter":
        default_model = OpenRouterLLMClient.DEFAULT_MODEL
        cfg = LLMConfig(model=model or default_model, provider=provider)
        client = OpenRouterLLMClient(cfg)
    else:
        default_model = LLMConfig().model
        cfg = LLMConfig(model=model or default_model, provider=provider)
        client = AnthropicLLMClient(cfg)
    return resolve_escalated(seg, val_result, client, cfg)


def build_output(path: str, include_eval: bool, include_text: bool,
                 blocks_only: bool, model_path: str | None = None,
                 min_confidence: float = 0.5, min_margin: float = 0.15,
                 llm_provider: str | None = None, llm_model: str | None = None,
                 llm_verify_all: bool = False) -> dict:
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
    
    # B3: ML Semantic Classification (+ B4 validation when a model is given)
    if model_path:
        try:
            classifier = SectionClassifier.load(model_path)
            cls_result = classify_blocks(seg, classifier)
            out["classification"] = cls_result.model_dump()
            out["integrity_b3"] = cls_result.integrity.model_dump()
            policy = ValidationPolicy(confidence_threshold=min_confidence,
                                      margin_threshold=min_margin,
                                      llm_verify_all=llm_verify_all)
            val_result = validate_classification(seg, cls_result, policy)
            out["validation"] = val_result.model_dump()
            out["integrity_b4"] = val_result.integrity.model_dump()
            # B5: LLM fallback for escalated blocks only (opt-in via --llm).
            llm_result = None
            if llm_provider:
                try:
                    llm_result = _run_llm_fallback(
                        seg, val_result, llm_provider, llm_model)
                    out["llm_resolution"] = llm_result.model_dump()
                    out["integrity_b5"] = llm_result.integrity.model_dump()
                except Exception as e:
                    out["llm_error"] = {"code": "LLM_FAILED", "message": str(e)}
            # B6: final trusted section output (works with or without B5).
            final = build_final_sections(seg, cls_result, val_result, llm_result)
            out["final_sections"] = final.model_dump()
            out["integrity_b6"] = final.integrity.model_dump()
            # B7: entry segmentation (jobs within experience, etc.).
            entries = build_entries(seg, final)
            out["entries"] = entries.model_dump()
            out["integrity_b7"] = entries.integrity.model_dump()
            out["counts"]["entries"] = entries.n_entries
            # B8: date/event extraction (which dates belong to each entry).
            # mode="json": reference_date is a datetime.date, not JSON-native.
            dated = extract_entry_dates(entries)
            out["entry_dates"] = dated.model_dump(mode="json")
            out["integrity_b8"] = dated.integrity.model_dump()
            out["counts"]["entries_with_dates"] = dated.n_with_dates
            out["counts"]["date_ranges"] = dated.n_ranges
            # B9: timeline + gap detection (chronology over entry dates).
            timeline = build_timeline(dated, entries)
            out["timeline"] = timeline.model_dump(mode="json")
            out["integrity_b9"] = timeline.integrity.model_dump()
            out["counts"]["timeline_events"] = timeline.n_events
            out["counts"]["career_gaps"] = timeline.n_gaps
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
    ap = argparse.ArgumentParser(description="Resume -> validated JSON (B0 through B9, complete pipeline)")
    ap.add_argument("resume", help="Path to the resume file")
    ap.add_argument("--compact", action="store_true", help="Single-line JSON output")
    ap.add_argument("--eval", action="store_true", help="Include evaluation summary")
    ap.add_argument("--no-text", action="store_true", help="Omit full text blobs")
    ap.add_argument("--blocks-only", action="store_true", help="Omit per-line appendix")
    default_model = str(Path(__file__).resolve().parent / "model_b3_real.pkl")
    ap.add_argument("--model", type=str, default=default_model if Path(default_model).exists() else None,
                    help="Path to trained B3 classifier model (.pkl)")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="B4 confidence floor (default 0.5)")
    ap.add_argument("--min-margin", type=float, default=0.15,
                    help="B4 top1-top2 margin floor (default 0.15)")
    ap.add_argument("--llm", choices=["anthropic", "nvidia", "openrouter"], default=None,
                    help="B5 provider for escalated blocks: Anthropic, NVIDIA NIM, or OpenRouter")
    ap.add_argument("--llm-model", type=str, default=None,
                    help="B5 model id (NVIDIA default: nvidia/nemotron-3-ultra-550b-a55b, OpenRouter default: nvidia/nemotron-3-ultra-550b-a55b)")
    ap.add_argument("--verify-all", action="store_true",
                    help="Send every block to B5 for LLM verification (requires --llm)")
    a = ap.parse_args(argv)
    if not a.llm:
        if os.environ.get("NVIDIA_API_KEY"):
            a.llm = "nvidia"
        elif os.environ.get("OPENROUTER_API_KEY"):
            a.llm = "openrouter"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            a.llm = "anthropic"

    if a.llm and not a.llm_model:
        if a.llm == "nvidia" or a.llm == "openrouter":
            a.llm_model = "nvidia/nemotron-3-ultra-550b-a55b"

    verify_all = a.verify_all or bool(a.llm)
    try:
        out = build_output(a.resume, include_eval=a.eval, include_text=not a.no_text,
                           blocks_only=a.blocks_only, model_path=a.model,
                           min_confidence=a.min_confidence, min_margin=a.min_margin,
                           llm_provider=a.llm, llm_model=a.llm_model,
                           llm_verify_all=a.verify_all)
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