#!/usr/bin/env python3
"""Print a resume as verifiable JSON: every B2 block with its text + B3 section.

Usage:
    venv/bin/python print_resume_blocks.py <resume-path> [--model MODEL_PATH] [--compact]

Output is pure JSON: summary counts plus one record per block
(block_id, section, confidence, trusted, source, line range, full text),
so a human can check each block's section assignment for correctness.
Pipeline stages after B6 (entries/dates/timeline) are summarized, not repeated.
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

from parse_resume import build_output

DEFAULT_MODEL = str(Path(__file__).resolve().parent / "model_b3_real.pkl")


def to_verifiable(out: dict) -> dict:
    final = {s["block_id"]: s for s in out.get("final_sections", {}).get("sections", [])}
    val = {v["block_id"]: v for v in out.get("validation", {}).get("verdicts", [])}
    cls = {c["block_id"]: c for c in out.get("classification", {}).get("classifications", [])}
    blocks = []
    for b in out.get("candidate_blocks", []):
        bid = b["block_id"]
        f, v, c = final.get(bid, {}), val.get(bid, {}), cls.get(bid, {})
        blocks.append({
            "block_id": bid,
            "section": f.get("final_section", c.get("predicted_section")),
            "confidence": round(float(f.get("confidence", c.get("confidence", 0.0) or 0.0)), 4),
            "cosine_similarity": c.get("cosine_similarity", 0.0),
            "matches_schema": c.get("matches_schema", True),
            "schema_section": c.get("schema_section", "workExperience"),
            "trusted": f.get("trusted"),
            "source": f.get("source"),
            "ml_section": c.get("predicted_section"),
            "alternatives": [{"section": a["section"],
                              "confidence": round(a["confidence"], 4)}
                             for a in c.get("alternatives", [])],
            "b4_verdict": v.get("verdict"),
            "b4_reasons": v.get("reasons", []),
            "line_ids": b["line_ids"],
            "page": b.get("page_indices", [None])[0],
            "text": b["text"],
        })

    span_cov = out.get("classification", {}).get("model_metadata", {}).get("semantic_span_coverage", {})
    schema_matched = sum(1 for b in blocks if b.get("matches_schema"))
    avg_conf = sum(b["confidence"] for b in blocks) / len(blocks) if blocks else 0.0
    avg_cos = sum(b.get("cosine_similarity") or 0.0 for b in blocks) / len(blocks) if blocks else 0.0

    return {
        "ok": True,
        "filename": out["filename"],
        "counts": out["counts"],
        "schema_matching": {
            "total_blocks": len(blocks),
            "matched_blocks": schema_matched,
            "unmatched_blocks": len(blocks) - schema_matched,
            "schema_match_rate": round(100.0 * schema_matched / max(len(blocks), 1), 2),
            "schema_match_accuracy": f"{round(100.0 * schema_matched / max(len(blocks), 1), 1)}%",
            "average_confidence": round(avg_conf, 4),
            "average_cosine_similarity": round(avg_cos, 4),
        },
        "coverage": {
            "blocks": len(blocks),
            "trusted": sum(1 for b in blocks if b["trusted"]),
            "untrusted": sum(1 for b in blocks if not b["trusted"]),
            "total_lines": out.get("counts", {}).get("total_lines", 0),
            "nonblank_lines": out.get("counts", {}).get("nonblank_lines", 0),
            "missing_lines": span_cov.get("missing_lines", 0),
            "missing_line_ids": span_cov.get("missing_line_ids", []),
            "unassigned_lines": span_cov.get("unassigned_lines", 0),
            "line_coverage_percent": span_cov.get("coverage_percent", 100.0),
        },
        "blocks": blocks,
        "timeline_summary": {
            "events": out.get("timeline", {}).get("n_events"),
            "gaps": out.get("timeline", {}).get("n_gaps"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resume -> verifiable per-block JSON")
    ap.add_argument("resume", help="Path to the resume file")
    ap.add_argument("--model", default=DEFAULT_MODEL if Path(DEFAULT_MODEL).exists() else None,
                    help="Path to trained B3 classifier (.pkl)")
    ap.add_argument("--compact", action="store_true", help="Single-line JSON output")
    ap.add_argument("--llm", choices=["anthropic", "nvidia", "openrouter"], default=None,
                    help="B5 provider for escalated blocks: Anthropic, NVIDIA NIM, or OpenRouter")
    ap.add_argument("--llm-model", type=str, default=None, help="B5 model id")
    ap.add_argument("--verify-all", action="store_true", default=False, help="Send every block for LLM verification")
    ap.add_argument("--escalated-only", action="store_true", default=False, help="Only verify escalated blocks instead of all blocks")
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

    verify_all = (a.verify_all or not a.escalated_only) if a.llm else False
    try:
        out = build_output(a.resume, include_eval=False, include_text=False,
                           blocks_only=True, model_path=a.model,
                           llm_provider=a.llm, llm_model=a.llm_model,
                           llm_verify_all=verify_all)
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)[:500]}, indent=2) + "\n")
        return 1
    if out.get("classification_error"):
        sys.stdout.write(json.dumps({"ok": False, **out}, indent=2) + "\n")
        return 1
    res = to_verifiable(out)
    sys.stdout.write((json.dumps(res, ensure_ascii=False) if a.compact
                      else json.dumps(res, indent=2, ensure_ascii=False)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
