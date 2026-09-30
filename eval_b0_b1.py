"""Evaluation harness entry point: python eval_b0_b1.py [file ...]"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from resume_parser.conversion import convert_bytes, convert_file
from resume_parser.evaluation import evaluate_document, print_report
from resume_parser.normalization import normalize_document

SAMPLE = """# Ada Lovelace
Senior Engineer — Analytical Engines • ada@example.com • https://example.com/ada

## Experience
- First Programmer, Analytical Engine (1843)
- Notes on the Engine | 1843 | London

| Role | Year |
| Engineer | 1843 |

```text
punch cards
```
"""


def run_on_text(text: str, filename: str = "sample.txt"):
    conv = convert_bytes(text.encode("utf-8"), filename=filename, content_type="text/plain")
    # Keep harness faithful to B0: use MarkItDown output as-is.
    doc = normalize_document(conv)
    return doc, evaluate_document(doc)


def main(argv: list[str]) -> int:
    targets = argv[1:]
    if not targets:
        doc, rep = run_on_text(SAMPLE)
        print(print_report(rep))
        print("---")
        print(json.dumps(rep.model_dump(), indent=2)[:2000])
        print("---\nfirst 5 lines:")
        for ln in doc.lines[:5]:
            print(f"  {ln.line_id} [{ln.line_kind.value}] raw=({ln.raw_start_char}:{ln.raw_end_char}) "
                  f"norm=({ln.normalized_start_char}:{ln.normalized_end_char}) {ln.normalized_text!r}")
        print(f"\nintegrity.passed={doc.integrity.passed} warnings={len(doc.warnings)}")
        return 0
    for t in targets:
        doc = normalize_document(convert_file(t))
        rep = evaluate_document(doc)
        print(f"== {t} ==")
        print(print_report(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
