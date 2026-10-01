# Final Project Report: ML-Driven Resume Parsing & Verification Engine

## Executive Summary

This project implements a hybrid **Machine Learning + Deterministic Rule Engine + LLM Fallback** architecture for converting unstructured resume documents (PDF, DOCX, TXT) into structured JSON adhering to the standard Resume-Matcher schema.

The primary design principle is **ML-first followed by strict deterministic rules**, ensuring fast, consistent, and explainable block-by-block parsing, with optional LLM verification via OpenRouter (using models like `nvidia/nemotron-3-ultra-550b-a55b`) or direct NVIDIA NIM / Anthropic APIs.

---

## 1. Core Logic & Architecture ($B_0 \rightarrow B_9$)

The pipeline divides processing into 10 decoupled stages:

| Stage | Name | Description | Key Modules |
|-------|------|-------------|-------------|
| **$B_0$** | Document Conversion | Converts binary formats (PDF, DOCX) to raw Markdown with SHA-256 integrity tracking. | `resume_parser.conversion` |
| **$B_1$** | Text Normalization | Splits Markdown into immutable `LineRecord`s (`L000000`..); ensures zero data loss. | `resume_parser.normalization` |
| **$B_2$** | Candidate Segmentation | Identifies structural blocks (`CandidateBlock`) based on headings and paragraph spacing. | `resume_parser.segmentation` |
| **$B_3$** | ML Section Classification | Classifies each block into 14 semantic categories using TF-IDF + LogisticRegression (`model_b3_real.pkl`). Computes **cosine similarity** against schema prototypes. | `resume_parser.classification` |
| **$B_4$** | Deterministic Validation | Evaluates ML confidence ($\ge 0.50$), prediction margin ($\ge 0.15$), and structure. | `resume_parser.validation` |
| **$B_5$** | LLM Verification | Sends unverified or all blocks to an LLM (e.g. NVIDIA Nemotron via OpenRouter) for confirmation or correction. | `resume_parser.resolution` |
| **$B_6$** | Final Trusted Sections | Merges ML decisions with LLM resolutions; marks confidence and provenance. | `resume_parser.final_sections` |
| **$B_7$** | Entry Segmentation | Splits blocks into distinct job or education entries. | `resume_parser.entries` |
| **$B_8$** | Date Extraction | Identifies employment/education date intervals and normalizes them. | `resume_parser.date_extraction` |
| **$B_9$** | Timeline & Gap Analysis | Builds career chronologies and flags gaps between roles. | `resume_parser.timeline` |

---

## 2. Key Metrics & Output Contract

### Per-Block Metrics
1. **ML Confidence (`mlConfidence`)**: Calibrated probability from the LogisticRegression model for the assigned section.
2. **Cosine Similarity (`cosineSimilarity`)**: Vector similarity between the block's TF-IDF embedding and the target section prototype text.
3. **Schema Compliance (`matchesSchema`, `schemaSection`)**: Validates that every predicted section maps to an allowed property in the output schema.
4. **Line Coverage (`lineCoveragePercent`, `missingLines`, `missingLineIds`)**: Verifies that 100% of non-blank lines in the original document are assigned to blocks without duplication or omission.

### LLM Verification Status
- `"CONFIRMED"`: The LLM agreed with the ML prediction.
- `"CORRECTED"`: The LLM identified a misclassification and corrected the section.
- `"LLM_REVIEWED_NEEDS_SPAN_HANDLING"`: The LLM reviewed a complex multi-section block requiring span-level handling.
- `"NOT_VERIFIED"`: The block was processed without LLM verification (pure ML + deterministic).

---

## 3. Auditing the Project Directory & Data Files

An audit of the repository was conducted to remove redundant artifacts while preserving essential resources:

- **The `real/` Directory**: Kept. Contains 22 real resume texts (`.txt`) and annotated ground-truth labels (`.labels.csv`). This is the evaluation and training set for `model_b3_real.pkl` (recorded in `model_b3_real.metrics.json`). Removing it would break model evaluation and retraining.
- **`model_b3_real.pkl`**: Primary production classification model.
- **Removed files**: Ephemeral execution dumps (`ankur_checked.json`, `out.json`, orphaned worktrees) were pruned to keep the repository clean.
- **`.gitignore`**: Added to prevent build artifacts, cache directories (`__pycache__`), virtual environments, and temporary output JSON files from polluting the workspace.

---

## 4. Execution Commands

### Clean Resume JSON
```bash
python print_resume_json.py "my_resumes/KAILAS SANGRAM KAWALE.pdf" --model model_b3_real.pkl
```

### With OpenRouter & NVIDIA Nemotron Verification
```bash
export OPENROUTER_API_KEY="your-key-here"

python print_resume_json.py "my_resumes/KAILAS SANGRAM KAWALE.pdf" \
  --model model_b3_real.pkl \
  --llm openrouter \
  --llm-model nvidia/nemotron-3-ultra-550b-a55b
```

### Inspecting Block-by-Block ML and Cosine Similarity
```bash
python print_resume_blocks.py "my_resumes/KAILAS SANGRAM KAWALE.pdf" --model model_b3_real.pkl
```

---

## 5. How to Explain This Project to Others

When presenting this project:
1. **Explain the Hybrid Strategy**: We do not rely exclusively on slow, expensive LLMs or fragile regex rules. Instead, an ML classifier trained on hundreds of resumes classifies blocks in milliseconds. Deterministic rules validate the predictions, and an LLM is only called to verify or arbitrate ambiguities.
2. **Explain Provenance**: Every extracted item tracks its exact line numbers (`L000001`..) back to the raw source file. We can prove where every single word came from.
3. **Explain Calibration & Fallbacks**: If an API key is not present or an external provider fails, the pipeline fails gracefully and returns the verified ML output rather than crashing.
