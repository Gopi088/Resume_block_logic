# Resume Parser & Semantic Sectioning Engine

A production-grade, hybrid ML + deterministic resume parsing pipeline with optional LLM verification (Anthropic, NVIDIA NIM, and OpenRouter).

---

## Architecture Overview: The $B_0 \rightarrow B_9$ Pipeline

The system converts raw resume files into structured JSON matching the Resume-Matcher schema using a strict pipeline:

1. **$B_0$ Conversion (`resume_parser.conversion`)**: Uses `markitdown` to convert PDFs, DOCX, and text files into Markdown with character spans and integrity tracking.
2. **$B_1$ Normalization (`resume_parser.normalization`)**: Cleans whitespace and extracts non-blank lines into immutable `LineRecord`s without destructive loss.
3. **$B_2$ Segmentation (`resume_parser.segmentation`)**: Splits the document into structural `CandidateBlock`s using whitespace boundaries, bullet points, and headers.
4. **$B_3$ ML Classification (`resume_parser.classification`)**: Categorizes blocks into resume sections using a trained TF-IDF + LogisticRegression model (`model_b3_real.pkl`) with cosine similarity scoring against section prototypes.
5. **$B_4$ Validation (`resume_parser.validation`)**: Deterministically checks classifier confidence, margins between top-1 and top-2 predictions, continuation coherence, and singleton section constraints.
6. **$B_5$ LLM Verification (`resume_parser.resolution`)**: Reviews blocks using Anthropic Claude, NVIDIA NIM, or OpenRouter (e.g. `nvidia/nemotron-3-ultra-550b-a55b`).
7. **$B_6$ Final Sections (`resume_parser.final_sections`)**: Merges validated ML predictions and LLM resolutions with provenance tracking.
8. **$B_7$ Entry Segmentation (`resume_parser.entries`)**: Splits experience and education blocks into individual job/degree entries.
9. **$B_8$ Date Extraction (`resume_parser.date_extraction`)**: Extracts start/end dates and durations using regular expressions.
10. **$B_9$ Timeline & Gaps (`resume_parser.timeline`)**: Orders career history and detects employment gaps.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Virtual environment with dependencies:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### 1. Generating Resume JSON (One Simple Command)
To parse any resume and produce clean structured JSON:
```bash
python print_resume_json.py "my_resumes/KAILAS SANGRAM KAWALE.pdf"
```
*Note: The CLI automatically defaults to `model_b3_real.pkl`. If `NVIDIA_API_KEY` or `OPENROUTER_API_KEY` is exported in your environment, it will also automatically run full LLM verification with `nvidia/nemotron-3-ultra-550b-a55b` without requiring extra flags!*

### 2. LLM Verification Options
If you have an API key set, you can run directly or override explicitly:
- **Direct NVIDIA NIM**:
  ```bash
  export NVIDIA_API_KEY="nvapi-your-key-here"
  python print_resume_json.py "my_resumes/A.Bhargava.docx"
  ```
- **Via OpenRouter**:
  ```bash
  export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
  python print_resume_json.py "my_resumes/A.Bhargava.docx"
  ```

### 3. Block-by-Block Inspection (Cosine Similarity, Line Coverage, Confidence)
To inspect every block's confidence, cosine similarity against section centroids, and schema mapping:
```bash
python print_resume_blocks.py "my_resumes/A.Bhargava.docx"
```

---

## Output Schema
The generated JSON follows the standard structure:
```json
{
  "personalInfo": {
    "name": "...",
    "title": "...",
    "email": "...",
    "phone": "...",
    "location": "...",
    "website": "...",
    "linkedin": "...",
    "github": "...",
    "mlConfidence": 0.85,
    "llmVerification": "CONFIRMED"
  },
  "summary": {
    "text": "...",
    "mlConfidence": 0.90,
    "llmVerification": "CONFIRMED"
  },
  "workExperience": [
    {
      "id": 1,
      "title": "Manager",
      "company": "State Street Bank",
      "location": "",
      "years": "2011 - 2016",
      "description": [ ... ],
      "mlConfidence": 0.82,
      "llmVerification": "CONFIRMED"
    }
  ],
  "education": [ ... ],
  "personalProjects": [ ... ],
  "additional": {
    "technicalSkills": [ ... ],
    "languages": [ ... ],
    "certificationsTraining": [ ... ],
    "awards": [ ... ],
    "mlConfidence": { ... },
    "llmVerification": { ... }
  },
  "customSections": {},
  "overallReport": {
    "accuracy": "NOT_MEASURED",
    "accuracyStatus": "NEEDS_GOLD_LABELS",
    "accuracyDefinition": "A resume-specific accuracy score requires human-verified reference labels.",
    "candidateBlocks": 20,
    "classifiedBlocks": 20,
    "missingBlocks": 0,
    "lineCoveragePercent": 100.0,
    "missingLines": 0,
    "duplicateLineAssignments": 0,
    "unassignedLines": 0,
    "spanCount": 21,
    "missingSections": [ "Projects", "Awards" ],
    "llmVerification": {
      "scope": "all_blocks",
      "provider": "openrouter",
      "model": "nvidia/nemotron-3-ultra-550b-a55b",
      "totalBlocks": 20,
      "blocksSent": 20,
      "blocksReviewed": 20,
      "blocksConfirmed": 16,
      "blocksCorrected": 4,
      "blocksWithFinalLLMSection": 20,
      "mlLlmAgreementPercent": 80.0,
      "mlLlmAgreementMeaning": "agreement with the LLM among blocks given a final LLM section; not accuracy",
      "blocksNotReviewed": 0,
      "blocksRejectedOrFailed": 0,
      "failureReasons": [],
      "blocksStillUntrusted": 0,
      "reviewCoveragePercent": 100.0,
      "status": "ALL_REVIEWED_AND_TRUSTED",
      "meaning": "LLM review results describe agreement/correction, not ground-truth accuracy.",
      "error": null,
      "lineCoveragePercent": 100.0
    }
  }
}
```

---

## Running Tests
Run the test suite using pytest:
```bash
./venv/bin/pytest
```
All 240 tests pass.

