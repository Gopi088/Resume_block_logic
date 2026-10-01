# Canonical resume normalization and evaluation

The existing classifier, trained model artifact and B0–B9 pipeline are retained. Normalization now runs between the existing deterministic projection and evaluation. No resume-specific heading or employer mapping was added.

## Files and functions changed

| File | Functions and purpose |
| --- | --- |
| `resume_parser/section_normalization.py` | New `heading_key`, `normalize_heading`, `technical_content`, `genuine_heading`, `normalize_resume`, `personal_fields`: shared aliases, conservative content/model-context fallback, canonical aggregation and independent contact recognition. |
| `resume_parser/source_evaluation.py` | `source_blocks`: identify actual headings and retain fragments inside canonical sections. `comparable`: normalize contact labels/phone formatting. `evaluate_extraction`: canonical aggregation, personal field comparisons, duplicate/association diagnostics. `verify_extraction`: send corresponding canonical structured content and veto contradicted approvals. `concise_evaluation`: requested nested confidence/evaluation shape and readable unknown-section titles. |
| `resume_parser/structure_projection.py` | `bounded_entries`: reuse normalized source boundaries in the existing projection; preserve genuine unknown section titles. |
| `print_resume_json.py` | `structured_resume`: shared contact recognizer. `group_jobs`: keep a date-led role/company header together, preventing discarded job fragments. `main`: normalize the existing projection before evaluation and public serialization. |
| `tests/test_section_normalization.py` | Heading families, punctuation, semantic fallback, unknown sections, short skills, canonical aggregation, phone formatting, field assignment, headingless model boundaries, date-led jobs and duplicate diagnostics. |
| `tests/test_source_evaluation.py` | Update canonical project evaluation and public confidence-object assertions; retain legacy-input compatibility coverage. |
| `resume_evaluation_output.json` | Regenerated with the requested real PDF command. |

## Canonical schema

The allowed resume root keys are `personalInfo`, `summary`, `workExperience`, `education`, `technicalSkills`, `projects`, `certifications`, `languages`, `awards`, `additionalSections`. Sections absent from the source are omitted and are not reported missing. Legacy `personalProjects`, `certificationsTraining`, `additional`, and `customSections` are accepted internally and normalized at the product boundary.

`personalInfo` is a field object; summary retains its text object; jobs, education and projects retain structured entry lists; skills, certifications, languages and awards retain content lists. Unknown genuine sections are preserved as `additionalSections: [{"title": "PUBLICATIONS", "content": ["..."]}]`. Multiple sections sharing a canonical key are combined without discarding their values. Internal provenance identifiers are stripped from public output.

## Normalization and evaluation

Known classifier heading aliases are reused, with additional heading families. Case, whitespace, punctuation, ampersands, slashes, hyphens and safe plurals normalize consistently. Unknown toolkit-like headings need technical content; short nontechnical headings can also use existing model context. Action narratives do not automatically become skills. Known noncanonical sections retain their original titles; ambiguous titles need structural evidence. Bullets, skill names, organization names and institution names do not become headings.

Source boundaries and the existing projection share the normalization utility. Explicit section context takes precedence over an unrelated model prediction. In headingless documents, existing model span boundaries remain available. Date/company patterns and narrative context separate interleaved technology lists from job content. The evaluator pools all spans for each canonical section, compares content and named contact fields, then submits one corresponding original/structured pair per block to the configured judge.

The public output remains `{"resume": {...}, "evaluation": {"blocks": [...], "missingBlocks": [...], "evaluationSummary": {...}, "metricDefinitions": {...}}`. Each block contains:

```json
{
  "block": "technicalSkills",
  "modelConfidence": {"value": 0.1486, "type": "raw_model_probability"},
  "extractionEvaluation": {
    "status": "CORRECT",
    "extractionConfidence": 1.0,
    "confidenceType": "evidence_based",
    "sourceMatch": true,
    "sourceCoverage": 100.0,
    "cosineSimilarity": 1.0,
    "missingContent": [],
    "incorrectContent": [],
    "wrongBlockContent": [],
    "hallucinatedContent": []
  },
  "llmVerification": {
    "status": "LLM_ERROR",
    "confidence": null,
    "reason": "Set OPENROUTER_API_KEY in the environment to use OpenRouter.",
    "missing": [],
    "incorrect": [],
    "hallucinated": []
  }
}
```

Unknown-section evaluation uses `block: "additionalSections"` with a separate `title`; contact evaluation additionally exposes individual `fields`. The actual classifier probability is retained as diagnostic evidence, using the minimum available assigned-section probability over contributing spans. If normalization changes the model's favored section, the corresponding class probability is recovered from the unchanged model and original inputs, with a reproducibility check. Extraction confidence separately combines source precision, completeness, line coverage, supported fields and cosine similarity, plus successful judge evidence when available. It is an uncalibrated evidence score, not semantic accuracy or a model probability.

## Real PDF validation

Executed:

```bash
python print_resume_json.py "/mnt/c/Users/Gopi Gedar/Downloads/A.Bhargava.pdf"
```

Full output is saved in `resume_evaluation_output.json`. Eight logical sections are evaluated; all 102 technical-skill values, including MS SQL, ETL, R, T-SQL and ELT, are retained together. Skill fragments, NETWORK LTD. and U.P. BOARD are content rather than evaluation block names. The five personal fields are individually correct. Seven work entries are retained. Projects are absent from this source and are not reported missing.

Measured summary:

```json
{
  "totalBlocks": 8,
  "correctBlocks": 7,
  "partialBlocks": 0,
  "incorrectBlocks": 1,
  "missingBlocks": 0,
  "notVerifiedBlocks": 0,
  "averageModelConfidence": 0.255,
  "averageExtractionConfidence": 0.9997,
  "averageCosineSimilarity": 0.9999,
  "sourceGroundedCorrectness": 99.18,
  "sourceContentRetentionF1": 99.59,
  "sourceCoverage": 100.0,
  "llmVerification": {
    "status": "LLM_ERROR",
    "eligible": 8,
    "submitted": 0,
    "reviewed": 0,
    "verifiedCorrect": 0,
    "verifiedPartial": 0,
    "verifiedIncorrect": 0,
    "errors": 8,
    "notVerified": 0,
    "verificationCoverage": 0.0,
    "reason": "Set OPENROUTER_API_KEY in the environment to use OpenRouter.",
    "llmVerifiedAccuracy": null
  }
}
```

Validation: `venv/bin/python -m pytest -q` completed with **311 passed**. The bootstrap-training warning comes from the existing temporary test fixture. The requested real command completed successfully; output invariants confirmed the canonical root keys, one skills block, preserved short skills, no missing content, preserved raw probabilities and zero successful LLM reviews.

## Remaining limitations

The real run cannot perform live LLM verification without a provider API key. Errors are preserved, review coverage remains zero and LLM verified accuracy is null. Timeout and successful-judge behavior are covered with test clients, not represented as live provider results.

Work-experience projection still repeats some source content; evaluation reports this as INCORRECT even though every source line is retained. High token retention or evidence scores do not override that status. Job-field associations in heavily interleaved layouts can still require semantic review.

Unknown headings without typography, a recognizable title or reliable content/model context remain ambiguous. The approach is reusable across heading variations but has not been validated against 1,500 human-labelled resumes; that scale requires a representative gold evaluation corpus. Source metrics measure converted resume text, so PDF conversion losses and OCR failures remain outside their scope.
