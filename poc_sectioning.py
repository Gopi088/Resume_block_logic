#!/usr/bin/env python3
"""
Hybrid Resume Sectioning PoC
============================

Architecture under test:

    PDF/DOCX
       |
       v
   MarkItDown
       |
       v
  loss-preserving document model
       |
       v
  candidate block segmentation
       |
       v
  ML section classification
       |
       v
  deterministic validation
       |
       +---- confident ----------------------+
       |                                     |
       +---- ambiguous / conflict ----------> LLM validation
                                             |
                                             v
                                  final section assignment
                                             |
                                             v
                                      coverage validator

IMPORTANT:
- This is a PoC for SECTION/BLOCK detection, not timeline extraction.
- The embedded ML training examples are only a bootstrap classifier.
- For a real evaluation, replace them with a labelled training set.
- The LLM is optional. If no LLM configuration is supplied, the system
  reports unresolved blocks instead of silently guessing.
- The source document is never deduplicated or discarded internally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from markitdown import MarkItDown

# ML is intentionally lightweight for this PoC.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# 1. SECTION TAXONOMY
# ============================================================

SECTIONS = [
    "contact",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "awards",
    "publications",
    "languages",
    "volunteering",
    "interests",
    "references",
    "other",
]

SECTION_ALIASES = {
    "summary": {
        "summary", "professional summary", "profile", "career summary",
        "objective", "career objective", "about me", "professional profile",
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history", "career history",
        "professional background", "career profile", "work background",
    },
    "education": {
        "education", "academic background", "educational background",
        "qualifications", "academic qualifications",
    },
    "skills": {
        "skills", "technical skills", "technology skills", "technical expertise",
        "core competencies", "competencies", "technical & skills",
        "technology & skills", "technical abilities", "key skills",
    },
    "projects": {
        "projects", "key projects", "project experience", "project details",
        "selected projects", "academic projects", "personal projects",
    },
    "certifications": {
        "certifications", "certificates", "professional certifications",
        "certification", "licenses & certifications",
    },
    "awards": {
        "awards", "honors", "honours", "achievements", "honors & awards",
        "honours & awards", "honors-awards", "honours-awards",
    },
    "publications": {
        "publications", "research", "papers", "research & publications",
    },
    "languages": {"languages", "language"},
    "volunteering": {
        "volunteering", "volunteer experience", "community involvement",
    },
    "interests": {"interests", "hobbies", "hobbies & interests"},
    "references": {"references", "referees"},
}

ALIAS_TO_SECTION = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


# ============================================================
# 2. REGEX / STRUCTURAL SIGNALS
# ============================================================

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"
)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)

DATE_RANGE_RE = re.compile(
    r"""
    (?:
        (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|
        may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|
        oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)
        \s*[\-/]?\s*\d{4}
        |
        \d{1,2}\s*[\-/]\s*\d{4}
        |
        \d{4}
    )
    \s*(?:-|–|—|to)\s*
    (?:
        (?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|
        may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|
        oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)
        \s*[\-/]?\s*\d{4}
        |
        \d{1,2}\s*[\-/]\s*\d{4}
        |
        \d{4}
        |
        present|current
    )
    """,
    re.I | re.X,
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

BULLET_RE = re.compile(r"^\s*(?:[•●▪◦*-]|\d+[.)])\s+")

ROLE_WORDS = {
    "engineer", "developer", "analyst", "manager", "consultant",
    "architect", "lead", "director", "specialist", "administrator",
    "designer", "intern", "officer", "executive", "associate",
    "scientist", "trainee", "coordinator", "product owner",
    "scrum master", "business analyst", "project manager", "developer",
    "team lead", "system engineer",
}

COMPANY_WORDS = {
    "ltd", "limited", "inc", "corp", "corporation", "llc",
    "technologies", "technology", "solutions", "services", "consulting",
    "consultancy", "systems", "industries", "group", "bank", "tcs",
    "wipro", "infosys", "accenture", "cognizant", "capgemini", "ibm",
    "microsoft", "amazon", "google",
}

AWARD_WORDS = {
    "award", "awards", "honor", "honours", "winner", "recognition",
    "recognized", "recognised",
}

SKILL_WORDS = {
    "python", "java", "javascript", "sql", "react", "power bi", "excel",
    "tableau", "jira", "confluence", "docker", "aws", "azure", "gcp",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "servicenow", "itil", "power automate",
}

EDUCATION_WORDS = {
    "bachelor", "master", "b.tech", "m.tech", "mba", "phd",
    "university", "college", "degree", "graduated", "school",
}

CERT_WORDS = {
    "certification", "certified", "certificate", "foundation certificate",
}

SUMMARY_WORDS = {
    "years of experience", "professional", "expertise", "experienced",
    "proven track record", "career", "specialize", "specialised",
}


# ============================================================
# 3. LOSS-PRESERVING DOCUMENT MODEL
# ============================================================

@dataclass
class SourceLine:
    line_id: int
    page: Optional[int]
    raw: str
    normalized: str
    is_blank: bool
    is_bullet: bool


@dataclass
class Block:
    block_id: str
    start_line: int
    end_line: int
    text: str
    line_ids: list[int]
    heading: Optional[str] = None

    # ML
    ml_section: Optional[str] = None
    ml_confidence: float = 0.0

    # deterministic
    deterministic_section: Optional[str] = None
    deterministic_score: float = 0.0
    rule_evidence: list[str] | None = None

    # final
    final_section: Optional[str] = None
    final_confidence: float = 0.0
    decision: str = "unresolved"
    needs_llm: bool = False

    def __post_init__(self):
        if self.rule_evidence is None:
            self.rule_evidence = []


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def normalize_heading(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"^[\W_]+", "", value)
    value = re.sub(r"[\W_]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def is_bullet(line: str) -> bool:
    return bool(BULLET_RE.match(line))


def remove_bullet(line: str) -> str:
    return BULLET_RE.sub("", line).strip()


# ============================================================
# 5. MARKITDOWN
# ============================================================

def extract_markdown(file_path: str) -> str:
    converter = MarkItDown()
    result = converter.convert(file_path)

    if hasattr(result, "markdown") and result.markdown:
        return str(result.markdown)

    if hasattr(result, "text") and result.text:
        return str(result.text)

    raise RuntimeError("MarkItDown returned no markdown/text")


def build_document(raw_text: str) -> list[SourceLine]:
    """
    IMPORTANT:
    Do not delete blank lines from the internal representation.
    They can be structural evidence.

    We preserve every extracted line. Blank lines are represented explicitly.
    """
    result = []

    # MarkItDown generally produces one logical line per split line.
    # We keep page as None because this PoC does not infer page boundaries
    # from markdown. A production adapter should preserve page metadata when
    # the converter provides it.
    for idx, raw in enumerate(raw_text.splitlines(), start=1):
        normalized = normalize_text(raw)
        result.append(
            SourceLine(
                line_id=idx,
                page=None,
                raw=raw,
                normalized=normalized,
                is_blank=(normalized == ""),
                is_bullet=is_bullet(normalized),
            )
        )

    return result


# ============================================================
# 6. HEADING DETECTION
# ============================================================

def explicit_heading(line: str) -> Optional[str]:
    normalized = normalize_heading(line)
    if not normalized:
        return None

    return ALIAS_TO_SECTION.get(normalized)


def looks_like_heading(line: str) -> bool:
    """
    Conservative formatting-independent heading heuristic.

    We do NOT classify it semantically here. We only identify a candidate
    boundary that may help segmentation.
    """
    text = normalize_text(line)
    if not text or len(text) > 80 or is_bullet(text):
        return False

    if explicit_heading(text):
        return True

    # Markdown heading produced by MarkItDown
    if text.startswith("#"):
        return True

    # Typical all-caps heading
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) >= 4 and letters.isupper():
        return True

    # Common section punctuation
    if text.endswith(":") and len(text.split()) <= 6:
        return True

    return False


# ============================================================
# 7. CANDIDATE BLOCK SEGMENTATION
# ============================================================

def has_date(text: str) -> bool:
    return bool(DATE_RANGE_RE.search(text) or YEAR_RE.search(text))


def block_has_experience_shape(lines: list[str]) -> bool:
    text = " ".join(lines).lower()

    date = bool(DATE_RANGE_RE.search(text))
    role = any(word in text for word in ROLE_WORDS)
    company = any(word in text for word in COMPANY_WORDS)
    bullets = sum(is_bullet(x) for x in lines)

    # This is only a structural segmentation signal.
    return date and (role or company) and bullets >= 1


def should_start_new_implicit_block(
    current: list[SourceLine],
    new_line: SourceLine,
    lookahead: list[SourceLine],
) -> bool:
    """
    Detect boundaries even when there is NO section heading.

    Particularly useful for resumes like Ankur's, where multiple jobs appear
    one after another without a WORK EXPERIENCE heading.
    """
    if not current:
        return False

    # Never split a bullet sequence merely because another bullet arrives.
    if new_line.is_bullet:
        return False

    current_text = [x.normalized for x in current if x.normalized]
    future = [x.normalized for x in lookahead if x.normalized]

    if not current_text:
        return False

    # A new non-bullet title/date sequence after an experience-shaped block
    # is a strong boundary.
    if block_has_experience_shape(current_text):
        future_text = " ".join(future[:6])

        has_future_date = bool(DATE_RANGE_RE.search(future_text))
        has_future_role = any(w in future_text.lower() for w in ROLE_WORDS)

        if has_future_date and has_future_role:
            return True

    return False


def segment_blocks(lines: list[SourceLine]) -> list[Block]:
    """
    Structural segmentation only.

    No semantic section decision is made here.
    Every non-empty source line is assigned to exactly one candidate block.
    """
    blocks: list[Block] = []
    current: list[SourceLine] = []
    block_counter = 1

    def flush():
        nonlocal current, block_counter
        nonempty = [x for x in current if not x.is_blank]
        if not nonempty:
            current = []
            return

        block_id = f"B{block_counter:04d}"
        block_counter += 1

        text = "\n".join(x.normalized for x in nonempty)

        heading = None
        if len(nonempty) <= 2 and looks_like_heading(nonempty[0].normalized):
            heading = nonempty[0].normalized

        blocks.append(
            Block(
                block_id=block_id,
                start_line=nonempty[0].line_id,
                end_line=nonempty[-1].line_id,
                text=text,
                line_ids=[x.line_id for x in nonempty],
                heading=heading,
            )
        )
        current = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.is_blank:
            # Blank lines are structural boundaries, but they are NOT lost.
            flush()
            i += 1
            continue

        explicit = explicit_heading(line.normalized)

        # Explicit heading starts a new candidate block.
        if explicit:
            flush()
            current.append(line)
            i += 1
            continue

        # New heading-like line after existing content.
        if looks_like_heading(line.normalized) and current:
            flush()
            current.append(line)
            i += 1
            continue

        # Implicit boundary detection.
        lookahead = lines[i : min(i + 7, len(lines))]
        if should_start_new_implicit_block(current, line, lookahead):
            flush()

        current.append(line)
        i += 1

    flush()
    return blocks


# ============================================================
# 8. BOOTSTRAP ML CLASSIFIER
# ============================================================

TRAINING_DATA = [
    # contact
    ("John Doe john@example.com +91 9999999999 linkedin.com/in/johndoe", "contact"),
    ("Jane Smith jane@gmail.com Bengaluru India linkedin.com/in/jane", "contact"),

    # summary
    ("Professional summary with 8 years of experience in software engineering and leadership", "summary"),
    ("Experienced business analyst with proven track record delivering enterprise solutions", "summary"),
    ("Career profile specializing in data analysis stakeholder management and product delivery", "summary"),

    # experience
    ("Senior Business Analyst Tata Consultancy Services 02/2024 – 08/2025 Sydney Australia "
     "analysed requirements coordinated developers stakeholders delivered projects", "experience"),
    ("Software Engineer ABC Technologies Jan 2022 – Present Bengaluru "
     "developed applications fixed defects and collaborated with engineering teams", "experience"),
    ("Project Manager XYZ Ltd 2020 – 2023 led teams managed delivery and coordinated stakeholders", "experience"),

    # education
    ("Bachelor of Technology Computer Science Engineering University 2016", "education"),
    ("Master Software Engineering BITS Pilani 2022", "education"),
    ("B.Tech IMS Engineering College graduated computer science", "education"),

    # skills
    ("Python Java SQL React Docker AWS JavaScript", "skills"),
    ("Technical Skills Python SQL Power BI Excel JIRA Confluence", "skills"),
    ("Technology & Skills Avaloq Banking Suite ServiceNow ITIL Microsoft Power BI", "skills"),

    # projects
    ("Resume Parser project developed using Python FastAPI React and PostgreSQL", "projects"),
    ("AI Resume Timeline Parser project NLP machine learning", "projects"),

    # certifications
    ("AWS Certified Solutions Architect certification 2025", "certifications"),
    ("ITIL Foundation Certificate 2018 Microsoft Power BI certification", "certifications"),

    # awards
    ("L1 Team Lead Award Ownership Award Best Team Award", "awards"),
    ("Honors Awards Continuous Performance Award Winner", "awards"),

    # publications
    ("Research paper published in IEEE conference transformer NLP phishing detection", "publications"),

    # languages
    ("Languages English Fluent Hindi Native", "languages"),

    # volunteering
    ("Volunteer experience community service NSS teaching students", "volunteering"),

    # interests
    ("Interests photography travel reading music", "interests"),

    # references
    ("References available upon request", "references"),

    # other
    ("Additional information miscellaneous professional details", "other"),
]


def train_ml_classifier() -> Pipeline:
    texts = [x[0] for x in TRAINING_DATA]
    labels = [x[1] for x in TRAINING_DATA]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(texts, labels)
    return model


def ml_predict(model: Pipeline, text: str) -> tuple[str, float, dict[str, float]]:
    probabilities = model.predict_proba([text])[0]
    labels = model.classes_

    ranking = sorted(
        zip(labels, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )

    best_label, best_probability = ranking[0]

    return (
        str(best_label),
        float(best_probability),
        {str(label): float(prob) for label, prob in ranking[:5]},
    )


# ============================================================
# 9. DETERMINISTIC VALIDATION
# ============================================================

def score_deterministic(section: str, block: Block) -> tuple[float, list[str]]:
    text = block.text.lower()
    lines = block.text.splitlines()
    score = 0.0
    evidence: list[str] = []

    heading_section = explicit_heading(block.heading or "")

    if heading_section == section:
        score += 0.70
        evidence.append("explicit_section_heading")

    date = bool(DATE_RANGE_RE.search(block.text))
    bullets = sum(is_bullet(x) for x in lines)
    role = any(word in text for word in ROLE_WORDS)
    company = any(word in text for word in COMPANY_WORDS)
    education = any(word in text for word in EDUCATION_WORDS)
    award = any(word in text for word in AWARD_WORDS)
    cert = any(word in text for word in CERT_WORDS)
    skills = sum(1 for word in SKILL_WORDS if word in text)

    if section == "experience":
        if date:
            score += 0.20
            evidence.append("date_range")
        if role:
            score += 0.20
            evidence.append("role_language")
        if company:
            score += 0.15
            evidence.append("company_language")
        if bullets >= 2:
            score += 0.15
            evidence.append("responsibility_bullets")

    elif section == "education":
        if education:
            score += 0.45
            evidence.append("education_language")
        if date:
            score += 0.10
            evidence.append("education_date")

    elif section == "skills":
        if skills >= 2:
            score += min(0.60, 0.15 * skills)
            evidence.append(f"skill_terms={skills}")

    elif section == "awards":
        if award:
            score += 0.55
            evidence.append("award_language")

    elif section == "certifications":
        if cert:
            score += 0.55
            evidence.append("certification_language")

    elif section == "languages":
        if re.search(r"\b(native|fluent|conversational|proficient)\b", text):
            score += 0.45
            evidence.append("language_proficiency")

    elif section == "contact":
        if EMAIL_RE.search(block.text):
            score += 0.40
            evidence.append("email")
        if PHONE_RE.search(block.text):
            score += 0.30
            evidence.append("phone")
        if URL_RE.search(block.text):
            score += 0.15
            evidence.append("url")

    elif section == "summary":
        matches = sum(1 for word in SUMMARY_WORDS if word in text)
        if matches:
            score += min(0.55, matches * 0.12)
            evidence.append(f"summary_language={matches}")

    elif section == "projects":
        if "project" in text:
            score += 0.35
            evidence.append("project_language")

    elif section == "volunteering":
        if "volunteer" in text or "community" in text or "nss" in text:
            score += 0.55
            evidence.append("volunteer_language")

    elif section == "interests":
        if any(x in text for x in ["hobbies", "interest", "photography", "travel", "music"]):
            score += 0.45
            evidence.append("interest_language")

    elif section == "references":
        if "reference" in text:
            score += 0.55
            evidence.append("reference_language")

    elif section == "publications":
        if any(x in text for x in ["published", "publication", "paper", "research"]):
            score += 0.45
            evidence.append("publication_language")

    return min(score, 1.0), evidence


def validate_ml_prediction(block: Block, alternatives: dict[str, float]) -> None:
    """
    Deterministic layer decides whether ML can be accepted, rejected,
    or sent to the LLM.

    This does NOT replace ML. It validates ML using independent evidence.
    """
    ml_section = block.ml_section
    ml_conf = block.ml_confidence

    det_score, evidence = score_deterministic(ml_section, block)
    block.deterministic_section = ml_section
    block.deterministic_score = det_score
    block.rule_evidence = evidence

    # Explicit heading is strong evidence.
    if block.heading:
        explicit = explicit_heading(block.heading)
        if explicit:
            block.final_section = explicit
            block.final_confidence = max(ml_conf, det_score, 0.95)
            block.decision = "accepted_deterministically"
            block.needs_llm = False
            return

    # Strong ML + supporting structural evidence.
    if ml_conf >= 0.78 and det_score >= 0.35:
        block.final_section = ml_section
        block.final_confidence = min(0.99, 0.55 * ml_conf + 0.45 * det_score)
        block.decision = "accepted_ml_plus_rules"
        block.needs_llm = False
        return

    # Strong deterministic evidence can rescue a weak ML prediction.
    if det_score >= 0.75:
        block.final_section = ml_section
        block.final_confidence = det_score
        block.decision = "accepted_deterministically"
        block.needs_llm = False
        return

    # Check disagreement with the runner-up.
    runner_up = None
    runner_up_prob = 0.0
    for label, prob in alternatives.items():
        if label != ml_section:
            runner_up = label
            runner_up_prob = prob
            break

    margin = ml_conf - runner_up_prob

    if ml_conf < 0.60 or margin < 0.15 or det_score < 0.35:
        block.decision = "needs_llm"
        block.needs_llm = True
        return

    block.final_section = ml_section
    block.final_confidence = ml_conf
    block.decision = "accepted_ml"
    block.needs_llm = False


# ============================================================
# 10. OPTIONAL LLM ADAPTER
# ============================================================

LLM_PROMPT = """You are validating one resume block.

Choose exactly one section from:
contact, summary, experience, education, skills, projects,
certifications, awards, publications, languages, volunteering,
interests, references, other.

Rules:
- Do not invent content.
- Classify the supplied block only.
- Prefer EXPERIENCE when a block contains a job title/company/date range
  and responsibilities, even if there is no WORK EXPERIENCE heading.
- Return JSON only.

Schema:
{
  "section": "experience",
  "confidence": 0.0,
  "reason": "short factual explanation"
}

Resume block:
{block}
"""


def llm_validate(block: Block) -> Optional[dict[str, Any]]:
    """
    Optional adapter.

    To keep this PoC provider-neutral, the actual HTTP/API call is not
    hard-coded. If LLM_API_URL is supplied, a production implementation
    can call the selected provider here.

    By default this returns None, meaning:
        unresolved -> explicitly unresolved
    rather than silently guessing.
    """
    if not os.getenv("LLM_API_URL"):
        return None

    # Intentionally left provider-neutral.
    # Add the team's chosen LLM client here.
    #
    # The important architectural contract is:
    # input = ONE ambiguous block + limited context
    # output = structured section/confidence/reason
    return None


# ============================================================
# 11. COVERAGE VALIDATION
# ============================================================

def coverage_report(
    source_lines: list[SourceLine],
    blocks: list[Block],
) -> dict[str, Any]:
    meaningful = {
        line.line_id
        for line in source_lines
        if not line.is_blank
    }

    assigned = set()
    duplicates = set()

    for block in blocks:
        for line_id in block.line_ids:
            if line_id in assigned:
                duplicates.add(line_id)
            assigned.add(line_id)

    missing = sorted(meaningful - assigned)
    extra = sorted(assigned - meaningful)

    coverage = (
        1.0
        if not meaningful
        else len(meaningful & assigned) / len(meaningful)
    )

    unresolved_lines = []
    for block in blocks:
        if block.decision == "needs_llm":
            unresolved_lines.extend(block.line_ids)

    return {
        "meaningful_source_lines": len(meaningful),
        "assigned_source_lines": len(meaningful & assigned),
        "missing_source_lines": missing,
        "duplicate_source_lines": sorted(duplicates),
        "extra_source_lines": extra,
        "coverage": round(coverage, 4),
        "coverage_pass": not missing and not duplicates and not extra,
        "unresolved_blocks": sum(
            1 for block in blocks if block.decision == "needs_llm"
        ),
        "unresolved_source_lines": sorted(set(unresolved_lines)),
    }


# ============================================================
# 12. FINAL SECTION MAP
# ============================================================

def final_section_map(blocks: list[Block]) -> dict[str, list[dict[str, Any]]]:
    result = {section: [] for section in SECTIONS}

    for block in blocks:
        section = block.final_section or "other"
        if section not in result:
            section = "other"

        result[section].append(
            {
                "block_id": block.block_id,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "text": block.text,
                "confidence": round(block.final_confidence, 4),
                "decision": block.decision,
                "needs_llm": block.needs_llm,
            }
        )

    return result


# ============================================================
# 13. PIPELINE
# ============================================================

def parse_resume(file_path: str) -> dict[str, Any]:
    raw_text = extract_markdown(file_path)
    source_lines = build_document(raw_text)

    # IMPORTANT:
    # Segmentation comes before semantic classification.
    blocks = segment_blocks(source_lines)

    model = train_ml_classifier()

    predictions = []

    for block in blocks:
        ml_section, ml_conf, alternatives = ml_predict(model, block.text)

        block.ml_section = ml_section
        block.ml_confidence = ml_conf

        validate_ml_prediction(block, alternatives)

        llm_result = None

        if block.needs_llm:
            llm_result = llm_validate(block)

            if llm_result:
                block.final_section = llm_result.get("section", "other")
                block.final_confidence = float(
                    llm_result.get("confidence", 0.0)
                )
                block.decision = "accepted_llm"
                block.needs_llm = False

        predictions.append(
            {
                "block_id": block.block_id,
                "ml_section": block.ml_section,
                "ml_confidence": round(block.ml_confidence, 4),
                "deterministic_score": round(block.deterministic_score, 4),
                "deterministic_evidence": block.rule_evidence,
                "final_section": block.final_section,
                "final_confidence": round(block.final_confidence, 4),
                "decision": block.decision,
                "needs_llm": block.needs_llm,
                "alternatives": alternatives,
                "llm_result": llm_result,
            }
        )

    coverage = coverage_report(source_lines, blocks)

    return {
        "success": True,
        "architecture": "MarkItDown -> blocks -> ML -> deterministic -> LLM(ambiguous)",
        "source_file": str(Path(file_path)),
        "document_stats": {
            "total_source_lines": len(source_lines),
            "meaningful_source_lines": sum(
                not x.is_blank for x in source_lines
            ),
            "candidate_blocks": len(blocks),
        },
        "coverage": coverage,
        "blocks": [
            {
                "block_id": b.block_id,
                "start_line": b.start_line,
                "end_line": b.end_line,
                "line_ids": b.line_ids,
                "heading": b.heading,
                "text": b.text,
                "ml_section": b.ml_section,
                "ml_confidence": round(b.ml_confidence, 4),
                "deterministic_section": b.deterministic_section,
                "deterministic_score": round(b.deterministic_score, 4),
                "rule_evidence": b.rule_evidence,
                "final_section": b.final_section,
                "final_confidence": round(b.final_confidence, 4),
                "decision": b.decision,
                "needs_llm": b.needs_llm,
            }
            for b in blocks
        ],
        "section_map": final_section_map(blocks),
        "evaluation": {
            "note": (
                "Section accuracy cannot be claimed from this PoC alone. "
                "Create a labelled test set and compare predictions against "
                "ground truth."
            ),
        },
    }


def print_human_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("HYBRID RESUME SECTIONING POC")
    print("=" * 72)

    stats = result["document_stats"]
    coverage = result["coverage"]

    print(f"Source: {result['source_file']}")
    print(f"Source lines: {stats['total_source_lines']}")
    print(f"Meaningful lines: {stats['meaningful_source_lines']}")
    print(f"Candidate blocks: {stats['candidate_blocks']}")

    print("\nCOVERAGE")
    print("-" * 72)
    print(f"Coverage: {coverage['coverage'] * 100:.2f}%")
    print(f"Coverage PASS: {coverage['coverage_pass']}")
    print(f"Missing lines: {coverage['missing_source_lines']}")
    print(f"Duplicate lines: {coverage['duplicate_source_lines']}")
    print(f"LLM/unresolved blocks: {coverage['unresolved_blocks']}")

    print("\nBLOCK DECISIONS")
    print("-" * 72)

    for block in result["blocks"]:
        preview = block["text"].replace("\n", " | ")
        if len(preview) > 110:
            preview = preview[:107] + "..."

        print(
            f"{block['block_id']} "
            f"[L{block['start_line']}-{block['end_line']}] "
            f"ML={block['ml_section']}({block['ml_confidence']:.2f}) "
            f"FINAL={block['final_section']} "
            f"DECISION={block['decision']}\n"
            f"    {preview}"
        )

    print("\nSECTION DISTRIBUTION")
    print("-" * 72)

    for section, items in result["section_map"].items():
        if items:
            print(f"{section:18} {len(items)} block(s)")

    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid ML + deterministic + optional LLM resume sectioning PoC"
    )
    parser.add_argument("resume", help="Path to PDF/DOCX resume")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output file",
    )

    args = parser.parse_args()

    path = Path(args.resume)
    if not path.exists():
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"File not found: {path}",
                },
                indent=2,
            )
        )
        return 1

    try:
        result = parse_resume(str(path))

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_human_summary(result)

            if args.output:
                print(f"\nFull JSON saved to: {args.output}")

        return 0

    except Exception as exc:
        error = {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

