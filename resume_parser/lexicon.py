"""Section-header lexicon for deterministic label proposals (B2).

Parity with the sectioning PoC (``poc_sectioning.py``): same TYPES, same
LEXICON/REV mapping, same matching rule (exact match on normalized text,
at most 5 words). B2 labels are PROPOSALS (``proposal_source``); the ML
stage may replace them later.
"""

from __future__ import annotations

import re

TYPES = ["contact_header", "summary_objective", "experience", "education", "skills",
         "projects", "certifications", "awards", "publications", "languages",
         "volunteering", "interests", "references", "other"]

LEXICON = {
    "summary_objective": ["summary", "professional summary", "objective", "career objective", "profile", "about me", "about"],
    "experience": ["experience", "work experience", "professional experience", "employment", "employment history", "work history", "career history"],
    "education": ["education", "academic background", "academics", "education and training", "qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "tech stack", "technologies"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "certifications": ["certifications", "licenses and certifications", "certificates", "courses"],
    "awards": ["awards", "honors and awards", "achievements", "honors"],
    "publications": ["publications", "research papers", "papers"],
    "languages": ["languages"],
    "volunteering": ["volunteering", "volunteer experience", "community service"],
    "interests": ["interests", "hobbies", "hobbies and interests"],
    "references": ["references"],
}
REV = {p: t for t, ps in LEXICON.items() for p in ps}

MAX_HEADER_WORDS = 5


def header_norm(s: str) -> str:
    """Match-only normalization (PoC parity). Never applied to output text."""
    s = re.sub(r"^[#\s*_|>\-]+", "", s.strip())
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def rule_header(line: str) -> str | None:
    """Exact lexicon match on a short line; None when no rule fires."""
    n = header_norm(line)
    return REV.get(n) if n and len(n.split()) <= MAX_HEADER_WORDS else None
