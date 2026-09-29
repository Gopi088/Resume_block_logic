"""B3: ML Semantic Classification for Candidate Blocks.

This module implements the PRIMARY semantic classifier for resume sections.
It uses a TF-IDF + Logistic Regression baseline as specified in the architecture.

Architecture:
- B2 produces structural CandidateBlocks (no semantic labels)
- B3 consumes CandidateBlocks and assigns semantic section labels
- ML is the PRIMARY classifier; deterministic validation (B4) and LLM (B5) come later
- Provenance from B1/B2 is preserved throughout

Model: TF-IDF (word n-grams 1-2) + Logistic Regression (balanced class weights)
Features: Text content + structural features from B2

Implementation status (v1 baseline):
- Block-level classification IS implemented (one label per candidate block).
- Semantic span detection is NOT yet inferred: the data model supports spans
  (BlockClassification.semantic_spans, LabelledBlock.semantic_spans, multi-row
  CSV annotations), but inference assigns a single block-level label. Mixed
  blocks are therefore labelled as a whole in v1; span splitting is future work.
- The v1 model uses TF-IDF TEXT features only. extract_semantic_features /
  extract_structural_features are implemented, tested helpers reserved for a
  future fusion stage; the `feature_weight` plumbing is stored but inactive.
- sklearn label boundary: sklearn mangles str-Enum members (LabelEncoder /
  pipeline classes_ come back as plain strings, not SectionLabel members), so
  ALL labels are coerced to plain value strings before any sklearn call and
  converted back to SectionLabel at the API boundary (see label_values()).

Boilerplate decision: B2 excludes boilerplate-flagged lines from candidate
blocks (they live in SegmentationResult.boilerplate, never deleted). B3
therefore never receives boilerplate text; the SectionLabel.BOILERPLATE value
is reserved for B4+ use.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from resume_parser.models import (
    AlternativePrediction,
    BlockClassification,
    CandidateBlock,
    ClassificationResult,
    IntegrityReport,
    SectionLabel,
    SegmentationResult,
    SemanticSpan,
)


# ============================================================
# Feature Extraction
# ============================================================

# Semantic vocabulary features (used as ML features, NOT deterministic rules)
ROLE_WORDS = {
    "engineer", "developer", "analyst", "manager", "consultant",
    "architect", "lead", "director", "specialist", "administrator",
    "designer", "intern", "officer", "executive", "associate",
    "scientist", "trainee", "coordinator", "product owner",
    "scrum master", "business analyst", "project manager",
    "team lead", "system engineer", "software", "data",
}

COMPANY_WORDS = {
    "ltd", "limited", "inc", "corp", "corporation", "llc",
    "technologies", "technology", "solutions", "services", "consulting",
    "consultancy", "systems", "industries", "group", "bank",
    "tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini",
    "ibm", "microsoft", "amazon", "google", "meta", "apple",
}

EDUCATION_WORDS = {
    "bachelor", "master", "b.tech", "m.tech", "mba", "phd",
    "university", "college", "degree", "graduated", "school",
    "b.sc", "m.sc", "b.e", "m.e", "bca", "mca",
}

SKILL_WORDS = {
    "python", "java", "javascript", "sql", "react", "power bi", "excel",
    "tableau", "jira", "confluence", "docker", "aws", "azure", "gcp",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "servicenow", "itil", "power automate", "kubernetes", "git",
    "linux", "html", "css", "node", "django", "flask", "spring",
}

CERT_WORDS = {
    "certification", "certified", "certificate", "foundation certificate",
    "license", "licensed", "credential", "accreditation",
}

AWARD_WORDS = {
    "award", "awards", "honor", "honours", "winner", "recognition",
    "recognized", "recognised", "excellence", "achievement",
}

PUBLICATION_WORDS = {
    "publication", "publications", "paper", "papers", "research",
    "published", "journal", "conference", "proceedings", "ieee",
}

LANGUAGE_WORDS = {
    "english", "hindi", "spanish", "french", "german", "chinese",
    "japanese", "native", "fluent", "conversational", "proficient",
    "basic", "intermediate", "advanced",
}

VOLUNTEER_WORDS = {
    "volunteer", "volunteering", "community", "service", "ngo",
    "nonprofit", "charity", "social", "outreach",
}

PROJECT_WORDS = {
    "project", "projects", "built", "developed", "created", "designed",
    "implemented", "github", "repository", "open source",
}

INTEREST_WORDS = {
    "interest", "interests", "hobby", "hobbies", "photography",
    "travel", "reading", "music", "sports", "gaming",
}

REFERENCE_WORDS = {
    "reference", "references", "referee", "referees", "available upon request",
}

CONTACT_WORDS = {
    "email", "phone", "linkedin", "github", "address", "city",
    "country", "mobile", "contact", "@", ".com",
}

SUMMARY_WORDS = {
    "summary", "profile", "objective", "about", "experience",
    "years", "professional", "expertise", "specialist", "passionate",
    "driven", "results", "track record", "background",
}


def label_values(labels: list[SectionLabel] | list[str]) -> list[str]:
    """Coerce labels to plain value strings for sklearn.

    REQUIRED: sklearn's encoders mangle str-Enum members (pipeline classes_
    come back as strings, breaking SectionLabel(...) lookups and silently
    zeroing every metric). Never pass SectionLabel members to fit/predict/
    scoring directly — convert at this boundary and back at the API edge.
    """
    return [l.value if isinstance(l, SectionLabel) else str(l) for l in labels]


def extract_semantic_features(text: str) -> dict[str, float]:
    """Extract semantic vocabulary features from text.
    
    These are used as ML features, not deterministic rules.
    Returns normalized feature counts.
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    total_words = max(len(words), 1)
    
    features = {}
    features["role_word_ratio"] = len(words & ROLE_WORDS) / total_words
    features["company_word_ratio"] = len(words & COMPANY_WORDS) / total_words
    features["education_word_ratio"] = len(words & EDUCATION_WORDS) / total_words
    features["skill_word_ratio"] = len(words & SKILL_WORDS) / total_words
    features["cert_word_ratio"] = len(words & CERT_WORDS) / total_words
    features["award_word_ratio"] = len(words & AWARD_WORDS) / total_words
    features["publication_word_ratio"] = len(words & PUBLICATION_WORDS) / total_words
    features["language_word_ratio"] = len(words & LANGUAGE_WORDS) / total_words
    features["volunteer_word_ratio"] = len(words & VOLUNTEER_WORDS) / total_words
    features["project_word_ratio"] = len(words & PROJECT_WORDS) / total_words
    features["interest_word_ratio"] = len(words & INTEREST_WORDS) / total_words
    features["reference_word_ratio"] = len(words & REFERENCE_WORDS) / total_words
    features["contact_word_ratio"] = len(words & CONTACT_WORDS) / total_words
    features["summary_word_ratio"] = len(words & SUMMARY_WORDS) / total_words
    
    # Date-like pattern feature
    import re
    date_pattern = re.compile(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}"
        r"|\d{1,2}[-/]\d{4}"
        r"|\d{4}\s*[-–—]\s*\d{4}"
        r"|present|current",
        re.IGNORECASE,
    )
    features["has_date_pattern"] = 1.0 if date_pattern.search(text) else 0.0
    
    # Bullet ratio
    lines = text.split("\n")
    bullet_lines = sum(1 for l in lines if l.strip().startswith(("•", "-", "*", "·", "●", "▪", "◦")))
    features["bullet_ratio"] = bullet_lines / max(len(lines), 1)
    
    # Line count
    features["line_count"] = float(len(lines))
    
    # Character count
    features["char_count"] = float(len(text))
    
    return features


def extract_structural_features(block: CandidateBlock) -> dict[str, float]:
    """Extract structural features from B2 candidate block."""
    features = {}
    features["block_line_count"] = float(len(block.line_ids))
    features["block_page_count"] = float(len(block.page_indices))
    features["has_header"] = 1.0 if block.header_line_id else 0.0
    features["is_continuation"] = 1.0 if block.is_continuation else 0.0
    
    # Boundary signal counts
    from collections import Counter
    signal_counts = Counter(block.boundary_signals)
    for signal in [
        "blank_line", "heading_like", "markdown_heading", "indentation_change",
        "bullet_transition", "table_boundary", "code_fence_boundary",
        "page_break", "formatting_change", "continuation_detected"
    ]:
        features[f"boundary_{signal}"] = float(signal_counts.get(signal, 0))
    
    # Display line characteristics
    display_kinds = [d.line_kind.value for d in block.display_lines]
    features["display_text_lines"] = float(len(display_kinds))
    features["display_bullet_ratio"] = display_kinds.count("bullet") / max(len(display_kinds), 1)
    features["display_heading_ratio"] = display_kinds.count("markdown_heading") / max(len(display_kinds), 1)
    features["display_table_ratio"] = display_kinds.count("table_row") / max(len(display_kinds), 1)
    features["display_code_ratio"] = display_kinds.count("code_fence") / max(len(display_kinds), 1)
    
    return features


def build_feature_text(block: CandidateBlock) -> str:
    """Build the primary text feature for TF-IDF from a candidate block.
    
    Uses the block's display text which preserves hyphen joins and formatting.
    """
    return block.text


# ============================================================
# Training Data Structures
# ============================================================

@dataclass
class LabelledBlock:
    """A candidate block with gold label from human annotation."""
    document_id: str
    block_id: str
    line_ids: list[str]
    start_line_index: int
    end_line_index: int
    text: str
    gold_section: SectionLabel
    # For mixed blocks: list of (start_line_id, end_line_id, section)
    semantic_spans: list[tuple[str, str, SectionLabel]] | None = None


@dataclass
class LabelledDocument:
    """A resume document with gold labels for all its candidate blocks."""
    document_id: str
    filename: str
    blocks: list[LabelledBlock]


class TrainingDataset:
    """Container for labelled training data with resume-level grouping."""
    
    def __init__(self, documents: list[LabelledDocument]):
        self.documents = documents
        self._validate()
    
    def _validate(self) -> None:
        """Validate the dataset integrity."""
        doc_ids = set()
        for doc in self.documents:
            if doc.document_id in doc_ids:
                raise ValueError(f"Duplicate document_id: {doc.document_id}")
            doc_ids.add(doc.document_id)
            for block in doc.blocks:
                if block.gold_section not in SectionLabel:
                    raise ValueError(f"Invalid gold_section: {block.gold_section}")
    
    def get_all_blocks(self) -> list[LabelledBlock]:
        """Flatten all blocks from all documents."""
        return [b for doc in self.documents for b in doc.blocks]
    
    def get_texts_and_labels(self) -> tuple[list[str], list[SectionLabel], list[str]]:
        """Get texts, labels, and document groups for training."""
        blocks = self.get_all_blocks()
        texts = [b.text for b in blocks]
        labels = [b.gold_section for b in blocks]
        groups = [b.document_id for b in blocks]
        return texts, labels, groups
    
    def split_by_resume(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                         test_ratio: float = 0.15, seed: int = 42) -> tuple["TrainingDataset", "TrainingDataset", "TrainingDataset"]:
        """Split dataset at RESUME level (not block level) to prevent leakage.
        
        This is CRITICAL - blocks from the same resume are highly correlated.
        """
        import random
        random.seed(seed)
        
        n_docs = len(self.documents)
        indices = list(range(n_docs))
        random.shuffle(indices)
        
        n_train = int(n_docs * train_ratio)
        n_val = int(n_docs * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_docs = [self.documents[i] for i in train_indices]
        val_docs = [self.documents[i] for i in val_indices]
        test_docs = [self.documents[i] for i in test_indices]
        
        return (
            TrainingDataset(train_docs),
            TrainingDataset(val_docs),
            TrainingDataset(test_docs),
        )
    
    def get_class_distribution(self) -> dict[SectionLabel, int]:
        """Get class distribution across all blocks."""
        from collections import Counter
        return Counter(b.gold_section for b in self.get_all_blocks())
    
    def __len__(self) -> int:
        return len(self.get_all_blocks())
    
    def __repr__(self) -> str:
        dist = self.get_class_distribution()
        return f"TrainingDataset({len(self.documents)} docs, {len(self)} blocks, classes={len(dist)})"


# ============================================================
# Model Training & Persistence
# ============================================================

class SectionClassifier:
    """TF-IDF + Logistic Regression classifier for resume section classification.
    
    This is the PRIMARY semantic classifier (B3).
    It produces probabilities for all section classes.
    """
    
    MODEL_VERSION = "1.0"
    
    def __init__(
        self,
        tfidf_params: dict[str, Any] | None = None,
        lr_params: dict[str, Any] | None = None,
        feature_weight: float = 0.0,  # Weight for structural/semantic features (0 = text only)
    ):
        self.feature_weight = feature_weight
        self.tfidf_params = tfidf_params or {
            "lowercase": True,
            "ngram_range": (1, 2),
            "min_df": 1,
            "sublinear_tf": True,
            "max_features": 10000,
        }
        self.lr_params = lr_params or {
            "max_iter": 2000,
            "class_weight": "balanced",
            "C": 1.0,
            "random_state": 42,
            "solver": "lbfgs",
        }
        
        self.pipeline: Pipeline | None = None
        self.classes_: list[SectionLabel] | None = None
        self.metadata: dict[str, Any] = {}
    
    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline."""
        return Pipeline([
            ("tfidf", TfidfVectorizer(**self.tfidf_params)),
            ("classifier", LogisticRegression(**self.lr_params)),
        ])
    
    def train(
        self,
        texts: list[str],
        labels: list[SectionLabel],
        groups: list[str] | None = None,
        eval_texts: list[str] | None = None,
        eval_labels: list[SectionLabel] | None = None,
    ) -> dict[str, Any]:
        """Train the classifier.
        
        Args:
            texts: Training texts (block text content)
            labels: Gold section labels
            groups: Document IDs for each block (for GroupKFold CV)
            eval_texts: Optional validation texts
            eval_labels: Optional validation labels
        
        Returns:
            Training metadata including evaluation metrics
        """
        self.pipeline = self._build_pipeline()
        self.classes_ = sorted(set(labels), key=lambda x: x.value)

        # Train on full training set (labels coerced: see label_values()).
        fit_labels = label_values(labels)
        self.pipeline.fit(texts, fit_labels)
        
        # Evaluate on validation set if provided
        eval_metrics = {}
        if eval_texts and eval_labels:
            eval_metrics = self.evaluate(eval_texts, eval_labels)
        
        # Cross-validation if groups provided
        cv_metrics = {}
        if groups and len(set(groups)) >= 3:
            cv_metrics = self._cross_validate(texts, labels, groups)
        
        self.metadata = {
            "model_version": self.MODEL_VERSION,
            "tfidf_params": self.tfidf_params,
            "lr_params": self.lr_params,
            "feature_weight": self.feature_weight,
            "classes": [c.value for c in self.classes_],
            "n_training_samples": len(texts),
            "n_classes": len(self.classes_),
            "eval_metrics": eval_metrics,
            "cv_metrics": cv_metrics,
        }
        
        return self.metadata
    
    def _cross_validate(
        self, texts: list[str], labels: list[SectionLabel], groups: list[str]
    ) -> dict[str, Any]:
        """Perform GroupKFold cross-validation (resume-level)."""
        n_splits = min(5, len(set(groups)))
        gkf = GroupKFold(n_splits=n_splits)
        
        all_preds = []
        all_true = []
        
        for train_idx, test_idx in gkf.split(texts, labels, groups):
            train_texts = [texts[i] for i in train_idx]
            train_labels = label_values([labels[i] for i in train_idx])
            test_texts = [texts[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
            
            fold_pipeline = self._build_pipeline()
            fold_pipeline.fit(train_texts, train_labels)
            preds = fold_pipeline.predict(test_texts)
            
            all_preds.extend(preds)
            all_true.extend(test_labels)
        
        return self._compute_metrics(all_true, all_preds, prefix="cv_")
    
    def _compute_metrics(
        self, y_true: list[SectionLabel], y_pred: list[SectionLabel], prefix: str = ""
    ) -> dict[str, Any]:
        """Compute comprehensive classification metrics.

        Both inputs are coerced to plain value strings (see label_values())
        so SectionLabel members and raw sklearn string outputs compare equal.
        """
        yt = label_values(y_true)
        yp = label_values(y_pred)
        # Score every class observed on either side. A test resume may contain
        # sections absent from training (never predicted: honest zeros) or
        # predictions outside the gold set — every sample stays counted and
        # sklearn's labels∩y_true requirement is satisfied by construction.
        scored = sorted(set(yt) | set(yp))
        if not scored:
            return {
                f"{prefix}accuracy": 0.0,
                f"{prefix}macro_precision": 0.0,
                f"{prefix}macro_recall": 0.0,
                f"{prefix}macro_f1": 0.0,
                f"{prefix}per_class": {},
                f"{prefix}confusion_matrix": {"labels": [], "matrix": []},
            }
        accuracy = accuracy_score(yt, yp)
        precision, recall, f1, support = precision_recall_fscore_support(
            yt, yp, average=None, labels=scored, zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0
        )
        cm = confusion_matrix(yt, yp, labels=scored)

        # Per-class metrics
        per_class = {}
        for i, cls_value in enumerate(scored):
            per_class[cls_value] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        return {
            f"{prefix}accuracy": float(accuracy),
            f"{prefix}macro_precision": float(macro_precision),
            f"{prefix}macro_recall": float(macro_recall),
            f"{prefix}macro_f1": float(macro_f1),
            f"{prefix}per_class": per_class,
            f"{prefix}confusion_matrix": {
                "labels": scored,
                "matrix": [[int(v) for v in row] for row in cm],
            },
        }
    
    def evaluate(self, texts: list[str], labels: list[SectionLabel]) -> dict[str, Any]:
        """Evaluate on a test set."""
        if not self.pipeline:
            raise RuntimeError("Model not trained")
        preds = self.pipeline.predict(texts)
        return self._compute_metrics(labels, preds, prefix="eval_")
    
    def predict(
        self, texts: list[str], return_alternatives: int = 3
    ) -> list[tuple[SectionLabel, float, list[AlternativePrediction]]]:
        """Predict section labels with confidence and alternatives.
        
        Args:
            texts: List of block texts to classify
            return_alternatives: Number of top alternatives to return
        
        Returns:
            List of (predicted_section, confidence, alternatives)
        """
        if not self.pipeline:
            raise RuntimeError("Model not trained")
        
        # Get probabilities for all classes
        probas = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        
        results = []
        for proba in probas:
            # Sort by probability descending
            sorted_idx = np.argsort(proba)[::-1]
            
            best_idx = sorted_idx[0]
            predicted = SectionLabel(classes[best_idx])
            confidence = float(proba[best_idx])
            
            # Top-k alternatives
            alternatives = []
            for idx in sorted_idx[1:return_alternatives+1]:
                alternatives.append(AlternativePrediction(
                    section=SectionLabel(classes[idx]),
                    confidence=float(proba[idx]),
                ))
            
            results.append((predicted, confidence, alternatives))
        
        return results
    
    def predict_single(self, text: str, return_alternatives: int = 3) -> tuple[SectionLabel, float, list[AlternativePrediction]]:
        """Predict for a single text."""
        return self.predict([text], return_alternatives)[0]
    
    def save(self, path: str | Path) -> None:
        """Save the trained model and metadata."""
        if not self.pipeline:
            raise RuntimeError("No trained model to save")
        
        save_data = {
            "pipeline": self.pipeline,
            "metadata": self.metadata,
            "feature_weight": self.feature_weight,
            "tfidf_params": self.tfidf_params,
            "lr_params": self.lr_params,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "SectionClassifier":
        """Load a trained model."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        clf = cls(
            tfidf_params=save_data.get("tfidf_params"),
            lr_params=save_data.get("lr_params"),
            feature_weight=save_data.get("feature_weight", 0.0),
        )
        clf.pipeline = save_data["pipeline"]
        clf.metadata = save_data.get("metadata", {})
        # pipeline.classes_ are plain value strings (see label_values());
        # restore SectionLabel members at the API edge.
        clf.classes_ = [SectionLabel(s) for s in clf.pipeline.classes_]
        return clf


# ============================================================
# B3 Classification Pipeline
# ============================================================

def classify_blocks(
    seg_result: SegmentationResult,
    classifier: SectionClassifier,
    return_alternatives: int = 3,
) -> ClassificationResult:
    """Classify all candidate blocks from a SegmentationResult.
    
    This is the main B3 inference function.
    """
    if not classifier.pipeline:
        raise RuntimeError("Classifier not trained")
    
    # Prepare texts for classification. NOTE: B2 excludes boilerplate-flagged
    # lines from candidate blocks (they live in SegmentationResult.boilerplate),
    # so every block here is content and gets classified. Boilerplate provenance
    # is preserved at B2; the BOILERPLATE label is reserved for B4+ use.
    texts = []
    blocks_to_classify = []
    
    for block in seg_result.blocks:
        texts.append(block.text)
        blocks_to_classify.append(block)
    
    if not texts:
        return ClassificationResult(
            document_id=seg_result.document_id,
            classifications=[],
            model_metadata=classifier.metadata,
            integrity=IntegrityReport(
                passed=True,
                violations=[],
                checks={"no_blocks": True},
            ),
        )
    
    # Get predictions
    predictions = classifier.predict(texts, return_alternatives=return_alternatives)
    
    # Build classification results with provenance
    classifications = []
    for block, (predicted, confidence, alternatives) in zip(blocks_to_classify, predictions):
        cls = BlockClassification(
            block_id=block.block_id,
            document_id=seg_result.document_id,
            predicted_section=predicted,
            confidence=confidence,
            alternatives=alternatives,
            source_line_ids=block.line_ids,
            start_line_index=block.start_line_index,
            end_line_index=block.end_line_index,
            classification_status="classified" if confidence >= 0.5 else "low_confidence",
        )
        classifications.append(cls)
    
    # Integrity check
    integrity = build_classification_integrity(seg_result, classifications)
    
    return ClassificationResult(
        document_id=seg_result.document_id,
        classifications=classifications,
        model_metadata=classifier.metadata,
        integrity=integrity,
    )


def build_classification_integrity(
    seg_result: SegmentationResult,
    classifications: list[BlockClassification],
) -> IntegrityReport:
    """Verify B3 output integrity against B2 input."""
    violations = []
    checks = {}
    
    # All blocks classified
    block_ids = {b.block_id for b in seg_result.blocks}
    classified_ids = {c.block_id for c in classifications}
    
    missing = block_ids - classified_ids
    extra = classified_ids - block_ids
    
    checks["all_blocks_classified"] = len(missing) == 0
    if missing:
        violations.append(f"Missing classifications for blocks: {missing}")
    if extra:
        violations.append(f"Extra classifications for unknown blocks: {extra}")
    
    # All classifications have valid provenance
    valid_line_ids = set()
    for block in seg_result.blocks:
        valid_line_ids.update(block.line_ids)
    
    provenance_ok = True
    for c in classifications:
        for lid in c.source_line_ids:
            if lid not in valid_line_ids:
                provenance_ok = False
                violations.append(f"Classification {c.block_id} references unknown line_id: {lid}")
    
    checks["provenance_valid"] = provenance_ok
    
    # Confidence in valid range
    confidence_ok = all(0.0 <= c.confidence <= 1.0 for c in classifications)
    checks["confidence_range_valid"] = confidence_ok
    if not confidence_ok:
        violations.append("Some confidences outside [0, 1] range")
    
    # Alternatives have valid confidences
    alt_conf_ok = all(
        0.0 <= a.confidence <= 1.0
        for c in classifications
        for a in c.alternatives
    )
    checks["alternatives_confidence_valid"] = alt_conf_ok
    
    return IntegrityReport(
        passed=len(violations) == 0,
        violations=violations,
        checks=checks,
    )


# ============================================================
# Training Pipeline
# ============================================================

def train_section_classifier(
    dataset: TrainingDataset,
    tfidf_params: dict[str, Any] | None = None,
    lr_params: dict[str, Any] | None = None,
    feature_weight: float = 0.0,
) -> tuple[SectionClassifier, dict[str, Any]]:
    """Complete training pipeline with resume-level splits.
    
    Args:
        dataset: Labelled training data
        tfidf_params: TF-IDF vectorizer parameters
        lr_params: Logistic Regression parameters
        feature_weight: Weight for additional features (0 = text only)
    
    Returns:
        (trained classifier, training metadata)
    """
    # Split by resume (critical: no leakage)
    train_ds, val_ds, test_ds = dataset.split_by_resume()
    
    # Get texts and labels
    train_texts, train_labels, train_groups = train_ds.get_texts_and_labels()
    val_texts, val_labels, _ = val_ds.get_texts_and_labels()
    test_texts, test_labels, _ = test_ds.get_texts_and_labels()
    
    # Train classifier
    classifier = SectionClassifier(
        tfidf_params=tfidf_params,
        lr_params=lr_params,
        feature_weight=feature_weight,
    )
    
    metadata = classifier.train(
        train_texts, train_labels, train_groups,
        eval_texts=val_texts, eval_labels=val_labels,
    )
    
    # Final evaluation on test set (skipped explicitly when the split is empty,
    # e.g. tiny development datasets — never silently scored).
    if test_texts and test_labels:
        test_metrics = classifier.evaluate(test_texts, test_labels)
    else:
        test_metrics = {"skipped": True, "reason": "empty test split"}
    metadata["test_metrics"] = test_metrics
    metadata["train_dataset_size"] = len(train_ds)
    metadata["val_dataset_size"] = len(val_ds)
    metadata["test_dataset_size"] = len(test_ds)
    metadata["train_docs"] = len(train_ds.documents)
    metadata["val_docs"] = len(val_ds.documents)
    metadata["test_docs"] = len(test_ds.documents)
    
    return classifier, metadata


# ============================================================
# Annotation / Labelled Data Utilities
# ============================================================

def create_labelled_dataset_from_csv(
    annotations_path: str | Path,
    segmentation_results: dict[str, SegmentationResult],
) -> TrainingDataset:
    """Create a TrainingDataset from CSV annotations.
    
    Expected CSV format (one row per block):
    document_id,block_id,start_line_id,end_line_id,gold_section
    OR for mixed blocks (multiple rows per block with different spans):
    document_id,block_id,start_line_id,end_line_id,gold_section
    """
    import csv
    
    annotations_by_doc: dict[str, list[dict]] = {}
    
    with open(annotations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["document_id"]
            annotations_by_doc.setdefault(doc_id, []).append(row)
    
    documents = []
    for doc_id, seg_result in segmentation_results.items():
        if doc_id not in annotations_by_doc:
            warnings.warn(f"No annotations for document {doc_id}")
            continue
        
        ann_rows = annotations_by_doc[doc_id]
        blocks = []
        
        # Group by block_id to handle mixed blocks
        from collections import defaultdict
        by_block = defaultdict(list)
        for row in ann_rows:
            by_block[row["block_id"]].append(row)
        
        for block in seg_result.blocks:
            if block.block_id not in by_block:
                warnings.warn(f"No annotation for block {block.block_id} in {doc_id}")
                continue
            
            rows = by_block[block.block_id]
            if len(rows) == 1:
                # Simple case: single label for entire block
                row = rows[0]
                blocks.append(LabelledBlock(
                    document_id=doc_id,
                    block_id=block.block_id,
                    line_ids=block.line_ids,
                    start_line_index=block.start_line_index,
                    end_line_index=block.end_line_index,
                    text=block.text,
                    gold_section=SectionLabel(row["gold_section"]),
                ))
            else:
                # Mixed block: multiple spans with different labels
                spans = []
                for row in rows:
                    spans.append((
                        row["start_line_id"],
                        row["end_line_id"],
                        SectionLabel(row["gold_section"]),
                    ))
                # Use majority label for block-level classification
                from collections import Counter
                majority = Counter(r["gold_section"] for r in rows).most_common(1)[0][0]
                blocks.append(LabelledBlock(
                    document_id=doc_id,
                    block_id=block.block_id,
                    line_ids=block.line_ids,
                    start_line_index=block.start_line_index,
                    end_line_index=block.end_line_index,
                    text=block.text,
                    gold_section=SectionLabel(majority),
                    semantic_spans=spans,
                ))
        
        documents.append(LabelledDocument(
            document_id=doc_id,
            filename=seg_result.document_id,  # or actual filename if available
            blocks=blocks,
        ))
    
    return TrainingDataset(documents)


# ============================================================
# Bootstrap Training Data (for development/testing)
# ============================================================

# Minimal bootstrap data matching the PoC taxonomy
# In production, this should be replaced with REAL labelled data
BOOTSTRAP_TRAINING_DATA = [
    # Contact
    ("John Doe john@example.com +91 9999999999 linkedin.com/in/johndoe", SectionLabel.CONTACT),
    ("Jane Smith jane@gmail.com Bengaluru India linkedin.com/in/jane", SectionLabel.CONTACT),
    ("Aarav Sharma +91 98765 43210 aarav@email.com github.com/aarav", SectionLabel.CONTACT),

    # Summary
    ("Professional summary with 8 years of experience in software engineering and leadership", SectionLabel.SUMMARY),
    ("Experienced business analyst with proven track record delivering enterprise solutions", SectionLabel.SUMMARY),
    ("Career profile specializing in data analysis stakeholder management and product delivery", SectionLabel.SUMMARY),
    ("Results-driven engineer with 5 years of experience building data products and leading teams", SectionLabel.SUMMARY),

    # Experience
    ("Senior Business Analyst Tata Consultancy Services 02/2024 – 08/2025 Sydney Australia analysed requirements coordinated developers stakeholders delivered projects", SectionLabel.EXPERIENCE),
    ("Software Engineer ABC Technologies Jan 2022 – Present Bengaluru developed applications fixed defects and collaborated with engineering teams", SectionLabel.EXPERIENCE),
    ("Project Manager XYZ Ltd 2020 – 2023 led teams managed delivery and coordinated stakeholders", SectionLabel.EXPERIENCE),
    ("Data Scientist Microsoft 2021 – Present built ML models improved accuracy by 15%", SectionLabel.EXPERIENCE),
    ("Backend Developer Startup Inc 2019 – 2021 designed APIs optimized database queries reduced latency", SectionLabel.EXPERIENCE),

    # Education
    ("Bachelor of Technology Computer Science Engineering University 2016", SectionLabel.EDUCATION),
    ("Master Software Engineering BITS Pilani 2022", SectionLabel.EDUCATION),
    ("B.Tech IMS Engineering College graduated computer science", SectionLabel.EDUCATION),
    ("MBA Indian Institute of Management 2020", SectionLabel.EDUCATION),
    ("PhD Computer Science Stanford University 2018", SectionLabel.EDUCATION),

    # Skills
    ("Python Java SQL React Docker AWS JavaScript", SectionLabel.SKILLS),
    ("Technical Skills Python SQL Power BI Excel JIRA Confluence", SectionLabel.SKILLS),
    ("Technology & Skills Avaloq Banking Suite ServiceNow ITIL Microsoft Power BI", SectionLabel.SKILLS),
    ("Core Competencies Machine Learning Deep Learning NLP TensorFlow PyTorch", SectionLabel.SKILLS),
    ("Tech Stack Go Kubernetes gRPC PostgreSQL Redis", SectionLabel.SKILLS),

    # Projects
    ("Resume Parser project developed using Python FastAPI React and PostgreSQL", SectionLabel.PROJECTS),
    ("AI Resume Timeline Parser project NLP machine learning", SectionLabel.PROJECTS),
    ("Personal Project E-commerce platform built with Node.js and MongoDB", SectionLabel.PROJECTS),
    ("Academic Project Distributed systems research paper implementation", SectionLabel.PROJECTS),
    ("Open Source Contributor to pandas and scikit-learn libraries", SectionLabel.PROJECTS),

    # Certifications
    ("AWS Certified Solutions Architect certification 2025", SectionLabel.CERTIFICATIONS),
    ("ITIL Foundation Certificate 2018 Microsoft Power BI certification", SectionLabel.CERTIFICATIONS),
    ("Google Cloud Professional Data Engineer 2023", SectionLabel.CERTIFICATIONS),
    ("Certified Scrum Master CSM 2021", SectionLabel.CERTIFICATIONS),
    ("PMP Project Management Professional 2020", SectionLabel.CERTIFICATIONS),

    # Awards
    ("L1 Team Lead Award Ownership Award Best Team Award", SectionLabel.AWARDS),
    ("Honors Awards Continuous Performance Award Winner", SectionLabel.AWARDS),
    ("Employee of the Year 2022 Excellence Award", SectionLabel.AWARDS),
    ("Hackathon Winner First Place 2023", SectionLabel.AWARDS),
    ("Dean's List Academic Achievement Award", SectionLabel.AWARDS),

    # Publications
    ("Research paper published in IEEE conference transformer NLP phishing detection", SectionLabel.PUBLICATIONS),
    ("Published paper ACM SIGIR conference information retrieval", SectionLabel.PUBLICATIONS),
    ("Journal article machine learning applications healthcare", SectionLabel.PUBLICATIONS),
    ("Conference proceedings neural network optimization", SectionLabel.PUBLICATIONS),
    ("Technical report distributed systems consensus algorithms", SectionLabel.PUBLICATIONS),

    # Languages
    ("Languages English Fluent Hindi Native", SectionLabel.LANGUAGES),
    ("English Native Spanish Fluent French Conversational", SectionLabel.LANGUAGES),
    ("German Proficient Japanese Basic", SectionLabel.LANGUAGES),
    ("Mandarin Native English Fluent", SectionLabel.LANGUAGES),

    # Volunteering
    ("Volunteer experience community service NSS teaching students", SectionLabel.VOLUNTEERING),
    ("Volunteer mentor coding bootcamp 2022 present", SectionLabel.VOLUNTEERING),
    ("Community service food bank weekends", SectionLabel.VOLUNTEERING),
    ("Nonprofit board member education foundation", SectionLabel.VOLUNTEERING),

    # Interests
    ("Interests photography travel reading music", SectionLabel.INTERESTS),
    ("Hobbies hiking cooking gaming chess", SectionLabel.INTERESTS),
    ("Passionate about photography and travel", SectionLabel.INTERESTS),

    # References
    ("References available upon request", SectionLabel.REFERENCES),
    ("Professional references provided on request", SectionLabel.REFERENCES),

    # Other
    ("Additional information miscellaneous professional details", SectionLabel.OTHER),
    ("Security clearance level TS/SCI active", SectionLabel.OTHER),
    ("Willing to relocate internationally", SectionLabel.OTHER),
]


def create_bootstrap_training_dataset() -> TrainingDataset:
    """Create a minimal bootstrap dataset for development/testing.
    
    WARNING: This is NOT real labelled data. It uses synthetic examples
    matching the PoC taxonomy. Replace with real human-annotated data
    for production use.
    """
    # Create dummy SegmentationResult objects for each example
    from resume_parser.conversion import convert_bytes
    from resume_parser.normalization import normalize_document
    from resume_parser.segmentation import segment_document
    
    documents = []
    for i, (text, label) in enumerate(BOOTSTRAP_TRAINING_DATA):
        conv = convert_bytes(text.encode("utf-8"), filename=f"bootstrap_{i}.txt", content_type="text/plain")
        conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
        doc = normalize_document(conv)
        seg = segment_document(doc)
        
        # Use the first block (or all blocks merged for bootstrap)
        for block in seg.blocks:
            documents.append(LabelledDocument(
                document_id=f"bootstrap_{i}",
                filename=f"bootstrap_{i}.txt",
                blocks=[LabelledBlock(
                    document_id=f"bootstrap_{i}",
                    block_id=block.block_id,
                    line_ids=block.line_ids,
                    start_line_index=block.start_line_index,
                    end_line_index=block.end_line_index,
                    text=block.text,
                    gold_section=label,
                )],
            ))
    
    return TrainingDataset(documents)


__all__ = [
    "SectionClassifier",
    "TrainingDataset",
    "LabelledDocument",
    "LabelledBlock",
    "classify_blocks",
    "train_section_classifier",
    "extract_semantic_features",
    "extract_structural_features",
    "build_feature_text",
    "label_values",
    "create_labelled_dataset_from_csv",
    "create_bootstrap_training_dataset",
    "ROLE_WORDS",
    "COMPANY_WORDS",
    "EDUCATION_WORDS",
    "SKILL_WORDS",
    "CERT_WORDS",
    "AWARD_WORDS",
    "PUBLICATION_WORDS",
    "LANGUAGE_WORDS",
    "VOLUNTEER_WORDS",
    "PROJECT_WORDS",
    "INTEREST_WORDS",
    "REFERENCE_WORDS",
    "CONTACT_WORDS",
    "SUMMARY_WORDS",
    "BOOTSTRAP_TRAINING_DATA",
]