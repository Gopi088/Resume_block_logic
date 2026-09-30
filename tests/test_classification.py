"""B3 ML semantic classification tests.

Covers the required areas: dataset loading, label validation, resume-level
splits, leakage, training, persistence, loading, prediction, confidence,
top-k alternatives, provenance, UNKNOWN/OTHER, conventional + no-heading +
mixed + boilerplate + empty/short + rare-vocab cases, reproducibility, and
B0/B1/B2 compatibility.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from resume_parser.classification import (
    BOOTSTRAP_TRAINING_DATA,
    LabelledBlock,
    LabelledDocument,
    SectionClassifier,
    TrainingDataset,
    build_feature_text,
    classify_blocks,
    create_bootstrap_training_dataset,
    create_labelled_dataset_from_csv,
    extract_semantic_features,
    extract_structural_features,
    label_values,
    train_section_classifier,
)
from resume_parser.conversion import convert_bytes
from resume_parser.models import SECTION_LABELS, SectionLabel
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import segment_document
from train_section_classifier import main as train_cli


# ---------------------------------------------------------------- helpers

def _chain(text: str):
    conv = convert_bytes(text.encode("utf-8"), filename="r.txt", content_type="text/plain")
    conv = conv.model_copy(update={"extracted_markdown": text, "extracted_char_count": len(text)})
    doc = normalize_document(conv)
    return doc, segment_document(doc)


def _tiny_docs(n_OPTS=2):
    """Handcrafted labelled docs (no pipeline cost). 2 classes x 3 docs."""
    docs = []
    for i in range(3):
        docs.append(LabelledDocument(
            document_id=f"tiny_exp_{i}", filename=f"e{i}.txt",
            blocks=[LabelledBlock(document_id=f"tiny_exp_{i}", block_id="B000000",
                                  line_ids=["L000000"], start_line_index=0, end_line_index=0,
                                  text="Senior Software Engineer at Infosys 2020 Present built systems",
                                  gold_section=SectionLabel.EXPERIENCE)]))
    for i in range(3):
        docs.append(LabelledDocument(
            document_id=f"tiny_ski_{i}", filename=f"s{i}.txt",
            blocks=[LabelledBlock(document_id=f"tiny_ski_{i}", block_id="B000000",
                                  line_ids=["L000000"], start_line_index=0, end_line_index=0,
                                  text="Python SQL Docker AWS Git Linux",
                                  gold_section=SectionLabel.SKILLS)]))
    return docs


@pytest.fixture(scope="module")
def bootstrap_ds():
    return create_bootstrap_training_dataset()


@pytest.fixture(scope="module")
def trained():
    ds = create_bootstrap_training_dataset()
    clf, meta = train_section_classifier(ds)
    return clf, meta, ds


# ---------------------------------------------------------------- taxonomy

def test_taxonomy_uses_repo_section_labels():
    assert SectionLabel.CONTACT.value == "contact"
    assert SectionLabel.EXPERIENCE.value == "experience"
    assert SectionLabel.OTHER.value == "other"
    assert SectionLabel.UNKNOWN.value == "unknown"
    assert SectionLabel.BOILERPLATE.value == "boilerplate"
    assert set(SECTION_LABELS) == set(SectionLabel)
    assert len(BOOTSTRAP_TRAINING_DATA) > 0


# ---------------------------------------------------------------- dataset loading + validation

def test_dataset_loading_from_csv(tmp_path):
    _, seg = _chain("Jane Doe\njane@mail.com\n\nSKILLS\nPython SQL\n")
    csv_path = tmp_path / "ann.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["document_id", "block_id", "start_line_id", "end_line_id", "gold_section"])
        golds = {seg.blocks[0].block_id: "contact", seg.blocks[-1].block_id: "skills"}
        for b in seg.blocks:
            g = golds.get(b.block_id, "other")
            w.writerow([seg.document_id, b.block_id, b.line_ids[0], b.line_ids[-1], g])
    ds = create_labelled_dataset_from_csv(csv_path, {seg.document_id: seg})
    assert len(ds.documents) == 1
    by_id = {b.block_id: b.gold_section for b in ds.get_all_blocks()}
    assert by_id[seg.blocks[0].block_id] == SectionLabel.CONTACT
    assert by_id[seg.blocks[-1].block_id] == SectionLabel.SKILLS


def test_label_validation_rejects_bad_section(tmp_path):
    _, seg = _chain("Jane Doe\n")
    b = seg.blocks[0]
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["document_id", "block_id", "start_line_id", "end_line_id", "gold_section"])
        w.writerow([seg.document_id, b.block_id, b.line_ids[0], b.line_ids[-1], "chief_executive"])
    with pytest.raises(ValueError):
        create_labelled_dataset_from_csv(csv_path, {seg.document_id: seg})


def test_label_validation_rejects_duplicate_docs():
    docs = _tiny_docs()
    docs.append(docs[0])
    with pytest.raises(ValueError, match="Duplicate document_id"):
        TrainingDataset(docs)


def test_csv_mixed_spans_majority_and_spans(tmp_path):
    _, seg = _chain("John Doe\njohn@x.com\nSoftware Engineer\nABC Ltd\n")
    b = seg.blocks[0]
    csv_path = tmp_path / "mix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["document_id", "block_id", "start_line_id", "end_line_id", "gold_section"])
        w.writerow([seg.document_id, b.block_id, b.line_ids[0], b.line_ids[1], "contact"])
        w.writerow([seg.document_id, b.block_id, b.line_ids[2], b.line_ids[-1], "experience"])
        w.writerow([seg.document_id, b.block_id, b.line_ids[2], b.line_ids[-1], "experience"])
    ds = create_labelled_dataset_from_csv(csv_path, {seg.document_id: seg})
    lb = ds.get_all_blocks()[0]
    assert lb.gold_section == SectionLabel.EXPERIENCE  # majority of span rows
    assert lb.semantic_spans is not None and len(lb.semantic_spans) == 3


# ---------------------------------------------------------------- splits + leakage

def test_resume_level_split_disjoint_and_complete(bootstrap_ds):
    tr, va, te = bootstrap_ds.split_by_resume(seed=42)
    ids = lambda d: {x.document_id for x in d.documents}
    assert ids(tr) | ids(va) | ids(te) == ids(bootstrap_ds)
    assert ids(tr) & ids(va) == set() and ids(tr) & ids(te) == set() and ids(va) & ids(te) == set()
    assert len(tr.documents) > len(va.documents) and len(te.documents) > 0


def test_no_train_test_resume_leakage_explicit(bootstrap_ds):
    tr, va, te = bootstrap_ds.split_by_resume(seed=7)
    _, _, tr_groups = tr.get_texts_and_labels()
    _, _, va_groups = va.get_texts_and_labels()
    _, _, te_groups = te.get_texts_and_labels()
    assert set(tr_groups) & set(va_groups) == set()
    assert set(tr_groups) & set(te_groups) == set()
    # groups align 1:1 with blocks
    assert len(tr_groups) == len(tr)


def test_split_reproducible_same_seed(bootstrap_ds):
    a = bootstrap_ds.split_by_resume(seed=123)
    b = bootstrap_ds.split_by_resume(seed=123)
    for x, y in zip(a, b):
        assert [d.document_id for d in x.documents] == [d.document_id for d in y.documents]


# ---------------------------------------------------------------- training + persistence + loading

def test_model_training_produces_metadata(trained):
    clf, meta, ds = trained
    assert clf.pipeline is not None
    assert meta["n_training_samples"] > 0 and meta["n_classes"] >= 10
    assert meta["train_docs"] + meta["val_docs"] + meta["test_docs"] == len(ds.documents)
    assert "random_state" in meta["lr_params"]  # reproducibility pinned


def test_model_learns_separable_classes():
    ds = TrainingDataset(_tiny_docs())
    clf = SectionClassifier()
    clf.train(*ds.get_texts_and_labels()[:2])
    s, c, _ = clf.predict_single("Python SQL Docker Kubernetes")
    assert s == SectionLabel.SKILLS and c > 0.5
    s2, _, _ = clf.predict_single("Senior Engineer at Infosys 2021 Present")
    assert s2 == SectionLabel.EXPERIENCE


def test_model_persistence_roundtrip(tmp_path, trained):
    clf, meta, _ = trained
    p = tmp_path / "model.pkl"
    clf.save(p)
    assert p.exists()
    loaded = SectionClassifier.load(p)
    assert [c.value for c in loaded.classes_] == [c.value for c in clf.classes_]
    assert loaded.metadata["lr_params"] == meta["lr_params"]
    texts = ["Python SQL Docker", "B.Tech University 2018"]
    assert [x[0] for x in loaded.predict(texts)] == [x[0] for x in clf.predict(texts)]
    assert [x[1] for x in loaded.predict(texts)] == [x[1] for x in clf.predict(texts)]


def test_unfitted_model_raises():
    clf = SectionClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(["text"])
    with pytest.raises(RuntimeError):
        clf.save("/tmp/never.pkl")
    with pytest.raises(RuntimeError):
        clf.evaluate(["text"], [SectionLabel.OTHER])
    with pytest.raises(FileNotFoundError):
        SectionClassifier.load("/nonexistent/model.pkl")


# ---------------------------------------------------------------- prediction / confidence / top-k

def test_prediction_structure_and_ranges(trained):
    clf, _, _ = trained
    out = clf.predict(["Senior Engineer built systems", "Python SQL"])
    assert len(out) == 2
    for section, conf, alts in out:
        assert isinstance(section, SectionLabel)
        assert 0.0 <= conf <= 1.0
        assert isinstance(alts, list) and len(alts) == 3


def test_confidence_matches_model_probabilities(trained):
    clf, _, _ = trained
    texts = ["Senior Engineer built systems"]
    proba = clf.pipeline.predict_proba(texts)[0]
    section, conf, _ = clf.predict_single(texts[0])
    assert conf == pytest.approx(float(proba.max()))
    assert section == SectionLabel(clf.pipeline.classes_[int(np.argmax(proba))])


def test_topk_alternatives_sorted_exclude_best(trained):
    clf, _, _ = trained
    section, conf, alts = clf.predict_single("B.Tech Computer Science University", return_alternatives=5)
    assert all(a.section != section for a in alts)
    confs = [a.confidence for a in alts]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert conf + sum(confs) <= 1.0 + 1e-9


def test_sklearn_boundary_keeps_section_labels():
    """Regression: sklearn mangled str-Enum members (metrics went 0.0, predict
    raised). Labels must cross the sklearn boundary as plain value strings."""
    assert label_values([SectionLabel.CONTACT, "skills"]) == ["contact", "skills"]
    clf = SectionClassifier()
    ds = TrainingDataset(_tiny_docs())
    texts, labels, _ = ds.get_texts_and_labels()
    clf.train(texts, labels)
    assert all(isinstance(c, str) and not isinstance(c, SectionLabel)
               for c in clf.pipeline.classes_)
    s, _, _ = clf.predict_single(texts[0])
    assert isinstance(s, SectionLabel)


# ---------------------------------------------------------------- evaluation metrics

def test_evaluation_metrics_complete(trained):
    _, meta, _ = trained
    tm = meta["test_metrics"]
    for key in ("eval_accuracy", "eval_macro_f1", "eval_macro_precision",
                "eval_macro_recall", "eval_per_class", "eval_confusion_matrix"):
        assert key in tm, key
    assert 0.0 <= tm["eval_accuracy"] <= 1.0
    cm = tm["eval_confusion_matrix"]
    assert len(cm["labels"]) > 0  # classes observed in the test split
    assert set(cm["labels"]) <= {c.value for c in SECTION_LABELS}
    n = len(cm["labels"])
    assert len(cm["matrix"]) == n and all(len(r) == n for r in cm["matrix"])
    assert sum(sum(r) for r in cm["matrix"]) == meta["test_dataset_size"]
    assert set(tm["eval_per_class"]) == set(cm["labels"])
    for cls, m in tm["eval_per_class"].items():
        assert set(m) == {"precision", "recall", "f1", "support"}
        assert 0.0 <= m["f1"] <= 1.0


def test_unseen_label_in_eval_does_not_crash():
    docs = [d for d in _tiny_docs()
            if d.blocks[0].gold_section == SectionLabel.EXPERIENCE][:2]
    docs.append(LabelledDocument(
        document_id="tiny_edu_0", filename="ed0.txt",
        blocks=[LabelledBlock(document_id="tiny_edu_0", block_id="B000000",
                              line_ids=["L000000"], start_line_index=0, end_line_index=0,
                              text="B.Tech Computer Science University 2016",
                              gold_section=SectionLabel.EDUCATION)]))
    clf = SectionClassifier()
    clf.train(*TrainingDataset(docs).get_texts_and_labels()[:2])
    m = clf.evaluate(["Python SQL Docker"], [SectionLabel.SKILLS])  # unseen in train
    assert 0.0 <= m["eval_accuracy"] <= 1.0


# ---------------------------------------------------------------- provenance

def test_provenance_preservation(trained):
    clf, _, _ = trained
    doc, seg = _chain("John Doe\njohn@x.com\n\nWORK EXPERIENCE\nSenior Engineer\n2022 - 2024\n")
    res = classify_blocks(seg, clf)
    assert res.document_id == seg.document_id == doc.document_id
    assert [c.block_id for c in res.classifications] == [b.block_id for b in seg.blocks]
    for c, b in zip(res.classifications, seg.blocks):
        assert c.source_line_ids == b.line_ids
        assert (c.start_line_index, c.end_line_index) == (b.start_line_index, b.end_line_index)
    assert res.integrity.passed, res.integrity.violations


# ---------------------------------------------------------------- unknown/other + statuses

def test_unknown_other_taxonomy_and_status_invariant(trained):
    clf, _, _ = trained
    texts = ["Xylophone quantum zebrafish", "Willing to relocate internationally",
             "Python Java SQL React Docker AWS"]
    for text in texts:
        _, seg = _chain(text + "\n")
        res = classify_blocks(seg, clf)
        for c in res.classifications:
            assert c.classification_status in ("classified", "low_confidence")
            assert (c.classification_status == "low_confidence") == (c.confidence < 0.5)
            assert c.predicted_section in SectionLabel


def test_rare_unseen_vocabulary_no_crash(trained):
    clf, _, _ = trained
    s, c, alts = clf.predict_single("Xylophone quantum zebrafish antidisestablishmentarianism floccinaucinihilipilification")
    assert isinstance(s, SectionLabel) and 0.0 <= c <= 1.0 and len(alts) == 3


# ---------------------------------------------------------------- resume scenarios

def test_conventional_resume_end_to_end(trained):
    clf, _, _ = trained
    text = ("John Doe\njohn@x.com | +91 99999\n\nWORK EXPERIENCE\nSenior Engineer | ABC Ltd\n"
            "2022 - 2024\n- Built APIs\n\nEDUCATION\nB.Tech 2018\n\nSKILLS\nPython SQL\n")
    doc, seg = _chain(text)
    assert doc.integrity.passed and seg.integrity.passed
    res = classify_blocks(seg, clf)
    assert res.integrity.passed
    assert len(res.classifications) == len(seg.blocks) >= 4
    assert {c.predicted_section for c in res.classifications} <= set(SectionLabel)


def test_no_heading_resume():
    """Spec example: contact + experience in one structural block is fine for B2;
    B3 must classify the block as a whole and keep full provenance."""
    clf = SectionClassifier()
    clf.train(*TrainingDataset(_tiny_docs()).get_texts_and_labels()[:2])
    text = ("John Doe\nemail@example.com\nBangalore\nSoftware Engineer\nABC Technologies\n"
            "2022 - 2024\nDeveloped backend APIs\nWorked with Python\n")
    _, seg = _chain(text)
    assert len(seg.blocks) == 1  # B2: one structural block, no semantic split
    res = classify_blocks(seg, clf)
    assert len(res.classifications) == 1
    assert res.classifications[0].source_line_ids == seg.blocks[0].line_ids
    assert len(res.classifications[0].source_line_ids) == 8
    assert res.integrity.passed


def test_mixed_candidate_block_baseline_documented():
    """v1 assigns ONE block-level label (documented baseline). The model must
    support spans (semantic_spans field) for future span detection."""
    clf = SectionClassifier()
    clf.train(*TrainingDataset(_tiny_docs()).get_texts_and_labels()[:2])
    _, seg = _chain("John Doe\njohn@x.com\nSoftware Engineer\nABC Ltd\n")
    assert len(seg.blocks) == 1
    res = classify_blocks(seg, clf)
    c = res.classifications[0]
    assert c.semantic_spans == []  # span detection: future work, not silent majority
    assert c.source_line_ids == seg.blocks[0].line_ids  # nothing dropped


def test_repeated_boilerplate_excluded_but_kept(trained):
    clf, _, _ = trained
    text = "Softwa\nSoftwa\nWORK EXPERIENCE\nAcme\n\fSoftwa\nSoftwa\nSKILLS\nPython\n"
    _, seg = _chain(text)
    assert seg.boilerplate, "expected boilerplate flagged"
    flagged = {lid for b in seg.boilerplate for lid in b.line_ids}
    res = classify_blocks(seg, clf)
    assert res.integrity.passed
    for c in res.classifications:
        assert not (set(c.source_line_ids) & flagged), c.block_id


def test_empty_and_very_short_blocks(trained):
    clf, _, _ = trained
    # Whitespace-only resume: B1 lines exist but no content -> zero blocks.
    _, seg_empty = _chain("   \n  \n")
    assert len(seg_empty.blocks) == 0
    res = classify_blocks(seg_empty, clf)
    assert res.classifications == [] and res.integrity.passed
    _, seg_hi = _chain("Hi\n")
    res_hi = classify_blocks(seg_hi, clf)
    assert len(res_hi.classifications) == 1 and res_hi.integrity.passed


# ---------------------------------------------------------------- features + reproducibility + compat

def test_feature_helpers():
    feats = extract_semantic_features("Senior Engineer at Infosys Jan 2020 - Present\n- Built systems")
    assert feats["has_date_pattern"] == 1.0
    assert feats["bullet_ratio"] == 0.5
    assert feats["role_word_ratio"] > 0 and feats["company_word_ratio"] > 0
    assert feats["education_word_ratio"] == 0.0
    _, seg = _chain("SKILLS\n- Python\n- SQL\n")
    sfeats = extract_structural_features(seg.blocks[-1])
    assert sfeats["block_line_count"] == float(len(seg.blocks[-1].line_ids))
    assert sfeats["has_header"] in (0.0, 1.0)
    assert build_feature_text(seg.blocks[-1]) == seg.blocks[-1].text


def test_reproducibility(trained):
    clf, _, _ = trained
    texts = ["Python SQL Docker", "Senior Engineer built systems"]
    assert [x[1] for x in clf.predict(texts)] == [x[1] for x in clf.predict(texts)]
    clf2 = SectionClassifier()
    clf2.train(*TrainingDataset(_tiny_docs()).get_texts_and_labels()[:2])
    c1 = SectionClassifier()
    c1.train(*TrainingDataset(_tiny_docs()).get_texts_and_labels()[:2])
    assert [x[1] for x in c1.predict(texts)] == [x[1] for x in clf2.predict(texts)]


def test_b0_b1_b2_compatibility(trained):
    """B3 consumes the real chain; B0/B1/B2 contracts hold end to end."""
    clf, _, _ = trained
    text = "Jane Doe\njane@mail.com\n\nWORK EXPERIENCE\nAcme Corp\n\nEDUCATION\nB.Tech\n"
    doc, seg = _chain(text)
    assert doc.extracted_markdown == text  # B0 immutable through the chain
    assert doc.integrity.passed and seg.integrity.passed
    assert all(b.assignment_status.value == "unassigned_pending_ml_classification" for b in seg.blocks)
    res = classify_blocks(seg, clf)
    assert res.integrity.passed and res.integrity.checks["all_blocks_classified"]
    covered = [lid for c in res.classifications for lid in c.source_line_ids]
    meaningful = [ln.line_id for ln in doc.lines
                  if ln.normalized_text.strip() and ln.line_id
                  not in {b for bp in seg.boilerplate for b in bp.line_ids}]
    assert sorted(covered) == sorted(meaningful)


def test_train_cli_bootstrap(tmp_path):
    out = tmp_path / "model.pkl"
    assert train_cli(["--bootstrap", "--output", str(out)]) == 0
    assert out.exists()
    loaded = SectionClassifier.load(str(out))
    s, c, _ = loaded.predict_single("Python SQL")
    assert isinstance(s, SectionLabel)


def test_error_types_survive_exception_protocol():
    """Regression: frozen-dataclass errors broke traceback chaining and died
    inside generator-based context managers (FrozenInstanceError)."""
    import contextlib
    import dataclasses
    import pickle
    from resume_parser.errors import IntegrityError, UnsupportedFormatError

    e = UnsupportedFormatError(detail="empty")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.code = "X"

    @contextlib.contextmanager
    def cm():
        yield

    with cm():
        with pytest.raises(UnsupportedFormatError):
            raise UnsupportedFormatError(detail="empty")

    Farrell = IntegrityError(detail="d", violations=("v1",))
    rt = pickle.loads(pickle.dumps(Farrell))
    assert (rt.code, rt.detail, rt.violations) == (
        "INTEGRITY_VIOLATION", "d", ("v1",))
