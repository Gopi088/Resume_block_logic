#!/usr/bin/env python3
"""Training CLI for B3 Section Classifier.

Usage:
    python train_section_classifier.py --bootstrap --output model.pkl
    python train_section_classifier.py --data labelled_data.csv --seg-dir segmentation_outputs/ --output model.pkl

This script trains the TF-IDF + Logistic Regression classifier for resume section
classification using resume-level train/val/test splits to prevent data leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from resume_parser.classification import (
    SectionClassifier,
    train_section_classifier,
    create_bootstrap_training_dataset,
    create_labelled_dataset_from_csv,
)
from resume_parser.conversion import convert_bytes
from resume_parser.normalization import normalize_document
from resume_parser.segmentation import segment_document


def load_segmentation_results(seg_dir: Path) -> dict[str, any]:
    """Load SegmentationResult objects from JSON files in a directory."""
    results = {}
    for json_file in seg_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        # Reconstruct SegmentationResult (simplified - in production use proper deserialization)
        # For now, we'll use the bootstrap dataset
        pass
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train B3 Section Classifier")
    ap.add_argument("--bootstrap", action="store_true", 
                    help="Use built-in bootstrap training data (for development only)")
    ap.add_argument("--data", type=str, 
                    help="Path to CSV annotations file")
    ap.add_argument("--seg-dir", type=str, 
                    help="Directory with SegmentationResult JSON files")
    ap.add_argument("--output", type=str, required=True,
                    help="Output model path (.pkl)")
    ap.add_argument("--tfidf-max-features", type=int, default=10000,
                    help="TF-IDF max features")
    ap.add_argument("--tfidf-ngram-max", type=int, default=2,
                    help="TF-IDF max n-gram")
    ap.add_argument("--lr-c", type=float, default=1.0,
                    help="Logistic Regression C parameter")
    ap.add_argument("--lr-max-iter", type=int, default=2000,
                    help="Logistic Regression max iterations")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible splits")
    ap.add_argument("--train-ratio", type=float, default=0.7,
                    help="Train split ratio")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="Validation split ratio")
    ap.add_argument("--test-ratio", type=float, default=0.15,
                    help="Test split ratio")
    
    a = ap.parse_args(argv)
    
    # Validate arguments
    if not a.bootstrap and (not a.data or not a.seg_dir):
        print("Error: Either --bootstrap or both --data and --seg-dir are required", file=sys.stderr)
        return 1
    
    if a.bootstrap:
        print("Creating bootstrap training dataset...")
        warnings.warn(
            "Using bootstrap data - NOT suitable for production! "
            "Replace with real human-annotated data.",
            UserWarning
        )
        dataset = create_bootstrap_training_dataset()
    else:
        print(f"Loading segmentation results from {a.seg_dir}...")
        seg_results = load_segmentation_results(Path(a.seg_dir))
        print(f"Loading annotations from {a.data}...")
        dataset = create_labelled_dataset_from_csv(a.data, seg_results)
    
    print(f"Dataset: {dataset}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Train
    tfidf_params = {
        "lowercase": True,
        "ngram_range": (1, a.tfidf_ngram_max),
        "min_df": 1,
        "sublinear_tf": True,
        "max_features": a.tfidf_max_features,
    }
    lr_params = {
        "max_iter": a.lr_max_iter,
        "class_weight": "balanced",
        "C": a.lr_c,
        "random_state": a.seed,
        "solver": "lbfgs",
    }
    
    print("Training classifier with resume-level splits...")
    classifier, metadata = train_section_classifier(
        dataset,
        tfidf_params=tfidf_params,
        lr_params=lr_params,
    )
    
    # Save model
    classifier.save(a.output)
    print(f"Model saved to {a.output}")
    
    # Print metrics
    print("\n=== TRAINING METADATA ===")
    print(json.dumps(metadata, indent=2, default=str))
    
    if "test_metrics" in metadata:
        print("\n=== TEST SET METRICS ===")
        tm = metadata["test_metrics"]
        print(f"Accuracy: {tm.get('eval_accuracy', 0):.4f}")
        print(f"Macro F1: {tm.get('eval_macro_f1', 0):.4f}")
        print(f"Macro Precision: {tm.get('eval_macro_precision', 0):.4f}")
        print(f"Macro Recall: {tm.get('eval_macro_recall', 0):.4f}")
        print("\nPer-class metrics:")
        for cls, metrics in tm.get("eval_per_class", {}).items():
            if metrics["support"] > 0:
                print(f"  {cls:20s} P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} (n={metrics['support']})")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())