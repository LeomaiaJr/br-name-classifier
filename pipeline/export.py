"""Export trained models to JSON for Python and TypeScript consumption."""

import json
import pickle

import numpy as np

from config import MODELS_DIR, OUTPUT_DIR, MODEL_VERSION
from constants import FEATURE_NAMES


def export_ngram_model():
    """Export the n-gram TF-IDF + SGDClassifier to JSON."""
    print("Exporting n-gram model...")

    with open(MODELS_DIR / "ngram_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Vocabulary: ngram string → index
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}

    # IDF weights
    idf = [round(float(x), 4) for x in vectorizer.idf_]

    # Classes
    classes = [str(c) for c in clf.classes_]

    # Sparse coefficients (only non-zero)
    feature_names = vectorizer.get_feature_names_out()
    coef_sparse = {}
    for i, cls in enumerate(classes):
        coefs = clf.coef_[i]
        sparse = {}
        for j in range(len(coefs)):
            if abs(coefs[j]) > 1e-6:
                sparse[feature_names[j]] = round(float(coefs[j]), 4)
        coef_sparse[cls] = sparse

    # Intercepts
    intercepts = [round(float(x), 4) for x in clf.intercept_]

    model = {
        "version": MODEL_VERSION,
        "vocabulary": vocabulary,
        "idf": idf,
        "classes": classes,
        "coef_sparse": coef_sparse,
        "intercept": intercepts,
        # TF-IDF config needed for TypeScript reimplementation
        "config": {
            "analyzer": "char_wb",
            "ngram_range": [2, 4],
            "sublinear_tf": True,
            "norm": "l2",
        }
    }

    path = OUTPUT_DIR / "ngram_model.json"
    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"))

    size_kb = path.stat().st_size / 1024
    total_nonzero = sum(len(v) for v in coef_sparse.values())
    print(f"  Vocabulary size: {len(vocabulary)}")
    print(f"  Non-zero coefficients: {total_nonzero}")
    print(f"  File size: {size_kb:.0f} KB")
    return model


def export_meta_model():
    """Export the meta-classifier weights to JSON."""
    print("\nExporting meta-classifier...")

    with open(MODELS_DIR / "meta_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    model = {
        "version": MODEL_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "coef": [round(float(x), 6) for x in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
    }

    path = OUTPUT_DIR / "meta_model.json"
    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"), indent=None)

    size_bytes = path.stat().st_size
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  File size: {size_bytes} bytes")
    return model


def export_all():
    """Export all model artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ngram = export_ngram_model()
    meta = export_meta_model()

    # Size summary
    print("\n=== Export Summary ===")
    total = 0
    for f in OUTPUT_DIR.glob("*.json"):
        size = f.stat().st_size
        total += size
        print(f"  {f.name}: {size / 1024:.0f} KB")
    print(f"  TOTAL: {total / (1024 * 1024):.1f} MB")

    if total > 5 * 1024 * 1024:
        print(f"\n  WARNING: Total exceeds 5 MB target ({total / (1024*1024):.1f} MB)")
        print("  Consider increasing FREQUENCY_MIN_OCCURRENCES in config.py")


if __name__ == "__main__":
    export_all()
