"""Train the character n-gram classifier and meta-classifier."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED


def train_ngram_classifier():
    """Train the character n-gram TF-IDF + LogisticRegression pipeline."""
    print("Loading training data...")
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train, y_train = train["name"].values, train["label"].values
    X_val, y_val = val["name"].values, val["label"].values
    X_test, y_test = test["name"].values, test["label"].values

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Build pipeline
    print("\nTraining n-gram classifier...")
    print("  Config: char_wb, ngram_range=(2,4), max_features=15000, L1 LogReg")
    t0 = time.time()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            max_features=15000,
            min_df=2,
            lowercase=True,
            norm="l2",
            use_idf=True,
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l1",
            alpha=1e-4,
            max_iter=100,
            tol=1e-3,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate on validation set
    print("\n=== Validation Set Results ===")
    y_val_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=4))

    # Evaluate on test set
    print("=== Test Set Results ===")
    y_test_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))

    print("Confusion Matrix (test):")
    cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
    print(f"  Labels: {list(pipeline.classes_)}")
    for row in cm:
        print(f"  {row}")

    # Inspect top features per class
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()

    print("\n=== Top 15 Most Predictive N-grams Per Class ===")
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_indices = np.argsort(coefs)[-15:][::-1]
        top_features = [(feature_names[j], round(coefs[j], 3)) for j in top_indices]
        print(f"\n  {cls}:")
        for feat, weight in top_features:
            print(f"    '{feat}': {weight}")

    # Sparsity report
    total_params = clf.coef_.size
    nonzero = np.count_nonzero(clf.coef_)
    sparsity = 1 - (nonzero / total_params)
    print(f"\n=== Model Sparsity ===")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero: {nonzero:,}")
    print(f"  Sparsity: {sparsity:.1%}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ngram_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved model to {model_path}")

    # Test on specific names
    print("\n=== Spot Check: Known Names ===")
    test_names = [
        "FERREIRA GUSTAVO DA SILVA",
        "GENIVALDO DE SOUZA",
        "DASILVA THIAGO",
        "HERNANDEZ JUAN CARLOS",
        "GONZALEZ MARIA DE LOS ANGELES",
        "SMITH JOHN",
        "JOHNSON MICHAEL",
        "CLEUDIMAR SANTOS",
        "EDILEUSA OLIVEIRA",
        "GARCIA JOSE",
    ]
    probs = pipeline.predict_proba(test_names)
    for name, prob in zip(test_names, probs):
        prob_dict = {cls: round(p, 3) for cls, p in zip(pipeline.classes_, prob)}
        pred = max(prob_dict, key=prob_dict.get)
        print(f"  {name:40s} → {pred:10s} {prob_dict}")

    return pipeline


if __name__ == "__main__":
    train_ngram_classifier()
