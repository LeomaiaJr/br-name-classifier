"""Train the meta-classifier that combines all features into a final Brazilian probability."""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from constants import FEATURE_NAMES
from pipeline.extract_features import extract_features_batch


def train_meta_classifier():
    """Train binary LogReg on the exported feature vector."""
    print("Loading training data...")
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    # Use a subset for meta-classifier, but always include curated hard cases.
    n_meta = 50000
    if "source" in train.columns:
        hard = train[train["source"].astype(str).str.startswith("hard_")]
        base = train.drop(hard.index)
        random_n = max(0, n_meta - len(hard))
        base_sample = base.sample(n=min(random_n, len(base)), random_state=RANDOM_SEED)
        train_sample = pd.concat([base_sample, hard], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    else:
        train_sample = train.sample(n=min(n_meta, len(train)), random_state=RANDOM_SEED)
    test_sample = test.sample(n=min(10000, len(test)), random_state=RANDOM_SEED)

    print(f"  Meta-train: {len(train_sample):,}, Meta-test: {len(test_sample):,}")

    # Extract features
    print("\nExtracting training features...")
    t0 = time.time()
    X_train_df = extract_features_batch(train_sample["name"].tolist())
    print(f"  Time: {time.time() - t0:.1f}s")

    print("Extracting test features...")
    t0 = time.time()
    X_test_df = extract_features_batch(test_sample["name"].tolist())
    print(f"  Time: {time.time() - t0:.1f}s")

    feature_names = list(FEATURE_NAMES)
    X_train = X_train_df[feature_names].fillna(0).values
    y_train = train_sample["label"].isin({"brazilian", "lusophone"}).astype(int).values

    X_test = X_test_df[feature_names].fillna(0).values
    y_test = test_sample["label"].isin({"brazilian", "lusophone"}).astype(int).values

    # Train binary classifier
    print("\nTraining meta-classifier...")
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n=== Meta-classifier Test Results ===")
    print(classification_report(y_test, y_pred, target_names=["not_brazilian", "brazilian"], digits=4))

    # Feature importance
    print("=== Feature Importance ===")
    coefs = clf.coef_[0]
    for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:35s}: {coef:+.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODELS_DIR / "meta_classifier.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved to {meta_path}")

    # Spot check
    print("\n=== Spot Check: P(Brazilian) ===")
    from pipeline.extract_features import extract_features
    test_names = [
        "FERREIRA GUSTAVO DA SILVA",
        "GENIVALDO DE SOUZA",
        "CLEUDIMAR SANTOS",
        "EDILEUSA OLIVEIRA",
        "DASILVA THIAGO",
        "HERNANDEZ JUAN CARLOS",
        "SMITH JOHN",
        "GARCIA DA SILVA JOSE",
        "JOHNSON MICHAEL",
    ]
    for name in test_names:
        feats = extract_features(name)
        X = np.array([[feats[f] for f in feature_names]])
        prob = clf.predict_proba(X)[0][1]  # P(brazilian)
        score = int(round(prob * 100))
        print(f"  {name:40s} → score={score:3d} P(BR)={prob:.3f}")

    return clf


if __name__ == "__main__":
    train_meta_classifier()
