"""
Offline training script for the Bike Sharing demand classifier.

Usage
-----
    python train_model.py

What it does
------------
1. Loads (or generates) the Bike Sharing CSV from data/bike_sharing.csv.
2. Engineers a three-class demand label from the raw count column:
      Low    →  cnt ≤ 33rd percentile
      Medium →  33rd < cnt ≤ 66th percentile
      High   →  cnt >  66th percentile
3. Trains a Random Forest Classifier.
4. Evaluates on a held-out test split and prints accuracy + F1.
5. Persists the fitted model and label thresholds to model/bike_model.joblib.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "bike_sharing.csv")
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "bike_model.joblib")

# ── feature columns used for inference ──────────────────────────────────────
FEATURE_COLS = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
]


def load_or_generate_data() -> pd.DataFrame:
    """Return the dataset, generating synthetic data if the CSV is missing."""
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from {DATA_PATH} ...")
        return pd.read_csv(DATA_PATH)

    print(f"{DATA_PATH} not found — generating synthetic data ...")
    # Ensure data/ directory exists
    os.makedirs("data", exist_ok=True)
    # Import the generator from the same package
    sys.path.insert(0, os.path.dirname(__file__))
    from data.generate_data import generate_bike_sharing
    df = generate_bike_sharing()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {DATA_PATH}")
    return df


def make_demand_label(cnt: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Convert raw rental counts into a three-class demand label.

    Returns
    -------
    labels      : pd.Series of strings  ("Low" | "Medium" | "High")
    low_thresh  : upper bound for Low class
    high_thresh : lower bound for High class
    """
    p33 = float(cnt.quantile(0.33))
    p66 = float(cnt.quantile(0.66))

    labels = pd.cut(
        cnt,
        bins=[-1, p33, p66, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    return labels.astype(str), p33, p66


def train(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest Classifier and return a bundle with model artefacts.

    Returns a dict that can be saved with joblib containing:
      - model        : fitted RandomForestClassifier
      - feature_cols : list of feature names (same order as training)
      - low_thresh   : count threshold separating Low from Medium
      - high_thresh  : count threshold separating Medium from High
      - label_encoder: fitted LabelEncoder
    """
    y_str, low_thresh, high_thresh = make_demand_label(df["cnt"])

    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    X  = df[FEATURE_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")

    print("\n── Model Performance ──────────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  F1 (macro): {f1:.4f}")
    print("\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"\n  Class thresholds  →  Low ≤ {low_thresh:.0f}  |"
          f"  Medium ≤ {high_thresh:.0f}  |  High > {high_thresh:.0f}")

    return {
        "model":         clf,
        "feature_cols":  FEATURE_COLS,
        "low_thresh":    low_thresh,
        "high_thresh":   high_thresh,
        "label_encoder": le,
        "accuracy":      round(acc, 4),
        "f1_macro":      round(f1, 4),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df     = load_or_generate_data()
    bundle = train(df)

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"  (accuracy={bundle['accuracy']},  f1_macro={bundle['f1_macro']})")


if __name__ == "__main__":
    main()
