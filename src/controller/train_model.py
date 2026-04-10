"""
train_model.py — load gesture CSVs, train an SVM classifier, save the model.

Run from the project root (after recording data with collect_data.py):
    python src/controller/train_model.py

Output
------
model/gesture_classifier.pkl   Fitted sklearn Pipeline (StandardScaler → SVC).

The pipeline uses probability=True on the SVC so that classify_gesture() can
return a confidence score and gate predictions with MIN_CONFIDENCE at runtime.
Retrain whenever you change MIN_CONFIDENCE or add new gesture classes.

CSV format (one row per sample):
    <label>,<x0>,<y0>,<z0>,...,<x20>,<y20>,<z20>
    64 columns total: 1 label + 63 normalised landmark values.
"""

import os
import sys

# Allow running directly: python src/controller/train_model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from config import CSV_PATHS, CLASSIFIER_PATH as MODEL_PATH, MODEL_DIR, CV_FOLDS, SVM_KERNEL


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Read all gesture CSVs defined in config.CSV_PATHS.

    Skips missing files with a warning so a partial dataset still trains.
    Rows with the wrong column count are silently dropped.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 63)
        Normalised landmark features (already processed by normalize_landmarks
        at collection time — no further normalisation needed here).
    y : np.ndarray, shape (n_samples,)
        Gesture label strings (e.g. "fist", "palm", "peace").

    Raises
    ------
    ValueError
        If no rows were loaded across all CSV files.
    """
    X_rows, y_rows = [], []
    for path in CSV_PATHS.values():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 64:
                    continue
                y_rows.append(parts[0])
                X_rows.append([float(v) for v in parts[1:]])
    if not X_rows:
        raise ValueError("No data loaded. Run collect_data.py first.")
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=str)


def preprocess(X: np.ndarray) -> np.ndarray:
    """
    Passthrough — features are already normalised by normalize_landmarks() at
    collection time, so no additional preprocessing is needed before training.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 63)

    Returns
    -------
    np.ndarray
        X unchanged.
    """
    return X


def train_classifier(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train a StandardScaler → SVC pipeline and report cross-validation accuracy.

    The SVC is configured with probability=True so that predict_proba() is
    available at inference time for confidence gating (MIN_CONFIDENCE in config).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 63)
        Feature matrix from load_dataset().
    y : np.ndarray, shape (n_samples,)
        Label array from load_dataset().

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully fitted pipeline ready for joblib serialisation.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=SVM_KERNEL, probability=True)),
    ])
    scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS)
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    pipeline.fit(X, y)
    return pipeline


def save_model(model: Pipeline) -> None:
    """
    Serialise the fitted pipeline to model/gesture_classifier.pkl.

    Creates the model/ directory if it does not exist.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted pipeline from train_classifier().
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    X, y = load_dataset()
    print(f"Loaded {len(y)} samples — classes: {sorted(set(y))}")
    X = preprocess(X)
    model = train_classifier(X, y)
    save_model(model)


if __name__ == "__main__":
    main()
