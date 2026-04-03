"""
train_model.py — load landmark CSVs, train an SVM classifier, save the model.

Run:  python src/train_model.py  (from the project root)
      Requires data/fist.csv, data/palm.csv, and data/peace.csv to exist.
      Outputs model/gesture_classifier.pkl
"""

import os

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from config import CSV_PATHS, CLASSIFIER_PATH as MODEL_PATH, MODEL_DIR, CV_FOLDS, SVM_KERNEL


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Read data/fist.csv, data/palm.csv, and data/peace.csv.
    Each row format: label, x0, y0, z0, ..., x20, y20, z20
    Returns X of shape (n_samples, 63) and y of shape (n_samples,).
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
    Passthrough — normalization is already applied by normalize_landmarks()
    in gesture_utils.py at collection time.
    """
    return X


def train_classifier(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train an SVM classifier with StandardScaler preprocessing.
    Prints cross-validation accuracy.
    Returns the fitted model (sklearn Pipeline).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=SVM_KERNEL)),
    ])
    scores = cross_val_score(pipeline, X, y, cv=CV_FOLDS)
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    pipeline.fit(X, y)
    return pipeline


def save_model(model: Pipeline) -> None:
    """
    Save the trained model to model/gesture_classifier.pkl.
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
