"""
train_model.py — load landmark CSVs, train an SVM classifier, save the model.

Run:  python train_model.py
      Requires data/squeeze_in.csv and data/squeeze_out.csv to exist.
      Outputs model/gesture_classifier.pkl
"""

import numpy as np


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Read data/squeeze_in.csv and data/squeeze_out.csv.
    Each row format: label, x0, y0, z0, ..., x20, y20, z20
    Returns X of shape (n_samples, 63) and y of shape (n_samples,).
    """
    # TODO: loop over both CSV file paths
    # TODO: skip any file that doesn't exist yet (warn the user)
    # TODO: for each row, split the label (col 0) from the 63 feature values
    # TODO: collect all features into X and all labels into y
    # TODO: raise an error if no data was loaded
    # TODO: return X as a float numpy array and y as a string array
    pass


def preprocess(X: np.ndarray) -> np.ndarray:
    """
    Any additional feature preprocessing before training.
    Normalization is already applied in gesture_utils.normalize_landmarks,
    so this can be a passthrough or add extra steps.
    Returns the processed X array.
    """
    # TODO: apply any extra transformations if needed (e.g. feature selection)
    # TODO: return X (can be a passthrough for now)
    pass


def train_classifier(X: np.ndarray, y: np.ndarray):
    """
    Train an SVM classifier with StandardScaler preprocessing.
    Prints cross-validation accuracy.
    Returns the fitted model (sklearn Pipeline).
    """
    # TODO: create an sklearn Pipeline with StandardScaler + SVC(kernel="rbf")
    # TODO: run cross_val_score with cv=5 and print the mean accuracy
    # TODO: fit the pipeline on the full dataset
    # TODO: return the fitted pipeline
    pass


def save_model(model) -> None:
    """
    Save the trained model to model/gesture_classifier.pkl.
    """
    # TODO: create the model/ directory if it doesn't exist
    # TODO: use joblib.dump to save the model to the .pkl path
    # TODO: print a confirmation message
    pass


def main():
    # TODO: call load_dataset() and print the sample count and class names
    # TODO: call preprocess() on X
    # TODO: call train_classifier() and print the result
    # TODO: call save_model() to persist the model
    pass


if __name__ == "__main__":
    main()
