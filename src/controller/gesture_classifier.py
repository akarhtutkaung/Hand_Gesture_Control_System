"""
gesture_classifier.py — model loading, gesture classification, and landmark utilities.

This module is the bridge between raw MediaPipe landmarks and high-level gesture
labels. It is imported by hand_detector.py (for normalize_landmarks) and by
hand_processors.py (for classify_gesture).

Public API
----------
load_model()                  Load the trained SVM pipeline from disk.
classify_gesture(lm, model)   Return (label, confidence) for a landmark array.
normalize_landmarks(lm)       Translate + scale landmarks to a canonical form.
calc_finger_spread(lm)        Mean pairwise fingertip distance (spread metric).
"""

import os
import joblib
import numpy as np

from config import (
    CLASSIFIER_PATH as MODEL_PATH,
)


def load_model():
    """
    Load the trained gesture classifier from model/gesture_classifier.pkl.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline (StandardScaler → SVC with probability=True).

    Raises
    ------
    FileNotFoundError
        If the model file is missing. Run controller/train_model.py first.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run `python src/controller/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def classify_gesture(landmarks: np.ndarray, model) -> tuple[str, float]:
    """
    Run the trained classifier on a normalised landmark array.

    Parameters
    ----------
    landmarks : np.ndarray, shape (63,)
        Normalised flat landmark array produced by normalize_landmarks().
    model : sklearn.pipeline.Pipeline
        Loaded classifier returned by load_model().

    Returns
    -------
    label : str
        Predicted gesture name (e.g. "fist", "palm", "peace").
    confidence : float
        predict_proba score for the winning class, in [0.0, 1.0].
        Compare against MIN_CONFIDENCE in config.py before acting on the result.
    """
    x = landmarks.reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return model.classes_[idx], float(proba[idx])


def calc_finger_spread(landmarks: np.ndarray) -> float:
    """
    Compute the mean pairwise Euclidean distance between the 5 fingertips.

    Fingertip landmark indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky).
    Only the x/y components are used; z (depth) is ignored.

    Parameters
    ----------
    landmarks : np.ndarray, shape (21, 3) or (63,)
        Raw or reshaped landmark array. Must already be in (21, 3) shape when called.

    Returns
    -------
    float
        Average distance across all 10 fingertip pairs. Higher = more open hand.
    """
    fingertip_coords = landmarks[[4, 8, 12, 16, 20], :2]
    distances = []
    for i in range(len(fingertip_coords)):
        for j in range(i + 1, len(fingertip_coords)):
            dist = np.linalg.norm(fingertip_coords[i] - fingertip_coords[j])
            distances.append(dist)
    return np.mean(distances)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate and scale landmarks to a pose-invariant canonical form.

    Steps:
      1. Reshape the flat (63,) array to (21, 3).
      2. Subtract the wrist (landmark 0) so it sits at the origin.
      3. Divide by the max absolute value so all values fall in [-1, 1].

    This normalisation is applied at both collection time (collect_data.py) and
    inference time (hand_detector.py), ensuring training and runtime distributions match.

    Parameters
    ----------
    landmarks : np.ndarray, shape (63,)
        Flat array of 21 landmarks × (x, y, z) in MediaPipe's image-space coordinates.

    Returns
    -------
    np.ndarray, shape (63,)
        Normalised flat array ready for CSV storage or model input.
    """
    landmarks = landmarks.reshape(21, 3)
    landmarks = landmarks - landmarks[0]
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()
