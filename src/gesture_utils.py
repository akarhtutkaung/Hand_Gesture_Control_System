import numpy as np
import cv2


def calc_finger_spread(landmarks: np.ndarray) -> float:
    """
    Calculate the average pairwise Euclidean distance between the 5 fingertips.
    Fingertip landmark indices: [4, 8, 12, 16, 20]
    landmarks: flat array of shape (63,) — 21 points × (x, y, z)
    Returns a float representing how spread the fingers are.
    """
    # TODO: extract the 5 fingertip (x, y) coords from the landmarks array
    # TODO: compute all pairwise Euclidean distances between those 5 points
    # TODO: return the average of those distances
    pass


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate landmarks so the wrist (index 0) is at the origin,
    then scale so values fit in a consistent range.
    landmarks: flat array of shape (63,)
    Returns a normalized flat array of shape (63,).
    """
    # TODO: reshape landmarks to (21, 3)
    # TODO: subtract the wrist point (index 0) from all points
    # TODO: divide by the max absolute value to scale to [-1, 1]
    # TODO: return the result flattened back to (63,)
    pass


def draw_overlay(frame: np.ndarray, gesture: str) -> np.ndarray:
    """
    Draw the detected gesture label on the video frame.
    Returns the annotated frame.
    """
    # TODO: choose a color based on the gesture string
    # TODO: use cv2.putText to draw the gesture label on the frame
    # TODO: return the frame
    pass
