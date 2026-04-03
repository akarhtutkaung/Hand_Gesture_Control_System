import numpy as np
import cv2


def calc_finger_spread(landmarks: np.ndarray) -> float:
    """
    Calculate the average pairwise Euclidean distance between the 5 fingertips.
    Fingertip landmark indices: [4, 8, 12, 16, 20]
    landmarks: flat array of shape (63,) — 21 points × (x, y, z)
    Returns a float representing how spread the fingers are.
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
    Translate landmarks so the wrist (index 0) is at the origin,
    then scale so values fit in a consistent range.
    landmarks: flat array of shape (63,)
    Returns a normalized flat array of shape (63,).
    """
    landmarks = landmarks.reshape(21, 3)
    landmarks = landmarks - landmarks[0]
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()


def draw_overlay(frame: np.ndarray, gesture: str) -> np.ndarray:
    """
    Draw the detected gesture label on the video frame.
    Returns the annotated frame. OpenCV use BGR color format.
    """
    color = (255, 255, 255)  # default to white
    if gesture == "open":
        color = (0, 255, 0)  # green
    elif gesture == "closed":
        color = (255, 0, 0)  # blue
    elif gesture == "stop":
        color = (0, 0, 255)  # red

    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame
