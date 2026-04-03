"""
collect_data.py — record labeled hand-landmark samples to CSV.

Controls:
  i  →  save current frame as "squeeze_in"
  o  →  save current frame as "squeeze_out"
  q  →  quit

Run:  python collect_data.py
      Aim for ~100 samples of each gesture.
"""

import cv2
import numpy as np
from gesture_utils import normalize_landmarks, draw_overlay


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """
    Open the webcam at the given device index.
    Returns a cv2.VideoCapture object.
    """
    # TODO: create a cv2.VideoCapture with the given index
    # TODO: raise an error if the camera fails to open
    # TODO: create a named window so it appears on macOS
    # TODO: return the capture object
    pass


def detect_landmarks(frame: np.ndarray, hands) -> np.ndarray | None:
    """
    Run MediaPipe Hands on a single BGR frame.
    Draws the hand skeleton on the frame in place.
    Returns a normalized flat (63,) landmark array, or None if no hand found.
    """
    # TODO: convert frame from BGR to RGB
    # TODO: run hands.process() on the RGB frame
    # TODO: return None if no hand is detected
    # TODO: extract the first detected hand's 21 landmarks as (x, y, z) coords
    # TODO: draw the landmarks on the frame using mp_draw.draw_landmarks
    # TODO: call normalize_landmarks() and return the result
    pass


def save_sample(landmarks: np.ndarray, label: str) -> None:
    """
    Append one labeled landmark row to the correct CSV file in data/.
    Row format: label, x0, y0, z0, ..., x20, y20, z20
    """
    # TODO: create the data/ directory if it doesn't exist
    # TODO: open the correct CSV file for the given label (append mode)
    # TODO: write one row: [label] + landmarks as a list
    # TODO: print how many samples have been saved for that label
    pass


def main():
    # TODO: call open_camera() to get the capture object
    # TODO: initialize mediapipe Hands with detection/tracking confidence
    # TODO: loop: read a frame from the camera
    # TODO: call detect_landmarks() on each frame
    # TODO: call draw_overlay() to show the current gesture on screen
    # TODO: show sample counts and key hints on the frame with cv2.putText
    # TODO: call cv2.imshow to display the frame
    # TODO: read key with cv2.waitKey — handle i, o, q
    # TODO: on i/o keypress, call save_sample() if landmarks are available
    # TODO: on q, break the loop
    # TODO: release the camera and destroy windows
    pass


if __name__ == "__main__":
    main()
