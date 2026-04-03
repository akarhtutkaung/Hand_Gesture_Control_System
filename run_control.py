"""
run_control.py — live gesture inference, prints motor commands to terminal.

Run:  python run_control.py
      Requires model/gesture_classifier.pkl to exist (run train_model.py first).
      Press q to quit.
"""

import numpy as np
import cv2
from gesture_utils import normalize_landmarks, draw_overlay


def load_model():
    """
    Load the trained gesture classifier from model/gesture_classifier.pkl.
    Returns the loaded sklearn Pipeline object.
    """
    # TODO: use joblib.load to load the .pkl file
    # TODO: raise a clear error if the file doesn't exist (remind user to train first)
    # TODO: return the model
    pass


def detect_gesture(frame: np.ndarray, hands) -> np.ndarray | None:
    """
    Run MediaPipe Hands on the current frame.
    Draws landmarks on the frame in place.
    Returns a normalized flat (63,) landmark array, or None if no hand detected.
    """
    # TODO: convert frame BGR → RGB
    # TODO: run hands.process() on the RGB frame
    # TODO: return None if no hand is detected
    # TODO: extract and normalize the first hand's landmarks
    # TODO: draw the skeleton on the frame
    # TODO: return the flat (63,) landmark array
    pass


def classify_gesture(landmarks: np.ndarray, model) -> str:
    """
    Run the trained classifier on a landmark array.
    Returns the predicted label string: "squeeze_in" or "squeeze_out".
    """
    # TODO: reshape landmarks to (1, 63) for sklearn
    # TODO: call model.predict() and return the label string
    pass


def send_command(gesture: str) -> None:
    """
    Act on the recognized gesture.
    Currently prints the motor command to the terminal.
    Later: replace print() with GPIO calls on the Raspberry Pi.
    """
    # TODO: map gesture labels to motor commands
    #       "squeeze_in"  → print "FORWARD"
    #       "squeeze_out" → print "BACKWARD"
    # TODO: GPIO — replace print with RPi.GPIO or gpiozero motor calls
    pass


def main():
    # TODO: call load_model() to get the classifier
    # TODO: open the webcam with cv2.VideoCapture(0)
    # TODO: initialize mediapipe Hands
    # TODO: loop: read a frame from the camera
    # TODO: call detect_gesture() on each frame
    # TODO: if landmarks found, call classify_gesture() to get the label
    # TODO: call send_command() only when the gesture changes (avoid repeated prints)
    # TODO: call draw_overlay() and cv2.imshow to display the annotated frame
    # TODO: break on q keypress
    # TODO: release camera and destroy windows on exit
    pass


if __name__ == "__main__":
    main()
