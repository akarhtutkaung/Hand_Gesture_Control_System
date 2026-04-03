"""
run_control.py — live gesture inference, prints motor commands to terminal.

Run:  python src/run_control.py  (from the project root)
      Requires model/gesture_classifier.pkl to exist (run train_model.py first).
      Press q to quit.
"""

import os
import time

import cv2
import numpy as np
import joblib
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from gesture_utils import normalize_landmarks

MODEL_PATH = "model/gesture_classifier.pkl"
LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")

COMMANDS = {
    "palm":  "FORWARD",
    "fist":  "STOP",
    "peace": "BACKWARD",
}

# BGR skeleton colors per gesture — matches gesture_utils.draw_overlay
SKELETON_COLORS = {
    "palm":  (0, 255, 0),    # green
    "fist":  (0, 0, 255),    # red
    "peace": (255, 0, 0),    # blue
}


def load_model():
    """
    Load the trained gesture classifier from model/gesture_classifier.pkl.
    Returns the loaded sklearn Pipeline object.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run `python src/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def build_landmarker() -> HandLandmarker:
    """
    Create a MediaPipe HandLandmarker in VIDEO mode.
    """
    path = os.path.normpath(LANDMARKER_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Hand landmark model not found: {path}\n"
            "Download it with:\n"
            "  curl -L -o model/hand_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def detect_gesture(frame: np.ndarray, landmarker: HandLandmarker, timestamp_ms: int):
    """
    Run the HandLandmarker on the current frame.
    Returns (normalized coords (63,), raw hand landmarks) or (None, None) if no hand detected.
    Drawing is deferred so the skeleton color can reflect the classified gesture.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None, None
    hand = result.hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
    return normalize_landmarks(coords), hand


def classify_gesture(landmarks: np.ndarray, model) -> str:
    """
    Run the trained classifier on a landmark array.
    Returns the predicted label string: "fist", "palm", or "peace".
    """
    return model.predict(landmarks.reshape(1, -1))[0]


def send_command(gesture: str) -> None:
    """
    Act on the recognized gesture.
    Currently prints the motor command to the terminal.
    # TODO: GPIO — replace print() with RPi.GPIO or gpiozero motor calls on the Pi.
    """
    command = COMMANDS.get(gesture, "UNKNOWN")
    print(command)


def main():
    model = load_model()
    landmarker = build_landmarker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Failed to open camera")
    cv2.namedWindow("Gesture Control")

    prev_gesture = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        frame = cv2.flip(frame, 1)
        timestamp_ms = time.time_ns() // 1_000_000
        landmarks, hand = detect_gesture(frame, landmarker, timestamp_ms)

        if landmarks is not None:
            gesture = classify_gesture(landmarks, model)
            color = SKELETON_COLORS.get(gesture, (255, 255, 255))
            spec = DrawingSpec(color=color)
            draw_landmarks(frame, hand, HandLandmarksConnections.HAND_CONNECTIONS,
                           landmark_drawing_spec=spec, connection_drawing_spec=spec)
            if gesture != prev_gesture:
                send_command(gesture)
                prev_gesture = gesture
            command = COMMANDS.get(gesture, "")
            font, scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 2.0, 5
            (w, _), _ = cv2.getTextSize(command, font, scale, thickness)
            cv2.putText(frame, command, (frame.shape[1] - w - 20, 70),
                        font, scale, color, thickness)
        else:
            prev_gesture = None

        cv2.putText(frame, "q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
