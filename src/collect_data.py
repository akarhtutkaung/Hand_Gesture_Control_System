"""
collect_data.py — record labeled hand-landmark samples to CSV.

Controls:
  f  →  save current frame as "fist"   (STOP)
  p  →  save current frame as "palm"   (FORWARD)
  v  →  save current frame as "peace"  (BACKWARD)
  q  →  quit

Run:  python src/collect_data.py  (from the project root)
      Aim for ~100 samples of each gesture.

Requires model/hand_landmarker.task — download once with:
  curl -L -o model/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import os
import time

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from gesture_utils import normalize_landmarks, draw_overlay
from config import (
    LANDMARKER_PATH,
    DATA_DIR,
    NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CAMERA_INDEX,
    MIRROR_MODE,
    COLLECT_KEYS,
    QUIT_KEY,
    HINT_COLOR,
    WINDOW_COLLECT,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", LANDMARKER_PATH)


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Open the webcam at the given device index.
    Returns a cv2.VideoCapture object.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise ValueError(f"Failed to open camera at index {index}")
    cv2.namedWindow(WINDOW_COLLECT)
    return cap


def build_landmarker() -> HandLandmarker:
    """
    Create a MediaPipe HandLandmarker in VIDEO mode.
    Raises FileNotFoundError with download instructions if the model is missing.
    """
    model_path = os.path.normpath(MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o model/hand_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


def detect_landmarks(frame: np.ndarray, landmarker: HandLandmarker, timestamp_ms: int) -> np.ndarray | None:
    """
    Run the HandLandmarker on a single BGR frame.
    Draws the hand skeleton on the frame in place.
    Returns a normalized flat (63,) landmark array, or None if no hand found.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    draw_landmarks(frame, hand, HandLandmarksConnections.HAND_CONNECTIONS)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
    return normalize_landmarks(coords)


def save_sample(landmarks: np.ndarray, label: str) -> None:
    """
    Append one labeled landmark row to the correct CSV file in data/.
    Row format: label, x0, y0, z0, ..., x20, y20, z20
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = f"{DATA_DIR}/{label}.csv"
    with open(csv_path, "a") as f:
        f.write(f"{label}," + ",".join(map(str, landmarks)) + "\n")
    count = sum(1 for _ in open(csv_path))
    print(f"[{label}] {count} samples saved")


def main():
    cap = open_camera()
    landmarker = build_landmarker()
    current_label = ""
    hint = "  ".join(f"{k}={v}" for k, v in COLLECT_KEYS.items()) + f"  {QUIT_KEY}=quit"
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)
        timestamp_ms = time.time_ns() // 1_000_000
        landmarks = detect_landmarks(frame, landmarker, timestamp_ms)
        draw_overlay(frame, current_label)
        cv2.putText(frame, hint, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HINT_COLOR, 2)
        cv2.imshow(WINDOW_COLLECT, frame)
        key = cv2.waitKey(1) & 0xFF
        for char, label in COLLECT_KEYS.items():
            if key == ord(char) and landmarks is not None:
                save_sample(landmarks, label)
                current_label = label
                break
        else:
            if key == ord(QUIT_KEY):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()