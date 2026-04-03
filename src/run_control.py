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
from config import (
    CLASSIFIER_PATH as MODEL_PATH,
    LANDMARKER_PATH as _LANDMARKER_REL,
    COMMANDS,
    GESTURE_COLORS,
    HINT_COLOR,
    NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CAMERA_INDEX,
    MIRROR_MODE,
    QUIT_KEY,
    COMMAND_FONT,
    COMMAND_FONT_SCALE,
    COMMAND_THICKNESS,
    COMMAND_MARGIN,
    WINDOW_CONTROL,
    DEBOUNCE_FRAMES,
    MIN_CONFIDENCE,
)

LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), "..", _LANDMARKER_REL)


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
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
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


def classify_gesture(landmarks: np.ndarray, model) -> tuple[str, float]:
    """
    Run the trained classifier on a landmark array.
    Returns (label, confidence) where confidence is the max predict_proba score.
    """
    x = landmarks.reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return model.classes_[idx], float(proba[idx])


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
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise ValueError("Failed to open camera")
    cv2.namedWindow(WINDOW_CONTROL)

    prev_gesture = None
    pending_gesture = None
    debounce_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)
        timestamp_ms = time.time_ns() // 1_000_000
        landmarks, hand = detect_gesture(frame, landmarker, timestamp_ms)

        if landmarks is not None:
            raw_gesture, confidence = classify_gesture(landmarks, model)

            # Confidence gate — ignore low-confidence predictions
            gesture = raw_gesture if confidence >= MIN_CONFIDENCE else None

            # Debounce — only commit a gesture after DEBOUNCE_FRAMES consecutive frames
            if gesture == pending_gesture:
                debounce_count += 1
            else:
                pending_gesture = gesture
                debounce_count = 1

            committed = pending_gesture if debounce_count >= DEBOUNCE_FRAMES else None

            color = GESTURE_COLORS.get(committed or "", GESTURE_COLORS[""])
            spec = DrawingSpec(color=color)
            draw_landmarks(frame, hand, HandLandmarksConnections.HAND_CONNECTIONS,
                           landmark_drawing_spec=spec, connection_drawing_spec=spec)

            if committed is not None and committed != prev_gesture:
                send_command(committed)
                prev_gesture = committed

            command = COMMANDS.get(committed, "") if committed else ""
            if command:
                (w, _), _ = cv2.getTextSize(command, COMMAND_FONT, COMMAND_FONT_SCALE, COMMAND_THICKNESS)
                cv2.putText(frame, command, (frame.shape[1] - w - COMMAND_MARGIN, 70),
                            COMMAND_FONT, COMMAND_FONT_SCALE, color, COMMAND_THICKNESS)
        else:
            pending_gesture = None
            debounce_count = 0
            prev_gesture = None

        cv2.putText(frame, f"{QUIT_KEY}=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HINT_COLOR, 2)
        cv2.imshow(WINDOW_CONTROL, frame)
        if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
