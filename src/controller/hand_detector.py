import os
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from controller.gesture_classifier import normalize_landmarks

from config import (
    LANDMARKER_PATH as _LANDMARKER_REL,
    NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIRROR_MODE,
    MIRROR_SWAP_HANDEDNESS,
)

LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), "..", _LANDMARKER_REL)

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


def detect_all_hands(frame: np.ndarray, landmarker: HandLandmarker, timestamp_ms: int) -> dict:
    """
    Run the HandLandmarker on the current frame.
    Returns a dict keyed by corrected handedness label:
        {"Left": (norm_landmarks (63,), raw_hand), "Right": (...)}
    Only includes keys for hands that were detected.
    When MIRROR_MODE and MIRROR_SWAP_HANDEDNESS are both True, Left↔Right labels
    are swapped to match the user's actual hands (MediaPipe inverts handedness on
    mirrored frames).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    hands = {}
    for i, hand in enumerate(result.hand_landmarks):
        label = result.handedness[i][0].category_name  # "Left" or "Right"
        if MIRROR_MODE and MIRROR_SWAP_HANDEDNESS:
            label = "Right" if label == "Left" else "Left"
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
        hands[label] = (normalize_landmarks(coords), hand)
    return hands
