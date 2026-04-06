"""
config.py — central configuration for the Hand Gesture Control System.

Import from any script:
    from config import GESTURES, COLORS, COMMANDS, ...
"""

import cv2
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_PATH  = os.path.normpath(os.path.join(_HERE, "..", "model", "gesture_classifier.pkl"))
LANDMARKER_PATH  = os.path.normpath(os.path.join(_HERE, "..", "model", "hand_landmarker.task"))
DATA_DIR = "data"
MODEL_DIR = "model"

# ---------------------------------------------------------------------------
# Gesture labels
# ---------------------------------------------------------------------------

# All recognized gesture names. Order matters for collect_data key bindings.
GESTURES = ["fist", "palm", "peace"]

# CSV data file per gesture
CSV_PATHS = {gesture: f"{DATA_DIR}/{gesture}.csv" for gesture in GESTURES}

# ---------------------------------------------------------------------------
# Gesture → motor command mapping
# ---------------------------------------------------------------------------

COMMANDS = {
    "palm":  "FORWARD",
    "fist":  "STOP",
    "peace": "BACKWARD",
}

# ---------------------------------------------------------------------------
# Colors — BGR format (OpenCV convention)
# ---------------------------------------------------------------------------

# Per-gesture overlay / skeleton color
GESTURE_COLORS = {
    "palm":     (0, 255, 0),    # green  — FORWARD
    "fist":     (0, 0, 255),    # red    — STOP
    "peace":    (255, 0, 0),    # blue   — BACKWARD
    "steering": (0, 255, 255),  # yellow — steering mode active (right hand)
    "":         (255, 255, 255), # white  — no gesture / default
}

# UI hint text color (key legend, etc.)
HINT_COLOR = (200, 200, 200)  # light grey

# ---------------------------------------------------------------------------
# Overlay / font
# ---------------------------------------------------------------------------

# Gesture label drawn by draw_overlay() in gesture_utils.py
OVERLAY_FONT       = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_SCALE = 1.0
OVERLAY_THICKNESS  = 2
OVERLAY_POSITION   = (10, 30)   # (x, y) pixels from top-left

# Large command text drawn in run_control.py
COMMAND_FONT       = cv2.FONT_HERSHEY_DUPLEX
COMMAND_FONT_SCALE = 2.0
COMMAND_THICKNESS  = 5
COMMAND_MARGIN     = 20         # pixels from the right edge

# ---------------------------------------------------------------------------
# MediaPipe hand landmarker settings
# ---------------------------------------------------------------------------

NUM_HANDS                    = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE  = 0.5
MIN_TRACKING_CONFIDENCE       = 0.5

# When MIRROR_MODE is True, MediaPipe sees a flipped image and reports handedness
# inverted relative to the user's actual hands. Set True to correct this.
MIRROR_SWAP_HANDEDNESS = True

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0       # default webcam
MIRROR_MODE  = True    # flip frame horizontally (mirror view)

# ---------------------------------------------------------------------------
# UDP socket (USE_SOCKET=1 mode)
# ---------------------------------------------------------------------------

SOCKET_HOST = "127.0.0.1"
SOCKET_PORT = 5005

# ---------------------------------------------------------------------------
# collect_data.py key bindings  {key_char: gesture_label}
# ---------------------------------------------------------------------------

COLLECT_KEYS = {
    "f": "fist",
    "p": "palm",
    "v": "peace",
}

QUIT_KEY = "q"

# ---------------------------------------------------------------------------
# Window titles
# ---------------------------------------------------------------------------

WINDOW_COLLECT = "Hand Gesture Data Collection"
WINDOW_CONTROL = "Gesture Control"

# ---------------------------------------------------------------------------
# Training (train_model.py)
# ---------------------------------------------------------------------------

CV_FOLDS = 5   # cross-validation folds
SVM_KERNEL = "rbf"

# ---------------------------------------------------------------------------
# Inference guards (run_control.py)
# ---------------------------------------------------------------------------

# Debounce: number of consecutive frames a gesture must hold before a command fires.
# Higher = more stable but slightly more latency. At ~30 fps, 5 frames ≈ 170 ms.
DEBOUNCE_FRAMES = 5

# Confidence threshold: minimum predict_proba score required to accept a prediction.
# Range 0.0–1.0. Predictions below this are ignored (treated as no detection).
MIN_CONFIDENCE = 0.85

# ---------------------------------------------------------------------------
# Steering (right hand, run_control.py)
# ---------------------------------------------------------------------------

# Sensitivity multiplier on the hand rotation delta (already in degrees).
# 1.0 = 1:1 mapping; increase for more sensitive steering.
STEER_SCALE     = 0.5

# Maximum steering angle in either direction (degrees).
STEER_MAX_ANGLE = 45.0

# Minimum angle change (degrees) required to re-send a steering command.
# Filters out jitter without adding latency.
STEER_DEADZONE  = 2.0
