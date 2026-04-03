"""
config.py — central configuration for the Hand Gesture Control System.

Import from any script:
    from config import GESTURES, COLORS, COMMANDS, ...
"""

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LANDMARKER_PATH = "model/hand_landmarker.task"
CLASSIFIER_PATH = "model/gesture_classifier.pkl"
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
    "palm":  (0, 255, 0),    # green  — FORWARD
    "fist":  (0, 0, 255),    # red    — STOP
    "peace": (255, 0, 0),    # blue   — BACKWARD
    "":      (255, 255, 255), # white  — no gesture / default
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

NUM_HANDS                    = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE  = 0.5
MIN_TRACKING_CONFIDENCE       = 0.5

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0       # default webcam
MIRROR_MODE  = True    # flip frame horizontally (mirror view)

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
