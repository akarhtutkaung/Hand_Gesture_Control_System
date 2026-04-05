"""
run_control.py — live gesture inference, prints motor commands to terminal.

Run:  python src/run_control.py  (from the project root)
      Requires model/gesture_classifier.pkl to exist (run train_model.py first).
      Press q to quit.

Dual-hand control:
  Left hand  — gesture-based: palm=FORWARD, fist=STOP, peace=BACKWARD
  Right hand — steering: hold fist to activate, wrist X relative to activation
               point maps to STEER:<angle> (negative=left, positive=right, 0=straight)
"""

import math
import os
import time

import cv2
import numpy as np
import joblib
import socket
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
    MIRROR_SWAP_HANDEDNESS,
    QUIT_KEY,
    COMMAND_FONT,
    COMMAND_FONT_SCALE,
    COMMAND_THICKNESS,
    COMMAND_MARGIN,
    WINDOW_CONTROL,
    DEBOUNCE_FRAMES,
    MIN_CONFIDENCE,
    STEER_SCALE,
    STEER_MAX_ANGLE,
    STEER_DEADZONE,
    SOCKET_HOST,
    SOCKET_PORT,
)

LANDMARKER_PATH = os.path.join(os.path.dirname(__file__), "..", _LANDMARKER_REL)
ON_PI      = os.path.exists("/sys/firmware/devicetree/base/model")
USE_SOCKET = os.environ.get("USE_SOCKET", "0") == "1"


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


def classify_gesture(landmarks: np.ndarray, model) -> tuple[str, float]:
    """
    Run the trained classifier on a landmark array.
    Returns (label, confidence) where confidence is the max predict_proba score.
    """
    x = landmarks.reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return model.classes_[idx], float(proba[idx])


def calc_hand_angle(raw) -> float:
    """
    Compute the hand's orientation angle (degrees) in the image plane using
    the vector from wrist (landmark 0) to middle-finger MCP (landmark 9).
    Returns a value in [-180, 180].
    """
    wrist   = raw[0]
    mid_mcp = raw[9]
    return math.degrees(math.atan2(mid_mcp.y - wrist.y, mid_mcp.x - wrist.x))


def calc_steer_angle(current_angle: float, origin_angle: float) -> float:
    """
    Compute steering angle (degrees) as the rotation of the hand relative to
    its orientation when the fist was first committed.
    Handles wrap-around at ±180°. Clamped to ±STEER_MAX_ANGLE.
    Negative = left, positive = right.
    """
    delta = current_angle - origin_angle
    # Normalise to [-180, 180] to handle wrap-around
    delta = (delta + 180) % 360 - 180
    return max(-STEER_MAX_ANGLE, min(STEER_MAX_ANGLE, delta * STEER_SCALE))


def send_command(gesture: str) -> None:
    """
    Act on a recognized left-hand gesture.
    Routes to the correct backend based on the detected runtime.
    """
    command = COMMANDS.get(gesture, "UNKNOWN")
    if ON_PI:
        pass  # TODO: GPIO motor calls
    elif USE_SOCKET:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(command.encode(), (SOCKET_HOST, SOCKET_PORT))
        except Exception as e:
            print(f"Socket error: {e}")
    else:
        print(f"Command: {command}")


def send_steer(angle: float) -> None:
    """
    Send a continuous steering angle command formatted as 'STEER:<angle>'.
    Negative = left, positive = right, 0 = straight.
    """
    command = f"STEER:{angle:.1f}"
    if ON_PI:
        pass  # TODO: map angle to GPIO PWM / servo
    elif USE_SOCKET:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(command.encode(), (SOCKET_HOST, SOCKET_PORT))
        except Exception as e:
            print(f"Socket error: {e}")
    else:
        print(f"Command: {command}")


def make_left_state() -> dict:
    return {"pending": None, "debounce": 0, "prev": None}


def make_right_state() -> dict:
    return {"fist_frames": 0, "active": False, "origin_angle": None, "prev_angle": None}


def process_left_hand(frame: np.ndarray, hand_data: tuple, model, state: dict) -> None:
    """
    Classify the left-hand gesture, apply debounce + confidence gate,
    draw the skeleton, and fire send_command() on gesture change.
    Mutates state in-place.
    """
    lm, raw = hand_data
    raw_g, conf = classify_gesture(lm, model)
    gesture = raw_g if conf >= MIN_CONFIDENCE else None

    if gesture == state["pending"]:
        state["debounce"] += 1
    else:
        state["pending"] = gesture
        state["debounce"] = 1

    committed = state["pending"] if state["debounce"] >= DEBOUNCE_FRAMES else None

    color = GESTURE_COLORS.get(committed or "", GESTURE_COLORS[""])
    spec = DrawingSpec(color=color)
    draw_landmarks(frame, raw, HandLandmarksConnections.HAND_CONNECTIONS,
                   landmark_drawing_spec=spec, connection_drawing_spec=spec)

    if committed is not None and committed != state["prev"]:
        send_command(committed)
        state["prev"] = committed

    cmd = COMMANDS.get(committed, "") if committed else ""
    if cmd:
        (w, _), _ = cv2.getTextSize(cmd, COMMAND_FONT, COMMAND_FONT_SCALE, COMMAND_THICKNESS)
        cv2.putText(frame, cmd,
                    (frame.shape[1] - w - COMMAND_MARGIN, 70),
                    COMMAND_FONT, COMMAND_FONT_SCALE, color, COMMAND_THICKNESS)


def reset_left_hand(state: dict) -> None:
    if state["prev"] is not None:
        send_command("fist")  # fist → STOP; only fires if a command was previously active
    state["pending"] = None
    state["debounce"] = 0
    state["prev"] = None


def process_right_hand(frame: np.ndarray, hand_data: tuple, model, state: dict) -> None:
    """
    Detect a right-hand fist to activate steering, then measure how much the
    hand has rotated (wrist→middle-MCP arc angle) relative to the activation
    pose and fire send_steer() on change.
    Mutates state in-place.
    """
    lm, raw = hand_data
    raw_g, conf = classify_gesture(lm, model)
    is_fist = (raw_g == "fist" and conf >= MIN_CONFIDENCE)

    if is_fist:
        state["fist_frames"] = min(state["fist_frames"] + 1, DEBOUNCE_FRAMES)
    else:
        state["fist_frames"] = 0
        if state["active"]:
            send_steer(0.0)
        state["active"] = False
        state["origin_angle"] = None
        state["prev_angle"]   = None

    if state["fist_frames"] >= DEBOUNCE_FRAMES and not state["active"]:
        state["active"]       = True
        state["origin_angle"] = calc_hand_angle(raw)

    skel_color = GESTURE_COLORS["steering"] if state["active"] else GESTURE_COLORS[""]
    spec = DrawingSpec(color=skel_color)
    draw_landmarks(frame, raw, HandLandmarksConnections.HAND_CONNECTIONS,
                   landmark_drawing_spec=spec, connection_drawing_spec=spec)

    if state["active"]:
        angle = calc_steer_angle(calc_hand_angle(raw), state["origin_angle"])
        if state["prev_angle"] is None or abs(angle - state["prev_angle"]) >= STEER_DEADZONE:
            send_steer(angle)
            state["prev_angle"] = angle

        cv2.putText(frame, f"STEER {angle:+.0f}deg", (10, 90),
                    COMMAND_FONT, COMMAND_FONT_SCALE * 0.7,
                    GESTURE_COLORS["steering"], COMMAND_THICKNESS - 1)


def reset_right_hand(state: dict) -> None:
    if state["active"]:
        send_steer(0.0)
    state["fist_frames"]  = 0
    state["active"]       = False
    state["origin_angle"] = None
    state["prev_angle"]   = None


def main():
    model = load_model()
    landmarker = build_landmarker()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise ValueError("Failed to open camera")
    cv2.namedWindow(WINDOW_CONTROL)

    left_state  = make_left_state()
    right_state = make_right_state()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break
        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)
        timestamp_ms = time.time_ns() // 1_000_000
        hands = detect_all_hands(frame, landmarker, timestamp_ms)

        if "Left" in hands:
            process_left_hand(frame, hands["Left"], model, left_state)
        else:
            reset_left_hand(left_state)

        if "Right" in hands:
            process_right_hand(frame, hands["Right"], model, right_state)
        else:
            reset_right_hand(right_state)

        cv2.putText(frame, f"{QUIT_KEY}=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HINT_COLOR, 2)
        cv2.imshow(WINDOW_CONTROL, frame)
        if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
