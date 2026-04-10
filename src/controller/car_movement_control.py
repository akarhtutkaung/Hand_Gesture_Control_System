"""
car_movement_control.py — main camera loop and entry point for gesture-based car control.

Orchestrates the full pipeline each frame:
  1. Capture frame from webcam and mirror if configured.
  2. Run MediaPipe hand detection (up to 2 hands).
  3. Route each detected hand to the appropriate processor:
       Left  hand → gesture classifier → FORWARD / STOP / BACKWARD
       Right hand → fist-activated steering → STEER:<angle>
  4. Reset the relevant state machine when a hand leaves frame.
  5. Render skeleton overlays and HUD text, then display the frame.

Run from the project root:
    python src/app.py

Requires model/gesture_classifier.pkl (run controller/train_model.py first).
Press the key defined by QUIT_KEY in config.py (default: q) to exit.
"""

import time

import cv2
from controller.gesture_classifier import load_model
from controller.hand_processors import (
    process_left_hand,
    reset_left_hand,
    process_right_hand,
    reset_right_hand,
)
from .hand_detector import build_landmarker, detect_all_hands
from config import (
    HINT_COLOR,
    CAMERA_INDEX,
    MIRROR_MODE,
    QUIT_KEY,
    WINDOW_CONTROL,
)


def make_left_state() -> dict:
    """
    Initialise the debounce state machine for the left hand.

    Keys
    ----
    pending   : str | None   Gesture label accumulating consecutive frames.
    debounce  : int          Consecutive-frame counter for the pending gesture.
    prev      : str | None   Last committed gesture; prevents re-sending the same command.
    """
    return {"pending": None, "debounce": 0, "prev": None}


def make_right_state() -> dict:
    """
    Initialise the two-stage steering state machine for the right hand.

    Keys
    ----
    fist_frames  : int          Consecutive fist frames counted toward activation.
    active       : bool         True once the fist has been held for DEBOUNCE_FRAMES.
    origin_angle : float | None Hand orientation angle captured at activation moment.
    prev_angle   : float | None Last sent steering angle; used for deadzone filtering.
    """
    return {"fist_frames": 0, "active": False, "origin_angle": None, "prev_angle": None}


def start_car_control() -> None:
    """
    Open the webcam and run the gesture-control loop until the user quits.

    Each iteration:
    - Reads a frame and optionally mirrors it (MIRROR_MODE).
    - Calls detect_all_hands() to get normalised landmarks for each visible hand.
    - Delegates to process_left_hand / process_right_hand (or their reset
      counterparts when a hand is absent) to update state and dispatch commands.
    - Draws a quit-key hint and shows the annotated frame via cv2.imshow().

    Raises
    ------
    ValueError
        If the camera cannot be opened at CAMERA_INDEX.
    FileNotFoundError
        If the gesture classifier model or MediaPipe landmarker task is missing.
    """
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
    start_car_control()
