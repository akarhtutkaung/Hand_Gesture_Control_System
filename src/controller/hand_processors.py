"""
hand_processors.py — per-hand state machines that translate detections into commands.

Each frame, car_movement_control.py calls the appropriate process_* function if a
hand is visible, or the matching reset_* function if the hand has left the frame.

Left hand  — gesture-based movement
    A debounce counter accumulates consecutive frames of the same high-confidence
    gesture. Once DEBOUNCE_FRAMES is reached the gesture is "committed" and
    send_command() is called. A command is only re-sent when the committed gesture
    changes, preventing repeated identical UDP packets.

Right hand — fist-activated steering
    Stage 1 (activation): the SVM must classify the hand as "fist" for
    DEBOUNCE_FRAMES consecutive frames. On commit, the current wrist→middle-MCP
    angle is captured as origin_angle.
    Stage 2 (continuous): while the fist is held, the rotation delta relative to
    origin_angle is computed each frame and sent as STEER:<degrees> whenever the
    change exceeds STEER_DEADZONE.

Public API
----------
process_left_hand(frame, hand_data, model, state)   Left-hand gesture pipeline.
reset_left_hand(state)                              Called when left hand absent.
process_right_hand(frame, hand_data, model, state)  Right-hand steering pipeline.
reset_right_hand(state)                             Called when right hand absent.
draw_overlay(frame, gesture)                        Render gesture label on frame.
"""

import cv2
import numpy as np

from controller.gesture_classifier import classify_gesture
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from comms.car_interface import send_command, send_steer
from controller.steering import calc_hand_angle, calc_steer_angle
from config import (
    GESTURE_COLORS,
    MIN_CONFIDENCE,
    DEBOUNCE_FRAMES,
    COMMANDS,
    COMMAND_FONT,
    COMMAND_FONT_SCALE,
    COMMAND_THICKNESS,
    COMMAND_MARGIN,
    STEER_DEADZONE,
    OVERLAY_POSITION,
    OVERLAY_FONT,
    OVERLAY_FONT_SCALE,
    OVERLAY_THICKNESS,
)


def process_left_hand(frame: np.ndarray, hand_data: tuple, model, state: dict) -> None:
    """
    Run the gesture classifier on the left hand and dispatch movement commands.

    Pipeline per frame:
      1. Classify landmarks; discard predictions below MIN_CONFIDENCE.
      2. Increment debounce counter if the gesture matches state["pending"],
         otherwise reset the counter to 1 with the new gesture.
      3. Commit the gesture once DEBOUNCE_FRAMES consecutive frames agree.
      4. Draw the hand skeleton in the gesture's colour.
      5. Call send_command() only when the committed gesture changes.
      6. Render the command text (e.g. "FORWARD") in the top-right corner.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame from the webcam; annotated in-place.
    hand_data : tuple
        (norm_landmarks (63,), raw_hand) as returned by detect_all_hands().
    model : sklearn.pipeline.Pipeline
        Loaded gesture classifier from gesture_classifier.load_model().
    state : dict
        Mutable debounce state from car_movement_control.make_left_state().
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
    """
    Handle the left hand leaving the frame.

    Sends a STOP command (via "fist") if any command was previously active,
    then clears all debounce state so the next detection starts fresh.

    Parameters
    ----------
    state : dict
        Mutable debounce state from car_movement_control.make_left_state().
    """
    if state["prev"] is not None:
        send_command("fist")  # fist → STOP
    state["pending"] = None
    state["debounce"] = 0
    state["prev"] = None


def process_right_hand(frame: np.ndarray, hand_data: tuple, model, state: dict) -> None:
    """
    Run the two-stage fist-activated steering pipeline on the right hand.

    Stage 1 — activation:
      Count consecutive fist frames. Once DEBOUNCE_FRAMES is reached and steering
      is not yet active, capture the current wrist→middle-MCP angle as origin_angle
      and set state["active"] = True.

    Stage 2 — continuous steering:
      Each frame while active, compute the rotation delta from origin_angle via
      calc_steer_angle(). Send STEER:<degrees> only when the change exceeds
      STEER_DEADZONE, suppressing jitter without adding latency.

    If the fist is released, immediately send STEER:0.0 and deactivate.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame from the webcam; annotated in-place.
    hand_data : tuple
        (norm_landmarks (63,), raw_hand) as returned by detect_all_hands().
    model : sklearn.pipeline.Pipeline
        Loaded gesture classifier from gesture_classifier.load_model().
    state : dict
        Mutable steering state from car_movement_control.make_right_state().
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
    """
    Handle the right hand leaving the frame.

    Sends STEER:0.0 if steering was active, then clears all steering state.

    Parameters
    ----------
    state : dict
        Mutable steering state from car_movement_control.make_right_state().
    """
    if state["active"]:
        send_steer(0.0)
    state["fist_frames"]  = 0
    state["active"]       = False
    state["origin_angle"] = None
    state["prev_angle"]   = None


def draw_overlay(frame: np.ndarray, gesture: str) -> np.ndarray:
    """
    Render the current gesture label on the video frame.

    The text color is looked up from GESTURE_COLORS so it matches the skeleton
    color drawn by draw_landmarks(). Used by collect_data.py during recording.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame; annotated in-place and also returned.
    gesture : str
        Gesture label to display (e.g. "fist", "palm"). Empty string renders nothing visible.

    Returns
    -------
    np.ndarray
        The same frame with the label drawn on it.
    """
    color = GESTURE_COLORS.get(gesture, GESTURE_COLORS[""])
    cv2.putText(frame, gesture, OVERLAY_POSITION, OVERLAY_FONT, OVERLAY_FONT_SCALE, color, OVERLAY_THICKNESS)
    return frame
