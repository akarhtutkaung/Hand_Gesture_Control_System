"""
steering.py — pure math for right-hand rotation-based steering.

No camera, model, or socket dependencies — all functions operate on raw
MediaPipe landmark objects and plain floats, making them straightforward to
unit-test without any hardware or ML setup.

The steering pipeline (called each frame by hand_processors.process_right_hand):
  1. calc_hand_angle(raw)                  → current orientation angle (degrees)
  2. calc_steer_angle(current, origin)     → clamped delta from activation pose

Public API
----------
calc_hand_angle(raw)                  Orientation angle of the hand in the image plane.
calc_steer_angle(current, origin)     Steering delta relative to the activation angle.
"""

import math

from config import (
    STEER_SCALE,
    STEER_MAX_ANGLE,
)


def calc_hand_angle(raw) -> float:
    """
    Compute the hand's orientation angle in the image plane.

    Uses the vector from the wrist (landmark 0) to the middle-finger MCP
    (landmark 9) as the hand's principal axis. atan2 is applied to the
    (dy, dx) components of that vector.

    Parameters
    ----------
    raw : list of MediaPipe NormalizedLandmark
        Full 21-landmark list for one hand, as returned by detect_all_hands().
        Landmark coordinates are in normalised image space [0, 1].

    Returns
    -------
    float
        Angle in degrees, in the range [-180, 180].
        0° = pointing right, 90° = pointing down, -90° = pointing up
        (image y-axis is inverted relative to standard math convention).
    """
    wrist   = raw[0]
    mid_mcp = raw[9]
    return math.degrees(math.atan2(mid_mcp.y - wrist.y, mid_mcp.x - wrist.x))


def calc_steer_angle(current_angle: float, origin_angle: float) -> float:
    """
    Compute the steering angle as the hand's rotation from its activation pose.

    The delta is normalised to [-180, 180] to handle wrap-around at the ±180°
    boundary, then scaled by STEER_SCALE and clamped to ±STEER_MAX_ANGLE.

    Parameters
    ----------
    current_angle : float
        Hand orientation this frame, from calc_hand_angle().
    origin_angle : float
        Hand orientation captured when steering was activated (fist committed).

    Returns
    -------
    float
        Steering angle in degrees.
        Negative = turn left, positive = turn right, 0 = straight.
        Always in the range [-STEER_MAX_ANGLE, +STEER_MAX_ANGLE].
    """
    delta = current_angle - origin_angle
    delta = (delta + 180) % 360 - 180  # normalise to [-180, 180]
    return max(-STEER_MAX_ANGLE, min(STEER_MAX_ANGLE, delta * STEER_SCALE))
