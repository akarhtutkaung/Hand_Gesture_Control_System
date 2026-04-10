"""
car_interface.py — runtime-adaptive command dispatcher.

Detects the execution environment at import time and routes every outgoing
command to the correct backend:

    ON_PI        /sys/firmware/devicetree/base/model exists → Raspberry Pi GPIO
    USE_SOCKET   USE_SOCKET=1 env var                       → UDP socket
    fallback     anything else (Mac dev machine, CI, …)     → stdout print

Public API
----------
send_command(gesture)   Fire a movement command derived from a left-hand gesture.
send_steer(angle)       Fire a continuous steering angle from the right hand.

Environment variables
---------------------
USE_SOCKET   Set to "1" to enable UDP output (default: "0").

Socket target is configured via SOCKET_HOST / SOCKET_PORT in config.py.
"""

import os
import socket

from config import (
    COMMANDS,
    SOCKET_HOST,
    SOCKET_PORT,
)

ON_PI      = os.path.exists("/sys/firmware/devicetree/base/model")
USE_SOCKET = os.environ.get("USE_SOCKET", "0") == "1"


def send_command(gesture: str) -> None:
    """
    Translate a left-hand gesture label into a movement command and dispatch it.

    Gesture → command mapping is defined in config.COMMANDS:
        "palm"  → "FORWARD"
        "fist"  → "STOP"
        "peace" → "BACKWARD"

    Parameters
    ----------
    gesture : str
        Gesture label produced by the classifier (e.g. "palm", "fist", "peace").
        An unrecognised label sends "UNKNOWN".
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
    Send a continuous steering angle command formatted as "STEER:<angle>".

    The Webots controller and Pi GPIO handler both parse this format.
    A value of 0.0 means straight ahead; negative = left, positive = right.
    The angle is clamped upstream (in steering.py) to ±STEER_MAX_ANGLE before
    this function is called.

    Parameters
    ----------
    angle : float
        Steering angle in degrees. Sent with one decimal place of precision,
        e.g. "STEER:-30.0" or "STEER:12.5".
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
