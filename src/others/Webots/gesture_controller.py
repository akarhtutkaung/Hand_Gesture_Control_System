"""
gesture_controller.py — Webots controller for e-puck gesture-based control.

Place this file at:
  Webots/controllers/gesture_controller/gesture_controller.py

How to run:
  Terminal 1 (gesture detection):
    cd /Users/akarkaung/Desktop/Robotic/Hand_Gesture_Control_System
    source venv/bin/activate
    USE_SOCKET=1 python src/run_control.py

  Terminal 2 (Webots):
    Press the play button in Webots

Controls:
  open hand         → forward
  fist              → backward
  flat hand         → stop
  hand leaves frame → auto-stop
"""

import sys
import socket

# ── Webots import ─────────────────────────────────────────────────────────────
try:
    from controller import Robot
except ModuleNotFoundError:
    sys.path.append("/Applications/Webots.app/lib/controller/python")
    from controller import Robot

# ── robot setup ───────────────────────────────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

print("=== E-puck ready ===")

# ── socket setup ──────────────────────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))
sock.setblocking(False)

print("Listening for gesture commands on port 5005...")

# ── main loop ─────────────────────────────────────────────────────────────────
current_command = "stop"  # default

while robot.step(timestep) != -1:
    try:
        data, _ = sock.recvfrom(1024)
        current_command = data.decode().strip().lower()
        print(f"Received: {current_command}")
    except BlockingIOError:
        pass  # no new command, keep current_command as is

    # apply current command every timestep regardless
    if current_command == "forward":
        left_motor.setVelocity(3)
        right_motor.setVelocity(3)
    elif current_command == "backward":
        left_motor.setVelocity(-3)
        right_motor.setVelocity(-3)
    elif current_command == "stop":
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

sock.close()