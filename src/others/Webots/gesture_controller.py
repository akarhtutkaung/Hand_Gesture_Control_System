"""
gesture_controller.py — Webots controller for e-puck dual-hand gesture control.

Place this file at:
  Webots/controllers/gesture_controller/gesture_controller.py

How to run:
  Terminal 1 (gesture detection):
    cd /Users/akarkaung/Desktop/Robotic/Hand_Gesture_Control_System
    source venv/bin/activate
    USE_SOCKET=1 python src/run_control.py

  Terminal 2 (Webots):
    Press the play button in Webots

Controls — left hand (movement):
  palm (open hand)  → FORWARD
  fist              → STOP
  peace (V sign)    → BACKWARD
  hand leaves frame → auto-stop

Controls — right hand (steering):
  hold fist         → activate steering mode
  rotate wrist CW   → STEER:+<angle>  (turn right)
  rotate wrist CCW  → STEER:-<angle>  (turn left)
  open / remove     → STEER:0.0       (go straight)

Command format over UDP:
  Movement : "FORWARD" | "STOP" | "BACKWARD"
  Steering : "STEER:<degrees>"  e.g. "STEER:-30.0"
  (host/port configured via SOCKET_HOST / SOCKET_PORT in src/config.py)
"""

import sys
import socket

# ── Webots import ─────────────────────────────────────────────────────────────
try:
    from controller import Robot
except ModuleNotFoundError:
    sys.path.append("/Applications/Webots.app/lib/controller/python")
    from controller import Robot

# ── constants ─────────────────────────────────────────────────────────────────
MAX_SPEED       = 3.0   # rad/s — match e-puck's rated speed
STEER_MAX_ANGLE = 45.0  # degrees — must match config.py STEER_MAX_ANGLE
DECEL_RATE      = 0.05  # velocity step per timestep toward target speed (tune to taste)

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
sock.bind(("127.0.0.1", 5005))  # keep in sync with SOCKET_HOST / SOCKET_PORT in config.py
sock.setblocking(False)

print("Listening for gesture commands on port 5005...")

# ── state ─────────────────────────────────────────────────────────────────────
drive_speed   = 0.0   # target speed set by FORWARD / STOP / BACKWARD
current_speed = 0.0   # actual applied speed — ramps toward drive_speed each timestep
steer_angle   = 0.0   # set by STEER:<degrees>

# ── main loop ─────────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:
    # drain all pending packets; keep only the most recent per channel
    try:
        while True:
            data, _ = sock.recvfrom(1024)
            command = data.decode().strip()
            print(f"Received: {command}")

            if command == "FORWARD":
                drive_speed = MAX_SPEED
            elif command == "BACKWARD":
                drive_speed = -MAX_SPEED
            elif command == "STOP":
                drive_speed = 0.0
            elif command.startswith("STEER:"):
                try:
                    steer_angle = float(command.split(":")[1])
                except ValueError:
                    pass
    except BlockingIOError:
        pass  # no new packets this timestep — keep current state

    # ramp current_speed toward drive_speed
    diff = drive_speed - current_speed
    if abs(diff) <= DECEL_RATE:
        current_speed = drive_speed
    else:
        current_speed += DECEL_RATE * (1.0 if diff > 0 else -1.0)

    # combine drive + steer into per-wheel velocities
    # steer_factor in [-1, 1]: positive = right turn (left wheel faster)
    steer_factor = steer_angle / STEER_MAX_ANGLE
    left_v  = current_speed * (1.0 + steer_factor)
    right_v = current_speed * (1.0 - steer_factor)

    # clamp both wheels to ±MAX_SPEED
    left_v  = max(-MAX_SPEED, min(MAX_SPEED, left_v))
    right_v = max(-MAX_SPEED, min(MAX_SPEED, right_v))

    left_motor.setVelocity(left_v)
    right_motor.setVelocity(right_v)

    direction = "FORWARD" if current_speed > 0.01 else "BACKWARD" if current_speed < -0.01 else "STOPPED"
    print(f"[{direction}] speed={current_speed:+.2f} rad/s  left={left_v:+.2f}  right={right_v:+.2f}  steer={steer_angle:+.1f}deg")

sock.close()