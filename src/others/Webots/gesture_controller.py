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
MAX_SPEED       = 6.0   # rad/s — e-puck rated max speed
STEER_MAX_ANGLE = 45.0  # degrees — must match config.py STEER_MAX_ANGLE
DECEL_RATE      = 0.2   # speed ramp per timestep — lower = smoother acceleration
STEER_RATE      = 2.0   # steering ramp per timestep — higher = snappier turning

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

# ── state ─────────────────────────────────────────────────────────────────────
drive_speed    = 0.0   # target speed from gesture command
current_speed  = 0.0   # actual applied speed — ramps toward drive_speed
steer_angle    = 0.0   # target steering angle from STEER command
current_steer  = 0.0   # actual applied steering — ramps toward steer_angle
last_direction = None  # for change-only printing
last_steer_log = None  # for change-only printing

# ── main loop ─────────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:

    # drain all pending packets — keep only the most recent per channel
    latest_move  = None
    latest_steer = None

    try:
        while True:
            data, _ = sock.recvfrom(1024)
            command = data.decode().strip()
            if command in ("FORWARD", "BACKWARD", "STOP"):
                latest_move = command
            elif command.startswith("STEER:"):
                latest_steer = command
    except BlockingIOError:
        pass  # no more packets this timestep

    # apply latest movement command
    if latest_move == "FORWARD":
        drive_speed = MAX_SPEED
    elif latest_move == "BACKWARD":
        drive_speed = -MAX_SPEED
    elif latest_move == "STOP":
        drive_speed = 0.0

    # apply latest steering command
    if latest_steer:
        try:
            steer_angle = float(latest_steer.split(":")[1])
        except ValueError:
            pass

    # ramp current_speed toward drive_speed
    speed_diff = drive_speed - current_speed
    if abs(speed_diff) <= DECEL_RATE:
        current_speed = drive_speed
    else:
        current_speed += DECEL_RATE * (1.0 if speed_diff > 0 else -1.0)

    # ramp current_steer toward steer_angle
    steer_diff = steer_angle - current_steer
    if abs(steer_diff) <= STEER_RATE:
        current_steer = steer_angle
    else:
        current_steer += STEER_RATE * (1.0 if steer_diff > 0 else -1.0)

    # combine speed + steering into per-wheel velocities
    # steer_factor > 0 = turn right (left wheel faster than right)
    # steer_factor < 0 = turn left  (right wheel faster than left)
    steer_factor = current_steer / STEER_MAX_ANGLE
    left_v  = current_speed * (1.0 + steer_factor)
    right_v = current_speed * (1.0 - steer_factor)

    # clamp both wheels to ±MAX_SPEED
    left_v  = max(-MAX_SPEED, min(MAX_SPEED, left_v))
    right_v = max(-MAX_SPEED, min(MAX_SPEED, right_v))

    left_motor.setVelocity(left_v)
    right_motor.setVelocity(right_v)

    # only print when something meaningfully changes
    direction = (
        "FORWARD"  if current_speed >  0.01 else
        "BACKWARD" if current_speed < -0.01 else
        "STOPPED"
    )
    steer_changed = last_steer_log is None or abs(current_steer - last_steer_log) > 1.0
    if direction != last_direction or steer_changed:
        print(f"[{direction}] speed={current_speed:+.2f} rad/s  steer={current_steer:+.1f}deg  L={left_v:+.2f}  R={right_v:+.2f}")
        last_direction = direction
        last_steer_log = current_steer

sock.close()