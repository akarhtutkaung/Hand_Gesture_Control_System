# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Real-time dual-hand gesture detection on Mac webcam using MediaPipe. The left hand controls forward/backward movement via gesture classification; the right hand controls steering via wrist rotation. Commands are sent to a Raspberry Pi or Webots simulator over UDP.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data model
```

> Apple Silicon: use `pip install mediapipe-silicon` instead of `mediapipe`.

## Run Order

```bash
python src/collect_data.py   # record ~100 samples per gesture (f=fist, p=palm, v=peace, q=quit)
python src/train_model.py    # train SVM → saves model/gesture_classifier.pkl
python src/run_control.py    # live inference → routes commands based on runtime
```

## Architecture

All scripts share `src/gesture_utils.py` — implement its three helpers first, everything else depends on them. All hardcoded values (colors, paths, thresholds, gestures) live in `src/config.py` — edit there, not in individual scripts.

| File | Role |
|---|---|
| `src/config.py` | Single source of truth for all constants — colors, paths, gestures, thresholds, font settings, socket config |
| `src/gesture_utils.py` | `calc_finger_spread`, `normalize_landmarks`, `draw_overlay` — imported by all other scripts |
| `src/collect_data.py` | Opens webcam, runs MediaPipe, saves labeled rows to `data/fist.csv` / `data/palm.csv` / `data/peace.csv` |
| `src/train_model.py` | Loads CSVs → trains sklearn SVM Pipeline with `probability=True` → saves `model/gesture_classifier.pkl` |
| `src/run_control.py` | Loads model, detects both hands with handedness, runs left/right state machines, sends commands |
| `src/others/Webots/gesture_controller.py` | Webots e-puck controller — receives UDP commands and drives motors with gradual deceleration |

**Landmark format:** MediaPipe returns 21 hand landmarks. Each is stored as `(x, y, z)`, giving a flat array of shape `(63,)`. `normalize_landmarks()` in `gesture_utils.py` translates the wrist to origin and scales to `[-1, 1]` before any CSV write or model call.

---

## Dual-Hand Control

### Left hand — movement (gesture-based)

The SVM classifier runs on the left hand each frame. Commands fire after `DEBOUNCE_FRAMES` consecutive frames and require `MIN_CONFIDENCE`.

| Gesture | Command |
|---|---|
| `palm` | `FORWARD` |
| `fist` | `STOP` |
| `peace` | `BACKWARD` |

When the left hand leaves frame, a `STOP` command is sent automatically.

### Right hand — steering (rotation-based)

The right hand uses a two-stage system — no retraining required.

**Stage 1 — activation:** The SVM must classify the right hand as `fist` for `DEBOUNCE_FRAMES` consecutive frames. On commit, the current wrist→middle-MCP orientation angle is captured as `origin_angle`.

**Stage 2 — continuous steering:** While the fist is held, the hand's orientation angle is computed each frame using `atan2` on the wrist (landmark 0) → middle-finger MCP (landmark 9) vector. The delta from `origin_angle` is the steering angle:

```
delta = current_angle − origin_angle   (normalised to [−180, 180])
steer = clamp(delta × STEER_SCALE, −STEER_MAX_ANGLE, +STEER_MAX_ANGLE)
```

Commands are sent as `STEER:<degrees>` (e.g. `STEER:-30.0`). A change smaller than `STEER_DEADZONE` is suppressed. Opening the right hand or removing it sends `STEER:0.0`.

**Steering tuning in `config.py`:**

| Constant | Default | Effect |
|---|---|---|
| `STEER_SCALE` | `1.0` | Multiplier on rotation delta — increase for sensitivity |
| `STEER_MAX_ANGLE` | `45.0` | Hard cap in degrees |
| `STEER_DEADZONE` | `2.0` | Min change (°) before re-sending |

---

## Inference Guards

- **Debounce** — `DEBOUNCE_FRAMES` (default 5) consecutive frames of the same gesture required before any command fires (~170 ms at 30 fps).
- **Confidence threshold** — `MIN_CONFIDENCE` (default 0.85) minimum `predict_proba` score; predictions below this are discarded. Requires `probability=True` on the SVC — retrain after any change.

---

## Handedness

MediaPipe inverts Left/Right labels when the frame is mirrored. `MIRROR_SWAP_HANDEDNESS = True` (default) corrects this so `"Left"` / `"Right"` keys in `detect_all_hands()` match the user's actual hands. If hands appear swapped, toggle this flag in `config.py`.

---

## Runtime Detection

`run_control.py` detects the environment at startup and routes both `send_command()` and `send_steer()` to the correct backend — no code changes needed when deploying.

| Flag | Condition | Backend |
|---|---|---|
| `ON_PI` | `/sys/firmware/devicetree/base/model` exists | GPIO stub (`# TODO: GPIO`) |
| `USE_SOCKET` | `USE_SOCKET=1` env var | UDP to `SOCKET_HOST:SOCKET_PORT` (default `127.0.0.1:5005`) |
| fallback | Mac / other | `print(f"Command: {command}")` |

Socket target is configured via `SOCKET_HOST` / `SOCKET_PORT` in `config.py`. The Webots controller (`gesture_controller.py`) must bind to the same host/port.

---

## Webots Controller

`src/others/Webots/gesture_controller.py` runs inside Webots as the e-puck controller. It:

- Listens on UDP and drains all pending packets each timestep
- Maintains `drive_speed` (from `FORWARD`/`STOP`/`BACKWARD`) and `steer_angle` (from `STEER:<degrees>`) as separate state
- Applies gradual deceleration: `current_speed` ramps toward `drive_speed` by `DECEL_RATE` per timestep
- Combines both into differential wheel velocities: `left = current_speed × (1 + factor)`, `right = current_speed × (1 − factor)`

`DECEL_RATE` (default `0.05`) and `MAX_SPEED` / `STEER_MAX_ANGLE` must be kept in sync with `config.py`.

---

## config.py Notes

- Paths (`CLASSIFIER_PATH`, `LANDMARKER_PATH`) are resolved as absolute paths relative to `src/` using `os.path.abspath(__file__)`, so scripts work regardless of the working directory they are launched from.
- `NUM_HANDS = 2` — required for dual-hand detection.
- All steering and socket constants live here; the Webots controller duplicates `MAX_SPEED` / `STEER_MAX_ANGLE` locally since it cannot import from `config.py`.
