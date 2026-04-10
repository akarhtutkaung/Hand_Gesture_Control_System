# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Real-time dual-hand gesture detection on Mac webcam using MediaPipe. The left hand controls forward/backward movement via gesture classification; the right hand controls steering via wrist rotation. Commands are sent to a Raspberry Pi or Webots simulator over UDP. A facial recognition + hand-gesture unlock flow guards entry before control starts.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data model
```

> Apple Silicon: use `pip install mediapipe-silicon` instead of `mediapipe`.

Download the MediaPipe hand landmarker model once:
```bash
curl -L -o model/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run Order

```bash
# 1. Record ~100 samples per gesture (f=fist, p=palm, v=peace, q=quit)
python src/controller/collect_data.py

# 2. Train SVM → saves model/gesture_classifier.pkl
python src/controller/train_model.py

# 3. Run the full app (auth → gesture control)
python src/app.py
```

## Architecture

All hardcoded values (colors, paths, thresholds, gestures, socket config) live in `src/config.py` — edit there, not in individual scripts.

```
src/
├── app.py                          Entry point: auth flow → start_car_control()
├── config.py                       Single source of truth for all constants
│
├── auth/                           (stubs — not yet implemented)
│   ├── facial_recognition.py       Step 1: register/verify face
│   ├── hand_gesture_auth.py        Step 2: register/verify unlock gesture
│   └── unlock_manager.py           Orchestrates Step 1 + Step 2
│
├── comms/
│   └── car_interface.py            send_command(), send_steer() → socket/GPIO/print
│
└── controller/
    ├── collect_data.py             Dev: webcam loop → saves labeled CSVs to data/
    ├── train_model.py              Dev: loads CSVs → trains SVM → gesture_classifier.pkl
    ├── gesture_classifier.py       load_model(), classify_gesture(), normalize_landmarks()
    ├── hand_detector.py            build_landmarker(), detect_all_hands()
    ├── hand_processors.py          process/reset left & right hand state machines
    ├── steering.py                 calc_hand_angle(), calc_steer_angle() — pure math
    └── car_movement_control.py     Camera loop + rendering; called by app.py
```

### Import rules

`controller/` modules use **relative imports** for siblings (e.g. `from .gesture_classifier import ...`) and bare imports for top-level `src/` modules (`config`, `comms`).

`collect_data.py` and `train_model.py` are dev scripts run directly. They insert `src/` onto `sys.path` at the top so they can resolve `config` and use `from controller.X import Y` for siblings that have relative imports internally.

---

## Dual-Hand Control

### Left hand — movement (gesture-based)

The SVM classifier runs on the left hand each frame. Commands fire after `DEBOUNCE_FRAMES` consecutive frames and require `MIN_CONFIDENCE`.

| Gesture | Command  |
|---------|----------|
| `palm`  | FORWARD  |
| `fist`  | STOP     |
| `peace` | BACKWARD |

When the left hand leaves frame, STOP is sent automatically.

### Right hand — steering (rotation-based)

Two-stage system — no retraining required.

**Stage 1 — activation:** SVM must classify the right hand as `fist` for `DEBOUNCE_FRAMES` consecutive frames. On commit, the current wrist→middle-MCP orientation angle is captured as `origin_angle`.

**Stage 2 — continuous steering:** While the fist is held, the delta from `origin_angle` is computed each frame:

```
delta = current_angle − origin_angle   (normalised to [−180, 180])
steer = clamp(delta × STEER_SCALE, −STEER_MAX_ANGLE, +STEER_MAX_ANGLE)
```

Commands are sent as `STEER:<degrees>`. A change smaller than `STEER_DEADZONE` is suppressed. Releasing the fist or removing the hand sends `STEER:0.0`.

**Tuning constants in `config.py`:**

| Constant         | Default | Effect                                      |
|------------------|---------|---------------------------------------------|
| `STEER_SCALE`    | `0.5`   | Multiplier on rotation delta                |
| `STEER_MAX_ANGLE`| `45.0`  | Hard cap in degrees                         |
| `STEER_DEADZONE` | `2.0`   | Min change (°) before re-sending            |

---

## Inference Guards

- **Debounce** — `DEBOUNCE_FRAMES` (default 5) consecutive frames required before any command fires (~170 ms at 30 fps).
- **Confidence threshold** — `MIN_CONFIDENCE` (default 0.85) minimum `predict_proba` score. Requires `probability=True` on the SVC — retrain after any change.

---

## Handedness

MediaPipe inverts Left/Right labels on mirrored frames. `MIRROR_SWAP_HANDEDNESS = True` (default) corrects this. If hands appear swapped, toggle this flag in `config.py`.

---

## Runtime Detection (`comms/car_interface.py`)

| Flag         | Condition                                          | Backend              |
|--------------|----------------------------------------------------|----------------------|
| `ON_PI`      | `/sys/firmware/devicetree/base/model` exists       | GPIO stub (TODO)     |
| `USE_SOCKET` | `USE_SOCKET=1` env var                             | UDP → `SOCKET_HOST:SOCKET_PORT` |
| fallback     | Mac / other                                        | `print()`            |

Socket target: `SOCKET_HOST` / `SOCKET_PORT` in `config.py` (default `127.0.0.1:5005`).

---

## Webots Controller

`src/others/Webots/` contains e-puck and Pioneer controllers that run inside Webots. Each:

- Listens on UDP and drains all pending packets each timestep
- Maintains `drive_speed` (FORWARD/STOP/BACKWARD) and `steer_angle` (STEER:\<degrees\>) as separate state
- Applies gradual deceleration: `current_speed` ramps toward `drive_speed` by `DECEL_RATE` per timestep
- Combines both into differential wheel velocities

`DECEL_RATE` and `MAX_SPEED` / `STEER_MAX_ANGLE` are duplicated locally in each Webots controller (they cannot import `config.py`). Keep them in sync.

---

## config.py Notes

- `CLASSIFIER_PATH` and `LANDMARKER_PATH` are resolved as absolute paths relative to `src/` via `os.path.abspath(__file__)` — scripts work regardless of working directory.
- `NUM_HANDS = 2` — required for dual-hand detection.
- `CSV_PATHS` is derived from `GESTURES` — add a new gesture name there and a new CSV is automatically targeted.
