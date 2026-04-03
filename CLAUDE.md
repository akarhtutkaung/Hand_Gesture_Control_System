# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Real-time hand gesture detection on Mac webcam using MediaPipe. Classifies finger gestures (squeeze-in / squeeze-out) and prints motor commands (`FORWARD` / `BACKWARD`) to the terminal. Designed to later transfer to a Raspberry Pi where `send_command()` gets swapped from `print()` to GPIO motor calls.

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
python collect_data.py   # record ~100 samples per gesture (i=squeeze_in, o=squeeze_out, q=quit)
python train_model.py    # train SVM → saves model/gesture_classifier.pkl
python run_control.py    # live inference → prints FORWARD / BACKWARD
```

## Architecture

All scripts share `gesture_utils.py` — implement its three helpers first, everything else depends on them.

| File | Role |
|---|---|
| `gesture_utils.py` | `calc_finger_spread`, `normalize_landmarks`, `draw_overlay` — imported by all other scripts |
| `collect_data.py` | Opens webcam, runs MediaPipe, saves labeled rows to `data/squeeze_in.csv` / `data/squeeze_out.csv` |
| `train_model.py` | Loads CSVs → trains sklearn SVM Pipeline → saves `model/gesture_classifier.pkl` |
| `run_control.py` | Loads model, classifies live frames, calls `send_command()` on gesture change |

**Landmark format:** MediaPipe returns 21 hand landmarks. Each is stored as `(x, y, z)`, giving a flat array of shape `(63,)`. `normalize_landmarks()` in `gesture_utils.py` translates the wrist to origin and scales to `[-1, 1]` before any CSV write or model call.

**Gesture change guard:** `run_control.py` should only call `send_command()` when the gesture changes from the previous frame — avoids flooding the terminal (and later, the motor controller).

**Pi migration point:** `send_command()` in `run_control.py` contains a `# TODO: GPIO` comment. That is the only function that needs to change when moving to the Raspberry Pi.
