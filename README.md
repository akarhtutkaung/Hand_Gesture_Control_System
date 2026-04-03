# Gesture Car — Hand Gesture Control System

A real-time hand gesture detection system using a Mac webcam. Detects finger gestures and prints motor commands to the terminal. Built to later transfer to a Raspberry Pi to control a physical robot car via GPIO.

---

## What It Uses

- **MediaPipe HandLandmarker** — 21-point hand landmark detection via the Tasks API
- **OpenCV** — webcam capture and frame display
- **scikit-learn** — SVM classifier trained on recorded gesture samples
- **NumPy** — landmark array manipulation

---

## Project Structure

```
Hand_Gesture_Control_System/
├── src/
│   ├── gesture_utils.py      ← shared helpers (landmark normalization, spread calc, overlay)
│   ├── collect_data.py       ← record labeled gesture samples to CSV
│   ├── train_model.py        ← train SVM classifier on collected data
│   └── run_control.py        ← live inference, prints FORWARD / BACKWARD
├── data/
│   ├── squeeze_in.csv        ← auto-generated when collecting
│   └── squeeze_out.csv       ← auto-generated when collecting
├── model/
│   ├── hand_landmarker.task  ← download once (see Setup below)
│   └── gesture_classifier.pkl  ← auto-generated after training
├── requirements.txt
└── Plan.md                   ← detailed function-level implementation guide
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data model
```

Download the MediaPipe hand landmark model (required — one time only):

```bash
curl -L -o model/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

macOS will prompt for camera permission on first run — click Allow.

---

## How to Run

Run all commands from the project root.

### Step 1 — Collect gesture samples

```bash
python src/collect_data.py
```

| Key | Action |
|-----|--------|
| `i` | Save current frame as **squeeze in** |
| `o` | Save current frame as **squeeze out** |
| `q` | Quit |

Aim for ~100 samples per gesture. Data saves automatically to `data/`.

### Step 2 — Train the classifier

```bash
python src/train_model.py
```

Reads `data/squeeze_in.csv` and `data/squeeze_out.csv`, trains an SVM, and saves the model to `model/gesture_classifier.pkl`. Prints cross-validation accuracy.

### Step 3 — Run live inference

```bash
python src/run_control.py
```

Opens the webcam, classifies each frame, and prints `FORWARD` or `BACKWARD` to the terminal whenever the gesture changes. Press `q` to quit.

---

## Raspberry Pi Migration

When the robot kit arrives, only one function needs to change: `send_command()` in `src/run_control.py`. Replace the `print()` calls with GPIO motor commands.

```bash
# On the Pi
pip install -r requirements.txt
pip install RPi.GPIO   # or gpiozero
```

Everything else (landmark detection, model, gesture logic) stays exactly the same.
