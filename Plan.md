# Gesture Car — Mac Setup Guide

## File Structure

```
Computer_Vision/
├── collect_data.py       ← Step 1: record hand samples
├── train_model.py        ← Step 2: train the classifier
├── run_control.py        ← Step 3: live inference + commands
├── gesture_utils.py      ← shared helper functions
├── data/
│   ├── squeeze_in.csv    ← auto-generated when collecting
│   └── squeeze_out.csv   ← auto-generated when collecting
├── model/
│   └── gesture_classifier.pkl  ← auto-generated after training
├── requirements.txt
└── Plan.md
```

---

## Step 1 — Set Up the Environment

Open Terminal and run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python scikit-learn numpy
mkdir data model
```

Create `requirements.txt`:

```
mediapipe
opencv-python
scikit-learn
numpy
```

> **Apple Silicon (M1/M2/M3)?** Use `pip install mediapipe-silicon` instead.

macOS will ask for camera permission the first time OpenCV runs. Click Allow.

If the camera window doesn't appear, add this near the top of any script:

```python
import cv2
cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
```

---

## Step 2 — What to Connect on Mac

Just your **built-in webcam** — nothing else needed at this stage.

---

## Step 3 — The 4 Scripts

### `gesture_utils.py` — shared helpers (build this first)

Everything else imports from here.

| Function | What it does |
|---|---|
| `calc_finger_spread(landmarks)` | Calculates average distance between fingertips |
| `normalize_landmarks(landmarks)` | Scales landmarks to 0–1 range for consistency |
| `draw_overlay(frame, gesture)` | Draws the detected gesture label on the video frame |

---

### `collect_data.py` — record training samples

Opens your webcam, detects hand landmarks, saves them to CSV when you press a key.

| Function | What it does |
|---|---|
| `open_camera()` | Opens webcam with OpenCV |
| `detect_landmarks(frame)` | Runs MediaPipe on a frame, returns 21 landmark points |
| `save_sample(landmarks, label)` | Appends one row to the correct CSV file |

**How to use:**
- Press `i` to save a "squeeze in" sample
- Press `o` to save a "squeeze out" sample
- Press `q` to quit
- Aim for ~100 samples of each gesture

---

### `train_model.py` — train the classifier

Reads the CSVs, trains a simple SVM classifier, saves the model.

| Function | What it does |
|---|---|
| `load_dataset()` | Reads both CSVs, returns X (features) and y (labels) |
| `preprocess(data)` | Normalizes feature values |
| `train_classifier(X, y)` | Trains an SVM, prints accuracy |
| `save_model(model)` | Saves to `model/gesture_classifier.pkl` |

---

### `run_control.py` — live inference

Loads the saved model, classifies each webcam frame in real time, prints the command.

| Function | What it does |
|---|---|
| `load_model()` | Loads `gesture_classifier.pkl` |
| `detect_gesture(frame)` | Gets landmarks from the current frame |
| `classify_gesture(landmarks)` | Runs the model, returns `"squeeze_in"` or `"squeeze_out"` |
| `send_command(gesture)` | Prints `FORWARD` or `BACKWARD` ← **Pi GPIO goes here later** |

The `send_command()` function will have a `# TODO: GPIO` comment as a placeholder for when you move to the Raspberry Pi.

---

## Step 4 — Run Order

```
1. python collect_data.py    → fills data/ folder
2. python train_model.py     → creates model/gesture_classifier.pkl
3. python run_control.py     → live camera + terminal output
4. (later) transfer to Pi + swap print() for GPIO
```

---

## Pi Migration Checklist (for later)

When your robot kit arrives:

- [ ] Copy the entire `Computer_Vision/` folder to the Pi
- [ ] Run `pip install -r requirements.txt` on the Pi
- [ ] Install `pip install RPi.GPIO` or `gpiozero`
- [ ] In `run_control.py`, replace the `send_command()` print statements with GPIO motor calls
- [ ] Connect the Pi camera or USB webcam
- [ ] Done — everything else stays exactly the same
