# How to Train the Gesture Classifier

## Prerequisites

- Venv activated: `source venv/bin/activate`
- Model file downloaded: `model/hand_landmarker.task` (see README.md)
- Standing in the project root

---

## Step 1 — Collect gesture data

```bash
python src/collect_data.py
```

The webcam opens in mirror mode. MediaPipe draws a white skeleton on your hand when it detects one.

### Gestures to record

| Key | Gesture | How to make it | Command |
|-----|---------|----------------|---------|
| `f` | Fist | Close all fingers tightly into a fist | STOP |
| `p` | Palm | Open hand flat, all fingers spread out | FORWARD |
| `v` | Peace | Index and middle finger up, others folded | BACKWARD |
| `q` | — | Quit the collector | — |

### Tips for good data

- **Aim for ~100 samples per gesture** — the terminal prints a count after each save.
- **Vary your position** — move your hand slightly between presses (distance, angle, tilt). Don't hold perfectly still for all 100.
- **Wait for the skeleton** — if no skeleton is drawn, MediaPipe hasn't detected your hand. Keypresses are ignored when no hand is detected.
- **Good lighting** — face a light source so your hand is well lit.
- **One gesture at a time** — do all fist samples, then all palm samples, then all peace samples to stay focused.

### Suggested collection flow

1. Make a fist → press `f` repeatedly ~100 times with slight hand variation
2. Open your palm → press `p` repeatedly ~100 times
3. Make a peace sign → press `v` repeatedly ~100 times
4. Press `q` to quit

Data is saved to:
```
data/fist.csv
data/palm.csv
data/peace.csv
```

---

## Step 2 — Train the classifier

```bash
python src/train_model.py
```

This reads the three CSV files, trains an SVM classifier with cross-validation, and saves the model.

Expected output:
```
Loaded N samples: ['fist', 'palm', 'peace']
Cross-validation accuracy: 0.XX ± 0.XX
Model saved to model/gesture_classifier.pkl
```

Aim for **cross-validation accuracy above 0.90**. If it's lower, collect more samples or re-collect with better variation.

---

## Step 3 — Test live inference

```bash
python src/run_control.py
```

Make each gesture in front of the camera. The terminal should print:
- `STOP` when you show a fist
- `FORWARD` when you show a palm
- `BACKWARD` when you show a peace sign

Commands only print when the gesture **changes** — holding a gesture steady won't flood the terminal.

Press `q` to quit.

---

## Retraining

If accuracy is poor or you want to improve results:

1. Delete the old CSV files (or keep them to add to):
   ```bash
   rm data/fist.csv data/palm.csv data/peace.csv
   ```
2. Re-run `collect_data.py` and collect fresh samples
3. Re-run `train_model.py`

The model file `model/gesture_classifier.pkl` is overwritten each time you train.