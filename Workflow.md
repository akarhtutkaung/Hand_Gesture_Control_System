# Workflow — System Diagram

---

## Big Picture — 3 Phases

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   collect_data.py   │     │   train_model.py    │     │   run_control.py    │
│                     │     │                     │     │                     │
│  Webcam → CSV       │────▶│  CSV → .pkl model   │────▶│  Webcam → Command   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
       Phase 1                      Phase 2                      Phase 3
   (run once to collect)       (run once to train)          (run live)
```

---

## Phase 1 — `collect_data.py` Call Flow

```
main()
 │
 ├── open_camera()
 │     └── cv2.VideoCapture(CAMERA_INDEX)
 │
 ├── build_landmarker()
 │     └── HandLandmarker (VIDEO mode, num_hands=2)
 │
 └── [webcam loop]
       │
       ├── cap.read()                          ← grab frame
       ├── cv2.flip()                          ← mirror if MIRROR_MODE
       │
       ├── detect_landmarks(frame, landmarker, timestamp_ms)
       │     ├── cv2.cvtColor BGR→RGB
       │     ├── landmarker.detect_for_video() ← MediaPipe Tasks inference
       │     ├── draw_landmarks()              ← draw skeleton on frame
       │     └── normalize_landmarks()         ← gesture_utils.py
       │           ├── reshape (63,) → (21, 3)
       │           ├── subtract wrist (origin)
       │           └── scale to [-1, 1]
       │
       ├── draw_overlay(frame, label)          ← gesture_utils.py
       ├── cv2.imshow()
       │
       └── [keypress handler]                  ← COLLECT_KEYS in config.py
             ├── f → save_sample(landmarks, "fist")
             ├── p → save_sample(landmarks, "palm")
             ├── v → save_sample(landmarks, "peace")
             └── q → break
                   │
                   └── save_sample()
                         └── append row to data/{label}.csv
```

**Output:** `data/fist.csv`, `data/palm.csv`, `data/peace.csv`

---

## Phase 2 — `train_model.py` Call Flow

```
main()
 │
 ├── load_dataset()
 │     ├── read data/fist.csv
 │     ├── read data/palm.csv
 │     ├── read data/peace.csv
 │     └── return X (n, 63), y (n,)
 │
 ├── preprocess(X)
 │     └── passthrough — normalization already applied at collection time
 │
 ├── train_classifier(X, y)
 │     ├── build Pipeline: StandardScaler → SVC(kernel=rbf, probability=True)
 │     ├── cross_val_score(cv=5) → print accuracy
 │     └── pipeline.fit(X, y)
 │
 └── save_model(pipeline)
       └── joblib.dump → model/gesture_classifier.pkl
```

**Output:** `model/gesture_classifier.pkl`

---

## Phase 3 — `run_control.py` Call Flow

```
main()
 │
 ├── load_model()             → gesture_classifier.pkl
 ├── build_landmarker()       → HandLandmarker (VIDEO mode, num_hands=2)
 ├── cv2.VideoCapture(CAMERA_INDEX)
 │
 ├── make_left_state()        → {pending, debounce, prev}
 ├── make_right_state()       → {fist_frames, active, origin_angle, prev_angle}
 │
 └── [webcam loop]
       │
       ├── cap.read() + flip
       │
       ├── detect_all_hands(frame, landmarker, timestamp_ms)
       │     ├── landmarker.detect_for_video()
       │     ├── correct handedness (MIRROR_SWAP_HANDEDNESS)
       │     └── returns {"Left": (norm_lm, raw), "Right": (...)}
       │
       ├── [Left hand present?]
       │     ├── YES → process_left_hand()
       │     │           ├── classify_gesture()        ← SVM predict_proba
       │     │           ├── confidence gate           ← MIN_CONFIDENCE (0.85)
       │     │           ├── debounce                  ← DEBOUNCE_FRAMES (5)
       │     │           ├── draw skeleton (color by gesture)
       │     │           └── send_command() on gesture change
       │     │                 ├── palm  → "FORWARD"
       │     │                 ├── fist  → "STOP"
       │     │                 └── peace → "BACKWARD"
       │     │
       │     └── NO  → reset_left_hand() → send_command("fist") → "STOP"
       │
       └── [Right hand present?]
             ├── YES → process_right_hand()
             │           ├── classify_gesture()        ← SVM predict_proba
             │           ├── [Stage 1 — activation]
             │           │     └── hold fist ≥ DEBOUNCE_FRAMES → capture origin_angle
             │           ├── [Stage 2 — continuous steering]
             │           │     ├── calc_hand_angle()   ← atan2(wrist→mid-MCP)
             │           │     ├── calc_steer_angle()  ← delta × STEER_SCALE, clamped
             │           │     └── send_steer(angle)   ← "STEER:<degrees>"
             │           └── draw skeleton (yellow = steering active)
             │
             └── NO  → reset_right_hand() → send_steer(0.0)
```

---

## `send_command()` / `send_steer()` — Runtime Routing

```
send_command(gesture) / send_steer(angle)
 │
 ├── ON_PI?     (/sys/firmware/devicetree/base/model exists)
 │     └── TODO: GPIO motor calls
 │
 ├── USE_SOCKET=1?  (environment variable)
 │     └── UDP → SOCKET_HOST:SOCKET_PORT (127.0.0.1:5005)
 │                          │
 │                          ▼
 │               gesture_controller.py  (Webots)
 │                 listens on port 5005, drives robot motors
 │
 └── fallback (Mac / other)
       └── print(f"Command: {command}")
```

---

## Webots Controller — `gesture_controller.py`

```
[Webots simulation loop]
 │
 ├── sock.recvfrom()          ← drain all UDP packets each timestep
 │     ├── "FORWARD"          → drive_speed = +MAX_SPEED
 │     ├── "BACKWARD"         → drive_speed = -MAX_SPEED
 │     ├── "STOP"             → drive_speed = 0.0
 │     └── "STEER:<degrees>"  → steer_angle = float(degrees)
 │
 ├── ramp current_speed → drive_speed  (by DECEL_RATE per step)
 ├── ramp current_steer → steer_angle  (by STEER_RATE per step)
 │
 ├── steer_factor = current_steer / STEER_MAX_ANGLE   ∈ [-1, 1]
 ├── left_v  = current_speed × (1 + steer_factor)
 ├── right_v = current_speed × (1 − steer_factor)
 ├── clamp both to ±MAX_SPEED
 │
 └── motor.setVelocity(left_v / right_v)
```

---

## Shared Dependencies — `gesture_utils.py`

```
gesture_utils.py
       │
       ├── normalize_landmarks(landmarks: ndarray) → ndarray (63,)
       │     called by: detect_landmarks()   [collect_data.py]
       │                detect_all_hands()   [run_control.py]
       │
       ├── calc_finger_spread(landmarks) → float
       │     optional — rule-based spread metric (not used by SVM pipeline)
       │
       └── draw_overlay(frame, gesture)
             called by: main()  [collect_data.py]
```

---

## Data Flow — End to End

```
Webcam frame
     │
     ▼
MediaPipe HandLandmarker (Tasks API, VIDEO mode)
     │
     ▼
21 landmark points (x, y, z) per hand  ─────────────── shape: (21, 3)
     │
     ▼
normalize_landmarks()
  ├── translate wrist to origin
  └── scale to [-1, 1]
     │
     ▼
flat normalized array  ──────────────────────────────── shape: (63,)
     │
     ├──[collect phase]──▶  data/{label}.csv  ──▶  SVM Pipeline  ──▶  .pkl
     │
     └──[inference phase]
           │
           ├── LEFT HAND  → SVM predict_proba → debounce → send_command()
           │                                                      │
           │                                              "FORWARD" / "STOP" / "BACKWARD"
           │
           └── RIGHT HAND → SVM (fist gate) → atan2 angle delta → send_steer()
                                                                         │
                                                                 "STEER:<degrees>"
```

---

## MediaPipe — 21 Landmark Indices

```
                8   12  16  20
                |   |   |   |
                7   11  15  19
                |   |   |   |
                6   10  14  18
                |   |   |   |
            4   5   9   13  17
            |   |
            3   |
            |   |
            2   |
             \  |
              \ |
               \|
        1───────0  (wrist)
```

Fingertip indices: `[4, 8, 12, 16, 20]`  
Steering vector: wrist `(0)` → middle-finger MCP `(9)`
