# Workflow — System Diagram

---

## Big Picture — 3 Phases

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   collect_data.py   │     │   train_model.py     │     │   run_control.py    │
│                     │     │                      │     │                     │
│  Webcam → CSV       │────▶│  CSV → .pkl model    │────▶│  Webcam → Command   │
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
 │     └── cv2.VideoCapture(0)
 │
 ├── mp.solutions.hands.Hands(...)          ← MediaPipe init
 │
 └── [webcam loop]
       │
       ├── cap.read()                       ← grab frame
       │
       ├── detect_landmarks(frame, hands)
       │     ├── cv2.cvtColor BGR→RGB
       │     ├── hands.process(rgb)         ← MediaPipe inference
       │     ├── mp_draw.draw_landmarks()   ← draw skeleton on frame
       │     └── normalize_landmarks()      ← gesture_utils.py
       │           ├── reshape (63,) → (21,3)
       │           ├── subtract wrist (origin)
       │           └── scale to [-1, 1]
       │
       ├── draw_overlay(frame, gesture)     ← gesture_utils.py
       │
       ├── cv2.imshow()
       │
       └── [keypress handler]
             ├── i → save_sample(landmarks, "squeeze_in")
             ├── o → save_sample(landmarks, "squeeze_out")
             └── q → break
                   │
                   └── save_sample()
                         └── append row to data/{label}.csv
```

**Output:** `data/squeeze_in.csv`, `data/squeeze_out.csv`

---

## Phase 2 — `train_model.py` Call Flow

```
main()
 │
 ├── load_dataset()
 │     ├── read data/squeeze_in.csv
 │     ├── read data/squeeze_out.csv
 │     └── return X (n, 63), y (n,)
 │
 ├── preprocess(X)
 │     └── return X  (passthrough for now)
 │
 ├── train_classifier(X, y)
 │     ├── build Pipeline: StandardScaler → SVC(rbf)
 │     ├── cross_val_score(cv=5) → print accuracy
 │     └── model.fit(X, y)
 │
 └── save_model(model)
       └── joblib.dump → model/gesture_classifier.pkl
```

**Output:** `model/gesture_classifier.pkl`

---

## Phase 3 — `run_control.py` Call Flow

```
main()
 │
 ├── load_model()
 │     └── joblib.load("model/gesture_classifier.pkl")
 │
 ├── cv2.VideoCapture(0)
 ├── mp.solutions.hands.Hands(...)
 │
 └── [webcam loop]
       │
       ├── cap.read()                       ← grab frame
       │
       ├── detect_gesture(frame, hands)
       │     ├── cv2.cvtColor BGR→RGB
       │     ├── hands.process(rgb)         ← MediaPipe inference
       │     ├── mp_draw.draw_landmarks()
       │     └── normalize_landmarks()      ← gesture_utils.py
       │
       ├── classify_gesture(landmarks, model)
       │     ├── landmarks.reshape(1, 63)
       │     └── model.predict()            ← SVM inference
       │           └── returns "squeeze_in" or "squeeze_out"
       │
       ├── [gesture changed?]
       │     └── send_command(gesture)
       │           ├── "squeeze_in"  → print("FORWARD")
       │           └── "squeeze_out" → print("BACKWARD")
       │                               # TODO: GPIO motor calls here
       │
       ├── draw_overlay(frame, gesture)     ← gesture_utils.py
       ├── cv2.imshow()
       └── q → break
```

---

## Shared Dependencies — `gesture_utils.py`

```
gesture_utils.py
       │
       ├── normalize_landmarks()
       │     called by: detect_landmarks()  [collect_data.py]
       │                detect_gesture()    [run_control.py]
       │
       ├── calc_finger_spread()
       │     optional — can be used for rule-based detection
       │     instead of the SVM classifier
       │
       └── draw_overlay()
             called by: main()  [collect_data.py]
                        main()  [run_control.py]
```

---

## Data Flow — End to End

```
Webcam frame
     │
     ▼
MediaPipe Hands
     │
     ▼
21 landmark points (x, y, z)  ──────────────── shape: (21, 3)
     │
     ▼
normalize_landmarks()
     │
     ▼
flat normalized array  ─────────────────────── shape: (63,)
     │
     ├──[collect phase]──▶  save to CSV  ──▶  train SVM  ──▶  .pkl
     │
     └──[inference phase]─▶  model.predict()  ──▶  label  ──▶  send_command()
                                                                      │
                                                               print FORWARD
                                                               print BACKWARD
                                                               (or GPIO later)
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

Fingertip indices used by `calc_finger_spread`: `[4, 8, 12, 16, 20]`
