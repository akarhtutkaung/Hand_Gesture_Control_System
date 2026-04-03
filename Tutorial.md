# Tutorial — Implementation Guide

This guide tells you exactly which functions to implement, in what order, and why. Follow it top to bottom — each function unlocks the next.

---

## Phase 1 — `gesture_utils.py` (implement first, everything depends on it)

### 1. `normalize_landmarks(landmarks)`

**Why first:** Every other function that touches landmark data calls this. Nothing works correctly without it.

**What to do:**
- Reshape the flat `(63,)` array into `(21, 3)` — 21 points each with x, y, z
- Subtract `points[0]` (the wrist) from every point so the wrist becomes the origin
- Divide everything by the max absolute value so all values sit in `[-1, 1]`
- Flatten back to `(63,)` and return

**Test it:** Pass in a dummy `np.zeros(63)` — it should return all zeros without crashing.

---

### 2. `calc_finger_spread(landmarks)`

**Why second:** Used later for optional rule-based detection. Simpler than the classifier approach.

**What to do:**
- Reshape landmarks to `(21, 3)`, take only x and y (drop z)
- Extract the 5 fingertip rows at indices `[4, 8, 12, 16, 20]`
- Use `itertools.combinations(tips, 2)` to get all 10 pairs
- Compute `np.linalg.norm(a - b)` for each pair
- Return the average

**Test it:** Close fingers → small number. Spread fingers → larger number.

---

### 3. `draw_overlay(frame, gesture)`

**Why third:** Cosmetic only — safe to implement last within this file, but needed before running any webcam script.

**What to do:**
- Pick a color based on the gesture string (e.g. green for `squeeze_in`, orange for `squeeze_out`)
- Call `cv2.putText(frame, f"Gesture: {gesture}", (20, 50), ...)` 
- Return the frame

---

## Phase 2 — `collect_data.py` (build the dataset)

### 4. `open_camera(index)`

**Why next:** The entry point for all webcam work. Must work before anything else in this file.

**What to do:**
- Create `cv2.VideoCapture(index)`
- Check `cap.isOpened()` — raise `RuntimeError` if it fails
- Call `cv2.namedWindow("Collect Data", cv2.WINDOW_NORMAL)` — required on macOS
- Return `cap`

**Test it:** Run the script — a blank window should appear.

---

### 5. `detect_landmarks(frame, hands)`

**Why next:** Core sensing function. `save_sample` depends on its output.

**What to do:**
- Convert frame: `rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
- Run: `result = hands.process(rgb)`
- If `result.multi_hand_landmarks` is empty/None → return `None`
- Extract landmarks from `result.multi_hand_landmarks[0]` as a `(21, 3)` numpy array
- Draw skeleton: `mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)`
- Call `normalize_landmarks()` on the flat array and return it

**Test it:** Run the script — you should see dots and lines on your hand in the window.

---

### 6. `save_sample(landmarks, label)`

**Why next:** Needs a working `detect_landmarks` to have something to save.

**What to do:**
- `os.makedirs("data", exist_ok=True)`
- Open `data/{label}.csv` in append mode
- Write one row: `[label] + landmarks.tolist()`
- Print a count so you know samples are accumulating

**Test it:** Press `i` a few times → check that `data/squeeze_in.csv` exists and has rows.

---

### 7. `main()` in `collect_data.py`

**Why last in this file:** Wires all the above together.

**What to do:**
- Call `open_camera()`
- Initialize `mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)` inside a `with` block
- Loop: `ret, frame = cap.read()`
- Call `detect_landmarks(frame, hands)` → store result
- Call `draw_overlay(frame, current_gesture)`
- Add a `cv2.putText` hint: `"i=squeeze_in  o=squeeze_out  q=quit"`
- `cv2.imshow(...)` then `cv2.waitKey(1)`
- Handle keys: `i` → `save_sample(..., "squeeze_in")`, `o` → `save_sample(..., "squeeze_out")`, `q` → break
- After loop: `cap.release()`, `cv2.destroyAllWindows()`

**Done when:** You have ~100 rows in each CSV.

---

## Phase 3 — `train_model.py` (train the classifier)

### 8. `load_dataset()`

**What to do:**
- Loop over `["data/squeeze_in.csv", "data/squeeze_out.csv"]`
- Skip missing files with a warning
- For each row: `label = row[0]`, `features = list(map(float, row[1:]))`
- Return `np.array(X, dtype=np.float32)`, `np.array(y)`

---

### 9. `preprocess(X)`

**What to do:**
- For now, just `return X` — normalization already happened in `normalize_landmarks`
- Add extra steps here later if needed

---

### 10. `train_classifier(X, y)`

**What to do:**
- Build a Pipeline: `[("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10, gamma="scale"))]`
- Run `cross_val_score(model, X, y, cv=5)` and print the mean accuracy
- Fit the pipeline on all data
- Return the fitted pipeline

**Expect:** Accuracy should be above 90% with 100+ samples per class.

---

### 11. `save_model(model)`

**What to do:**
- `os.makedirs("model", exist_ok=True)`
- `joblib.dump(model, "model/gesture_classifier.pkl")`
- Print confirmation

---

### 12. `main()` in `train_model.py`

**What to do:**
- Call `load_dataset()` → print shape and class names
- Call `preprocess(X)`
- Call `train_classifier(X, y)`
- Call `save_model(model)`

---

## Phase 4 — `run_control.py` (live inference)

### 13. `load_model()`

**What to do:**
- `joblib.load("model/gesture_classifier.pkl")`
- Raise a clear `FileNotFoundError` with a message like `"Run train_model.py first"`
- Return the model

---

### 14. `detect_gesture(frame, hands)`

**What to do:**
- Same logic as `detect_landmarks` in `collect_data.py` — convert to RGB, run MediaPipe, draw skeleton, normalize, return array or `None`
- You can copy the logic directly; this is the inference-time version

---

### 15. `classify_gesture(landmarks, model)`

**What to do:**
- Reshape to `(1, 63)`: `landmarks.reshape(1, -1)`
- Call `model.predict(...)` → returns array with one label string
- Return `result[0]`

---

### 16. `send_command(gesture)`

**What to do:**
- Map `"squeeze_in"` → `print("FORWARD")`
- Map `"squeeze_out"` → `print("BACKWARD")`
- Add comment: `# TODO: GPIO — replace print() with motor calls`

---

### 17. `main()` in `run_control.py`

**What to do:**
- Call `load_model()`
- Open `cv2.VideoCapture(0)`
- Initialize MediaPipe Hands
- Track `last_gesture = None`
- Loop: read frame → `detect_gesture()` → `classify_gesture()` → if gesture != `last_gesture`: call `send_command()` and update `last_gesture`
- Call `draw_overlay()` and `cv2.imshow()`
- Break on `q`
- Release and destroy on exit

---

## Summary Order

```
gesture_utils.py
  1. normalize_landmarks
  2. calc_finger_spread
  3. draw_overlay

collect_data.py
  4. open_camera
  5. detect_landmarks
  6. save_sample
  7. main  → run it, collect data

train_model.py
  8.  load_dataset
  9.  preprocess
  10. train_classifier
  11. save_model
  12. main  → run it, get the .pkl

run_control.py
  13. load_model
  14. detect_gesture
  15. classify_gesture
  16. send_command
  17. main  → run it, see FORWARD / BACKWARD
```
