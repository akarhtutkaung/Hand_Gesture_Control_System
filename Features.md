# Features

## Summary

| Feature | Summary | Implemented |
|---|---|:---:|
| [Gesture-Based Motor Control](#gesture-based-motor-control) | Control the robot forward, backward, and stop using three hand gestures via webcam. | Yes |
| [Person Registration](#person-registration) | Scan and remember a specific person's appearance using a thumbs-up gesture. | No |
| [Follow Mode](#follow-mode) | After a follow gesture, the robot autonomously trails the registered target person. | No |
| [Obstacle Avoidance](#obstacle-avoidance) | Detect obstacles in the robot's path and autonomously navigate around them. | No |
| [Live Video Preview](#live-video-preview) | Stream the robot's camera feed in real time to a cloud endpoint for remote viewing. | No |
| [Remote Control Interface](#remote-control-interface) | Control the robot manually via a web or mobile interface without needing hand gestures. | No |

---

## Existing Features

### Gesture-Based Motor Control
Control a robotic vehicle with three hand gestures detected in real time via webcam.

| Gesture | Command | Description |
|---|---|---|
| Palm (open hand) | `FORWARD` | Robot moves forward |
| Fist (closed hand) | `STOP` | Robot halts |
| Peace sign (two fingers) | `BACKWARD` | Robot moves backward |

**How it works:**
- MediaPipe `HandLandmarker` extracts 21 3D hand landmarks per frame
- Landmarks are normalized (wrist at origin, scaled to `[-1, 1]`)
- An SVM classifier (sklearn Pipeline with StandardScaler) predicts the gesture label
- Commands are sent only on gesture change to avoid flooding the motor controller

**Inference guards:**
- **Confidence threshold** — predictions below `MIN_CONFIDENCE` (default 0.85) are discarded
- **Debounce** — the same gesture must hold for `DEBOUNCE_FRAMES` (default 5) consecutive frames before a command fires (~170 ms at 30 fps)

**Pi migration path:** `send_command()` in `run_control.py` contains a `# TODO: GPIO` marker. Swap `print()` for RPi.GPIO / gpiozero motor calls to deploy on hardware.

---

## Planned Features

### Person Registration

**Overview:**
The user shows a thumbs-up gesture while standing in front of the camera. The system captures their appearance, builds a compact embedding, and stores it as the target profile for Follow Mode.

**Registration flow:**
1. User shows a thumbs-up gesture while standing in front of the camera.
2. The system detects the person's bounding box and captures their appearance — body proportions and optionally a face embedding.
3. The appearance is compressed into a feature vector and saved as the target profile.
4. A confirmation overlay (e.g., green border flash) signals successful registration.

**Key technical components:**

| Component | Role |
|---|---|
| Person detector | Detect the human bounding box at registration time (e.g., MediaPipe Pose, YOLOv8-nano) |
| Appearance embedding | Compact feature vector built from the captured bounding box region |
| Profile storage | In-memory or on-disk store for the target embedding, loaded on startup |

**New gesture needed:**

| Gesture | Action |
|---|---|
| Thumbs-up | Register the person currently in frame as the target |

**Config additions planned (`config.py`):**
```python
REGISTER_GESTURE          = "thumbs_up"
REID_SIMILARITY_THRESHOLD = 0.75  # cosine similarity cutoff for re-identification
```

---

### Follow Mode

**Overview:**
After a wave gesture, the robot switches from manual control to autonomous tracking and continuously trails the registered target person.

**Follow Mode flow:**
1. User shows a wave gesture (or open-palm hold for 1 second).
2. The robot switches from manual gesture control to autonomous tracking.
3. A re-identification model continuously locates the registered person in the frame by comparing live embeddings against the stored target profile.
4. Motor commands are derived from the person's position relative to the frame center:
   - Person centered → `FORWARD` (maintain distance)
   - Person left/right of center → steer to re-center
   - Person too close → `STOP`
   - Person lost from frame → `STOP` and rotate to search
5. Fist gesture exits Follow Mode and returns to manual control.

**Key technical components:**

| Component | Role |
|---|---|
| Re-identification | Match live frames against the stored target embedding each frame |
| Steering logic | PID or proportional controller mapping horizontal offset → left/right motor differential |
| Mode state machine | `MANUAL` → `FOLLOWING` → `MANUAL` |

**New gestures needed:**

| Gesture | Action |
|---|---|
| Wave (or open-palm hold) | Enter Follow Mode |
| Fist | Exit Follow Mode → return to manual |

**Config additions planned (`config.py`):**
```python
FOLLOW_MODE_GESTURE    = "wave"
FOLLOW_CENTER_DEADZONE = 0.1   # fraction of frame width — no steer within this band
FOLLOW_STOP_DISTANCE   = 0.4   # bounding-box height fraction — stop if person is this close
```

**Pi migration path:** Steering differential maps directly to two independent motor channels. The same `send_command()` abstraction can be extended with `LEFT` / `RIGHT` / `SEARCH` commands.

---

### Obstacle Avoidance

**Overview:**
While the robot is moving (manually or in Follow Mode), it detects physical obstacles in its path and autonomously steers around them before resuming its original direction or target.

**Avoidance flow:**
1. Distance sensor(s) or depth camera continuously measures proximity in front of the robot.
2. When an obstacle is detected within `OBSTACLE_STOP_DISTANCE`, the robot halts.
3. The robot scans left and right to find the clearer side.
4. It steers around the obstacle with a fixed turn-and-advance maneuver.
5. Once clear, it resumes the previous command (`FORWARD` or Follow Mode tracking).

**Key technical components:**

| Component | Role |
|---|---|
| Proximity sensor | Ultrasonic sensor (e.g., HC-SR04) or depth camera measuring forward clearance |
| Obstacle detection | Threshold check on sensor reading each control loop tick |
| Avoidance planner | Simple reactive logic — stop, scan, choose clear side, arc around |
| State machine | Adds `AVOIDING` state: `MANUAL/FOLLOWING` → `AVOIDING` → `MANUAL/FOLLOWING` |

**Config additions planned (`config.py`):**
```python
OBSTACLE_STOP_DISTANCE   = 30    # cm — halt and begin avoidance below this distance
OBSTACLE_CLEAR_DISTANCE  = 50    # cm — resume forward motion once clearance exceeds this
AVOIDANCE_TURN_DURATION  = 0.6   # seconds — how long to arc during the go-around maneuver
```

**Pi migration path:** Sensor reads map to GPIO input pins (HC-SR04 trigger/echo) or a serial depth camera feed. The `AVOIDING` state emits existing `LEFT` / `RIGHT` / `FORWARD` / `STOP` commands, so no new motor interface is needed.

---

### Live Video Preview

**Overview:**
The robot's camera feed is encoded and streamed in real time to a cloud endpoint, allowing remote viewers to monitor what the robot sees from any device with a browser.

**Streaming flow:**
1. The Pi captures frames from the camera in the main control loop.
2. Each frame is JPEG-encoded and pushed to a cloud relay (e.g., WebRTC, RTMP, or an MJPEG-over-HTTP server).
3. A lightweight cloud service (e.g., AWS Kinesis Video Streams, a self-hosted WebRTC signalling server, or a simple Flask MJPEG endpoint behind a public URL) receives and redistributes the stream.
4. Remote viewers open the stream URL in a browser — no app install required.
5. Streaming runs as a background thread and does not block the gesture or motor control loop.

**Key technical components:**

| Component | Role |
|---|---|
| Frame encoder | Compress raw BGR frames to JPEG before transmission to reduce bandwidth |
| Streaming transport | WebRTC (low latency), RTMP (broad compatibility), or MJPEG-over-HTTP (simplest) |
| Cloud relay | Receives and redistributes the stream to viewer clients |
| Background thread | Runs the stream sender independently from the main control loop |
| Viewer endpoint | Public URL or dashboard where the stream can be watched remotely |

**Config additions planned (`config.py`):**
```python
STREAM_ENABLED      = True
STREAM_FPS          = 15       # frames per second to encode and push
STREAM_JPEG_QUALITY = 70       # JPEG compression quality (0–100)
STREAM_ENDPOINT_URL = ""       # cloud relay URL — set via environment variable in production
```

**Pi migration path:** Frame capture already happens in the main loop. A background `threading.Thread` reads from the same `cv2.VideoCapture`, encodes to JPEG, and pushes to the endpoint without interfering with motor commands.

---

### Remote Control Interface

**Overview:**
A web or mobile interface that lets a user send motor commands to the robot manually — forward, backward, stop, and steer — without needing hand gestures or physical proximity.

**Interface options (pick one):**

| Option | Pros | Cons |
|---|---|---|
| Web dashboard (browser) | No install, works on any device, easy to combine with live video preview | Requires a running web server on the Pi or cloud |
| Mobile app (React Native / Flutter) | Native feel, on-screen D-pad, haptic feedback | Requires build and install step |
| Gamepad / physical controller | Tactile, low latency, familiar UX | Requires USB or Bluetooth pairing |

**Control flow:**
1. A lightweight HTTP or WebSocket server runs on the Pi (e.g., Flask or FastAPI).
2. The interface sends button or joystick events as command messages (`FORWARD`, `BACKWARD`, `STOP`, `LEFT`, `RIGHT`).
3. The server receives the command and calls the same `send_command()` function used by gesture control.
4. The active control source (gesture vs. remote) is tracked by a mode flag — only one is active at a time.
5. An emergency stop button is always visible regardless of mode.

**Key technical components:**

| Component | Role |
|---|---|
| HTTP / WebSocket server | Receives command events from the interface (e.g., Flask-SocketIO, FastAPI WebSocket) |
| Command router | Passes incoming remote commands to `send_command()`, respects active mode |
| Web / mobile UI | D-pad or joystick buttons that emit commands on press and stop on release |
| Mode flag | Prevents gesture control and remote control from conflicting |
| Auth layer | Basic token or password check so the robot isn't open to the public internet |

**Config additions planned (`config.py`):**
```python
REMOTE_CONTROL_ENABLED = True
REMOTE_SERVER_HOST     = "0.0.0.0"
REMOTE_SERVER_PORT     = 5000
REMOTE_AUTH_TOKEN      = ""    # set via environment variable — never hardcode
```

**Pi migration path:** Flask or FastAPI runs as a background thread on the Pi alongside the main control loop. `send_command()` is already the single integration point, so remote commands slot in with no motor-layer changes.
