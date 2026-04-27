# SafeSwim Project

Real-time drowning-risk monitoring pipeline with:
- video ingestion from file
- human detection + tracking
- drowning-risk classification
- live dashboard (frontend) connected to backend via Socket.IO

---

## 1) Project Overview

SafeSwim processes a video stream frame-by-frame and displays:
- tracked swimmers
- risk signals (time below danger line, motion cues)
- classifier output (`OK` / `DANGER`, probability)
- live alerts in a dashboard UI

End-to-end flow:

`Video path (UI) -> Backend pipeline -> Detection/Tracking/Classification -> Socket events -> Frontend live render`

---

## 2) Repository Structure

- `backend/`
  - `server.py`: Flask-SocketIO server, pipeline orchestration, event API
- `frontend/`
  - React + Vite dashboard
  - `src/hooks/useWebSocket.js`: socket connection + commands
  - `src/components/VideoFeed.jsx`: frame rendering + overlays
- `src/`
  - `classification.py`: model loading + inference logic
  - `detection/`: detector + tracking modules
  - `preprocessing/`: frame preprocessing
- `models/`
  - `drowning_model.pkl`: serialized classifier package
- `data/`
  - demo/test videos (e.g. `swim_test.mp4`, `drowning_test.mp4`)

---

## 3) Current Architecture

### Backend

Backend uses:
- Flask
- Flask-SocketIO (`threading` mode for Windows stability)
- OpenCV for video processing

Main pipeline in `backend/server.py`:
1. Open video (`cv2.VideoCapture`)
2. Preprocess frame
3. Detect persons
4. Track persons across frames
5. Classify drowning risk per tracked person
6. Emit frame + metadata + alerts to frontend

### Frontend

Frontend uses:
- React + Vite
- `socket.io-client`
- `<img>` for decoded frame + `<canvas>` overlay for boxes/labels/line

Layout:
- Left panel: video + controls + Y chart
- Right panel (fixed): swimmers + alert log

---

## 4) Frontend <-> Backend Integration Contract

### Client -> Server events

- `start_pipeline` with payload:
  - `{ "video": "data/drowning_test.mp4" }`
- `stop_pipeline`
- `set_paused` with payload `{ "paused": true|false }`
- `set_speed` with payload `{ "speed": 0.3|0.5|1.0|2.0 }`

### Server -> Client events

- `status`:
  - running/paused/speed state
- `frame`:
  - base64 frame image
  - frame index, fps, session time
  - tracked persons list
  - active alerts
- `new_alert`:
  - alert/error/warning messages
- `pipeline_started`
- `pipeline_stopped`

---

## 5) Environment and Startup

## Prerequisites

- Python (use project `venv`)
- Node.js + npm
- Windows tested

## Backend setup

From project root:

```powershell
.\venv\Scripts\python.exe backend\server.py --port 5001
```

Backend health check:

```powershell
Invoke-WebRequest http://127.0.0.1:5001/api/status
```

## Frontend setup

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 3001
```

Open:
- Frontend: `http://127.0.0.1:3001`
- Backend: `http://127.0.0.1:5001`

---

## 6) Model and Inference

`src/classification.py` loads model from:
- `models/drowning_model.pkl`

Output per tracked person:
- `status`: `OK` or `DANGER`
- `proba`: risk probability (%)
- temporal confirmation counter

Important:
- Use exact filename `drowning_model.pkl` (no extra spaces)
- Bounding boxes are clamped to frame bounds before crop

---

## 7) Video Input Paths

Examples accepted by UI:
- `data/swim_test.mp4`
- `data/drowning_test.mp4`
- `data/drown.test.mp4`

Absolute path also works:
- `C:\Users\admin\Desktop\projet_CV\data\...`

---

## 8) Stability Improvements Already Applied

- Socket backend switched to `threading` mode (instead of eventlet)
- Reconnection-friendly frontend socket settings
- Safer pipeline lifecycle (`try/except/finally`)
- Frame-level error resilience (single frame failure does not kill run)
- Relative video path normalization from project root
- Alert transition deduplication (reduces event spam)
- Adaptive pacing in frame loop
- Catch-up mechanism to avoid slow-motion drift when processing lags

---

## 9) Performance / Quality Tuning

In `backend/server.py`:

- `STREAM_W`, `STREAM_H`
  - stream resolution sent to frontend
- `STREAM_JPEG_QUALITY`
  - encoded frame quality/size tradeoff
- `DETECT_EVERY_N_FRAMES`
  - run detector less often for speed
- `CLASSIFY_EVERY_N_FRAMES`
  - run classifier less often for speed
- `MAX_CATCHUP_GRABS`
  - maximum frames dropped to re-sync with real-time

Guidance:
- smoother playback: lower resolution/quality, higher skip cadence
- better visual quality: increase resolution/quality (higher CPU/network cost)

---

## 10) Troubleshooting

### Frontend shows DISCONNECTED

1. Check backend endpoint:
   - `http://127.0.0.1:5001/api/status`
2. Ensure frontend socket URL points to `http://localhost:5001`
3. Restart backend then hard refresh browser (`Ctrl+F5`)

### Port conflict error (WinError 10048)

- Another process is using the port
- Keep backend on `5001` (current project default runtime)

### Video not found

- Ensure path exists from project root
- Prefer `data/<name>.mp4`

### Playback feels slow

- This usually means processing cost > frame budget
- Tune values in section 9 (especially stream quality and cadence)

### Model load issues

- Confirm file exists: `models/drowning_model.pkl`
- Ensure dependencies installed in `venv`

---

## 11) Development Notes

- Frontend build command:

```powershell
cd frontend
npm run build
```

- Backend runs locally only:
  - `127.0.0.1` host binding in current setup

---

## 12) Quick Demo Checklist

1. Start backend on `5001`
2. Start frontend on `3001`
3. Open dashboard
4. Enter video path (example `data/drowning_test.mp4`)
5. Click `START`
6. Observe overlays, swimmer cards, and alert log updating in real time

