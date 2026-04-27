from __future__ import annotations
import sys, os, time, base64, threading, argparse
import cv2

from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing.preprocessing import FramePreprocessor
from detection.detection import HumanDetector, TrackingManager
from classification import classify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Threading mode is significantly more stable on Windows than eventlet.
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_interval=20,
    ping_timeout=60,
)

# ── State ────────────────────────────────────────────────────────────
pipeline_thread = None
pipeline_running = False
pipeline_paused  = False
pipeline_speed   = 1.0
alert_log        = []
lock             = threading.Lock()
last_alert_by_pid = {}
STREAM_W = 480
STREAM_H = 360
STREAM_JPEG_QUALITY = 62
EMIT_EVERY_N_FRAMES = 1
DETECT_EVERY_N_FRAMES = 1
CLASSIFY_EVERY_N_FRAMES = 2
MAX_CATCHUP_GRABS = 8

# ── REST endpoints ───────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    return jsonify({
        "running": pipeline_running,
        "paused":  pipeline_paused,
        "speed":   pipeline_speed,
    })

@app.route("/api/alerts")
def api_alerts():
    return jsonify(alert_log[-50:])

# ── Pipeline ─────────────────────────────────────────────────────────
def person_to_dict(person, session_start):
    cx, cy = person.centroid
    return {
        "person_id":        person.person_id,
        "bbox":             list(person.bbox),
        "centroid":         [cx, cy],
        "centroid_history": [[x, y] for x, y in list(person.centroid_history)[-20:]],
        "frames_lost":      person.frames_lost,
        "is_predicted":     person.is_predicted,
        "speed_px":         round(person.speed_px, 2),
        "time_below_line":  round(person.time_below_line, 2),
        "drowning_alert":   person.drowning_alert,
        "legs_suspicious":  person.legs_suspicious,
        "classification_status": getattr(person, "classification_status", "OK"),
        "classification_proba": getattr(person, "classification_proba", 0.0),
        "y_positions":      list(person.y_positions)[-50:],
        "timestamps":       [round(t - session_start, 2)
                             for t in list(person.timestamps)[-50:]],
    }

def run_pipeline(video_path: str):
    global pipeline_running, pipeline_paused, pipeline_speed, last_alert_by_pid

    cap = None
    try:
        preprocessor = FramePreprocessor()
        detector     = HumanDetector()
        tracker      = TrackingManager()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            socketio.emit("new_alert", {
                "time": time.strftime("%H:%M:%S"),
                "person_id": -1,
                "type": "error",
                "message": f"Cannot open video: {video_path}",
            })
            return

        fps_src       = cap.get(cv2.CAP_PROP_FPS) or 30
        session_start = time.time()
        frame_idx     = 0
        fps_tracker   = {"t": time.time(), "n": 0, "fps": 0.0}
        last_alert_by_pid = {}

        socketio.emit("pipeline_started", {
            "video": os.path.basename(video_path),
            "fps":   fps_src,
        })
        socketio.emit("status", {"running": True, "paused": False, "speed": pipeline_speed})

        last_detections = []
        last_classification_by_pid = {}

        while pipeline_running:
            loop_start = time.perf_counter()
            if pipeline_paused:
                socketio.sleep(0.05)
                continue

            # Keep playback near real-time by dropping overdue frames.
            expected_idx = int((time.time() - session_start) * fps_src * pipeline_speed)
            frames_behind = expected_idx - frame_idx
            if frames_behind > 1:
                to_grab = min(frames_behind - 1, MAX_CATCHUP_GRABS)
                grabbed = 0
                for _ in range(to_grab):
                    if not cap.grab():
                        break
                    grabbed += 1
                frame_idx += grabbed

            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Role 1
                clean_frame, _ = preprocessor.get_clean_frame(frame)

                # Role 2 (performance mode): reuse detections between frames
                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    detections = detector.detect(clean_frame)
                    last_detections = detections
                else:
                    detections = last_detections
                tracked_persons = tracker.update(detections)

                # Role 3 — Classification model
                active_alerts = []
                for p in tracked_persons:
                    if frame_idx % CLASSIFY_EVERY_N_FRAMES == 0:
                        result = classify(p, clean_frame)
                        last_classification_by_pid[p.person_id] = result
                    else:
                        result = last_classification_by_pid.get(
                            p.person_id,
                            {"person_id": p.person_id, "status": "OK", "proba": 0.0},
                        )
                    p.classification_status = result["status"]
                    p.classification_proba = result["proba"]
                    if result["status"] == "DANGER" or p.drowning_alert:
                        active_alerts.append(p.person_id)
                active_alerts = sorted(set(active_alerts))

                # Encode frame (balanced for smooth real-time streaming)
                display = cv2.resize(clean_frame, (STREAM_W, STREAM_H))
                ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if not ok:
                    frame_idx += 1
                    socketio.sleep(0.001)
                    continue
                b64 = base64.b64encode(buf).decode()

                # FPS counter
                fps_tracker["n"] += 1
                if fps_tracker["n"] >= 15:
                    fps_tracker["fps"] = 15 / max(1e-6, (time.time() - fps_tracker["t"]))
                    fps_tracker["t"]   = time.time()
                    fps_tracker["n"]   = 0

                persons_json = [person_to_dict(p, session_start) for p in tracked_persons]

                if frame_idx % EMIT_EVERY_N_FRAMES == 0:
                    socketio.emit("frame", {
                        "frame":         b64,
                        "frame_idx":     frame_idx,
                        "fps":           round(fps_tracker["fps"], 1),
                        "persons":       persons_json,
                        "active_alerts": active_alerts,
                        "session_time":  round(time.time() - session_start, 1),
                        "red_line_y":    tracker.red_line_y,
                    })

                # Emit alert events only on transitions to reduce socket floods
                current = set(active_alerts)
                for pid in current:
                    if not last_alert_by_pid.get(pid, False):
                        event = {
                            "time":      time.strftime("%H:%M:%S"),
                            "person_id": pid,
                            "type":      "drowning",
                            "message":   f"Swimmer ID {pid} below danger line",
                        }
                        socketio.emit("new_alert", event)
                        with lock:
                            alert_log.append(event)
                        last_alert_by_pid[pid] = True

                for pid in list(last_alert_by_pid.keys()):
                    if pid not in current:
                        last_alert_by_pid[pid] = False

            except Exception as frame_err:
                # Keep streaming even if one frame fails.
                socketio.emit("new_alert", {
                    "time": time.strftime("%H:%M:%S"),
                    "person_id": -1,
                    "type": "warning",
                    "message": f"Frame processing error: {frame_err}",
                })

            frame_idx += 1

            # Adaptive pacing: keep near source FPS without artificial slowdowns.
            target_dt = 1.0 / max(1e-6, (fps_src * pipeline_speed))
            elapsed = time.perf_counter() - loop_start
            delay = max(0.0, target_dt - elapsed)
            if delay > 0:
                socketio.sleep(delay)

    except Exception as e:
        socketio.emit("new_alert", {
            "time": time.strftime("%H:%M:%S"),
            "person_id": -1,
            "type": "error",
            "message": f"Pipeline crashed: {e}",
        })
    finally:
        if cap is not None:
            cap.release()
        pipeline_running = False
        pipeline_paused = False
        socketio.emit("pipeline_stopped", {})
        socketio.emit("status", {"running": False, "paused": False, "speed": pipeline_speed})

# ── Socket events ────────────────────────────────────────────────────
@socketio.on("start_pipeline")
def on_start(data):
    global pipeline_thread, pipeline_running, pipeline_paused
    if pipeline_running:
        return
    video = str(data.get("video", "")).strip()
    if not video:
        socketio.emit("new_alert", {
            "time": time.strftime("%H:%M:%S"),
            "person_id": -1,
            "type": "error",
            "message": "Video path is empty",
        })
        return
    if not os.path.isabs(video):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        video = os.path.normpath(os.path.join(project_root, video))
    if not os.path.exists(video):
        socketio.emit("new_alert", {
            "time": time.strftime("%H:%M:%S"),
            "person_id": -1,
            "type": "error",
            "message": f"Video not found: {video}",
        })
        return
    pipeline_running = True
    pipeline_paused  = False
    pipeline_thread  = socketio.start_background_task(run_pipeline, video)

@socketio.on("stop_pipeline")
def on_stop():
    global pipeline_running
    pipeline_running = False

@socketio.on("set_paused")
def on_paused(data):
    global pipeline_paused
    pipeline_paused = data.get("paused", False)
    socketio.emit("status", {
        "running": pipeline_running,
        "paused":  pipeline_paused,
        "speed":   pipeline_speed,
    })

@socketio.on("set_speed")
def on_speed(data):
    global pipeline_speed
    pipeline_speed = float(data.get("speed", 1.0))
    socketio.emit("status", {
        "running": pipeline_running,
        "paused":  pipeline_paused,
        "speed":   pipeline_speed,
    })

# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", help="Default video path")
    parser.add_argument("--port",  default=5000, type=int)
    args = parser.parse_args()
    print(f"[SafeSwim] Server starting on http://localhost:{args.port}")
    socketio.run(app, host="127.0.0.1", port=args.port, debug=False, allow_unsafe_werkzeug=True)