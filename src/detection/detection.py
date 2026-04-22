"""
detection.py — Rôle 2
======================
Adapté du code avancé (yolov8-pose + Kalman + Re-ID + graphe Y).
Compatible avec :
    - Rôle 1 : reçoit get_clean_frame() → (bgr_clean, gray_frame)
    - Rôle 3 : produit TrackedPerson avec tous les champs attendus
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ── Constantes ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.4
KEYPOINT_CONF_THRESHOLD = 0.4
DROWNING_TIME_THRESHOLD = 3.0
DEFAULT_RED_LINE_Y      = 300
MAX_PLOT_POINTS         = 100
HISTORY_SECONDS         = 3
IOU_THRESHOLD           = 0.3
REID_THRESHOLD          = 0.85


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_mock_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    resized = cv2.resize(image, (32, 32))
    return resized.astype(np.float32).flatten().reshape(1, -1) / 255.0


def calculate_iou(box1, box2) -> float:
    xA = max(box1[0], box2[0]);  yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]);  yB = min(box1[3], box2[3])
    if xB < xA or yB < yA:
        return 0.0
    inter = (xB - xA) * (yB - yA)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


def leg_motion_suspicious(keypoints: np.ndarray) -> bool:
    try:
        if len(keypoints) < 16:
            return False
        lk = keypoints[12][:2];  la = keypoints[14][:2]
        rk = keypoints[13][:2];  ra = keypoints[15][:2]
        if any(np.all(p == 0) for p in [lk, la, rk, ra]):
            return False
        lv = np.linalg.norm(np.array(la) - np.array(lk))
        rv = np.linalg.norm(np.array(ra) - np.array(rk))
        return (lv + rv) / 2 < 2
    except Exception:
        return False


# ─────────────────────────────────────────────
# Structure Detection (interface HumanDetector → TrackingManager)
# ─────────────────────────────────────────────

@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    embedding:  Optional[np.ndarray] = None
    keypoints:  Optional[np.ndarray] = None   # pour l'analyse des jambes

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


# ─────────────────────────────────────────────
# Structure TrackedPerson (interface → Rôle 3)
# ─────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """
    Nageur suivi — interface vers le Rôle 3.

    Champs principaux :
        person_id        : identifiant stable
        bbox             : (x, y, w, h)
        centroid_history : 60 dernières positions
        frames_lost      : frames sans détection (sous l'eau)
        is_predicted     : True si Kalman prédit
        speed_px         : vitesse pixels/frame
        time_below_line  : secondes cumulées sous la ligne rouge
        drowning_alert   : True si time_below_line >= 3s
        legs_suspicious  : True si jambes immobiles détectées
    """

    person_id:        int
    bbox:             tuple[int, int, int, int]
    centroid_history: list[tuple[int, int]] = field(default_factory=list)
    frames_lost:      int   = 0
    is_predicted:     bool  = False
    speed_px:         float = 0.0

    time_below_line:  float = 0.0
    drowning_alert:   bool  = False
    legs_suspicious:  bool  = False

    y_positions:  deque = field(default_factory=lambda: deque(maxlen=MAX_PLOT_POINTS))
    timestamps:   deque = field(default_factory=lambda: deque(maxlen=MAX_PLOT_POINTS))
    start_time:   float = field(default_factory=time.time)

    MAX_HISTORY: int = 60

    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def update_centroid(self, cx: int, cy: int) -> None:
        self.centroid_history.append((cx, cy))
        if len(self.centroid_history) > self.MAX_HISTORY:
            self.centroid_history.pop(0)


# ─────────────────────────────────────────────
# Kalman vertical (sur Y uniquement)
# ─────────────────────────────────────────────

class VerticalKalmanTracker:
    def __init__(self, initial_y: int) -> None:
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix    = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([[1, 0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(2, dtype=np.float32)
        self.kf.measurementNoiseCov = np.array([[10]], dtype=np.float32)
        self.kf.errorCovPost        = np.eye(2, dtype=np.float32) * 1000.0
        self.kf.statePost           = np.array([[float(initial_y)], [0.0]], dtype=np.float32)

    def update(self, y: float) -> tuple[float, float]:
        self.kf.predict()
        c = self.kf.correct(np.array([[float(y)]], dtype=np.float32))
        return float(c[0][0]), float(c[1][0])

    def predict(self) -> tuple[float, float]:
        p = self.kf.predict()
        return float(p[0][0]), float(p[1][0])


# ─────────────────────────────────────────────
# Détecteur YOLO (avec support pose)
# ─────────────────────────────────────────────

class HumanDetector:
    """
    Détecteur YOLO avec support optionnel du modèle pose.
    Cherche dans l'ordre :
        1. src/detection/yolov8n-pose.pt  (pose — keypoints)
        2. src/detection/yolov8n.pt       (standard)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.model_path = self._resolve_model_path(model_path)
        self.model = self._load_model()
        self._has_pose = "pose" in str(self.model_path)

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            return Path(model_path)
        base = Path(__file__).resolve().parent
        for name in ["yolov8n-pose.pt", "yolov8n.pt"]:
            p = base / name
            if p.exists():
                return p
        # Télécharge yolov8n.pt automatiquement si rien trouvé
        return base / "yolov8n.pt"

    def _load_model(self):
        if YOLO is None:
            raise ImportError("ultralytics non installé : pip install ultralytics")
        LOGGER.info("Chargement YOLO : %s", self.model_path)
        return YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Accepte BGR ou gris.
        Retourne liste de Detection avec keypoints si modèle pose disponible.
        """
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
        detections: list[Detection] = []

        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1 = max(0, min(x1, frame.shape[1]-1))
            y1 = max(0, min(y1, frame.shape[0]-1))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            emb  = get_mock_embedding(crop)

            # Keypoints si modèle pose
            kps = None
            if self._has_pose and hasattr(results, 'keypoints') \
               and results.keypoints is not None and i < len(results.keypoints):
                try:
                    raw = results.keypoints[i].xy.cpu().numpy()
                    conf = results.keypoints[i].conf
                    if conf is not None:
                        conf = conf.cpu().numpy()
                        # Filtre par confiance
                        valid = conf[0] > KEYPOINT_CONF_THRESHOLD
                        kps = raw[0].copy()
                        kps[~valid] = 0
                    else:
                        kps = raw[0]
                except Exception:
                    kps = None

            detections.append(Detection(
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                confidence=float(box.conf[0]),
                embedding=emb,
                keypoints=kps,
            ))

        return detections


# ─────────────────────────────────────────────
# Graphe Y-position (OpenCV, pas matplotlib)
# ─────────────────────────────────────────────

def create_y_graph(persons: list[TrackedPerson],
                   frame_w: int, frame_h: int,
                   red_line_y: int = DEFAULT_RED_LINE_Y) -> np.ndarray:
    """Graphe OpenCV de la position Y dans le temps — affiché en overlay."""
    graph = np.ones((300, 400, 3), dtype=np.uint8) * 255

    # Ligne rouge de danger
    ry = int(red_line_y * 300 / frame_h)
    cv2.line(graph, (0, ry), (400, ry), (0, 0, 255), 2)
    cv2.putText(graph, "Danger", (10, ry - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Axes
    cv2.line(graph, (50, 250), (50, 50),   (0, 0, 0), 2)
    cv2.line(graph, (50, 250), (350, 250), (0, 0, 0), 2)
    cv2.putText(graph, "t (s)",  (180, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.putText(graph, "Y pos",  (5, 150),   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.putText(graph, "Y-Position Tracking", (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

    COLORS = [(0,180,255),(0,255,100),(255,120,0),(200,0,255),(0,255,255)]

    for idx, p in enumerate(persons):
        if len(p.timestamps) < 2:
            continue
        color = (0, 0, 255) if p.drowning_alert else COLORS[idx % len(COLORS)]
        max_t = max(p.timestamps) if p.timestamps else 1

        pts = []
        for i in range(len(p.timestamps)):
            x = int(50 + (p.timestamps[i] / max_t * 300)) if max_t > 0 else 50
            y = int(250 - (p.y_positions[i] / frame_h * 200))
            pts.append((x, y))

        for i in range(1, len(pts)):
            cv2.line(graph, pts[i-1], pts[i], color, 2)

        cv2.putText(graph, f"ID{p.person_id}",
                    (pts[-1][0], pts[-1][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return graph


# ─────────────────────────────────────────────
# TrackingManager
# ─────────────────────────────────────────────

class TrackingManager:
    """
    Tracking avec :
        - Association IoU prioritaire
        - Kalman vertical pour prédiction
        - Re-ID par similarité cosinus
        - Détection noyade par ligne rouge + temps
        - Analyse jambes (si keypoints disponibles)
    """

    MAX_LOST = 60

    def __init__(self, red_line_y: int = DEFAULT_RED_LINE_Y) -> None:
        self.red_line_y  = red_line_y
        self._persons:    dict[int, TrackedPerson]         = {}
        self._kalman:     dict[int, VerticalKalmanTracker] = {}
        self._embeddings: dict[int, Optional[np.ndarray]]  = {}
        self._last_below: dict[int, Optional[float]]       = {}
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        current_time = time.time()

        if not self._persons:
            for d in detections:
                self._register(d, current_time)
            return list(self._persons.values())

        matched_det = set()
        matched_pid = set()

        # Association IoU
        for pid, person in list(self._persons.items()):
            best_idx, best_iou = -1, 0.0
            for idx, det in enumerate(detections):
                if idx in matched_det:
                    continue
                iou = calculate_iou(person.bbox_xyxy, det.bbox_xyxy)
                if iou > IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                matched_det.add(best_idx)
                matched_pid.add(pid)
                self._update_person(pid, detections[best_idx], current_time)
            else:
                self._predict_person(pid, current_time)

        # Détections non associées → Re-ID ou nouveau tracker
        for idx, det in enumerate(detections):
            if idx in matched_det:
                continue
            reid_pid = self._find_reid(det, excluded=matched_pid)
            if reid_pid is not None:
                matched_pid.add(reid_pid)
                self._update_person(reid_pid, det, current_time)
            else:
                self._register(det, current_time)

        self._cleanup()
        return list(self._persons.values())

    # ── Interne ──────────────────────────────

    def _register(self, det: Detection, t: float) -> None:
        pid = self._next_id
        self._next_id += 1
        cx, cy = det.centroid

        p = TrackedPerson(person_id=pid, bbox=(det.x, det.y, det.w, det.h))
        p.update_centroid(cx, cy)
        p.y_positions.append(float(cy))
        p.timestamps.append(0.0)

        self._persons[pid]    = p
        self._kalman[pid]     = VerticalKalmanTracker(cy)
        self._embeddings[pid] = det.embedding
        self._last_below[pid] = t if det.y + det.h >= self.red_line_y else None

    def _update_person(self, pid: int, det: Detection, t: float) -> None:
        p    = self._persons[pid]
        prev = p.centroid
        cx, cy = det.centroid

        smoothed_y, vel_y = self._kalman[pid].update(cy)

        p.bbox        = (det.x, det.y, det.w, det.h)
        p.frames_lost = 0
        p.is_predicted = False
        p.speed_px    = float(np.hypot(cx - prev[0], smoothed_y - prev[1]))
        p.update_centroid(cx, int(smoothed_y))
        p.y_positions.append(float(cy))
        p.timestamps.append(t - p.start_time)
        self._embeddings[pid] = det.embedding

        # Analyse jambes
        if det.keypoints is not None:
            p.legs_suspicious = leg_motion_suspicious(det.keypoints)

        self._update_drowning(pid, det.y + det.h, t)

    def _predict_person(self, pid: int, t: float) -> None:
        p = self._persons[pid]
        pred_y, _ = self._kalman[pid].predict()
        x, y, w, h = p.bbox
        cx, _ = p.centroid
        new_y = int(pred_y - h / 2)

        p.bbox         = (x, new_y, w, h)
        p.frames_lost += 1
        p.is_predicted = True
        p.update_centroid(cx, int(pred_y))
        p.y_positions.append(float(pred_y))
        p.timestamps.append(t - p.start_time)

        self._update_drowning(pid, new_y + h, t)

    def _update_drowning(self, pid: int, bottom_y: int, t: float) -> None:
        p = self._persons[pid]
        if bottom_y >= self.red_line_y:
            last = self._last_below[pid]
            if last is not None:
                p.time_below_line += max(0.0, t - last)
            self._last_below[pid] = t
            p.drowning_alert = p.time_below_line >= DROWNING_TIME_THRESHOLD
        else:
            self._last_below[pid] = None
            p.time_below_line     = 0.0
            p.drowning_alert      = False

    def _find_reid(self, det: Detection,
                   excluded: set[int]) -> Optional[int]:
        best_pid, best_score = None, REID_THRESHOLD
        for pid, emb in self._embeddings.items():
            if pid in excluded or emb is None or det.embedding is None:
                continue
            score = cosine_similarity(emb, det.embedding)[0][0]
            if score > best_score:
                best_score = score
                best_pid   = pid
        return best_pid

    def _cleanup(self) -> None:
        to_del = [pid for pid, p in self._persons.items()
                  if p.frames_lost > self.MAX_LOST]
        for pid in to_del:
            del self._persons[pid]
            del self._kalman[pid]
            del self._embeddings[pid]
            del self._last_below[pid]