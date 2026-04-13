
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dépendance optionnelle à l'exécution
    YOLO = None


LOGGER = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.4
MAX_PLOT_POINTS = 100
DROWNING_TIME_THRESHOLD = 3.0
DEFAULT_RED_LINE_Y = 300


def get_mock_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    """Embedding simple pour garder une logique de ré-identification légère."""
    if image is None or image.size == 0:
        return None
    resized = cv2.resize(image, (32, 32))
    return resized.astype(np.float32).flatten().reshape(1, -1) / 255.0


def cosine_similarity_score(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    a_vec = a.reshape(-1)
    b_vec = b.reshape(-1)
    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a_vec, b_vec) / (a_norm * b_norm))


def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return inter / denom


@dataclass
class Detection:
    """Détection brute YOLO retournée par HumanDetector."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class TrackedPerson:
    """
    Nageur/personne suivi.

    Champs existants conservés pour compatibilité avec `src/main.py`.
    """

    person_id: int
    bbox: tuple[int, int, int, int]
    centroid_history: list[tuple[int, int]] = field(default_factory=list)
    frames_lost: int = 0
    is_predicted: bool = False
    speed_px: float = 0.0

    time_below_line: float = 0.0
    drowning_alert: bool = False
    y_positions: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_PLOT_POINTS))
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_PLOT_POINTS))
    start_time: float = field(default_factory=time.time)

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


class VerticalKalmanTracker:
    """Kalman minimaliste sur la position Y du centre, inspiré du code fourni."""

    def __init__(self, initial_y: int) -> None:
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32)
        self.kf.measurementNoiseCov = np.array([[10]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 1000.0
        self.kf.statePost = np.array([[float(initial_y)], [0.0]], dtype=np.float32)

    def update(self, center_y: float) -> tuple[float, float]:
        self.kf.predict()
        corrected = self.kf.correct(np.array([[float(center_y)]], dtype=np.float32))
        return float(corrected[0][0]), float(corrected[1][0])

    def predict(self) -> tuple[float, float]:
        predicted = self.kf.predict()
        return float(predicted[0][0]), float(predicted[1][0])


class HumanDetector:
    """
    Détecteur de personnes basé sur YOLO.

    Le code source fourni utilisait `yolov8n-pose.pt`. Ici on supporte :
        1. `src/detection/yolov8n-pose.pt` si présent
        2. sinon `src/detection/yolov8n.pt`
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.model_path = self._resolve_model_path(model_path)
        self.model = self._load_model()

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            return Path(model_path)

        base_dir = Path(__file__).resolve().parent
        preferred = base_dir / "yolov8n-pose.pt"
        fallback = base_dir / "yolov8n.pt"
        return preferred if preferred.exists() else fallback

    def _load_model(self):
        if YOLO is None:
            raise ImportError(
                "Ultralytics n'est pas installé. Ajoutez `ultralytics` dans les dépendances."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle YOLO introuvable : {self.model_path}. "
                "Ajoutez un poids comme `src/detection/yolov8n.pt`."
            )
        LOGGER.info("Chargement du modèle YOLO depuis %s", self.model_path)
        return YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Accepte soit une image BGR soit une image en niveaux de gris.
        """
        if frame.ndim == 2:
            color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            color_frame = frame

        results = self.model(color_frame, conf=self.confidence_threshold, verbose=False)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1 = max(0, min(x1, color_frame.shape[1] - 1))
            y1 = max(0, min(y1, color_frame.shape[0] - 1))
            x2 = max(0, min(x2, color_frame.shape[1]))
            y2 = max(0, min(y2, color_frame.shape[0]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = color_frame[y1:y2, x1:x2]
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    confidence=float(box.conf[0]),
                    embedding=get_mock_embedding(crop),
                )
            )

        return detections


class TrackingManager:
    """
    Tracking inspiré du code fourni :
        - association prioritaire par IoU
        - ré-identification légère par similarité d'apparence
        - prédiction Kalman quand une détection disparaît
    """

    MAX_LOST: int = 60
    IOU_THRESHOLD: float = 0.3
    REID_THRESHOLD: float = 0.85

    def __init__(self, red_line_y: int = DEFAULT_RED_LINE_Y) -> None:
        self.red_line_y = red_line_y
        self._persons: dict[int, TrackedPerson] = {}
        self._kalman: dict[int, VerticalKalmanTracker] = {}
        self._embeddings: dict[int, Optional[np.ndarray]] = {}
        self._last_below_timestamp: dict[int, Optional[float]] = {}
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        current_time = time.time()

        if not self._persons:
            for det in detections:
                self._register(det, current_time)
            return list(self._persons.values())

        matched_detection_indices: set[int] = set()
        matched_person_ids: set[int] = set()

        for pid, person in list(self._persons.items()):
            best_match_idx = -1
            best_iou = 0.0

            for idx, detection in enumerate(detections):
                if idx in matched_detection_indices:
                    continue
                iou = calculate_iou(person.bbox_xyxy, detection.bbox_xyxy)
                if iou > self.IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            if best_match_idx >= 0:
                matched_detection_indices.add(best_match_idx)
                matched_person_ids.add(pid)
                self._update_person(pid, detections[best_match_idx], current_time)
            else:
                self._predict_person(pid, current_time)

        for idx, detection in enumerate(detections):
            if idx in matched_detection_indices:
                continue

            reid_pid = self._find_reid_match(detection, excluded_ids=matched_person_ids)
            if reid_pid is not None:
                matched_person_ids.add(reid_pid)
                self._update_person(reid_pid, detection, current_time)
            else:
                self._register(detection, current_time)

        self._cleanup()
        return list(self._persons.values())

    def _register(self, detection: Detection, current_time: float) -> None:
        pid = self._next_id
        self._next_id += 1

        person = TrackedPerson(
            person_id=pid,
            bbox=(detection.x, detection.y, detection.w, detection.h),
        )
        cx, cy = detection.centroid
        person.update_centroid(cx, cy)
        person.y_positions.append(float(cy))
        person.timestamps.append(0.0)

        self._persons[pid] = person
        self._kalman[pid] = VerticalKalmanTracker(cy)
        self._embeddings[pid] = detection.embedding
        self._last_below_timestamp[pid] = current_time if detection.y + detection.h >= self.red_line_y else None

    def _update_person(self, pid: int, detection: Detection, current_time: float) -> None:
        person = self._persons[pid]
        previous_centroid = person.centroid

        center_x, center_y = detection.centroid
        smoothed_y, velocity_y = self._kalman[pid].update(center_y)

        person.bbox = (detection.x, detection.y, detection.w, detection.h)
        person.frames_lost = 0
        person.is_predicted = False
        person.speed_px = float(np.hypot(center_x - previous_centroid[0], smoothed_y - previous_centroid[1]))
        person.update_centroid(center_x, int(smoothed_y))
        person.y_positions.append(float(center_y))
        person.timestamps.append(current_time - person.start_time)
        self._embeddings[pid] = detection.embedding

        self._update_drowning_state(pid, detection.y + detection.h, current_time)
        if person.drowning_alert:
            person.speed_px = max(person.speed_px, abs(velocity_y))

    def _predict_person(self, pid: int, current_time: float) -> None:
        person = self._persons[pid]
        predicted_y, _ = self._kalman[pid].predict()
        x, y, w, h = person.bbox
        cx, _ = person.centroid
        new_y = int(predicted_y - h / 2)

        person.bbox = (x, new_y, w, h)
        person.frames_lost += 1
        person.is_predicted = True
        person.update_centroid(cx, int(predicted_y))
        person.y_positions.append(float(predicted_y))
        person.timestamps.append(current_time - person.start_time)

        self._update_drowning_state(pid, new_y + h, current_time)

    def _update_drowning_state(self, pid: int, bottom_y: int, current_time: float) -> None:
        person = self._persons[pid]
        if bottom_y >= self.red_line_y:
            last_ts = self._last_below_timestamp[pid]
            if last_ts is not None:
                person.time_below_line += max(0.0, current_time - last_ts)
            self._last_below_timestamp[pid] = current_time
            person.drowning_alert = person.time_below_line >= DROWNING_TIME_THRESHOLD
        else:
            self._last_below_timestamp[pid] = None
            person.time_below_line = 0.0
            person.drowning_alert = False

    def _find_reid_match(
        self, detection: Detection, excluded_ids: Optional[set[int]] = None
    ) -> Optional[int]:
        excluded = excluded_ids or set()
        best_pid = None
        best_score = self.REID_THRESHOLD
        for pid, embedding in self._embeddings.items():
            if pid in excluded:
                continue
            score = cosine_similarity_score(embedding, detection.embedding)
            if score > best_score:
                best_score = score
                best_pid = pid
        return best_pid

    def _cleanup(self) -> None:
        to_remove = [pid for pid, person in self._persons.items() if person.frames_lost > self.MAX_LOST]
        for pid in to_remove:
            del self._persons[pid]
            del self._kalman[pid]
            del self._embeddings[pid]
            del self._last_below_timestamp[pid]
