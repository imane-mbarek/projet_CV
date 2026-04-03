"""
SafeSwim - Module de Détection Humaine
Rôle 2 : Ingénieur Vision
Auteur  : [Ton Nom]

Stratégie :
  - HOG + SVM (plus robuste que Haar en contexte piscine)
  - Kalman Filter pour prédire la position quand le nageur disparaît
  - Memory-efficient : on ne stocke que les N derniers centroides
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """Une détection brute : bounding box + centroïde."""
    x: int
    y: int
    w: int
    h: int

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class TrackedPerson:
    """Un nageur suivi avec historique de positions."""
    person_id: int
    bbox: tuple[int, int, int, int]          # (x, y, w, h) courant
    centroid_history: list[tuple[int, int]] = field(default_factory=list)
    frames_lost: int = 0                     # frames sans détection
    is_predicted: bool = False               # True si position Kalman

    MAX_HISTORY = 30   # ~1 seconde à 30 fps

    def update_history(self, centroid: tuple[int, int]) -> None:
        self.centroid_history.append(centroid)
        if len(self.centroid_history) > self.MAX_HISTORY:
            self.centroid_history.pop(0)

    @property
    def centroid(self) -> tuple[int, int]:
        cx = self.bbox[0] + self.bbox[2] // 2
        cy = self.bbox[1] + self.bbox[3] // 2
        return (cx, cy)


# ─────────────────────────────────────────────
# Détecteur HOG
# ─────────────────────────────────────────────

class HumanDetector:
    """
    Détection de personnes via HOG + SVM d'OpenCV.
    Plus robuste que Haar pour les nageurs partiellement visibles.
    """

    def __init__(self, scale: float = 1.05, win_stride: int = 8,
                 min_neighbors: int = 2, min_size: tuple = (40, 80)):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.scale       = scale
        self.win_stride  = (win_stride, win_stride)
        self.padding     = (8, 8)
        self.min_size    = min_size
        self.min_neighbors = min_neighbors

    def detect(self, gray_frame: np.ndarray) -> list[Detection]:
        """
        Reçoit un frame en GRIS (venant du module preprocessing).
        Retourne une liste de Detection.
        """
        # HOG attend une image 3 canaux ou 1 canal — on convertit en BGR
        frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        rects, weights = self.hog.detectMultiScale(
            frame_bgr,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
        )

        detections: list[Detection] = []

        if len(rects) == 0:
            return detections

        # Non-Maximum Suppression pour éliminer les doublons
        rects = self._nms(rects, weights, overlap_thresh=0.65)

        for (x, y, w, h) in rects:
            d = Detection(x=int(x), y=int(y), w=int(w), h=int(h))
            if d.w >= self.min_size[0] and d.h >= self.min_size[1]:
                detections.append(d)

        return detections

    @staticmethod
    def _nms(rects: np.ndarray, weights: np.ndarray,
             overlap_thresh: float = 0.65) -> np.ndarray:
        """Non-Maximum Suppression : garde la meilleure boîte par groupe."""
        if len(rects) == 0:
            return rects

        x1 = rects[:, 0].astype(float)
        y1 = rects[:, 1].astype(float)
        x2 = (rects[:, 0] + rects[:, 2]).astype(float)
        y2 = (rects[:, 1] + rects[:, 3]).astype(float)
        areas = (x2 - x1) * (y2 - y1)
        weights = weights.flatten()
        order = weights.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= overlap_thresh)[0] + 1]

        return rects[keep]


# ─────────────────────────────────────────────
# Filtre de Kalman — prédit la position perdue
# ─────────────────────────────────────────────

class KalmanTracker:
    """
    Filtre de Kalman 2D pour prédire le centroïde
    quand le nageur plonge ou passe sous l'eau.
    État : [x, y, vx, vy]
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)   # 4 états, 2 mesures

        # Matrice de transition (mouvement à vitesse constante)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Matrice d'observation (on mesure x et y seulement)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Bruit de processus (confiance dans le modèle)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # Bruit de mesure (confiance dans la caméra)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    def init(self, centroid: tuple[int, int]) -> None:
        """Initialise le filtre avec le premier centroïde mesuré."""
        self.kf.statePre = np.array(
            [centroid[0], centroid[1], 0, 0], dtype=np.float32
        ).reshape(4, 1)
        self._initialized = True

    def update(self, centroid: tuple[int, int]) -> tuple[int, int]:
        """Met à jour avec une vraie mesure. Retourne la position corrigée."""
        if not self._initialized:
            self.init(centroid)

        self.kf.predict()
        measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        x = corrected.flatten()
        return (int(x[0]), int(x[1]))

    def predict(self) -> tuple[int, int]:
        """Prédit la position sans mesure (nageur invisible)."""
        predicted = self.kf.predict()
        x = predicted.flatten()
        return (int(x[0]), int(x[1]))

    @property
    def initialized(self) -> bool:
        return self._initialized


# ─────────────────────────────────────────────
# Manager de Tracking — orchestre tout
# ─────────────────────────────────────────────

class TrackingManager:
    """
    Associe les détections aux personnes suivies.
    Gère les pertes de signal avec le Kalman.
    Interface vers le Rôle 3 : fournit get_tracked_persons().
    """

    MAX_LOST_FRAMES = 20   # frames avant de supprimer un tracker perdu
    MAX_DISTANCE    = 80   # pixels max pour associer détection → tracker

    def __init__(self):
        self._persons: dict[int, TrackedPerson] = {}
        self._kalman:  dict[int, KalmanTracker] = {}
        self._next_id = 0

    # ── API publique ──────────────────────────────

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        """
        Appeler à chaque frame.
        Entrée  : liste de Detection (venant du HumanDetector)
        Sortie  : liste de TrackedPerson à jour
        """
        if detections:
            self._associate(detections)
        else:
            self._predict_all()

        self._cleanup()
        return list(self._persons.values())

    def get_tracked_persons(self) -> list[TrackedPerson]:
        """Interface simple pour le Rôle 3."""
        return list(self._persons.values())

    # ── Logique interne ──────────────────────────

    def _associate(self, detections: list[Detection]) -> None:
        """Algorithme glouton d'association détection ↔ tracker."""
        used_det = set()
        used_pid = set()

        # Distance centroïde détection ↔ centroïde tracker connu
        for pid, person in self._persons.items():
            best_dist = self.MAX_DISTANCE
            best_di   = -1

            for di, det in enumerate(detections):
                if di in used_det:
                    continue
                dist = self._euclidean(det.centroid, person.centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_di   = di

            if best_di >= 0:
                det = detections[best_di]
                used_det.add(best_di)
                used_pid.add(pid)

                # Mise à jour avec Kalman
                corrected = self._kalman[pid].update(det.centroid)
                x = corrected[0] - det.w // 2
                y = corrected[1] - det.h // 2
                person.bbox = (x, y, det.w, det.h)
                person.update_history(corrected)
                person.frames_lost  = 0
                person.is_predicted = False
            else:
                # Pas de détection proche → prédiction Kalman
                predicted = self._kalman[pid].predict()
                w, h = person.bbox[2], person.bbox[3]
                x = predicted[0] - w // 2
                y = predicted[1] - h // 2
                person.bbox = (x, y, w, h)
                person.update_history(predicted)
                person.frames_lost  += 1
                person.is_predicted  = True

        # Nouvelles détections non associées → nouveaux trackers
        for di, det in enumerate(detections):
            if di not in used_det:
                self._register(det)

    def _predict_all(self) -> None:
        """Aucune détection ce frame → on prédit tout le monde."""
        for pid, person in self._persons.items():
            if self._kalman[pid].initialized:
                predicted = self._kalman[pid].predict()
                w, h = person.bbox[2], person.bbox[3]
                person.bbox = (predicted[0] - w // 2, predicted[1] - h // 2, w, h)
                person.update_history(predicted)
                person.frames_lost  += 1
                person.is_predicted  = True

    def _register(self, det: Detection) -> None:
        """Crée un nouveau tracker pour une détection inconnue."""
        pid = self._next_id
        self._next_id += 1

        person = TrackedPerson(
            person_id=pid,
            bbox=(det.x, det.y, det.w, det.h)
        )
        person.update_history(det.centroid)

        kf = KalmanTracker()
        kf.init(det.centroid)

        self._persons[pid] = person
        self._kalman[pid]  = kf

    def _cleanup(self) -> None:
        """Supprime les trackers perdus depuis trop longtemps."""
        to_remove = [
            pid for pid, p in self._persons.items()
            if p.frames_lost > self.MAX_LOST_FRAMES
        ]
        for pid in to_remove:
            del self._persons[pid]
            del self._kalman[pid]

    @staticmethod
    def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))
