# src/classification.py
import cv2, pickle
import numpy as np
from skimage.feature import hog
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drowning_model.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)


MODEL  = data["model"]
SCALER = data["scaler"]  # None pour Random Forest

IMG_SIZE   = (64, 128)
HOG_PARAMS = {
    "orientations":    9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
}

danger_counters: dict = {}
SEUIL_CONFIRMATION   = 8

def preprocess(crop: np.ndarray) -> np.ndarray:
    """Identique au preprocessing de Personne 1 — NE PAS MODIFIER."""
    img = cv2.resize(crop, IMG_SIZE)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def classify(person, frame: np.ndarray) -> dict:
    pid = person.person_id
    x, y, w, h = person.bbox
    fh, fw = frame.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(fw, int(x + w))
    y2 = min(fh, int(y + h))

    # Étape 1 — Extraire le crop
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"person_id": pid, "status": "OK", "proba": 0.0}

    # Étape 2 — Preprocessing + HOG
    img      = preprocess(crop)
    features = hog(img, **HOG_PARAMS)

    # Étape 3 — Normalisation si scaler existe
    if SCALER is not None:
        features = SCALER.transform([features])[0]

    # Étape 4 — Prédiction
    proba_raw = MODEL.predict_proba([features])[0][1]
    proba = float(np.clip(proba_raw, 0.0, 1.0))
    score = proba

    # Étape 5 — Enrichir avec signaux Personne 2
    if person.drowning_alert:
        score = min(1.0, score + 0.30)
    if person.legs_suspicious:
        score = min(1.0, score + 0.15)
    if person.speed_px < 1.0:
        score = min(1.0, score + 0.10)
    if person.frames_lost > 60:
        score = min(1.0, score + 0.10)

    # Étape 6 — Confirmation temporelle
    if score > 0.6:
        danger_counters[pid] = danger_counters.get(pid, 0) + 1
    else:
        danger_counters[pid] = 0

    alerte = danger_counters[pid] >= SEUIL_CONFIRMATION

    return {
        "person_id": pid,
        "status":    "DANGER" if alerte else "OK",
        "proba":     round(score * 100, 1),
        "frames":    danger_counters[pid]
    }