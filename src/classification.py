# src/classification.py
import cv2, pickle
import numpy as np
from skimage.feature import hog

with open("models/drowning_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

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
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def classify(person, frame: np.ndarray) -> dict:
    pid = person.person_id
    x, y, w, h = person.bbox

    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return {"person_id": pid, "status": "OK", "proba": 0.0}

    img      = preprocess(crop)
    features = hog(img, **HOG_PARAMS)
    proba    = MODEL.predict_proba([features])[0][1]

    score = proba

    if person.drowning_alert:
        score = min(1.0, score + 0.30)
    if person.legs_suspicious:
        score = min(1.0, score + 0.15)
    if person.speed_px < 1.0:
        score = min(1.0, score + 0.10)
    if person.frames_lost > 60:
        score = min(1.0, score + 0.10)

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