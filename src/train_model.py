# src/train_model.py
import os, pickle
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report

IMG_SIZE   = (64, 128)
HOG_PARAMS = {
    "orientations":    9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
}

def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_dataset(folder: str):
    X, y = [], []
    for label, classe in [(1, "drowning"), (0, "normal")]:
        path = os.path.join(folder, classe)
        if not os.path.exists(path):
            print(f"Dossier introuvable : {path}")
            continue
        for fichier in os.listdir(path):
            img = cv2.imread(os.path.join(path, fichier))
            if img is None:
                continue
            img      = preprocess(img)
            features = hog(img, **HOG_PARAMS)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

print("Chargement des images de Personne 1...")
X_train, y_train = load_dataset("data/processed/train")
X_test,  y_test  = load_dataset("data/processed/test")
print(f"Train : {len(X_train)} images | Test : {len(X_test)} images")

print("Entraînement SVM en cours...")
model = SVC(kernel="rbf", class_weight="balanced", probability=True)
model.fit(X_train, y_train)

print("\nRésultats :")
print(classification_report(y_test, model.predict(X_test),
      target_names=["normal", "drowning"]))

os.makedirs("models", exist_ok=True)
with open("models/drowning_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Modèle sauvegardé → models/drowning_model.pkl")