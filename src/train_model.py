import numpy as np, pickle, os
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle, resample
import cv2

IMG_SIZE   = (64, 128)
HOG_PARAMS = {
    "orientations":    9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
}

def load_dataset(folder):
    X, y = [], []
    erreurs = 0
    for label, classe in [(1, "drowning"), (0, "normal")]:
        path = os.path.join(folder, classe)
        if not os.path.exists(path):
            print(f"Dossier introuvable : {path}")
            continue
        fichiers = [f for f in os.listdir(path)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f"Chargement {classe} : {len(fichiers)} fichiers...")
        for fichier in fichiers:
            chemin = os.path.join(path, fichier)
            try:
                with open(chemin, "rb") as f:
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    erreurs += 1
                    continue
                img      = cv2.resize(img, IMG_SIZE)
                features = hog(img, **HOG_PARAMS)
                X.append(features)
                y.append(label)
            except:
                erreurs += 1
                continue
    print(f"Erreurs ignorées : {erreurs}")
    return np.array(X), np.array(y)

# Charger features
if os.path.exists("/content/drive/MyDrive/X_train.npy"):
    print("Chargement features depuis Drive...")
    X_train = np.load("/content/drive/MyDrive/X_train.npy")
    y_train = np.load("/content/drive/MyDrive/y_train.npy")
    X_test  = np.load("/content/drive/MyDrive/X_test.npy")
    y_test  = np.load("/content/drive/MyDrive/y_test.npy")
else:
    print("Extraction features...")
    X_train, y_train = load_dataset("/content/processed/train")
    X_test,  y_test  = load_dataset("/content/processed/test")
    np.save("/content/drive/MyDrive/X_train.npy", X_train)
    np.save("/content/drive/MyDrive/y_train.npy", y_train)
    np.save("/content/drive/MyDrive/X_test.npy",  X_test)
    np.save("/content/drive/MyDrive/y_test.npy",  y_test)

print(f"Train : {len(X_train)} | Test : {len(X_test)}")

# Équilibrer avec TOUTES les images
X_drown = X_train[y_train == 1]
X_norm  = X_train[y_train == 0]
n = min(len(X_drown), len(X_norm))
X_drown = resample(X_drown, n_samples=n, random_state=42)
X_norm  = resample(X_norm,  n_samples=n, random_state=42)
X_bal   = np.vstack([X_drown, X_norm])
y_bal   = np.array([1]*n + [0]*n)
X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)
print(f"Dataset équilibré : {len(X_bal)} images")

# Random Forest amélioré
print("Entraînement Random Forest amélioré...")
model = RandomForestClassifier(
    n_estimators=500,       # plus d'arbres
    max_depth=20,           # arbres plus profonds
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_bal, y_bal)

print("\nRésultats :")
print(classification_report(y_test, model.predict(X_test),
      target_names=["normal", "drowning"]))

# Sauvegarder
os.makedirs("/content/models", exist_ok=True)
with open("/content/models/drowning_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": None}, f)
with open("/content/drive/MyDrive/drowning_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": None}, f)
print("Modèle sauvegardé !")

from google.colab import files
files.download("/content/models/drowning_model.pkl")S