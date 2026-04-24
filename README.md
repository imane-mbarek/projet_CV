# SafeSwim — Rôle 2 : Détection & Tracking

> **Auteur** : Imane Ait Mbarek  
> **Module** : Vision par Ordinateur — Python 3.12  
> **Objectif du rôle** : Localiser chaque nageur dans le flux vidéo, lui attribuer un identifiant stable, et suivre ses mouvements frame par frame pour fournir des données au Rôle 3.

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture complète](#2-architecture-complète)
3. [Structure des fichiers](#3-structure-des-fichiers)
4. [Installation](#4-installation)
5. [Pourquoi ces choix techniques](#5-pourquoi-ces-choix-techniques)
6. [detection.py — Le cœur du Rôle 2](#6-detectionpy--le-cœur-du-rôle-2)
7. [main.py — Le pipeline complet](#7-mainpy--le-pipeline-complet)
8. [test_tracking.py — Les tests](#8-test_trackingpy--les-tests)
9. [train_swimmer_detector.py — Fine-tuning](#9-train_swimmer_detectorpy--fine-tuning)
10. [Interface avec le Rôle 1](#10-interface-avec-le-rôle-1)
11. [Interface avec le Rôle 3](#11-interface-avec-le-rôle-3)
12. [Lancer le projet](#12-lancer-le-projet)
13. [Problèmes rencontrés et solutions](#13-problèmes-rencontrés-et-solutions)

---

## 1. Vue d'ensemble du projet

SafeSwim est un système de surveillance intelligente de piscine. Il analyse un flux vidéo en temps réel pour détecter si un nageur est en situation de noyade.

Le projet est divisé en **3 rôles** qui travaillent en séquence :

```
Caméra vidéo
    │
    ▼
┌──────────────────────────────────────────┐
│  RÔLE 1 — Prétraitement                  │
│  FramePreprocessor.get_clean_frame()     │
│  → resize 640×480 + GaussianBlur + gris  │
└──────────────────────────────────────────┘
    │ bgr_clean, gray_frame
    ▼
┌──────────────────────────────────────────┐
│  RÔLE 2 — Détection & Tracking  ← TOI   │
│  HumanDetector  → détecte les nageurs    │
│  TrackingManager → suit chaque nageur    │
│  → liste de TrackedPerson                │
└──────────────────────────────────────────┘
    │ TrackedPerson (position, vitesse,
    │ durée sous l'eau, alerte noyade...)
    ▼
┌──────────────────────────────────────────┐
│  RÔLE 3 — Classification & Alerte        │
│  HOG + SVM → normal ou noyade ?          │
│  → déclenchement de l'alarme             │
└──────────────────────────────────────────┘
```

**Ce que fait le Rôle 2 spécifiquement :**
- Prend le frame nettoyé du Rôle 1
- Détecte toutes les personnes visibles avec YOLO
- Attribue un ID unique à chaque nageur
- Maintient l'ID même quand le nageur plonge et disparaît
- Mesure la vitesse, la trajectoire, la durée sous l'eau
- Transmet tout ça au Rôle 3 sous forme d'objets `TrackedPerson`

---

## 2. Architecture complète

### Flux de données par frame

```
frame brut (caméra)
        │
        ▼ preprocessor.get_clean_frame()
(bgr_clean, gray_frame)   ← Rôle 1
        │
        ▼ detector.detect(gray_frame)
[Detection, Detection, ...]
        │  chaque Detection contient :
        │  x, y, w, h, confidence, embedding, keypoints
        │
        ▼ tracker.update(detections)
[TrackedPerson, TrackedPerson, ...]
        │  chaque TrackedPerson contient :
        │  person_id, bbox, centroid_history,
        │  speed_px, frames_lost, is_predicted,
        │  time_below_line, drowning_alert, legs_suspicious
        │
        ▼ Rôle 3
   classification + alerte
```

### Algorithme d'association (comment on sait qui est qui)

À chaque frame, le `TrackingManager` doit associer les nouvelles détections aux nageurs déjà connus. Il utilise 3 mécanismes en cascade :

```
Nouvelle détection
        │
        ▼ Étape 1 : IoU (Intersection over Union)
   La boîte se chevauche avec un tracker connu ?
   ┌── OUI → même personne, mise à jour du tracker
   └── NON ↓
        │
        ▼ Étape 2 : Re-ID (similarité cosinus)
   L'apparence visuelle ressemble à un tracker connu ?
   ┌── OUI → même personne retrouvée (ex: après plongée)
   └── NON ↓
        │
        ▼ Étape 3 : Nouveau tracker
   Nouvelle personne, nouvel ID
```

---

## 3. Structure des fichiers

```
projet_CV/
│
├── src/
│   ├── main.py                    ← Point d'entrée principal
│   ├── detection/
│   │   └── detection.py           ← Détection + Tracking (cœur du Rôle 2)
│   └── preprocessing/
│       └── preprocessing.py       ← Rôle 1 (fourni par l'équipe)
│
├── tests/
│   └── test_tracking.py           ← Script de test séparé
│
├── videos/                        ← Vidéos de test
├── models/                        ← Modèles YOLO (.pt)
├── train_swimmer_detector.py      ← Fine-tuning YOLO sur nageurs
├── download_test_videos.py        ← Téléchargement de vidéos de test
└── README.md                      ← Ce fichier
```

---

## 4. Installation

### Environnement virtuel (recommandé)

```bash
# Créer un environnement propre
python3 -m venv env_safeswim
source env_safeswim/bin/activate   # Linux/Mac
# ou
env_safeswim\Scripts\activate      # Windows
```

### Dépendances

```bash
pip install ultralytics            # YOLOv8
pip install opencv-python          # OpenCV
pip install numpy                  # Calcul matriciel
pip install scikit-learn           # cosine_similarity pour Re-ID
```

### Vérification

```bash
python -c "import cv2, ultralytics, numpy, sklearn; print('OK')"
```

---

## 5. Pourquoi ces choix techniques

### Pourquoi YOLOv8 et pas HOG ?

Le projet initial prévoyait HOG + SVM pour la détection. J'ai choisi YOLOv8 pour le Rôle 2 pour deux raisons :

**HOG est conçu pour les piétons debout, vus de face.** En piscine, les nageurs sont horizontaux, partiellement immergés, vus de côté ou d'en haut. HOG donne un taux de détection très faible dans ce contexte.

**YOLOv8n (nano)** est un réseau de neurones convolutif pré-entraîné sur des millions d'images. Il détecte les personnes dans des poses variées, des éclairages difficiles, et des occlusions partielles. Il tourne à environ 15 FPS sur CPU sans GPU.

> **Note importante** : HOG + SVM reste dans le projet au Rôle 3 pour la **classification** (normal vs noyade). Ce n'est pas la même tâche — le Rôle 3 analyse des crops 64×128 déjà extraits, pas des frames entiers.

### Pourquoi le Kalman Filter ?

Quand un nageur plonge, il disparaît de la caméra pendant 1 à 3 secondes. Sans Kalman, le tracker l'oublie et lui attribue un nouvel ID à la réapparition — ce qui casse l'analyse comportementale.

Le filtre de Kalman **prédit la position** du nageur à partir de sa vitesse et direction précédentes. Il maintient le tracker actif pendant l'absence et affiche une boîte pointillée bleue pour signaler que c'est une prédiction.

```
Frame 10 : nageur détecté à Y=250, vitesse vers le bas
Frame 11 : nageur plonge, plus détecté
Frame 12 : Kalman prédit Y=255 (continue vers le bas)
Frame 13 : Kalman prédit Y=260
...
Frame 20 : nageur réapparaît à Y=240 → même ID conservé
```

### Pourquoi la Re-ID par cosinus ?

Le Kalman peut se tromper si le nageur réapparaît dans une position très différente (ex: il a nagé sous l'eau sur 5 mètres). Dans ce cas, la distance cosinus compare **l'apparence visuelle** (couleur du maillot, silhouette) du nageur disparu avec la nouvelle détection.

Si la similarité cosinus est > 0.85, c'est le même nageur → même ID.

### Pourquoi la ligne rouge et `time_below_line` ?

L'idée vient du code de référence fourni par l'équipe. La ligne rouge représente la surface de l'eau dans la vidéo. Si le bas de la boîte englobante descend sous cette ligne pendant plus de 3 secondes cumulées, `drowning_alert` passe à `True`.

C'est un signal simple mais efficace que le Rôle 3 peut utiliser directement.

### Pourquoi les keypoints (jambes) ?

Si le modèle `yolov8n-pose.pt` est disponible, YOLO détecte aussi les 17 points clés du corps humain (épaules, coudes, genoux, chevilles...). L'analyse des jambes est simple : si la distance genou-cheville est inférieure à 2 pixels sur plusieurs frames, les jambes ne bougent pas → `legs_suspicious = True`.

Un nageur normal fait des mouvements de jambes réguliers. Un nageur en détresse s'agite verticalement avec les bras mais ses jambes s'immobilisent.

---

## 6. detection.py — Le cœur du Rôle 2

Ce fichier contient toute la logique de détection et de tracking.

### `Detection` (dataclass)

Structure qui représente une détection brute de YOLO sur un frame.

```python
@dataclass
class Detection:
    x: int              # coin supérieur gauche
    y: int
    w: int              # largeur de la boîte
    h: int              # hauteur de la boîte
    confidence: float   # score de confiance YOLO (0 à 1)
    embedding: ndarray  # empreinte visuelle 32×32 aplatie
    keypoints: ndarray  # 17 points du corps (si modèle pose)
```

### `TrackedPerson` (dataclass)

Structure qui représente un nageur suivi. C'est **l'interface officielle vers le Rôle 3**.

```python
@dataclass
class TrackedPerson:
    person_id:        int    # ID unique et stable
    bbox:             tuple  # (x, y, w, h) position actuelle
    centroid_history: list   # 60 dernières positions (x, y)
    frames_lost:      int    # frames sans détection (sous l'eau)
    is_predicted:     bool   # True = position Kalman, pas caméra
    speed_px:         float  # vitesse en pixels/frame
    time_below_line:  float  # secondes cumulées sous la ligne rouge
    drowning_alert:   bool   # True si time_below_line >= 3 secondes
    legs_suspicious:  bool   # True si jambes immobiles (keypoints)
    y_positions:      deque  # historique Y pour le graphe
    timestamps:       deque  # timestamps correspondants
```

### `VerticalKalmanTracker`

Filtre de Kalman 1D sur la position Y uniquement.

- **État** : `[y, vy]` (position verticale + vitesse verticale)
- **Mesure** : `y` seulement (on mesure où est le nageur)
- **Bruit de mesure** : 10 (on fait confiance à la caméra)
- **Bruit de processus** : 1 (le nageur peut accélérer)

```
Chaque frame :
    predict() → estime la nouvelle position à partir de la vitesse
    update(y_mesuré) → corrige l'estimation avec la détection réelle
```

### `HumanDetector`

Charge YOLO et détecte les personnes dans chaque frame.

```python
detector = HumanDetector()
# Cherche yolov8n-pose.pt, puis yolov8n.pt, puis télécharge automatiquement

detections = detector.detect(gray_frame)
# Accepte BGR ou gris
# Filtre uniquement la classe 0 (person) de YOLO
# Retourne liste de Detection
```

**Pourquoi accepter le frame gris du Rôle 1 ?**  
YOLO fonctionne en BGR, donc si on passe le frame gris, il le reconvertit automatiquement (`cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)`). On reste compatible avec l'interface du Rôle 1.

### `TrackingManager`

Le gestionnaire de tracking. C'est lui qui maintient les IDs stables.

```python
tracker = TrackingManager(red_line_y=300)
# red_line_y : position Y de la surface de l'eau dans la vidéo

persons = tracker.update(detections)
# À appeler à chaque frame
# Retourne liste de TrackedPerson à jour
```

**Algorithme `update()` en détail :**

```
1. Si aucun tracker existant → enregistrer toutes les détections

2. Pour chaque tracker existant :
   - Chercher la détection avec le meilleur IoU
   - Si IoU > 0.3 → association → mise à jour du tracker
   - Sinon → prédiction Kalman (nageur absent)

3. Pour chaque détection non associée :
   - Calculer la similarité cosinus avec les trackers existants
   - Si similarité > 0.85 → Re-ID → même tracker
   - Sinon → nouveau tracker, nouvel ID

4. Nettoyage :
   - Supprimer les trackers perdus depuis > 60 frames
```

### `create_y_graph()`

Génère un graphe OpenCV (sans matplotlib) montrant l'évolution de la position Y de chaque nageur dans le temps. Le graphe est intégré en overlay dans le coin supérieur droit de la vidéo.

La ligne rouge de danger est dessinée sur le graphe pour voir visuellement quand un nageur passe en dessous.

---

## 7. main.py — Le pipeline complet

Point d'entrée du Rôle 2. Orchestre tous les modules.

### Usage

```bash
# Basique
python src/main.py --video videos/swimmer_video.mp4

# Avec options
python src/main.py --video videos/test.mp4 --width 960 --speed 0.5

# Sauvegarder le résultat annoté
python src/main.py --video videos/test.mp4 --output out.mp4

# Ajuster la ligne de danger (défaut: Y=300)
python src/main.py --video videos/test.mp4 --red-line 250
```

### Touches clavier

| Touche | Action |
|--------|--------|
| `q` | Quitter |
| `Espace` | Pause / Reprendre |

### Ce qui s'affiche

**Sur le frame vidéo :**
- Boîte verte/rouge autour de chaque nageur avec son ID
- Boîte bleue pointillée si le nageur est sous l'eau (prédiction Kalman)
- Vitesse en pixels/frame sous chaque boîte
- Durée sous la ligne rouge au-dessus de la boîte
- "Jambes immobiles" si `legs_suspicious = True`
- Trace de mouvement colorée (les 60 dernières positions)
- Ligne rouge horizontale = ligne de danger
- Bordure rouge clignotante si `drowning_alert = True`

**En coin supérieur droit :**
- Graphe Y-position dans le temps pour chaque nageur
- La ligne rouge du graphe correspond à la ligne de danger

**En coin supérieur gauche :**
- FPS
- Numéro de frame
- Nombre de nageurs détectés
- État de chaque ID (OK / PERDU / vitesse)

---

## 8. test_tracking.py — Les tests

Script de test **séparé** de `main.py`. Il ne modifie pas le comportement du pipeline — il ajoute des métriques de qualité et supporte différents angles de caméra.

### Pourquoi un fichier de test séparé ?

`main.py` est le pipeline de production. `test_tracking.py` est l'outil de validation. On ne mélange pas les deux pour garder le code propre.

### Angles de caméra supportés

| Angle | Confidence YOLO | Description |
|-------|----------------|-------------|
| `normal` | 0.40 | Vue latérale standard (bord de bassin) |
| `overhead` | 0.30 | Vue de dessus (drone ou caméra plafond) |
| `angle45` | 0.35 | Vue inclinée à 45° |

La confidence est plus basse pour `overhead` car les corps vus de dessus sont moins reconnaissables par YOLO, qui a été entraîné principalement sur des vues latérales.

### Usage

```bash
# Vue normale
python tests/test_tracking.py --video videos/swimmer_video.mp4

# Vue de dessus avec stats par ID
python tests/test_tracking.py --video videos/overhead.mp4 --view overhead --multi

# Vue 45° avec sauvegarde
python tests/test_tracking.py --video videos/test.mp4 --view angle45 --output result.mp4
```

### Rapport de fin de test

À la fin de chaque test, un rapport s'affiche automatiquement :

```
==================================================
  RAPPORT DE TEST
==================================================
  Vue              : normal
  Frames traités   : 450
  Détections tot.  : 512
  Moy. det/frame   : 1.14
  Nouveaux IDs     : 2   (idéal = 0 pour 1 nageur)
==================================================
```

**Le compteur "Nouveaux IDs"** est l'indicateur clé de qualité : si un seul nageur est dans la piscine et qu'il y a 5 nouveaux IDs, ça veut dire que le tracker a perdu et retrouvé le nageur 5 fois — la Re-ID ne fonctionne pas bien. L'idéal est 0 ou 1.

---

## 9. train_swimmer_detector.py — Fine-tuning

### Pourquoi fine-tuner YOLO ?

Inspiré de l'article de recherche **Lygouras et al. (2019)** publié dans *Sensors* (MDPI) :  
*"Unsupervised Human Detection with an Embedded Vision System on a Fully Autonomous UAV for Search and Rescue Operations"*

L'article démontre qu'un modèle généraliste pré-entraîné sur COCO donne **~21% de mAP** sur des images de nageurs en eau, contre **67% après fine-tuning** sur des images de nageurs spécifiquement annotées.

La différence s'explique par les caractéristiques propres aux nageurs :
- Corps partiellement immergé (on ne voit que la tête et les épaules)
- Poses horizontales (YOLO est entraîné sur des personnes debout)
- Reflets et distorsion de l'eau
- Vue de dessus ou de côté

### Usage

```bash
# Étape 1 : extraire des frames de ta vidéo de piscine
python train_swimmer_detector.py --extract --video videos/piscine.mp4

# Étape 2 : annoter les images extraites avec LabelImg
pip install labelImg && labelImg
# → ouvrir dataset/swimmers/images/train/
# → format YOLO, classe "swimmer"

# Étape 3 : entraîner (~30-60 min sur CPU)
python train_swimmer_detector.py --train

# Étape 4 : comparer les deux modèles côte à côte
python train_swimmer_detector.py --test --video videos/test.mp4

# Étape 5 : mettre à jour le pipeline automatiquement
python train_swimmer_detector.py --update
```

Après l'étape 5, `detection.py` utilisera `models/swimmer_detector.pt` au lieu de `yolov8n.pt`.

---

## 10. Interface avec le Rôle 1

Le Rôle 1 fournit la fonction `get_clean_frame()` dans `preprocessing/preprocessing.py`.

**Ce qu'elle fait :**
```python
def get_clean_frame(frame):
    frame   = cv2.resize(frame, (640, 480))   # Standardise la taille
    blurred = cv2.GaussianBlur(frame, (5,5), 0)  # Réduit le bruit de l'eau
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # Convertit en gris
    return blurred, gray
```

**Comment le Rôle 2 l'utilise :**
```python
# Dans main.py, à chaque frame :
clean_frame, gray_frame = preprocessor.get_clean_frame(frame)
detections = detector.detect(gray_frame)   # YOLO reçoit le frame gris
persons    = tracker.update(detections)
```

**Point critique :** Le Rôle 2 passe `gray_frame` à YOLO. Le `HumanDetector` détecte que le frame est en niveaux de gris et le reconvertit en BGR (`cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)`) avant de le passer à YOLO. On reste compatible avec l'interface du Rôle 1 sans modifier leur code.

---

## 11. Interface avec le Rôle 3

Le Rôle 3 reçoit une liste de `TrackedPerson` à chaque frame. Voici exactement ce qu'il peut utiliser :

```python
# Dans la boucle principale, après tracker.update() :
for person in persons:

    # ── Localisation ──────────────────────────
    person.person_id          # int : ID unique du nageur
    person.bbox               # (x, y, w, h) : boîte englobante
    person.centroid           # (cx, cy) : centre de la boîte
    person.centroid_history   # list[(x,y)] : 60 dernières positions

    # ── Mouvement ────────────────────────────
    person.speed_px           # float : vitesse en pixels/frame
    person.is_predicted       # bool : True si sous l'eau (Kalman)
    person.frames_lost        # int : frames sans détection réelle

    # ── Signaux de noyade (déjà calculés) ────
    person.time_below_line    # float : secondes sous la ligne rouge
    person.drowning_alert     # bool : True si time_below_line >= 3s
    person.legs_suspicious    # bool : True si jambes immobiles

    # ── Historique pour graphes ───────────────
    person.y_positions        # deque : positions Y dans le temps
    person.timestamps         # deque : timestamps correspondants
```

**Ce que le Rôle 3 doit faire avec ces données :**

```python
# Exemple de logique classification Rôle 3 :
def classify(person):
    signals = 0
    if person.drowning_alert:    signals += 2   # signal fort
    if person.legs_suspicious:   signals += 1
    if person.speed_px < 1.0:    signals += 1   # nageur immobile
    if person.frames_lost > 60:  signals += 1   # sous l'eau longtemps
    return signals >= 2   # True = danger
```

---

## 12. Lancer le projet

### Prérequis

```bash
# 1. Cloner le projet
git clone https://github.com/imane-mbarek/projet_CV
cd projet_CV

# 2. Installer les dépendances
pip install ultralytics opencv-python numpy scikit-learn

# 3. Télécharger des vidéos de test (optionnel)
python download_test_videos.py
```

### Lancer le pipeline principal

```bash
# Sur ta vidéo de nageur
python src/main.py --video videos/swimmer_video.mp4

# Ajuster la ligne rouge selon ta caméra
python src/main.py --video videos/test.mp4 --red-line 280

# Ralentir la vidéo pour mieux observer
python src/main.py --video videos/test.mp4 --speed 0.3

# Sauvegarder le résultat annoté
python src/main.py --video videos/test.mp4 --output annotated.mp4
```

### Lancer les tests

```bash
# Test standard
python tests/test_tracking.py --video videos/swimmer_video.mp4

# Test multi-personnes avec stats
python tests/test_tracking.py --video videos/pedestrians.avi --multi

# Test vue de dessus
python tests/test_tracking.py --video videos/overhead.mp4 --view overhead
```

---

## 13. Problèmes rencontrés et solutions

### Problème 1 : Nageur perd son ID après chaque plongée

**Cause :** Le tracker de base utilisait seulement IoU. Quand le nageur disparaît et réapparaît à une position différente, IoU = 0 → nouvel ID.

**Solution :** Ajout du filtre de Kalman (prédit la position pendant l'absence) et de la Re-ID cosinus (compare l'apparence visuelle à la réapparition).

### Problème 2 : Vidéo très lente (< 5 FPS)

**Cause :** YOLO tournait sur chaque frame à pleine résolution.

**Solution :**
- Résolution d'entrée YOLO réduite à 416×416
- YOLO skip : ne détecte qu'1 frame sur 3, Kalman interpole les autres
- Filtrage sur la classe `person` uniquement (`classes=[0]`)

### Problème 3 : Conflit MediaPipe / TensorFlow / protobuf

**Cause :** MediaPipe 0.10.x force `numpy<2` et `protobuf~=4.25`, mais TensorFlow 2.21 exige `protobuf>=6.31`. Ces deux packages sont incompatibles dans le même environnement.

**Solution :** Créer un environnement virtuel séparé `env_safeswim` sans TensorFlow. La pose MediaPipe est optionnelle — le pipeline tourne parfaitement sans elle.

### Problème 4 : Disque plein lors de l'installation

**Cause :** PyTorch avec CUDA pèse ~900 MB et l'espace disque était à 99%.

**Solution :**
```bash
pip cache purge              # Libère le cache pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Version CPU seulement : ~200 MB au lieu de ~900 MB
```

### Problème 5 : `track_id` est une string dans DeepSORT

**Cause :** La librairie `deep-sort-realtime` retourne `track_id` comme string (`"1"`, `"2"`), pas comme int. L'opération `"1" % 50` plante en Python.

**Solution :** Cast explicite `int(t.track_id)` partout où on utilise l'ID.

---

## Dépendances

| Package | Version | Usage |
|---------|---------|-------|
| `ultralytics` | ≥ 8.0 | YOLOv8 détection |
| `opencv-python` | ≥ 4.8 | Traitement vidéo |
| `numpy` | ≥ 1.24 | Calcul matriciel |
| `scikit-learn` | ≥ 1.3 | `cosine_similarity` Re-ID |

---

## Références

- **Lygouras et al. (2019)** — *Unsupervised Human Detection with an Embedded Vision System on a Fully Autonomous UAV for Search and Rescue Operations* — Sensors, MDPI. [DOI: 10.3390/s19163542](https://doi.org/10.3390/s19163542)
- **Ultralytics YOLOv8** — [https://docs.ultralytics.com](https://docs.ultralytics.com)
- **OpenCV Kalman Filter** — [https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html](https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html)
- **Notebook inspiration couloirs** — Ron-po, *tracking-swimmers*, GitHub

---

*SafeSwim — Rôle 2 | Vision par Ordinateur | Python 3.12 | OpenCV + YOLOv8*
