
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
# Ce fichier est dans src/, les modules aussi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessing import FramePreprocessor        # Rôle 1
from detection.detection     import HumanDetector, TrackingManager, TrackedPerson
from classification.classification import BehaviorClassifier, draw_alerte


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeSwim — Rôle 2 : Détection & Tracking")
    p.add_argument("--video",  required=True,         help="Chemin vers la vidéo")
    p.add_argument("--output", default=None,          help="Sauvegarder le résultat (ex: out.mp4)")
    p.add_argument("--width",  default=960, type=int, help="Largeur affichage (défaut: 960)")
    p.add_argument("--speed",  default=1.0, type=float, help="Vitesse lecture (défaut: 1.0)")
    return p.parse_args()


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

COLORS = [
    (0, 255, 100),
    (0, 180, 255),
    (255, 120, 0),
    (200, 0, 255),
    (0, 255, 255),
    (255, 50, 150),
]


def get_color(pid: int) -> tuple:
    return COLORS[pid % len(COLORS)]


def draw_person(frame: np.ndarray, person: TrackedPerson) -> None:
    """Dessine la bbox et la vitesse."""
    x, y, w, h = person.bbox
    color = get_color(person.person_id)

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    label = f"ID {person.person_id}"

    # Label avec fond
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, label, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Vitesse
    if person.speed_px > 0.5:
        cv2.putText(frame, f"{person.speed_px:.1f} px/f",
                    (x, y + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, persons: list[TrackedPerson], fps: float) -> None:
    """Panneau d'information en haut à gauche."""
    lines = [
        f"FPS     : {fps:.1f}",
        f"Personnes : {len(persons)}",
    ]
    for p in persons:
        lines.append(f"  ID{p.person_id}: {p.speed_px:.1f}px/f")

    for i, line in enumerate(lines):
        y = 24 + i * 20
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _dashed_rect(frame, pt1, pt2, color, thickness=2, dash=10):
    x1, y1 = pt1
    x2, y2 = pt2

    def dline(a, b):
        dist = int(np.hypot(b[0]-a[0], b[1]-a[1]))
        if dist == 0:
            return
        for s in range(0, dist, dash * 2):
            e = min(s + dash, dist)
            t0, t1 = s / dist, e / dist
            p = (int(a[0]+t0*(b[0]-a[0])), int(a[1]+t0*(b[1]-a[1])))
            q = (int(a[0]+t1*(b[0]-a[0])), int(a[1]+t1*(b[1]-a[1])))
            cv2.line(frame, p, q, color, thickness)

    dline((x1,y1),(x2,y1))
    dline((x2,y1),(x2,y2))
    dline((x2,y2),(x1,y2))
    dline((x1,y2),(x1,y1))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[ERREUR] Vidéo introuvable : {args.video}")
        sys.exit(1)

    # ── Modules ─────────────────────────────
    preprocessor = FramePreprocessor()      # Rôle 1
    detector     = HumanDetector()          # Rôle 2 — détection
    tracker      = TrackingManager()  # Rôle 2 — tracking
    classifier   = BehaviorClassifier()    # Rôle 3 — classification

    # ── Vidéo ────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la vidéo.")
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    delay   = max(1, int(1000 / (fps_src * args.speed)))

    # Taille affichage
    # FramePreprocessor redimensionne à 640×480 — on adapte l'affichage
    PROC_W, PROC_H = 640, 480
    disp_scale = min(args.width / PROC_W, 1.0)
    dw = int(PROC_W * disp_scale)
    dh = int(PROC_H * disp_scale)

    print(f"[INFO] Vidéo : {args.video}")
    print(f"[INFO] Traitement : 640×480 | Affichage : {dw}×{dh} @ {fps_src:.0f}fps")
    print("[INFO] 'q' quitter | Espace pause")

    # ── Sortie vidéo ─────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (dw, dh))
        print(f"[INFO] Enregistrement → {args.output}")

    fps_t, fps_n, fps = time.time(), 0, 0.0
    paused = False

    # ── Boucle principale ────────────────────
    while ret:
        if not paused:

            # ── Rôle 1 : Prétraitement ────────
            clean_frame, gray_frame = preprocessor.get_clean_frame(frame)

            # ── Rôle 2 : Détection ───────────
            detections = detector.detect(clean_frame)

            # ── Rôle 2 : Tracking ────────────
            tracked_persons = tracker.update(detections)

            # ── Rôle 3 : Classification ──────
            for person in tracked_persons:
                 cx, cy = person.centroid
                 en_danger = classifier.update(person.person_id, cx, cy)
                 if en_danger:
                      draw_alerte(display, person.person_id)

            # ── Affichage ────────────────────
            display = cv2.resize(
                clean_frame, (dw, dh), interpolation=cv2.INTER_LINEAR
            )
            # Remet à l'échelle si l'affichage est réduit
            if disp_scale < 1.0:
                scaled = _scale_persons(tracked_persons, disp_scale)
            else:
                scaled = tracked_persons

            for person in scaled:
                draw_person(display, person)

            fps_n += 1
            if fps_n >= 15:
                fps   = 15 / (time.time() - fps_t)
                fps_t = time.time()
                fps_n = 0

            draw_hud(display, tracked_persons, fps)

            if writer:
                writer.write(display)

        cv2.imshow("SafeSwim - Role 2", display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

        if not paused:
            ret, frame = cap.read()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Terminé.")


def _scale_persons(
    persons: list[TrackedPerson], scale: float
) -> list[TrackedPerson]:
    """Crée des copies avec bbox/historique à l'échelle d'affichage."""
    from copy import copy
    out = []
    for p in persons:
        x, y, w, h = p.bbox
        q = copy(p)
        q.bbox = (
            int(x * scale), int(y * scale),
            int(w * scale), int(h * scale)
        )
        q.centroid_history = [
            (int(cx * scale), int(cy * scale))
            for cx, cy in p.centroid_history
        ]
        out.append(q)
    return out


if __name__ == "__main__":
    main()
