

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ajoute src/ au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # pas ROOT / "src"


from preprocessing.preprocessing import FramePreprocessor
from detection.detection import HumanDetector, TrackingManager, TrackedPerson


# ─────────────────────────────────────────────
# Configuration par type de vue
# ─────────────────────────────────────────────

VIEW_CONFIGS = {
    "normal": {
        "description": "Vue latérale (bord de bassin)",
        "confidence":  0.40,
    },
    "overhead": {
        "description": "Vue de dessus (drone / caméra plafond)",
        "confidence":  0.30,   # corps moins reconnaissables vus de haut
    },
    "angle45": {
        "description": "Vue à 45° (caméra inclinée)",
        "confidence":  0.35,
    },
}

COLORS = [
    (0, 255, 100), (0, 180, 255), (255, 120, 0),
    (200, 0, 255), (0, 255, 255), (255, 50, 150),
    (50, 200, 255), (255, 200, 0),
]


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeSwim — Test Rôle 2")
    p.add_argument("--video",  required=True,
                   help="Chemin vers la vidéo de test")
    p.add_argument("--view",   default="normal",
                   choices=["normal", "overhead", "angle45"],
                   help="Type de vue (défaut: normal)")
    p.add_argument("--multi",  action="store_true",
                   help="Afficher les stats par ID (multi-personnes)")
    p.add_argument("--output", default=None,
                   help="Sauvegarder la vidéo annotée (ex: out.mp4)")
    p.add_argument("--width",  default=960, type=int,
                   help="Largeur affichage (défaut: 960)")
    return p.parse_args()


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def get_color(pid: int) -> tuple:
    return COLORS[pid % len(COLORS)]


def draw_person(frame: np.ndarray, person: TrackedPerson,
                scale: float = 1.0) -> None:
    x, y, w, h = person.bbox
    x = int(x * scale); y = int(y * scale)
    w = int(w * scale); h = int(h * scale)
    cx, cy = x + w // 2, y + h // 2
    c = get_color(person.person_id)

    # Bbox : pointillée si sous l'eau, pleine sinon
    if person.is_predicted:
        _dashed_rect(frame, (x, y), (x+w, y+h), (60, 60, 255))
        label = f"ID{person.person_id} [sous l'eau]"
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
        label = f"ID{person.person_id}"

    # Label avec fond coloré
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), c, -1)
    cv2.putText(frame, label, (x+2, y-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Vitesse
    if person.speed_px > 0.5:
        cv2.putText(frame, f"{person.speed_px:.1f} px/f",
                    (x, y+h+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)

    # Centroïde
    cv2.circle(frame, (cx, cy), 4, c, -1)



def draw_hud(frame: np.ndarray, persons: list[TrackedPerson],
             fps: float, view: str, multi: bool) -> None:
    lines = [
        f"FPS     : {fps:.1f}",
        f"Vue     : {view}",
        f"Nageurs : {len(persons)}",
    ]
    if multi:
        for p in persons:
            state = f"PERDU({p.frames_lost}f)" if p.is_predicted else "OK"
            lines.append(f"  ID{p.person_id}: {state} | {p.speed_px:.1f}px/f")

    for i, line in enumerate(lines):
        y = 24 + i * 20
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_alert(frame: np.ndarray, persons: list[TrackedPerson]) -> None:
    for p in persons:
        if p.frames_lost > 45:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(
                frame,
                f"ALERTE — ID{p.person_id} sous l'eau ({p.frames_lost}f)",
                (w // 2 - 260, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3, cv2.LINE_AA,
            )
            break


def _dashed_rect(frame, pt1, pt2, color, thickness=2, dash=10):
    x1, y1 = pt1
    x2, y2 = pt2

    def dline(a, b):
        dist = int(np.hypot(b[0]-a[0], b[1]-a[1]))
        if dist == 0:
            return
        for s in range(0, dist, dash * 2):
            e  = min(s + dash, dist)
            t0 = s / dist
            t1 = e / dist
            p = (int(a[0]+t0*(b[0]-a[0])), int(a[1]+t0*(b[1]-a[1])))
            q = (int(a[0]+t1*(b[0]-a[0])), int(a[1]+t1*(b[1]-a[1])))
            cv2.line(frame, p, q, color, thickness)

    dline((x1, y1), (x2, y1))
    dline((x2, y1), (x2, y2))
    dline((x2, y2), (x1, y2))
    dline((x1, y2), (x1, y1))


def _scale_persons(persons: list[TrackedPerson],
                   scale: float) -> list[TrackedPerson]:
    """Copie légère avec bbox/historique à l'échelle d'affichage."""
    from copy import copy
    out = []
    for p in persons:
        x, y, w, h = p.bbox
        q = copy(p)
        q.bbox = (int(x*scale), int(y*scale),
                  int(w*scale), int(h*scale))
        q.centroid_history = [
            (int(cx*scale), int(cy*scale))
            for cx, cy in p.centroid_history
        ]
        out.append(q)
    return out


# ─────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    config = VIEW_CONFIGS[args.view]

    print(f"\n[TEST] SafeSwim — Rôle 2")
    print(f"[TEST] Vidéo      : {args.video}")
    print(f"[TEST] Vue        : {args.view} — {config['description']}")
    print(f"[TEST] Confidence : {config['confidence']}")
    print(f"[TEST] Multi      : {args.multi}\n")

    if not os.path.exists(args.video):
        print(f"[ERREUR] Vidéo introuvable : {args.video}")
        sys.exit(1)

    # ── Modules (identiques à main.py) ───────
    preprocessor = FramePreprocessor()
    detector     = HumanDetector(confidence_threshold=config["confidence"])
    tracker      = TrackingManager()

    # ── Vidéo ────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    ret, raw = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la vidéo.")
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    delay   = max(1, int(1000 / fps_src))

    # Après preprocessing → 640×480
    PROC_W, PROC_H = 640, 480
    disp_scale = args.width / PROC_W
    dw = int(PROC_W * disp_scale)
    dh = int(PROC_H * disp_scale)

    print(f"[TEST] Affichage : {dw}×{dh} @ {fps_src:.0f} fps")
    print("[TEST] 'q' quitter | Espace pause\n")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (dw, dh))
        print(f"[TEST] Enregistrement → {args.output}")

    # Fenêtre redimensionnable (évite le rendu "petit angle")
    window_name = f"SafeSwim — Test [{args.view}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, dw, dh)

    # ── Stats ────────────────────────────────
    total_frames     = 0
    total_detections = 0
    prev_ids         = set()
    id_changes       = 0

    fps_t, fps_n, fps = time.time(), 0, 0.0
    paused = False

    # ── Boucle principale ─────────────────────
    while ret:
        if not paused:

            # Rôle 1 — prétraitement
            clean_frame, gray_frame = preprocessor.get_clean_frame(raw)

            # Rôle 2 — détection
            detections = detector.detect(gray_frame)
            total_detections += len(detections)

            # Rôle 2 — tracking
            persons = tracker.update(detections)
            total_frames += 1

            # Compteur changements d'ID (qualité tracking)
            curr_ids = {p.person_id for p in persons}
            if prev_ids and (curr_ids - prev_ids):
                id_changes += len(curr_ids - prev_ids)
            prev_ids = curr_ids

            # Affichage
            display = cv2.resize(clean_frame, (dw, dh),
                                 interpolation=cv2.INTER_LINEAR)

            scaled = _scale_persons(persons, disp_scale)
            for p in scaled:
                draw_person(display, p)

            fps_n += 1
            if fps_n >= 15:
                fps   = 15 / (time.time() - fps_t)
                fps_t = time.time()
                fps_n = 0

            draw_hud(display, persons, fps, args.view, args.multi)
            draw_alert(display, persons)

            if writer:
                writer.write(display)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

        if not paused:
            ret, raw = cap.read()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Rapport final ─────────────────────────
    print("\n" + "=" * 50)
    print("  RAPPORT DE TEST")
    print("=" * 50)
    print(f"  Vue              : {args.view}")
    print(f"  Frames traités   : {total_frames}")
    print(f"  Détections tot.  : {total_detections}")
    print(f"  Moy. det/frame   : {total_detections / max(total_frames,1):.2f}")
    print(f"  Nouveaux IDs     : {id_changes}  (idéal = 0 pour 1 nageur)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
