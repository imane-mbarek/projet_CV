"""
main.py — SafeSwim Rôle 2
==========================
Pipeline complet avec graphe Y-position et alertes visuelles.

Usage :
    python src/main.py --video videos/swimmer_video.mp4
    python src/main.py --video videos/test.mp4 --width 960
    python src/main.py --video videos/test.mp4 --output out.mp4
    python src/main.py --video videos/test.mp4 --red-line 250

Touches : q quitter | Espace pause
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessing import FramePreprocessor
from detection.detection import (
    HumanDetector, TrackingManager, TrackedPerson,
    create_y_graph, DEFAULT_RED_LINE_Y,
)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
ALERT_COLOR  = (0, 0, 255)
COLORS = [
    (0, 255, 100), (0, 180, 255), (255, 120, 0),
    (200, 0, 255), (0, 255, 255), (255, 50, 150),
]


# ── Arguments ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeSwim — Rôle 2")
    p.add_argument("--video",    required=True)
    p.add_argument("--output",   default=None)
    p.add_argument("--width",    default=960, type=int)
    p.add_argument("--speed",    default=1.0, type=float)
    p.add_argument("--red-line", default=DEFAULT_RED_LINE_Y, type=int,
                   help=f"Y de la ligne de danger (défaut: {DEFAULT_RED_LINE_Y})")
    return p.parse_args()


# ── Visualisation ─────────────────────────────────────────────────────

def get_color(pid: int) -> tuple:
    return COLORS[pid % len(COLORS)]


def draw_person(frame: np.ndarray, person: TrackedPerson) -> None:
    x, y, w, h = person.bbox
    cx, cy     = person.centroid
    is_alert   = person.drowning_alert or person.legs_suspicious
    color      = ALERT_COLOR if is_alert else get_color(person.person_id)

    if person.is_predicted:
        _dashed_rect(frame, (x, y), (x+w, y+h), (60, 60, 255))
        label = f"ID {person.person_id} [sous l'eau]"
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"ID {person.person_id}" + (" ALERT!" if is_alert else "")

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x, y-th-6), (x+tw+4, y), color, -1)
    cv2.putText(frame, label, (x+2, y-3), FONT, 0.5, (0, 0, 0), 1)

    if person.time_below_line > 0:
        t_color = ALERT_COLOR if person.time_below_line > 1.0 else (255, 165, 0)
        cv2.putText(frame, f"{person.time_below_line:.1f}s",
                    (x, y-th-20), FONT, 0.45, t_color, 1)

    if person.legs_suspicious:
        cv2.putText(frame, "Jambes immobiles",
                    (x, y+h+30), FONT, 0.42, ALERT_COLOR, 1)

    if person.speed_px > 0.5:
        cv2.putText(frame, f"{person.speed_px:.1f}px/f",
                    (x, y+h+16), FONT, 0.42, get_color(person.person_id), 1)

    cv2.circle(frame, (cx, cy), 4, color, -1)

    pts = person.centroid_history
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, pts[i-1], pts[i], c, 2)


def draw_red_line(frame: np.ndarray, red_line_y: int) -> None:
    w = frame.shape[1]
    cv2.line(frame, (0, red_line_y), (w, red_line_y), ALERT_COLOR, 2)
    cv2.putText(frame, "ZONE DE DANGER",
                (10, red_line_y - 8), FONT, 0.55, ALERT_COLOR, 2)


def draw_hud(frame: np.ndarray, persons: list[TrackedPerson],
             fps: float, frame_count: int) -> None:
    lines = [
        f"FPS     : {fps:.1f}",
        f"Frame   : {frame_count}",
        f"Nageurs : {len(persons)}",
    ]
    for p in persons:
        state = f"PERDU({p.frames_lost}f)" if p.is_predicted else "OK"
        lines.append(f"  ID{p.person_id}: {state} | {p.speed_px:.1f}px/f")

    for i, line in enumerate(lines):
        yy = 24 + i * 20
        cv2.putText(frame, line, (10, yy), FONT, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, yy), FONT, 0.55, (255,255,255), 1, cv2.LINE_AA)


def draw_global_alert(frame: np.ndarray, persons: list[TrackedPerson],
                      frame_count: int) -> None:
    alerting = [p for p in persons if p.drowning_alert]
    if not alerting:
        return
    h, w = frame.shape[:2]
    thickness = 10 if (frame_count // 8) % 2 == 0 else 3
    cv2.rectangle(frame, (0, 0), (w, h), ALERT_COLOR, thickness)
    ids = ", ".join(f"ID{p.person_id}" for p in alerting)
    cv2.putText(frame, f"ALERTE NOYADE — {ids}",
                (w//2 - 200, h - 20), FONT, 0.9, ALERT_COLOR, 3, cv2.LINE_AA)


def overlay_graph(frame: np.ndarray, persons: list[TrackedPerson],
                  red_line_y: int) -> None:
    if not persons:
        return
    h, w = frame.shape[:2]
    graph = create_y_graph(persons, w, h, red_line_y)
    gw    = w // 4
    gh    = int(graph.shape[0] * gw / graph.shape[1])
    small = cv2.resize(graph, (gw, gh))
    x_off = w - gw - 10
    y_off = 10
    gh    = min(gh, h - y_off)
    gw    = min(gw, w - x_off)
    frame[y_off:y_off+gh, x_off:x_off+gw] = small[:gh, :gw]


def _dashed_rect(frame, pt1, pt2, color, thickness=2, dash=10):
    x1, y1 = pt1;  x2, y2 = pt2

    def dline(a, b):
        dist = int(np.hypot(b[0]-a[0], b[1]-a[1]))
        if dist == 0:
            return
        for s in range(0, dist, dash * 2):
            e  = min(s + dash, dist)
            t0 = s / dist;  t1 = e / dist
            p = (int(a[0]+t0*(b[0]-a[0])), int(a[1]+t0*(b[1]-a[1])))
            q = (int(a[0]+t1*(b[0]-a[0])), int(a[1]+t1*(b[1]-a[1])))
            cv2.line(frame, p, q, color, thickness)

    dline((x1,y1),(x2,y1)); dline((x2,y1),(x2,y2))
    dline((x2,y2),(x1,y2)); dline((x1,y2),(x1,y1))


def _scale_persons(persons: list[TrackedPerson],
                   scale: float) -> list[TrackedPerson]:
    from copy import copy
    out = []
    for p in persons:
        x, y, w, h = p.bbox
        q = copy(p)
        q.bbox = (int(x*scale), int(y*scale), int(w*scale), int(h*scale))
        q.centroid_history = [
            (int(cx*scale), int(cy*scale)) for cx, cy in p.centroid_history
        ]
        out.append(q)
    return out


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[ERREUR] Vidéo introuvable : {args.video}")
        sys.exit(1)

    preprocessor = FramePreprocessor()
    detector     = HumanDetector()
    tracker      = TrackingManager(red_line_y=args.red_line)

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Vidéo vide.")
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    delay   = max(1, int(1000 / (fps_src * args.speed)))

    PROC_W, PROC_H = 640, 480
    disp_scale = min(args.width / PROC_W, 1.0)
    dw = int(PROC_W * disp_scale)
    dh = int(PROC_H * disp_scale)

    print(f"[INFO] Vidéo          : {args.video}")
    print(f"[INFO] Affichage      : {dw}×{dh} @ {fps_src:.0f}fps")
    print(f"[INFO] Ligne de danger: Y = {args.red_line}")
    print("[INFO] 'q' quitter | Espace pause")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (dw, dh))
        print(f"[INFO] Enregistrement → {args.output}")

    fps_t, fps_n, fps = time.time(), 0, 0.0
    paused      = False
    frame_count = 0

    while ret:
        if not paused:
            frame_count += 1

            # ── Rôle 1 ────────────────────────
            clean_frame, gray_frame = preprocessor.get_clean_frame(frame)

            # ── Rôle 2 : Détection ────────────
            detections = detector.detect(gray_frame)

            # ── Rôle 2 : Tracking ─────────────
            persons = tracker.update(detections)

            # ── Affichage ─────────────────────
            display = cv2.resize(clean_frame, (dw, dh),
                                 interpolation=cv2.INTER_LINEAR)

            red_scaled = int(args.red_line * disp_scale)
            draw_red_line(display, red_scaled)

            scaled = _scale_persons(persons, disp_scale) if disp_scale < 1.0 else persons
            for p in scaled:
                draw_person(display, p)

            fps_n += 1
            if fps_n >= 15:
                fps   = 15 / (time.time() - fps_t)
                fps_t = time.time()
                fps_n = 0

            draw_hud(display, persons, fps, frame_count)
            draw_global_alert(display, persons, frame_count)
            overlay_graph(display, persons, red_scaled)

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


if __name__ == "__main__":
    main()