"""
scripts/preview_dataset.py
--------------------------
Displays a random grid of processed sample images, one row per class.
Useful for a quick visual sanity check before training.

Requirements: opencv-python, numpy (already needed for the pipeline).
Optional: matplotlib — if not installed, images are shown via cv2.imshow.

Usage
-----
From the project root:
    python -m scripts.preview_dataset
    python -m scripts.preview_dataset --split train --n 8
    python -m scripts.preview_dataset --save previews/grid.png
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.class_mapping import get_active_labels


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CELL_SIZE: int = 96    # display cell size in pixels (upscaled from 64×64)
PADDING:   int = 8     # pixels of white padding between cells
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SafeSwim — Preview random processed dataset samples."
    )
    parser.add_argument(
        "--processed",
        default="data/processed",
        help="Path to the processed dataset root.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which split to preview. Default: train",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Number of images per class row. Default: 8",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save the grid as an image file instead of displaying it.",
    )
    return parser.parse_args()


def _load_random_images(
    class_dir: Path,
    n: int,
) -> list[np.ndarray]:
    """Load up to n randomly sampled images from class_dir, as grayscale."""
    all_files = sorted(class_dir.glob("*.png"))
    selected  = random.sample(all_files, min(n, len(all_files)))
    images    = []
    for f in selected:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def _build_grid(
    label_images: dict[str, list[np.ndarray]],
    n_cols: int,
) -> np.ndarray:
    """
    Build a BGR preview grid: one row per class label.
    Labels without any images are shown as a row of blank cells.
    """
    rows = []
    cell = CELL_SIZE
    pad  = PADDING

    for label, images in label_images.items():
        # ---- Row header (label name) ----
        header_h = 28
        header_w = (cell + pad) * n_cols + pad
        header = np.full((header_h, header_w, 3), 240, dtype=np.uint8)
        cv2.putText(
            header, label.upper(), (pad, header_h - 8),
            FONT, 0.6, (30, 30, 30), 1, cv2.LINE_AA
        )
        rows.append(header)

        # ---- Image cells ----
        row_h = cell + 2 * pad
        row_w = (cell + pad) * n_cols + pad
        row   = np.full((row_h, row_w, 3), 255, dtype=np.uint8)

        for col_idx in range(n_cols):
            x = col_idx * (cell + pad) + pad
            y = pad

            if col_idx < len(images):
                img_gray = images[col_idx]
                img_up   = cv2.resize(img_gray, (cell, cell), interpolation=cv2.INTER_NEAREST)
                img_bgr  = cv2.cvtColor(img_up, cv2.COLOR_GRAY2BGR)
                row[y:y+cell, x:x+cell] = img_bgr
            else:
                # Empty cell — draw a grey rectangle
                cv2.rectangle(row, (x, y), (x+cell, y+cell), (200, 200, 200), -1)
                cv2.putText(
                    row, "none", (x + 12, y + cell//2),
                    FONT, 0.4, (120, 120, 120), 1
                )

        rows.append(row)

    # ---- Stack all rows ----
    grid = np.vstack(rows)
    return grid


def main() -> None:
    args    = parse_args()
    proc    = Path(args.processed)
    labels  = get_active_labels()
    n       = args.n

    if not proc.exists():
        print(f"[ERROR] Processed dataset not found: {proc}")
        sys.exit(1)

    print(f"\n  Previewing split='{args.split}', {n} samples per class ...")

    label_images: dict[str, list[np.ndarray]] = {}
    for label in sorted(labels):
        class_dir = proc / args.split / label
        if class_dir.exists():
            imgs = _load_random_images(class_dir, n)
            print(f"    {label}: {len(imgs)} loaded (of {n} requested)")
        else:
            imgs = []
            print(f"    {label}: [DIR NOT FOUND]")
        label_images[label] = imgs

    grid = _build_grid(label_images, n_cols=n)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)
        print(f"\n  Grid saved to: {out_path}")
    else:
        print("\n  Showing preview window. Press any key to close.")
        cv2.imshow("SafeSwim — Dataset Preview", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
