"""
cropper.py
----------
Reads images, applies parsed bounding boxes, extracts person crops,
standardizes them, and writes them to the processed dataset folder.

DESIGN DECISIONS
----------------

Output size: 64×64 pixels
    HOG operates on fixed-size windows. 64×64 gives ~3780 HOG features
    with default parameters — large enough for body pose, small enough
    for fast SVM training. A student project does not need 128×128.

Grayscale output:
    HOG is a gradient-based descriptor and does not benefit from color.
    Converting to grayscale halves memory usage and avoids artificially
    inflating the feature vector with redundant colour channels.

Output format: PNG
    Lossless. Avoids JPEG compression artefacts that could affect HOG
    gradients, especially on small 64×64 patches. File size difference
    is negligible at this resolution.

No pixel normalisation at this stage:
    Normalisation (0–1 float scaling) is Role 2's responsibility and
    should happen inside the HOG extraction step, not during I/O. Saving
    normalised floats to disk would require .npy files and complicates
    inspection.

Padding strategy for small crops:
    When a bounding box is valid but produces a crop smaller than
    OUTPUT_SIZE, we resize using INTER_LINEAR (bilinear interpolation).
    This is gentler than INTER_NEAREST for upscaling small patches.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from .yolo_parser import BoundingBox, parse_label_file, find_label_for_image
from .class_mapping import map_class_id

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_SIZE: tuple[int, int] = (64, 64)   # (width, height) in pixels
MIN_CROP_SIZE: int = 20                    # discard crops smaller than this
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------

@dataclass
class CropResult:
    """Statistics for one processed image."""
    image_path: Path
    crops_saved: int = 0
    crops_skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def crop_image(
    image_path: Path,
    labels_dir: Path,
    output_root: Path,
    split_name: str,
    crop_counter: dict[str, int],
) -> CropResult:
    """
    Extract all person crops from one image and save them to the
    processed dataset.

    Parameters
    ----------
    image_path   : path to the source image
    labels_dir   : directory containing matching .txt label files
    output_root  : root of the processed dataset (e.g. data/dataset_processed/)
    split_name   : "train" or "test"
    crop_counter : shared dict {label: count} used to generate unique filenames

    Returns
    -------
    CropResult with per-image statistics.
    """
    result = CropResult(image_path=image_path)

    # ---- 1. Read the image ----
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        result.error = "cv2.imread returned None (corrupted or unsupported format)"
        return result

    img_h, img_w = img_bgr.shape[:2]

    # ---- 2. Find matching label file ----
    label_path = find_label_for_image(image_path, labels_dir)
    if label_path is None:
        result.error = "No matching .txt label file found"
        return result

    # ---- 3. Parse the label file ----
    parse_result = parse_label_file(
        label_path=label_path,
        image_width=img_w,
        image_height=img_h,
        min_box_size=MIN_CROP_SIZE,
    )

    if not parse_result.ok:
        result.error = f"Label parse error: {parse_result.error}"
        return result

    if parse_result.skipped_lines:
        result.skip_reasons.append(
            f"{len(parse_result.skipped_lines)} malformed annotation line(s) skipped"
        )

    if not parse_result.boxes:
        result.skip_reasons.append("Label file has no valid boxes")
        return result

    # ---- 4. Process each bounding box ----
    for box in parse_result.boxes:
        saved, reason = _process_box(
            box=box,
            img_bgr=img_bgr,
            img_w=img_w,
            img_h=img_h,
            output_root=output_root,
            split_name=split_name,
            source_stem=image_path.stem,
            crop_counter=crop_counter,
        )
        if saved:
            result.crops_saved += 1
        else:
            result.crops_skipped += 1
            if reason:
                result.skip_reasons.append(reason)

    return result


def _process_box(
    box: BoundingBox,
    img_bgr: np.ndarray,
    img_w: int,
    img_h: int,
    output_root: Path,
    split_name: str,
    source_stem: str,
    crop_counter: dict[str, int],
) -> tuple[bool, str | None]:
    """
    Crop, standardize and save a single bounding box.

    Returns (True, None) on success or (False, reason_string) on skip.
    """
    # ---- Map class ID to target label ----
    label = map_class_id(box.class_id)
    if label is None:
        return False, f"class_id {box.class_id} not in CLASS_MAPPING — skipped"

    # ---- Validate box dimensions ----
    if box.width < MIN_CROP_SIZE or box.height < MIN_CROP_SIZE:
        return False, (
            f"Crop too small ({box.width}×{box.height} < {MIN_CROP_SIZE}px) — skipped"
        )

    # ---- Sanity check box is within image ----
    if box.x1 >= img_w or box.y1 >= img_h or box.x2 <= 0 or box.y2 <= 0:
        return False, "Box is entirely outside image bounds — skipped"

    # ---- Extract the crop (already clamped by parser) ----
    crop_bgr = img_bgr[box.y1:box.y2, box.x1:box.x2]

    if crop_bgr.size == 0:
        return False, "Crop array is empty after slicing — skipped"

    # ---- Convert to grayscale ----
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Resize to standard output size ----
    crop_resized = cv2.resize(
        crop_gray,
        OUTPUT_SIZE,
        interpolation=cv2.INTER_LINEAR,
    )

    # ---- Build output path ----
    out_dir = output_root / split_name / label
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_counter[label] = crop_counter.get(label, 0) + 1
    filename = f"{source_stem}_c{box.class_id}_{crop_counter[label]:05d}.png"
    out_path = out_dir / filename

    # ---- Save ----
    success = cv2.imwrite(str(out_path), crop_resized)
    if not success:
        return False, f"cv2.imwrite failed for {out_path}"

    return True, None


# ---------------------------------------------------------------------------
# Batch processor for one split
# ---------------------------------------------------------------------------

def process_split(
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    split_name: str,
    verbose: bool = True,
) -> list[CropResult]:
    """
    Process all images in one dataset split.

    Parameters
    ----------
    images_dir  : e.g. data/dataset_raw/train/images/
    labels_dir  : e.g. data/dataset_raw/train/labels/
    output_root : e.g. data/dataset_processed/
    split_name  : "train" or "test"
    verbose     : print per-image progress

    Returns
    -------
    List of CropResult, one per image.
    """
    if not images_dir.exists():
        print(f"  [SKIP] {images_dir} does not exist.")
        return []

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"  [SKIP] No images found in {images_dir}")
        return []

    results: list[CropResult] = []
    crop_counter: dict[str, int] = {}

    total = len(image_paths)
    print(f"\n  Processing split '{split_name}' — {total} images ...")

    for i, img_path in enumerate(image_paths, start=1):
        res = crop_image(
            image_path=img_path,
            labels_dir=labels_dir,
            output_root=output_root,
            split_name=split_name,
            crop_counter=crop_counter,
        )
        results.append(res)

        if verbose and (i % 100 == 0 or i == total):
            saved  = sum(r.crops_saved for r in results)
            errors = sum(1 for r in results if not r.ok)
            print(f"    [{i}/{total}] crops saved so far: {saved} | errors: {errors}")

    return results
