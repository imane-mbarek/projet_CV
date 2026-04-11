"""
cropper.py
==========
Extracts person patch crops from full images using YOLO bounding boxes.

Each crop becomes one training sample for HOG + SVM.
Multiple persons per image → multiple crops (all extracted).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .yolo_parser import BoundingBox, YoloParser

logger = logging.getLogger(__name__)


class Cropper:
    """
    Extracts and returns person crops as numpy arrays.

    Args:
        min_crop_size    : Minimum valid width AND height in pixels (default 20)
        padding_fraction : Extra context added around each box as a
                           fraction of box size (default 0.05 = 5%)
    """

    def __init__(self, min_crop_size: int = 20, padding_fraction: float = 0.05):
        self.min_crop_size = min_crop_size
        self.padding_fraction = padding_fraction
        self.parser = YoloParser()

    def crop_from_paths(
        self,
        image_path: str | Path,
        labels_dir: str | Path,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Parse label file automatically then crop all bounding boxes.

        Returns:
            List of (crop_array, class_id) — one entry per valid person crop.
        """
        boxes = self.parser.parse_for_image(image_path, labels_dir)
        if not boxes:
            logger.debug(f"No annotations for {Path(image_path).name}")
            return []
        return self.crop_image(image_path, boxes)

    def crop_image(
        self,
        image_path: str | Path,
        boxes: List[BoundingBox],
    ) -> List[Tuple[np.ndarray, int]]:
        """Crop all bounding boxes from one image."""
        img = self._load_image(Path(image_path))
        if img is None:
            return []

        img_h, img_w = img.shape[:2]
        crops = []

        for i, box in enumerate(boxes):
            crop = self._extract_crop(img, box, img_w, img_h, Path(image_path), i)
            if crop is not None:
                crops.append((crop, box.class_id))

        return crops

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Cannot load image: {path}")
        return img

    def _extract_crop(
        self,
        img: np.ndarray,
        box: BoundingBox,
        img_w: int,
        img_h: int,
        src: Path,
        idx: int,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box.to_pixel_coords(img_w, img_h)

        # Add padding for context around the person
        if self.padding_fraction > 0:
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * self.padding_fraction), int(bh * self.padding_fraction)
            x1 = max(0,      x1 - px)
            y1 = max(0,      y1 - py)
            x2 = min(img_w,  x2 + px)
            y2 = min(img_h,  y2 + py)

        cw, ch = x2 - x1, y2 - y1
        if cw < self.min_crop_size or ch < self.min_crop_size or cw <= 0 or ch <= 0:
            logger.debug(f"{src.name}[{idx}]: crop too small ({cw}×{ch}) — skipped")
            return None

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop
