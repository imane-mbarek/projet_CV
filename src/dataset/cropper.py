"""
cropper.py
==========
Extracts person patch crops from full images using YOLO bounding boxes.

Each crop becomes one training sample for HOG + SVM.
Multiple persons per image → multiple crops (all extracted).
"""

# ───────────────────────────────────────────────────────────────
# IMPORTS
# ───────────────────────────────────────────────────────────────

import logging                     # For debug / warning messages
from pathlib import Path           # Clean path handling
from typing import List, Optional, Tuple  # Type hints

import cv2                         # OpenCV for image processing
import numpy as np                 # Image arrays

from .yolo_parser import BoundingBox, YoloParser  # Custom YOLO parser

# Logger instance (used to print debug info)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# MAIN CLASS
# ───────────────────────────────────────────────────────────────

class Cropper:
    """
    Extracts and returns person crops as numpy arrays.

    Args:
        min_crop_size    : Minimum width AND height (pixels)
                           → avoids tiny useless crops
        padding_fraction : Adds extra space around the bounding box
                           → gives context (important for ML)
    """

    def __init__(self, min_crop_size: int = 20, padding_fraction: float = 0.05):

        # Minimum allowed size for a crop (width AND height)
        self.min_crop_size = min_crop_size

        # Percentage of padding added around bounding box
        self.padding_fraction = padding_fraction

        # YOLO parser (reads .txt annotation files)
        self.parser = YoloParser()


    # ───────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ───────────────────────────────────────────────────────────

    def crop_from_paths(
        self,
        image_path: str | Path,
        labels_dir: str | Path,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Automatically:
            1. Reads YOLO annotation file
            2. Extracts bounding boxes
            3. Crops all persons

        Returns:
            List of tuples:
                (image_crop_array, class_id)
        """

        # Parse YOLO labels → list of BoundingBox objects
        boxes = self.parser.parse_for_image(image_path, labels_dir)

        # If no annotations → no crops
        if not boxes:
            logger.debug(f"No annotations for {Path(image_path).name}")
            return []

        # Otherwise crop all bounding boxes
        return self.crop_image(image_path, boxes)


    # ───────────────────────────────────────────────────────────
    # PROCESS ONE IMAGE
    # ───────────────────────────────────────────────────────────

    def crop_image(
        self,
        image_path: str | Path,
        boxes: List[BoundingBox],
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Crops ALL bounding boxes from a single image.

        Example:
            1 image with 3 persons → returns 3 crops
        """

        # Load image from disk
        img = self._load_image(Path(image_path))

        # If image failed to load → return empty
        if img is None:
            return []

        # Get image dimensions
        img_h, img_w = img.shape[:2]

        crops = []  # list of (crop, class_id)

        # Loop over each bounding box
        for i, box in enumerate(boxes):

            # Extract crop for this bounding box
            crop = self._extract_crop(img, box, img_w, img_h, Path(image_path), i)

            # Only keep valid crops
            if crop is not None:
                crops.append((crop, box.class_id))

        return crops


    # ───────────────────────────────────────────────────────────
    # LOAD IMAGE
    # ───────────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Loads an image using OpenCV.

        Returns:
            numpy array (image) OR None if failed
        """

        img = cv2.imread(str(path))

        # If OpenCV fails → img = None
        if img is None:
            logger.warning(f"Cannot load image: {path}")

        return img


    # ───────────────────────────────────────────────────────────
    # EXTRACT ONE CROP
    # ───────────────────────────────────────────────────────────

    def _extract_crop(
        self,
        img: np.ndarray,
        box: BoundingBox,
        img_w: int,
        img_h: int,
        src: Path,
        idx: int,
    ) -> Optional[np.ndarray]:
        """
        Extracts ONE crop from the image using bounding box.

        Steps:
            1. Convert YOLO box → pixel coordinates
            2. Add padding (context)
            3. Clip to image boundaries
            4. Check size validity
            5. Return cropped image
        """

        # Convert normalized YOLO coords → pixel coords
        x1, y1, x2, y2 = box.to_pixel_coords(img_w, img_h)

        # ── Add padding around the bounding box ──
        if self.padding_fraction > 0:

            # Width & height of the box
            bw, bh = x2 - x1, y2 - y1

            # Padding size
            px = int(bw * self.padding_fraction)
            py = int(bh * self.padding_fraction)

            # Expand bounding box (with clipping to image borders)
            x1 = max(0,      x1 - px)
            y1 = max(0,      y1 - py)
            x2 = min(img_w,  x2 + px)
            y2 = min(img_h,  y2 + py)

        # ── Compute final crop size ──
        cw, ch = x2 - x1, y2 - y1

        # ── Validate crop size ──
        if cw < self.min_crop_size or ch < self.min_crop_size or cw <= 0 or ch <= 0:

            logger.debug(
                f"{src.name}[{idx}]: crop too small ({cw}×{ch}) — skipped"
            )
            return None

        # ── Extract crop from image ──
        # Note: numpy slicing → img[row, col] = img[y, x]
        crop = img[y1:y2, x1:x2]

        # If something went wrong → empty array
        if crop.size == 0:
            return None

        return crop