"""
dataset_builder.py
==================
Full pipeline orchestrator — converts the Roboflow YOLO dataset into a
HOG+SVM classification dataset.

Dataset specifics handled here:
    - Splits: train/ valid/ test/  (Roboflow naming convention)
    - Classes: 0=Drowning, 1=Person out of water, 2=Swimming
    - Mapping:  0 → "drowning"
                1 → "normal"  (merged: out of water = not drowning)
                2 → "normal"  (merged: swimming = not drowning)
    - Output: binary classification (drowning vs normal)

Image standardization:
    - Size: 64×128 px (canonical HOG pedestrian window, Dalal & Triggs 2005)
    - Grayscale: True (HOG is gradient-based, color adds no benefit)
    - Interpolation: INTER_AREA for downscale, INTER_LINEAR for upscale
"""

import hashlib
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .cropper import Cropper
from .yolo_parser import YoloParser

logger = logging.getLogger(__name__)

VALID_CLASSES     = {"normal", "drowning", "struggling"}
REVIEW_FOLDER     = "review"
SUPPORTED_IMAGES  = {".jpg", ".jpeg", ".png", ".bmp"}

# Roboflow uses "valid", not "val"
ROBOFLOW_SPLITS   = ["train", "valid", "test"]


class DatasetBuilder:
    """
    End-to-end pipeline:
        data/train + data/valid + data/test
            → crop all persons
            → standardize (64×128 grayscale)
            → 80/20 train/test split (stratified)
            → deduplicate
            → save to data/processed/

    Args:
        raw_dir         : Path to data/ folder
        processed_dir   : Output path (e.g. data/processed/)
        class_mapping   : Dict[class_id → class_name]
                          Example: {0:"drowning", 1:"normal", 2:"normal"}
        output_width    : Crop width  (default 64)
        output_height   : Crop height (default 128)
        grayscale       : Convert to grayscale (default True)
        min_crop_size   : Skip crops smaller than this (pixels, default 20)
        test_split_ratio: Fraction held out for test (default 0.2)
        seed            : Random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        class_mapping: Dict[int, str],
        output_width: int = 64,
        output_height: int = 128,
        grayscale: bool = True,
        min_crop_size: int = 20,
        test_split_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.class_mapping = class_mapping
        self.output_size   = (output_width, output_height)
        self.grayscale     = grayscale
        self.test_split    = test_split_ratio
        self.seed          = seed

        self.cropper = Cropper(min_crop_size=min_crop_size)
        self._stats: Dict = defaultdict(lambda: defaultdict(int))

    def build(self) -> Dict:
        """
        Run the full pipeline. Returns final crop counts per split/class.

        Steps:
            1. Validate class mapping
            2. Create output folder structure
            3. Process all splits (train / valid / test)
            4. 80/20 stratified split → train/ and test/
            5. Remove exact duplicate crops (MD5)
            6. Print summary
        """
        logger.info("Starting dataset build...")
        self._validate_mapping()
        self._create_output_dirs()

        # All crops go to staging first, then we split
        all_crops: Dict[str, List[Path]] = defaultdict(list)

        for split in ROBOFLOW_SPLITS:
            split_dir = self.raw_dir / split
            if split_dir.exists():
                self._process_split(split_dir, all_crops)
            else:
                logger.warning(f"Split not found, skipping: {split_dir}")

        self._split_and_save(all_crops)
        self._remove_duplicates()

        summary = self._count_output()
        self._print_summary(summary)
        return summary

    # ── Validation ──────────────────────────────────────────────────

    def _validate_mapping(self) -> None:
        for cid, name in self.class_mapping.items():
            if name not in VALID_CLASSES and name != REVIEW_FOLDER:
                raise ValueError(
                    f"Invalid class name '{name}' for class_id {cid}. "
                    f"Must be one of: {VALID_CLASSES}"
                )
        logger.info(f"Class mapping: {self.class_mapping}")

    # ── Directory setup ──────────────────────────────────────────────

    def _create_output_dirs(self) -> None:
        all_classes = set(self.class_mapping.values()) | {REVIEW_FOLDER}
        for split in ["train", "test"]:
            for cls in all_classes:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
        for cls in all_classes:
            (self.processed_dir / "staging" / cls).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output dirs created: {self.processed_dir}")

    # ── Processing ──────────────────────────────────────────────────

    def _process_split(self, split_dir: Path, all_crops: Dict) -> None:
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            logger.warning(f"No images/ in {split_dir}")
            return

        image_files = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGES
        ]
        logger.info(f"Processing {len(image_files)} images from {split_dir.name}/")

        for image_path in image_files:
            crops = self.cropper.crop_from_paths(image_path, labels_dir)

            for idx, (crop_arr, class_id) in enumerate(crops):
                class_name  = self.class_mapping.get(class_id, REVIEW_FOLDER)
                standardized = self._standardize(crop_arr)
                if standardized is None:
                    continue

                saved = self._save_staging(standardized, image_path, idx, class_name)
                if saved:
                    all_crops[class_name].append(saved)

    # ── Standardization ─────────────────────────────────────────────

    def _standardize(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        1. Grayscale (if enabled)
        2. Resize to target dimensions

        Interpolation:
            INTER_AREA   → best for downscaling (anti-aliasing)
            INTER_LINEAR → best for upscaling
        """
        try:
            if self.grayscale and len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            h, w = crop.shape[:2]
            tw, th = self.output_size
            interp = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR
            return cv2.resize(crop, (tw, th), interpolation=interp)

        except cv2.error as e:
            logger.warning(f"Standardization error: {e}")
            return None

    # ── Staging save ────────────────────────────────────────────────

    def _save_staging(
        self,
        img: np.ndarray,
        source: Path,
        idx: int,
        class_name: str,
    ) -> Optional[Path]:
        filename  = f"{source.stem}_crop{idx:03d}.jpg"
        save_path = self.processed_dir / "staging" / class_name / filename
        try:
            cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return save_path
        except Exception as e:
            logger.warning(f"Save failed for {filename}: {e}")
            return None

    # ── Train/test split ────────────────────────────────────────────

    def _split_and_save(self, all_crops: Dict[str, List[Path]]) -> None:
        """
        Stratified 80/20 split per class.
        Each class is shuffled and split independently so rare classes
        (drowning) are not accidentally all assigned to one split.
        """
        random.seed(self.seed)

        for class_name, paths in all_crops.items():
            if not paths:
                continue
            random.shuffle(paths)

            n_test = max(1, int(len(paths) * self.test_split))
            for p in paths[n_test:]:
                dest = self.processed_dir / "train" / class_name / p.name
                shutil.copy2(p, dest)

            for p in paths[:n_test]:
                dest = self.processed_dir / "test" / class_name / p.name
                shutil.copy2(p, dest)

        shutil.rmtree(self.processed_dir / "staging", ignore_errors=True)
        logger.info("Staging cleaned up.")

    # ── Deduplication ───────────────────────────────────────────────

    def _remove_duplicates(self) -> None:
        """Remove exact byte-level duplicate crops using MD5 hashing."""
        seen: set = set()
        removed = 0
        for img_path in sorted((self.processed_dir).rglob("*.jpg")):
            h = self._md5(img_path)
            if h in seen:
                img_path.unlink()
                removed += 1
            else:
                seen.add(h)
        if removed:
            logger.info(f"Removed {removed} exact duplicate crops.")

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── Summary ─────────────────────────────────────────────────────

    def _count_output(self) -> Dict:
        summary: Dict = {"train": {}, "test": {}}
        for split in ["train", "test"]:
            for cls_dir in (self.processed_dir / split).iterdir():
                if cls_dir.is_dir():
                    summary[split][cls_dir.name] = len(list(cls_dir.glob("*.jpg")))
        return summary

    def _print_summary(self, summary: Dict) -> None:
        print("\n" + "=" * 60)
        print("  DATASET BUILD COMPLETE")
        print("=" * 60)
        print(f"  Output    : {self.processed_dir}")
        print(f"  Crop size : {self.output_size[0]}×{self.output_size[1]} px")
        print(f"  Grayscale : {self.grayscale}")
        print(f"  Test %    : {int(self.test_split * 100)}%")

        total_all = 0
        for split in ["train", "test"]:
            print(f"\n  [{split.upper()}]")
            total = 0
            for cls, count in sorted(summary[split].items()):
                print(f"    {cls:<22}: {count} crops")
                total += count
            print(f"    {'TOTAL':<22}: {total} crops")
            total_all += total

        print(f"\n  Grand total: {total_all} crops")

        # Class imbalance warning
        train = summary.get("train", {})
        drowning_n = train.get("drowning", 0)
        normal_n   = train.get("normal", 0)
        if normal_n > 0 and drowning_n > 0:
            ratio = normal_n / drowning_n
            if ratio > 3:
                print(
                    f"\n  WARNING: Class imbalance detected "
                    f"(normal is {ratio:.1f}× more than drowning).\n"
                    f"  → Tell the SVM team to use class_weight='balanced'."
                )
        print("=" * 60 + "\n")
