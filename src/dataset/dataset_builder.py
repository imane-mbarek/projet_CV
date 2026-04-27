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

# ───────────────────────────────────────────────────────────────
# IMPORTS
# ───────────────────────────────────────────────────────────────

import hashlib            # Used to generate hash (MD5) for duplicate detection
import logging            # For logging progress and warnings
import random             # For random shuffling (train/test split)
import shutil             # For copying and deleting files
from collections import defaultdict  # Dictionary with default values
from pathlib import Path  # Clean path handling (instead of strings)
from typing import Dict, List, Optional, Tuple  # Type hints (readability)

import cv2                # OpenCV (image processing)
import numpy as np        # Numerical arrays (images)

from .cropper import Cropper       # Custom module: extracts crops from images
from .yolo_parser import YoloParser  # (Imported but not directly used here)

# Logger for debug/info messages
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# CONSTANTS (global configuration)
# ───────────────────────────────────────────────────────────────

VALID_CLASSES     = {"normal", "drowning", "struggling"}  # Allowed class names
REVIEW_FOLDER     = "review"  # Used for unknown/unmapped classes
SUPPORTED_IMAGES  = {".jpg", ".jpeg", ".png", ".bmp"}  # Allowed formats

# Roboflow uses "valid" instead of "val"
ROBOFLOW_SPLITS   = ["train", "valid", "test"]


# ───────────────────────────────────────────────────────────────
# MAIN CLASS
# ───────────────────────────────────────────────────────────────

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
        output_width    : Width of output image (default 64)
        output_height   : Height of output image (default 128)
        grayscale       : Convert to grayscale (default True)
        min_crop_size   : Ignore very small crops
        test_split_ratio: Percentage for test set
        seed            : Random seed (reproducibility)
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
        # Convert paths to Path objects
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Mapping YOLO class_id → class_name
        self.class_mapping = class_mapping

        # Output image size (width, height)
        self.output_size   = (output_width, output_height)

        # Whether to convert images to grayscale
        self.grayscale     = grayscale

        # Percentage for test set
        self.test_split    = test_split_ratio

        # Random seed (important for reproducibility)
        self.seed          = seed

        # Cropper object (handles bounding boxes extraction)
        self.cropper = Cropper(min_crop_size=min_crop_size)

        # Statistics dictionary
        self._stats: Dict = defaultdict(lambda: defaultdict(int))


    # ───────────────────────────────────────────────────────────
    # MAIN PIPELINE FUNCTION
    # ───────────────────────────────────────────────────────────

    def build(self) -> Dict:
        """
        Runs the full pipeline.

        Steps:
            1. Validate class mapping
            2. Create output directories
            3. Process all dataset splits
            4. Split into train/test
            5. Remove duplicates
            6. Print summary
        """

        logger.info("Starting dataset build...")

        # Step 1: check if mapping is correct
        self._validate_mapping()

        # Step 2: create folders
        self._create_output_dirs()

        # Temporary storage for all crops before splitting
        all_crops: Dict[str, List[Path]] = defaultdict(list)

        # Step 3: process each split
        for split in ROBOFLOW_SPLITS:
            split_dir = self.raw_dir / split

            if split_dir.exists():
                self._process_split(split_dir, all_crops)
            else:
                logger.warning(f"Split not found, skipping: {split_dir}")

        # Step 4: split into train/test
        self._split_and_save(all_crops)

        # Step 5: remove duplicate images
        self._remove_duplicates()

        # Step 6: summary
        summary = self._count_output()
        self._print_summary(summary)

        return summary


    # ───────────────────────────────────────────────────────────
    # VALIDATION
    # ───────────────────────────────────────────────────────────

    def _validate_mapping(self) -> None:
        """
        Ensures that all class names are valid.
        """
        for cid, name in self.class_mapping.items():
            if name not in VALID_CLASSES and name != REVIEW_FOLDER:
                raise ValueError(
                    f"Invalid class name '{name}' for class_id {cid}. "
                    f"Must be one of: {VALID_CLASSES}"
                )
        logger.info(f"Class mapping: {self.class_mapping}")


    # ───────────────────────────────────────────────────────────
    # DIRECTORY CREATION
    # ───────────────────────────────────────────────────────────

    def _create_output_dirs(self) -> None:
        """
        Creates folder structure:
            processed/
                train/
                test/
                staging/
        """

        all_classes = set(self.class_mapping.values()) | {REVIEW_FOLDER}

        # Create train/test folders
        for split in ["train", "test"]:
            for cls in all_classes:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

        # Create staging folder (temporary)
        for cls in all_classes:
            (self.processed_dir / "staging" / cls).mkdir(parents=True, exist_ok=True)

        logger.info(f"Output dirs created: {self.processed_dir}")


    # ───────────────────────────────────────────────────────────
    # PROCESS ONE SPLIT (train/valid/test)
    # ───────────────────────────────────────────────────────────

    def _process_split(self, split_dir: Path, all_crops: Dict) -> None:
        """
        Processes one dataset split:
            - Load images
            - Crop persons using YOLO labels
            - Standardize images
            - Save to staging
        """

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            logger.warning(f"No images/ in {split_dir}")
            return

        # Get all image files
        image_files = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGES
        ]

        logger.info(f"Processing {len(image_files)} images from {split_dir.name}/")

        # Loop over each image
        for image_path in image_files:

            # Get crops (image regions + class_id)
            crops = self.cropper.crop_from_paths(image_path, labels_dir)

            for idx, (crop_arr, class_id) in enumerate(crops):

                # Convert class_id → class_name
                class_name  = self.class_mapping.get(class_id, REVIEW_FOLDER)

                # Standardize image
                standardized = self._standardize(crop_arr)

                if standardized is None:
                    continue

                # Save to staging folder
                saved = self._save_staging(standardized, image_path, idx, class_name)

                if saved:
                    all_crops[class_name].append(saved)


    # ───────────────────────────────────────────────────────────
    # IMAGE STANDARDIZATION
    # ───────────────────────────────────────────────────────────

    def _standardize(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts image to:
            - Grayscale (optional)
            - Fixed size (64×128)

        Also chooses correct interpolation method.
        """

        try:
            # Convert to grayscale if needed
            if self.grayscale and len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Current size
            h, w = crop.shape[:2]

            # Target size
            tw, th = self.output_size

            # Choose interpolation method
            # Downscale → INTER_AREA (better quality)
            # Upscale   → INTER_LINEAR
            interp = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR

            # Resize image
            return cv2.resize(crop, (tw, th), interpolation=interp)

        except cv2.error as e:
            logger.warning(f"Standardization error: {e}")
            return None


    # ───────────────────────────────────────────────────────────
    # SAVE TO STAGING
    # ───────────────────────────────────────────────────────────

    def _save_staging(
        self,
        img: np.ndarray,
        source: Path,
        idx: int,
        class_name: str,
    ) -> Optional[Path]:
        """
        Saves image temporarily before train/test split.
        """

        filename  = f"{source.stem}_crop{idx:03d}.jpg"
        save_path = self.processed_dir / "staging" / class_name / filename

        try:
            # Save image with high quality
            cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return save_path
        except Exception as e:
            logger.warning(f"Save failed for {filename}: {e}")
            return None


    # ───────────────────────────────────────────────────────────
    # TRAIN / TEST SPLIT
    # ───────────────────────────────────────────────────────────

    def _split_and_save(self, all_crops: Dict[str, List[Path]]) -> None:
        """
        Splits dataset into train/test (stratified per class).
        """

        random.seed(self.seed)

        for class_name, paths in all_crops.items():

            if not paths:
                continue

            # Shuffle data randomly
            random.shuffle(paths)

            # Number of test samples
            n_test = max(1, int(len(paths) * self.test_split))

            # Train set
            for p in paths[n_test:]:
                dest = self.processed_dir / "train" / class_name / p.name
                shutil.copy2(p, dest)

            # Test set
            for p in paths[:n_test]:
                dest = self.processed_dir / "test" / class_name / p.name
                shutil.copy2(p, dest)

        # Remove staging folder after split
        shutil.rmtree(self.processed_dir / "staging", ignore_errors=True)

        logger.info("Staging cleaned up.")


    # ───────────────────────────────────────────────────────────
    # REMOVE DUPLICATES
    # ───────────────────────────────────────────────────────────

    def _remove_duplicates(self) -> None:
        """
        Removes exact duplicate images using MD5 hash.
        """

        seen: set = set()
        removed = 0

        for img_path in sorted((self.processed_dir).rglob("*.jpg")):

            # Compute hash
            h = self._md5(img_path)

            if h in seen:
                img_path.unlink()  # Delete duplicate
                removed += 1
            else:
                seen.add(h)

        if removed:
            logger.info(f"Removed {removed} exact duplicate crops.")


    @staticmethod
    def _md5(path: Path) -> str:
        """
        Compute MD5 hash of a file.
        """
        h = hashlib.md5()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)

        return h.hexdigest()


    # ───────────────────────────────────────────────────────────
    # SUMMARY
    # ───────────────────────────────────────────────────────────

    def _count_output(self) -> Dict:
        """
        Counts number of images per class per split.
        """

        summary: Dict = {"train": {}, "test": {}}

        for split in ["train", "test"]:
            for cls_dir in (self.processed_dir / split).iterdir():
                if cls_dir.is_dir():
                    summary[split][cls_dir.name] = len(list(cls_dir.glob("*.jpg")))

        return summary


    def _print_summary(self, summary: Dict) -> None:
        """
        Prints final dataset statistics.
        """

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

        # Check class imbalance
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