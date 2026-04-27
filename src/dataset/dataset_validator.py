"""
dataset_validator.py
--------------------
Validates the processed classification dataset.

PURPOSE
-------
After build_dataset.py runs, this module proves that:
  1. All expected split and class folders exist.
  2. Every image file is readable by OpenCV.
  3. All images have the expected shape (64×64 greyscale).
  4. No image is all-black or all-white (degenerate crop).
  5. Class and split counts are reported clearly.
  6. Any corrupted or anomalous files are listed.

Role 2 can run this before starting HOG+SVM training to be confident
the dataset is clean.
"""

# ───────────────────────────────────────────────────────────────
# IMPORTS
# ───────────────────────────────────────────────────────────────

from __future__ import annotations  # allows modern type hints like list[str]

import cv2                          # OpenCV for image loading
import numpy as np                  # numerical operations (std, etc.)
from pathlib import Path            # clean file path handling
from dataclasses import dataclass, field  # structured data containers
from typing import Optional

from .class_mapping import get_active_labels  # returns labels used in dataset


# ───────────────────────────────────────────────────────────────
# EXPECTED CONFIGURATION (must match dataset_builder)
# ───────────────────────────────────────────────────────────────

# Expected image shape AFTER preprocessing
EXPECTED_SHAPE: tuple[int, int] = (64, 64)   # (height, width)

# Expected pixel type (grayscale 0–255)
EXPECTED_DTYPE: str = "uint8"

# If std < 1 → image is almost constant → useless
DEGENERATE_STD_THRESHOLD: float = 1.0


# ───────────────────────────────────────────────────────────────
# DATA STRUCTURES (for clean reporting)
# ───────────────────────────────────────────────────────────────

@dataclass
class ImageCheckResult:
    """
    Stores validation result for ONE image.
    """

    path: Path                      # file path
    readable: bool = True           # can OpenCV read it?
    shape_ok: bool = True           # correct size?
    not_degenerate: bool = True     # not blank?
    actual_shape: Optional[tuple] = None  # real image shape
    issue: Optional[str] = None     # error message (if any)


@dataclass
class ClassStats:
    """
    Stores stats for one (split + class).
    Example:
        train + drowning
        test + normal
    """

    split: str
    label: str

    count: int = 0                  # total images
    corrupt_count: int = 0          # unreadable images
    shape_mismatch_count: int = 0   # wrong dimensions
    degenerate_count: int = 0       # blank images

    issues: list[str] = field(default_factory=list)  # detailed problems

    @property
    def ok(self) -> bool:
        """
        True if no errors at all.
        """
        return (
            self.corrupt_count == 0
            and self.shape_mismatch_count == 0
            and self.degenerate_count == 0
        )


@dataclass
class ValidationReport:
    """
    Global report for the entire dataset.
    """

    processed_root: Path

    class_stats: list[ClassStats] = field(default_factory=list)
    missing_dirs: list[str] = field(default_factory=list)

    total_images: int = 0
    total_corrupt: int = 0
    total_shape_mismatch: int = 0
    total_degenerate: int = 0

    @property
    def passed(self) -> bool:
        """
        Dataset is valid if:
            - no missing directories
            - no corrupted images
            - no shape errors
        """
        return (
            len(self.missing_dirs) == 0
            and self.total_corrupt == 0
            and self.total_shape_mismatch == 0
        )


# ───────────────────────────────────────────────────────────────
# CHECK ONE IMAGE
# ───────────────────────────────────────────────────────────────

def _check_image(path: Path) -> ImageCheckResult:
    """
    Runs ALL validation checks on a single image.

    Checks:
        1. Readable by OpenCV
        2. Correct shape (64x64)
        3. Not blank (low std)
    """

    result = ImageCheckResult(path=path)

    # ---- 1. READABILITY ----
    # Load as grayscale
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        # OpenCV failed to read file
        result.readable = False
        result.shape_ok = False
        result.not_degenerate = False
        result.issue = "cv2.imread returned None"
        return result

    # Save actual shape (for debugging)
    result.actual_shape = img.shape  # (H, W)

    # ---- 2. SHAPE CHECK ----
    if img.shape != EXPECTED_SHAPE:
        result.shape_ok = False
        result.issue = (
            f"Expected shape {EXPECTED_SHAPE}, got {img.shape}"
        )

    # ---- 3. DEGENERACY CHECK ----
    # Compute standard deviation of pixel values
    std = float(np.std(img))

    # If std is very small → image is almost uniform (bad crop)
    if std < DEGENERATE_STD_THRESHOLD:
        result.not_degenerate = False

        msg = f"Image std={std:.3f} — likely blank/degenerate"

        # Combine issues if already exists
        result.issue = (result.issue + "; " + msg) if result.issue else msg

    return result


# ───────────────────────────────────────────────────────────────
# MAIN VALIDATION FUNCTION
# ───────────────────────────────────────────────────────────────

def validate_dataset(processed_root: str | Path) -> ValidationReport:
    """
    Validates the whole dataset.

    Steps:
        - Check all expected directories
        - Loop over all images
        - Run validation checks
        - Aggregate stats
    """

    processed_root = Path(processed_root)
    report = ValidationReport(processed_root=processed_root)

    # Get labels used (e.g. ["normal", "drowning"])
    active_labels = get_active_labels()

    # Only train/test exist after build
    splits_to_check = ["train", "test"]

    # Loop over splits and classes
    for split_name in splits_to_check:
        for label in active_labels:

            class_dir = processed_root / split_name / label

            # ---- Check directory existence ----
            if not class_dir.exists():
                report.missing_dirs.append(str(class_dir))
                continue

            # Initialize stats for this class
            stats = ClassStats(split=split_name, label=label)

            # Load all PNG images
            png_files = sorted(class_dir.glob("*.png"))
            stats.count = len(png_files)

            # ---- Check each image ----
            for img_path in png_files:

                chk = _check_image(img_path)

                report.total_images += 1

                # ---- Corrupt ----
                if not chk.readable:
                    stats.corrupt_count += 1
                    report.total_corrupt += 1
                    stats.issues.append(f"CORRUPT: {img_path.name}")

                # ---- Shape ----
                if not chk.shape_ok:
                    stats.shape_mismatch_count += 1
                    report.total_shape_mismatch += 1

                    if chk.readable:  # avoid duplicate messages
                        stats.issues.append(
                            f"SHAPE: {img_path.name} → {chk.actual_shape}"
                        )

                # ---- Blank ----
                if not chk.not_degenerate:
                    stats.degenerate_count += 1
                    report.total_degenerate += 1
                    stats.issues.append(f"BLANK: {img_path.name}")

            # Save stats
            report.class_stats.append(stats)

    return report


# ───────────────────────────────────────────────────────────────
# PRINT REPORT
# ───────────────────────────────────────────────────────────────

def print_validation_report(report: ValidationReport) -> None:
    """
    Prints a clean, structured report in the terminal.
    """

    sep = "=" * 60

    # Final status
    status = "✓ PASSED" if report.passed else "✗ FAILED"

    print(f"\n{sep}")
    print(f"  SAFESWIM — DATASET VALIDATION REPORT   [{status}]")
    print(f"{sep}")

    print(f"  Processed root : {report.processed_root}")
    print(f"  Total images   : {report.total_images}")
    print(f"  Corrupt files  : {report.total_corrupt}")
    print(f"  Shape errors   : {report.total_shape_mismatch}")
    print(f"  Blank/degenerate: {report.total_degenerate}")

    # ---- Missing folders ----
    if report.missing_dirs:
        print(f"\n  [WARN] Missing expected directories:")
        for d in report.missing_dirs:
            print(f"    ✗ {d}")
    else:
        print("\n  All expected split/class directories exist. ✓")

    # ---- Per-class stats ----
    print(f"\n  {'─'*56}")
    print("  PER-CLASS STATISTICS")
    print(f"  {'─'*56}")

    from itertools import groupby

    # Group stats by split (train/test)
    sorted_stats = sorted(report.class_stats, key=lambda s: s.split)

    for split_name, group in groupby(sorted_stats, key=lambda s: s.split):

        print(f"\n  Split: {split_name.upper()}")

        for stats in group:
            icon = "✓" if stats.ok else "✗"

            print(f"    {icon} {stats.label:<15} : {stats.count} images", end="")

            # Show errors if any
            if not stats.ok:
                print(
                    f"  [corrupt={stats.corrupt_count}, "
                    f"shape_err={stats.shape_mismatch_count}, "
                    f"blank={stats.degenerate_count}]",
                    end="",
                )

            print()

            # Show first 5 issues
            for issue in stats.issues[:5]:
                print(f"         → {issue}")

            if len(stats.issues) > 5:
                print(f"         → ... and {len(stats.issues)-5} more issues")

    # ---- Final verdict ----
    print(f"\n  {'─'*56}")

    if report.passed:
        print("  ✓ Dataset is VALID and ready for Role 2 (HOG + SVM training).")
    else:
        print("  ✗ Dataset has issues. Fix errors above before training.")

    print(f"{sep}\n")