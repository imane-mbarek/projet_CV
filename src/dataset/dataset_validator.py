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

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .class_mapping import get_active_labels

# ---------------------------------------------------------------------------
# Expected configuration (must match cropper.py)
# ---------------------------------------------------------------------------

EXPECTED_SHAPE: tuple[int, int] = (64, 64)   # (height, width) — numpy row,col order
EXPECTED_DTYPE: str = "uint8"
DEGENERATE_STD_THRESHOLD: float = 1.0        # std < 1.0 → near-blank image


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageCheckResult:
    """Result of checking one image file."""
    path: Path
    readable: bool = True
    shape_ok: bool = True
    not_degenerate: bool = True
    actual_shape: Optional[tuple] = None
    issue: Optional[str] = None


@dataclass
class ClassStats:
    """Statistics for one (split, class) combination."""
    split: str
    label: str
    count: int = 0
    corrupt_count: int = 0
    shape_mismatch_count: int = 0
    degenerate_count: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.corrupt_count == 0
            and self.shape_mismatch_count == 0
            and self.degenerate_count == 0
        )


@dataclass
class ValidationReport:
    """Full validation report for the processed dataset."""
    processed_root: Path
    class_stats: list[ClassStats] = field(default_factory=list)
    missing_dirs: list[str] = field(default_factory=list)
    total_images: int = 0
    total_corrupt: int = 0
    total_shape_mismatch: int = 0
    total_degenerate: int = 0

    @property
    def passed(self) -> bool:
        return (
            len(self.missing_dirs) == 0
            and self.total_corrupt == 0
            and self.total_shape_mismatch == 0
        )


# ---------------------------------------------------------------------------
# Per-image check
# ---------------------------------------------------------------------------

def _check_image(path: Path) -> ImageCheckResult:
    """Run all checks on a single image file."""
    result = ImageCheckResult(path=path)

    # ---- Readability ----
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        result.readable = False
        result.shape_ok = False
        result.not_degenerate = False
        result.issue = "cv2.imread returned None"
        return result

    result.actual_shape = img.shape  # (H, W) for grayscale

    # ---- Shape ----
    if img.shape != EXPECTED_SHAPE:
        result.shape_ok = False
        result.issue = (
            f"Expected shape {EXPECTED_SHAPE}, got {img.shape}"
        )

    # ---- Degeneracy (near-blank) ----
    std = float(np.std(img))
    if std < DEGENERATE_STD_THRESHOLD:
        result.not_degenerate = False
        msg = f"Image std={std:.3f} — likely blank/degenerate"
        result.issue = (result.issue + "; " + msg) if result.issue else msg

    return result


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_dataset(processed_root: str | Path) -> ValidationReport:
    """
    Validate the entire processed dataset.

    Parameters
    ----------
    processed_root : path to data/dataset_processed/

    Returns
    -------
    ValidationReport with all findings.
    """
    processed_root = Path(processed_root)
    report = ValidationReport(processed_root=processed_root)

    active_labels = get_active_labels()
    splits_to_check = ["train", "test"]

    for split_name in splits_to_check:
        for label in active_labels:
            class_dir = processed_root / split_name / label

            if not class_dir.exists():
                report.missing_dirs.append(str(class_dir))
                continue

            stats = ClassStats(split=split_name, label=label)
            png_files = sorted(class_dir.glob("*.png"))
            stats.count = len(png_files)

            for img_path in png_files:
                chk = _check_image(img_path)
                report.total_images += 1

                if not chk.readable:
                    stats.corrupt_count += 1
                    report.total_corrupt += 1
                    stats.issues.append(f"CORRUPT: {img_path.name}")

                if not chk.shape_ok:
                    stats.shape_mismatch_count += 1
                    report.total_shape_mismatch += 1
                    if chk.readable:  # avoid double-reporting
                        stats.issues.append(
                            f"SHAPE: {img_path.name} → {chk.actual_shape}"
                        )

                if not chk.not_degenerate:
                    stats.degenerate_count += 1
                    report.total_degenerate += 1
                    stats.issues.append(f"BLANK: {img_path.name}")

            report.class_stats.append(stats)

    return report


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_validation_report(report: ValidationReport) -> None:
    """Print a structured validation report."""
    sep = "=" * 60
    status = "✓ PASSED" if report.passed else "✗ FAILED"

    print(f"\n{sep}")
    print(f"  SAFESWIM — DATASET VALIDATION REPORT   [{status}]")
    print(f"{sep}")
    print(f"  Processed root : {report.processed_root}")
    print(f"  Total images   : {report.total_images}")
    print(f"  Corrupt files  : {report.total_corrupt}")
    print(f"  Shape errors   : {report.total_shape_mismatch}")
    print(f"  Blank/degenerate: {report.total_degenerate}")

    # ---- Missing directories ----
    if report.missing_dirs:
        print(f"\n  [WARN] Missing expected directories:")
        for d in report.missing_dirs:
            print(f"    ✗ {d}")
    else:
        print("\n  All expected split/class directories exist. ✓")

    # ---- Per-class breakdown ----
    print(f"\n  {'─'*56}")
    print("  PER-CLASS STATISTICS")
    print(f"  {'─'*56}")

    # Group by split
    from itertools import groupby
    sorted_stats = sorted(report.class_stats, key=lambda s: s.split)
    for split_name, group in groupby(sorted_stats, key=lambda s: s.split):
        print(f"\n  Split: {split_name.upper()}")
        for stats in group:
            icon = "✓" if stats.ok else "✗"
            print(f"    {icon} {stats.label:<15} : {stats.count} images", end="")
            if not stats.ok:
                print(
                    f"  [corrupt={stats.corrupt_count}, "
                    f"shape_err={stats.shape_mismatch_count}, "
                    f"blank={stats.degenerate_count}]",
                    end="",
                )
            print()
            # Show first few issues
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
