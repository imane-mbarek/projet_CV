"""
dataset_builder.py
------------------
Top-level orchestrator for the dataset treatment pipeline.

SPLITTING STRATEGY (explained)
-------------------------------
The raw dataset already has train / valid / test splits.

Option A — Keep all three splits separately
    Pros: preserves original split logic.
    Cons: valid is typically small and HOG+SVM uses cross-validation
          internally (GridSearchCV), so a separate val split adds no value.

Option B — Merge train+valid → processed/train/, keep test → processed/test/
    Pros: maximises training data; test stays untouched for final evaluation.
    Cons: none for this use case.

Option C — Ignore all raw splits and re-split from scratch
    Pros: full control over class balance.
    Cons: risks data leakage if the same person appears in train and test
          (we cannot verify this without metadata); more complexity.

→ RECOMMENDATION: Option B
    Merge train and valid into processed/train/.
    Keep test as processed/test/ (untouched evaluation set).
    This is standard practice for academic HOG+SVM projects.
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field

from .cropper import process_split, CropResult
from .class_mapping import print_mapping_summary


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------

@dataclass
class BuildConfig:
    """All paths and settings for one pipeline run."""
    raw_root: Path       # data/dataset_raw/
    processed_root: Path # data/dataset_processed/
    verbose: bool = True


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------

@dataclass
class BuildReport:
    """Aggregated results from the full build."""
    config: BuildConfig
    split_results: dict[str, list[CropResult]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def total_saved(self) -> int:
        return sum(
            r.crops_saved
            for results in self.split_results.values()
            for r in results
        )

    def total_errors(self) -> int:
        return sum(
            1
            for results in self.split_results.values()
            for r in results
            if not r.ok
        )

    def total_skipped_crops(self) -> int:
        return sum(
            r.crops_skipped
            for results in self.split_results.values()
            for r in results
        )


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_dataset(config: BuildConfig) -> BuildReport:
    """
    Run the full dataset conversion pipeline:
        raw YOLO dataset → processed classification dataset

    Steps
    -----
    1. Print configuration summary.
    2. Merge train + valid into processed/train/.
    3. Convert test into processed/test/.
    4. Print final build report.

    Parameters
    ----------
    config : BuildConfig with source and destination paths.

    Returns
    -------
    BuildReport with per-split statistics.
    """
    t_start = time.time()
    report = BuildReport(config=config)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  SAFESWIM — DATASET BUILD PIPELINE")
    print(f"{sep}")
    print(f"  Source : {config.raw_root}")
    print(f"  Output : {config.processed_root}")
    print_mapping_summary()

    # ---- Ensure output root exists ----
    config.processed_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # TRAIN — merge raw 'train' and 'valid' into processed 'train'
    # ----------------------------------------------------------------
    train_results: list[CropResult] = []

    for raw_split in ("train", "valid"):
        raw_images = config.raw_root / raw_split / "images"
        raw_labels = config.raw_root / raw_split / "labels"

        results = process_split(
            images_dir=raw_images,
            labels_dir=raw_labels,
            output_root=config.processed_root,
            split_name="train",
            verbose=config.verbose,
        )
        train_results.extend(results)

    report.split_results["train"] = train_results

    # ----------------------------------------------------------------
    # TEST — raw 'test' → processed 'test' (kept separate)
    # ----------------------------------------------------------------
    test_images = config.raw_root / "test" / "images"
    test_labels = config.raw_root / "test" / "labels"

    test_results = process_split(
        images_dir=test_images,
        labels_dir=test_labels,
        output_root=config.processed_root,
        split_name="test",
        verbose=config.verbose,
    )
    report.split_results["test"] = test_results

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    report.elapsed_seconds = time.time() - t_start
    _print_build_report(report)
    return report


def _print_build_report(report: BuildReport) -> None:
    """Print a concise build summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  BUILD COMPLETE")
    print(f"{sep}")
    print(f"  Total crops saved   : {report.total_saved()}")
    print(f"  Total crops skipped : {report.total_skipped_crops()}")
    print(f"  Images with errors  : {report.total_errors()}")
    print(f"  Time elapsed        : {report.elapsed_seconds:.1f}s")

    # Per processed split breakdown
    for split_name, results in report.split_results.items():
        saved   = sum(r.crops_saved for r in results)
        skipped = sum(r.crops_skipped for r in results)
        errors  = sum(1 for r in results if not r.ok)
        print(f"\n  Split '{split_name}':")
        print(f"    crops saved   : {saved}")
        print(f"    crops skipped : {skipped}")
        print(f"    image errors  : {errors}")
        if errors:
            # Show first few error messages
            for r in results:
                if not r.ok:
                    print(f"      ✗ {r.image_path.name}: {r.error}")
                    if sum(1 for r2 in results if not r2.ok) > 5:
                        break  # Don't flood the console

    # Count per-class output files
    proc_root = report.config.processed_root
    print(f"\n  Output class counts:")
    for split_name in ("train", "test"):
        split_dir = proc_root / split_name
        if split_dir.exists():
            print(f"    {split_name}/")
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.png")))
                    print(f"      {class_dir.name:<15}: {count} images")

    print(f"\n  Output written to: {report.config.processed_root}\n")
