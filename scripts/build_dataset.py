#!/usr/bin/env python3
"""
build_dataset.py
================
SafeSwim -- Data Pipeline Entry Point

Dataset: Roboflow "swimming-drowning-2" (YOLOv8 format)
Classes from data.yaml:
    0 = Drowning          -> mapped to "drowning"
    1 = Person out of water -> mapped to "normal"
    2 = Swimming          -> mapped to "normal"

Output: binary classification dataset (drowning / normal)
    data/processed/
        train/  drowning/  normal/
        test/   drowning/  normal/

Usage:
    python scripts/build_dataset.py

Options:
    --raw      : path to data/ folder         (default: data)
    --out      : output path                  (default: data/processed)
    --width    : crop width  in px            (default: 64)
    --height   : crop height in px            (default: 128)
    --no-grayscale : keep RGB crops
    --test-ratio   : test split fraction      (default: 0.2)
    --min-crop     : minimum crop size in px  (default: 20)
    --seed         : random seed              (default: 42)
    --skip-analysis: skip the analysis step
    --yes          : skip confirmation prompt
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import DatasetAnalyzer, DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")

# ====================================================================
# CLASS MAPPING -- confirmed from data.yaml
#
#   names: ['Drowning', 'Person out of water', 'Swimming']
#
#   0 = Drowning            -> drowning
#   1 = Person out of water -> normal  (not in water, not in danger)
#   2 = Swimming            -> normal  (active swimmer, not in danger)
#
#   Result: binary problem -- drowning vs normal
# ====================================================================
CLASS_MAPPING = {
    0: "drowning",
    1: "normal",
    2: "normal",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeSwim -- Build HOG+SVM classification dataset from YOLO annotations"
    )
    p.add_argument("--raw",           default="data",           help="Path to data/ folder")
    p.add_argument("--out",           default="data/processed", help="Output directory")
    p.add_argument("--width",         type=int,   default=64,   help="Crop width (px)")
    p.add_argument("--height",        type=int,   default=128,  help="Crop height (px)")
    p.add_argument("--no-grayscale",  action="store_true",      help="Keep RGB")
    p.add_argument("--test-ratio",    type=float, default=0.2,  help="Test split ratio")
    p.add_argument("--min-crop",      type=int,   default=20,   help="Min crop size (px)")
    p.add_argument("--seed",          type=int,   default=42,   help="Random seed")
    p.add_argument("--skip-analysis", action="store_true",      help="Skip analysis step")
    p.add_argument("--yes",           action="store_true",      help="Skip confirmation")
    return p.parse_args()


def print_header(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  SafeSwim -- Dataset Build Pipeline")
    print("=" * 60)
    print(f"  Raw data  : {args.raw}")
    print(f"  Output    : {args.out}")
    print(f"  Crop size : {args.width}×{args.height} px")
    print(f"  Grayscale : {not args.no_grayscale}")
    print(f"  Test %    : {int(args.test_ratio * 100)}%")
    print("=" * 60)


def confirm(auto: bool) -> bool:
    print("\n" + "─" * 60)
    print("  CLASS MAPPING:")
    for cid, name in sorted(CLASS_MAPPING.items()):
        labels = ["Drowning", "Person out of water", "Swimming"]
        print(f"    {cid} ({labels[cid]:<22}) -> {name}")
    print("\n  This produces a BINARY dataset: drowning vs normal")
    print("─" * 60)
    if auto:
        print("  Auto-confirmed (--yes)")
        return True
    return input("  Proceed? [y/N]: ").strip().lower() == "y"


def print_next_steps(out: str) -> None:
    print(f"""
  NEXT STEPS
  ──────────
  1. Browse {out}/train/ and spot-check crops look correct.
  2. If imbalance warning appeared: tell SVM team to use
     class_weight='balanced' in sklearn.svm.SVC.
  3. Your processed dataset is ready for HOG extraction.
""")


def main() -> None:
    args = parse_args()
    print_header(args)

    # Step 1 -- Analyze
    if not args.skip_analysis:
        logger.info("Step 1/3 -- Analyzing dataset...")
        try:
            DatasetAnalyzer(raw_dir=args.raw).analyze()
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.info("Step 1/3 -- Analysis skipped.")

    # Step 2 -- Confirm
    if not confirm(args.yes):
        print("  Aborted.")
        sys.exit(0)

    # Step 3 -- Build
    logger.info("Step 3/3 -- Building classification dataset...")
    try:
        DatasetBuilder(
            raw_dir=args.raw,
            processed_dir=args.out,
            class_mapping=CLASS_MAPPING,
            output_width=args.width,
            output_height=args.height,
            grayscale=not args.no_grayscale,
            min_crop_size=args.min_crop,
            test_split_ratio=args.test_ratio,
            seed=args.seed,
        ).build()
    except ValueError as e:
        logger.error(f"Config error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

    print_next_steps(args.out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
