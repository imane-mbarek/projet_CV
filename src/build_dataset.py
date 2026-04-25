"""
scripts/build_dataset.py
------------------------
Entry point for the full dataset build pipeline.

Usage
-----
From the project root:
    python -m scripts.build_dataset
    python -m scripts.build_dataset --raw data --out data/processed

The script:
  1. Analyzes the raw dataset structure.
  2. Prints the class mapping that will be applied.
  3. Builds the processed classification dataset.
"""

import argparse
import sys
from pathlib import Path

# ---- Make sure the project root is on sys.path ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.dataset_analyzer import analyze_dataset, print_report
from src.dataset.dataset_builder import build_dataset, BuildConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SafeSwim — Build processed classification dataset from raw YOLO data."
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="data",
        help="Path to the raw YOLO dataset root. Default: data",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed",
        help="Path to write the processed dataset. Default: data/processed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_root = Path(args.raw)
    processed_root = Path(args.out)

    # ---- Step 1: Analyze raw dataset ----
    print("\n[STEP 1/2] Analyzing raw dataset ...")
    report = analyze_dataset(raw_root)
    print_report(report)

    # ---- Abort if raw root does not exist ----
    if not raw_root.exists():
        print(f"[ERROR] Raw dataset root not found: {raw_root}")
        print("        Make sure your dataset is under data/")
        sys.exit(1)

    # ---- Step 2: Build processed dataset ----
    print("[STEP 2/2] Building processed dataset ...")
    config = BuildConfig(
        raw_root=raw_root,
        processed_root=processed_root,
        verbose=not args.quiet,
    )
    build_dataset(config)

    print("\n[DONE] Run 'python -m scripts.validate_dataset' to verify the output.")


if __name__ == "__main__":
    main()
