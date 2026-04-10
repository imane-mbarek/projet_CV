"""
scripts/validate_dataset.py
---------------------------
Validates the processed classification dataset and produces a detailed
health report. Run this after build_dataset.py to confirm the dataset
is ready for Role 2 (HOG + SVM training).

Usage
-----
From the project root:
    python -m scripts.validate_dataset
    python -m scripts.validate_dataset --processed data/dataset_processed
    python -m scripts.validate_dataset --fail-on-warnings

Exit codes
----------
    0 — validation passed (or only warnings, without --fail-on-warnings)
    1 — validation failed (corrupt files, shape mismatches, missing dirs)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.dataset_validator import validate_dataset, print_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SafeSwim — Validate the processed classification dataset."
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="data/dataset_processed",
        help="Path to the processed dataset root. Default: data/dataset_processed",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with code 1 even for degenerate/blank image warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed)

    if not processed_root.exists():
        print(f"\n[ERROR] Processed dataset not found at: {processed_root}")
        print("        Run 'python -m scripts.build_dataset' first.")
        sys.exit(1)

    report = validate_dataset(processed_root)
    print_validation_report(report)

    # ---- Exit code ----
    if not report.passed:
        sys.exit(1)

    if args.fail_on_warnings and report.total_degenerate > 0:
        print(f"[WARN] {report.total_degenerate} degenerate images found (--fail-on-warnings is set)")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
