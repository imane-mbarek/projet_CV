"""
dataset_analyzer.py
====================
Validates and reports on the raw dataset structure.

Adapted for the SafeSwim Roboflow dataset which uses:
    data/
        train/  images/ + labels/
        valid/  images/ + labels/   ← "valid" not "val"
        test/   images/ + labels/
        data.yaml

Classes (from data.yaml):
    0 = Drowning
    1 = Person out of water
    2 = Swimming
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Reads and validates the Roboflow YOLO dataset structure.
    Run this FIRST before any processing — catch problems early.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    # Roboflow exports use "valid" not "val"
    SPLITS = ["train", "valid", "test"]

    # Human-readable class names from data.yaml
    CLASS_NAMES = {
        0: "Drowning",
        1: "Person out of water",
        2: "Swimming",
    }

    def __init__(self, raw_dir: str):
        """
        Args:
            raw_dir: Path to the data/ folder (e.g. 'data/')
        """
        self.raw_dir = Path(raw_dir)

    def analyze(self) -> Dict:
        """
        Full analysis pass. Validates structure, counts files, detects issues.

        Returns:
            Summary dict used by downstream modules.
        """
        logger.info(f"Analyzing dataset at: {self.raw_dir}")
        self._check_root_exists()

        summary = {
            "splits": {},
            "class_ids_found": set(),
            "total_images": 0,
            "total_labels": 0,
            "issues": [],
        }

        for split in self.SPLITS:
            split_dir = self.raw_dir / split
            if not split_dir.exists():
                logger.warning(f"Split '{split}' not found — skipping.")
                continue
            stats = self._analyze_split(split, summary["issues"])
            summary["splits"][split] = stats
            summary["total_images"] += stats["num_images"]
            summary["total_labels"] += stats["num_labels"]
            summary["class_ids_found"].update(stats["class_ids"])

        self._print_report(summary)
        return summary

    def _check_root_exists(self) -> None:
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.raw_dir}\n"
                f"Make sure 'data/' is in your project root."
            )

    def _analyze_split(self, split: str, issues: List[str]) -> Dict:
        images_dir = self.raw_dir / split / "images"
        labels_dir = self.raw_dir / split / "labels"

        for d, name in [(images_dir, "images"), (labels_dir, "labels")]:
            if not d.exists():
                issues.append(f"[{split}] Missing {name}/ directory")

        if not images_dir.exists() or not labels_dir.exists():
            return self._empty_stats()

        image_files = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
        ]
        label_files = {f.stem: f for f in labels_dir.glob("*.txt")}

        image_stems = {f.stem for f in image_files}
        label_stems = set(label_files.keys())

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            issues.append(f"[{split}] {len(missing_labels)} image(s) have no label")
        if missing_images:
            issues.append(f"[{split}] {len(missing_images)} label(s) have no image")

        class_id_counts, bad_lines = self._scan_labels(split, label_files, issues)

        return {
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "images_without_labels": len(missing_labels),
            "labels_without_images": len(missing_images),
            "class_id_counts": class_id_counts,
            "class_ids": set(class_id_counts.keys()),
            "bad_lines": bad_lines,
        }

    def _scan_labels(
        self, split: str, label_files: Dict, issues: List[str]
    ) -> Tuple[Dict, int]:
        class_id_counts: Dict[int, int] = defaultdict(int)
        bad_lines = 0

        for stem, label_path in label_files.items():
            try:
                lines = label_path.read_text().splitlines()
                if not lines:
                    issues.append(f"[{split}] Empty label file: {label_path.name}")
                    continue
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        bad_lines += 1
                        continue
                    try:
                        class_id_counts[int(parts[0])] += 1
                    except ValueError:
                        bad_lines += 1
            except OSError as e:
                issues.append(f"[{split}] Cannot read {label_path.name}: {e}")

        if bad_lines:
            issues.append(f"[{split}] {bad_lines} malformed annotation line(s)")

        return dict(class_id_counts), bad_lines

    @staticmethod
    def _empty_stats() -> Dict:
        return {
            "num_images": 0, "num_labels": 0,
            "images_without_labels": 0, "labels_without_images": 0,
            "class_id_counts": {}, "class_ids": set(), "bad_lines": 0,
        }

    def _print_report(self, summary: Dict) -> None:
        print("\n" + "=" * 60)
        print("  DATASET ANALYSIS REPORT")
        print("=" * 60)
        print(f"  Root         : {self.raw_dir}")
        print(f"  Total images : {summary['total_images']}")
        print(f"  Total labels : {summary['total_labels']}")

        for split, stats in summary["splits"].items():
            print(f"\n  [{split.upper()}]")
            print(f"    Images : {stats['num_images']}")
            print(f"    Labels : {stats['num_labels']}")
            if stats["class_id_counts"]:
                print(f"    Class distribution:")
                for cid, count in sorted(stats["class_id_counts"].items()):
                    name = self.CLASS_NAMES.get(cid, f"unknown_{cid}")
                    print(f"      ID {cid} ({name:<22}): {count} boxes")

        print(f"\n  Class IDs found: {sorted(summary['class_ids_found'])}")
        print(f"  Expected:        [0 = Drowning, 1 = Person out of water, 2 = Swimming]")

        if summary["issues"]:
            print(f"\n  WARNING — {len(summary['issues'])} issue(s) detected:")
            for issue in summary["issues"]:
                print(f"    - {issue}")
        else:
            print("\n  No structural issues detected.")

        print("=" * 60 + "\n")
