"""
dataset_analyzer.py
-------------------
Scans the raw YOLO-style dataset structure and produces a detailed
summary report: file counts, missing pairs, class distribution, and
YAML class definitions.

This module is READ-ONLY with respect to the raw dataset.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitReport:
    """Statistics for one dataset split (train / valid / test)."""
    name: str
    image_count: int = 0
    label_count: int = 0
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)
    class_counts: dict[int, int] = field(default_factory=dict)

    @property
    def paired_count(self) -> int:
        return self.image_count - len(self.images_without_labels)

    @property
    def is_consistent(self) -> bool:
        return (
            len(self.images_without_labels) == 0
            and len(self.labels_without_images) == 0
        )


@dataclass
class DatasetReport:
    """Full report across all splits."""
    raw_root: Path
    yaml_classes: Optional[dict[int, str]]  # None if data.yaml is missing/unreadable
    splits: dict[str, SplitReport] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(s.image_count for s in self.splits.values())

    @property
    def total_labels(self) -> int:
        return sum(s.label_count for s in self.splits.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_stems(folder: Path, extensions: set[str]) -> dict[str, Path]:
    """Return {stem: full_path} for all files matching the given extensions."""
    result: dict[str, Path] = {}
    if not folder.exists():
        return result
    for f in folder.iterdir():
        if f.suffix.lower() in extensions:
            result[f.stem] = f
    return result


def _parse_yaml_classes(yaml_path: Path) -> Optional[dict[int, str]]:
    """
    Read data.yaml and extract the class id→name mapping.

    Returns None if the file is missing or malformed.
    """
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)
        names = data.get("names", [])
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        return None
    except Exception as exc:
        print(f"[WARN] Could not parse {yaml_path}: {exc}")
        return None


def _count_classes_in_labels(label_folder: Path) -> dict[int, int]:
    """Return class_id → annotation_count for all .txt files in a folder."""
    counts: dict[int, int] = {}
    if not label_folder.exists():
        return counts
    for txt_file in label_folder.glob("*.txt"):
        try:
            for line in txt_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    counts[class_id] = counts.get(class_id, 0) + 1
        except Exception:
            pass  # malformed files are reported elsewhere
    return counts


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_dataset(raw_root: str | Path) -> DatasetReport:
    """
    Scan the raw YOLO dataset rooted at *raw_root* and return a DatasetReport.

    Parameters
    ----------
    raw_root : path to data/dataset_raw/

    Returns
    -------
    DatasetReport with per-split statistics and YAML class definitions.
    """
    raw_root = Path(raw_root)
    yaml_path = raw_root / "data.yaml"
    yaml_classes = _parse_yaml_classes(yaml_path)

    report = DatasetReport(raw_root=raw_root, yaml_classes=yaml_classes)

    for split_name in ("train", "valid", "test"):
        split_dir = raw_root / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        image_stems = _collect_stems(images_dir, IMAGE_EXTENSIONS)
        label_stems = _collect_stems(labels_dir, {".txt"})

        split = SplitReport(name=split_name)
        split.image_count = len(image_stems)
        split.label_count = len(label_stems)
        split.images_without_labels = [
            str(p) for stem, p in image_stems.items() if stem not in label_stems
        ]
        split.labels_without_images = [
            str(p) for stem, p in label_stems.items() if stem not in image_stems
        ]
        split.class_counts = _count_classes_in_labels(labels_dir)

        report.splits[split_name] = split

    return report


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_report(report: DatasetReport) -> None:
    """Print a human-readable version of the DatasetReport."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  SAFESWIM — RAW DATASET ANALYSIS REPORT")
    print(f"{sep}")
    print(f"  Root : {report.raw_root}")
    print(f"  Total images : {report.total_images}")
    print(f"  Total label files : {report.total_labels}")

    # ---- YAML classes ----
    print(f"\n  {'─'*56}")
    print("  YAML CLASS DEFINITIONS")
    print(f"  {'─'*56}")
    if report.yaml_classes is None:
        print("  [WARN] data.yaml not found or unreadable.")
        print("         You must manually define the class mapping.")
    else:
        for cid, cname in sorted(report.yaml_classes.items()):
            print(f"    class {cid} → '{cname}'")

    # ---- Per-split ----
    for split_name, split in report.splits.items():
        print(f"\n  {'─'*56}")
        print(f"  SPLIT: {split_name.upper()}")
        print(f"  {'─'*56}")
        print(f"    Images      : {split.image_count}")
        print(f"    Label files : {split.label_count}")
        print(f"    Paired      : {split.paired_count}")

        if split.images_without_labels:
            print(f"    [WARN] {len(split.images_without_labels)} image(s) with no label file")
            for p in split.images_without_labels[:5]:  # show first 5
                print(f"           {p}")
            if len(split.images_without_labels) > 5:
                print(f"           ... and {len(split.images_without_labels)-5} more")

        if split.labels_without_images:
            print(f"    [WARN] {len(split.labels_without_images)} label(s) with no image file")

        if split.class_counts:
            print("    Class distribution (annotations):")
            for cid, cnt in sorted(split.class_counts.items()):
                cname = (report.yaml_classes or {}).get(cid, "unknown")
                print(f"      class {cid} ('{cname}') : {cnt} boxes")
        else:
            print("    [WARN] No annotations found in this split.")

    print(f"\n{sep}\n")
