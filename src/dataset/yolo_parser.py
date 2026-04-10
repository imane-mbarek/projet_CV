"""
yolo_parser.py
--------------
Parses YOLO-format .txt annotation files and converts normalised
coordinates into pixel bounding boxes.

YOLO format (one line per object):
    class_id  x_center  y_center  width  height
All values are normalised to [0, 1] relative to image dimensions.

This module is pure data — it does NOT read images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    A single annotation converted to absolute pixel coordinates.

    Attributes
    ----------
    class_id : original YOLO class index
    x1, y1   : top-left corner (pixels, clamped to image bounds)
    x2, y2   : bottom-right corner (pixels, clamped to image bounds)
    """
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_valid(self, min_size: int = 1) -> bool:
        """Return True if the box has positive area above the minimum threshold."""
        return self.width >= min_size and self.height >= min_size


@dataclass
class ParseResult:
    """
    Result of parsing one label file.

    Attributes
    ----------
    path         : Path to the .txt file that was parsed
    boxes        : list of successfully parsed BoundingBox objects
    skipped_lines: raw lines that could not be parsed (for diagnostics)
    error        : set to an error message if the file could not be opened
    """
    path: Path
    boxes: list[BoundingBox]
    skipped_lines: list[str]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_label_file(
    label_path: str | Path,
    image_width: int,
    image_height: int,
    min_box_size: int = 1,
) -> ParseResult:
    """
    Parse a single YOLO .txt label file.

    Parameters
    ----------
    label_path   : path to the .txt annotation file
    image_width  : pixel width of the corresponding image
    image_height : pixel height of the corresponding image
    min_box_size : boxes smaller than this (in pixels, both axes) are skipped

    Returns
    -------
    ParseResult with all valid BoundingBox objects and any skipped lines.
    """
    label_path = Path(label_path)
    boxes: list[BoundingBox] = []
    skipped: list[str] = []

    # ---- Try to open the file ----
    try:
        raw_text = label_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ParseResult(
            path=label_path, boxes=[], skipped_lines=[],
            error=f"Cannot open file: {exc}"
        )

    # ---- Parse line by line ----
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue  # blank lines are fine

        parts = line.split()

        # A valid YOLO line has exactly 5 fields
        if len(parts) != 5:
            skipped.append(line)
            continue

        try:
            class_id = int(parts[0])
            xc_norm  = float(parts[1])
            yc_norm  = float(parts[2])
            w_norm   = float(parts[3])
            h_norm   = float(parts[4])
        except ValueError:
            skipped.append(line)
            continue

        # ---- Validate normalised values ----
        if not (0.0 <= xc_norm <= 1.0 and 0.0 <= yc_norm <= 1.0
                and 0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0):
            skipped.append(line)
            continue

        # ---- Convert to pixel coordinates ----
        xc = xc_norm * image_width
        yc = yc_norm * image_height
        w  = w_norm  * image_width
        h  = h_norm  * image_height

        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)

        # ---- Clamp to image bounds ----
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width,  x2)
        y2 = min(image_height, y2)

        box = BoundingBox(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2)

        if not box.is_valid(min_size=min_box_size):
            skipped.append(line)
            continue

        boxes.append(box)

    return ParseResult(path=label_path, boxes=boxes, skipped_lines=skipped)


# ---------------------------------------------------------------------------
# Utility: find matching label for an image
# ---------------------------------------------------------------------------

def find_label_for_image(image_path: Path, labels_dir: Path) -> Optional[Path]:
    """
    Return the .txt label path whose stem matches the image stem,
    or None if no matching label exists.
    """
    candidate = labels_dir / (image_path.stem + ".txt")
    return candidate if candidate.exists() else None
