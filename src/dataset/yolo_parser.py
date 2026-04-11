"""
yolo_parser.py
==============
Parses YOLO-format .txt annotation files into BoundingBox objects.

YOLO format per line:
    class_id  x_center  y_center  width  height
    (all coordinates normalized 0.0 → 1.0)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """
    One annotated object from a YOLO label file.

    class_id meanings (this dataset):
        0 = Drowning
        1 = Person out of water
        2 = Swimming
    """
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized YOLO coords → absolute pixel (x1, y1, x2, y2).

        YOLO stores center + size.
        We need top-left + bottom-right for numpy slicing: img[y1:y2, x1:x2]
        """
        cx = self.x_center * img_width
        cy = self.y_center * img_height
        w  = self.width    * img_width
        h  = self.height   * img_height

        x1 = max(0,          int(cx - w / 2))
        y1 = max(0,          int(cy - h / 2))
        x2 = min(img_width,  int(cx + w / 2))
        y2 = min(img_height, int(cy + h / 2))

        return x1, y1, x2, y2

    def is_valid(self) -> bool:
        return (
            0.0 <= self.x_center <= 1.0
            and 0.0 <= self.y_center <= 1.0
            and 0.0 < self.width  <= 1.0
            and 0.0 < self.height <= 1.0
        )


class YoloParser:
    """
    Reads YOLO .txt label files and returns lists of BoundingBox objects.
    Skips malformed lines with a warning instead of crashing.
    """

    def parse(self, label_path: str | Path) -> List[BoundingBox]:
        """Parse a single label file. Returns [] if missing or unreadable."""
        label_path = Path(label_path)
        boxes: List[BoundingBox] = []

        if not label_path.exists():
            logger.debug(f"No label file: {label_path.name}")
            return boxes

        try:
            lines = label_path.read_text().splitlines()
        except OSError as e:
            logger.warning(f"Cannot read {label_path}: {e}")
            return boxes

        for i, line in enumerate(lines, 1):
            box = self._parse_line(line.strip(), label_path, i)
            if box is not None:
                boxes.append(box)

        return boxes

    def parse_for_image(self, image_path: str | Path, labels_dir: str | Path) -> List[BoundingBox]:
        """Find and parse the label file for a given image path."""
        label_path = Path(labels_dir) / (Path(image_path).stem + ".txt")
        return self.parse(label_path)

    def _parse_line(self, line: str, src: Path, line_num: int) -> Optional[BoundingBox]:
        if not line:
            return None
        parts = line.split()
        if len(parts) != 5:
            logger.warning(f"{src.name}:{line_num} — expected 5 fields, got {len(parts)}")
            return None
        try:
            box = BoundingBox(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        except ValueError as e:
            logger.warning(f"{src.name}:{line_num} — parse error: {e}")
            return None

        if not box.is_valid():
            logger.warning(f"{src.name}:{line_num} — invalid coords: {box}")
            return None

        return box
