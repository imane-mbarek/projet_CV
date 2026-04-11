"""
video_frame_extractor.py
========================
Extracts frames from a .mp4 video file and saves them as images
into the dataset's raw folder structure, ready for annotation
and re-processing by the pipeline.

When is this used?
    When you want to EXPAND the dataset with new footage.
    For example:
        - A new pool camera recording is provided
        - You extract frames from it
        - Someone annotates them in Roboflow
        - You re-run build_dataset.py

This is YOUR responsibility as Data & Preprocessing Engineer
because it feeds YOUR pipeline — not the real-time detector.

Usage:
    from src.dataset.video_frame_extractor import VideoFrameExtractor

    extractor = VideoFrameExtractor(
        video_path="data/test_video.mp4",
        output_dir="data/raw/train/images",
    )
    extractor.extract()
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    Extracts frames from a video file and saves them as .jpg images.

    Design decisions:
        - Frame interval: we do NOT extract every single frame.
          Video is typically 25-30 fps. Adjacent frames are nearly
          identical — useless for training and creates duplicates.
          We extract 1 frame every N frames (default: every 10).

        - Quality check: black frames (camera cut, night vision off)
          and blurry frames (motion blur) are detected and skipped.
          They add noise to the dataset without adding information.

        - Naming: frames are named {video_stem}_frame_{index:06d}.jpg
          so they are always traceable back to their source video.

    Args:
        video_path      : Path to the .mp4 video file
        output_dir      : Where to save extracted frames
        frame_interval  : Extract 1 frame every N frames (default 10)
        max_frames      : Maximum number of frames to extract (None = no limit)
        min_brightness  : Skip frames darker than this (0-255, default 15)
        blur_threshold  : Skip frames blurrier than this (default 50.0)
                          Uses Laplacian variance — lower = blurrier
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 10,
        max_frames: Optional[int] = None,
        min_brightness: int = 15,
        blur_threshold: float = 50.0,
    ):
        self.video_path     = Path(video_path)
        self.output_dir     = Path(output_dir)
        self.frame_interval = frame_interval
        self.max_frames     = max_frames
        self.min_brightness = min_brightness
        self.blur_threshold = blur_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> dict:
        """
        Run the full extraction.

        Returns:
            Summary dict with counts of saved, skipped, and total frames.
        """
        self._validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps           = cap.get(cv2.CAP_PROP_FPS)
        duration_secs = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Video: {self.video_path.name} | "
            f"{total_frames} frames | {fps:.1f} fps | "
            f"{duration_secs:.1f}s"
        )

        stats = {
            "saved": 0,
            "skipped_interval": 0,
            "skipped_black": 0,
            "skipped_blur": 0,
            "total_read": 0,
        }

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            stats["total_read"] += 1

            # Skip frames not in our interval
            if frame_index % self.frame_interval != 0:
                stats["skipped_interval"] += 1
                frame_index += 1
                continue

            # Quality checks
            skip_reason = self._quality_check(frame)
            if skip_reason == "black":
                stats["skipped_black"] += 1
                frame_index += 1
                continue
            if skip_reason == "blur":
                stats["skipped_blur"] += 1
                frame_index += 1
                continue

            # Save the frame
            filename = f"{self.video_path.stem}_frame_{frame_index:06d}.jpg"
            save_path = self.output_dir / filename
            cv2.imwrite(
                str(save_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            stats["saved"] += 1

            # Stop if we hit the max
            if self.max_frames and stats["saved"] >= self.max_frames:
                logger.info(f"Reached max_frames limit ({self.max_frames}). Stopping.")
                break

            frame_index += 1

        cap.release()
        self._print_summary(stats, total_frames)
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Fail fast with a clear message if inputs are wrong."""
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}\n"
                f"Make sure the file is at the correct path."
            )
        if self.video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise ValueError(
                f"Unsupported video format: {self.video_path.suffix}\n"
                f"Supported: .mp4 .avi .mov .mkv"
            )
        if self.frame_interval < 1:
            raise ValueError("frame_interval must be >= 1")

    def _quality_check(self, frame: np.ndarray) -> Optional[str]:
        """
        Check if a frame is usable.

        Returns:
            None        → frame is good, save it
            "black"     → frame is too dark (camera cut / night)
            "blur"      → frame is too blurry (motion blur)

        Why check brightness?
            Black frames appear during camera transitions,
            recording gaps, or night-vision switching.
            They contain no useful information.

        Why check blur?
            Motion blur happens when someone moves very fast
            across the frame. A blurry crop teaches the HOG
            descriptor nothing about human shape.
            We use the Laplacian variance method:
            sharp images have high variance, blurry ones low.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.min_brightness:
            return "black"

        # Blur check (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.blur_threshold:
            return "blur"

        return None

    def _print_summary(self, stats: dict, total_video_frames: int) -> None:
        """Print a clear extraction summary."""
        print("\n" + "=" * 60)
        print("  FRAME EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Video          : {self.video_path.name}")
        print(f"  Output dir     : {self.output_dir}")
        print(f"  Frame interval : every {self.frame_interval} frames")
        print(f"  Total frames in video : {total_video_frames}")
        print(f"\n  Frames read    : {stats['total_read']}")
        print(f"  Saved          : {stats['saved']}")
        print(f"  Skipped (interval) : {stats['skipped_interval']}")
        print(f"  Skipped (black)    : {stats['skipped_black']}")
        print(f"  Skipped (blur)     : {stats['skipped_blur']}")
        print(f"\n  Output dir     : {self.output_dir}")
        print(
            f"\n  NEXT STEP: annotate the extracted frames in Roboflow,\n"
            f"  export as YOLOv8, place in data/raw/, re-run build_dataset.py"
        )
        print("=" * 60 + "\n")
