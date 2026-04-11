#!/usr/bin/env python3
"""
extract_frames.py
=================
Extract frames from a video file into data/raw/ for dataset expansion.

Usage:
    python scripts/extract_frames.py --video data/test_video.mp4

Options:
    --video     : path to video file           (required)
    --out       : output folder                (default: data/raw/train/images)
    --interval  : extract 1 frame every N      (default: 10)
    --max       : maximum frames to extract    (default: no limit)
    --min-brightness : skip dark frames        (default: 15)
    --blur-threshold : skip blurry frames      (default: 50.0)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.video_frame_extractor import VideoFrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeSwim -- Extract frames from video for dataset expansion"
    )
    p.add_argument("--video",            required=True,         help="Path to video file")
    p.add_argument("--out",              default="data/raw/train/images", help="Output directory")
    p.add_argument("--interval",         type=int,   default=10,  help="Extract 1 frame every N")
    p.add_argument("--max",              type=int,   default=None, help="Max frames to extract")
    p.add_argument("--min-brightness",   type=int,   default=15,  help="Skip frames darker than this")
    p.add_argument("--blur-threshold",   type=float, default=50.0, help="Skip frames blurrier than this")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  SafeSwim -- Frame Extractor")
    print("=" * 60)
    print(f"  Video    : {args.video}")
    print(f"  Output   : {args.out}")
    print(f"  Interval : every {args.interval} frames")
    print("=" * 60 + "\n")

    try:
        extractor = VideoFrameExtractor(
            video_path=args.video,
            output_dir=args.out,
            frame_interval=args.interval,
            max_frames=args.max,
            min_brightness=args.min_brightness,
            blur_threshold=args.blur_threshold,
        )
        extractor.extract()

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
