#!/usr/bin/env python3
"""
Prepare HMDB51 split CSV from raw video directory.

Usage:
    python scripts/prepare_data.py \
        --video_dir /path/to/hmdb51 \
        --output_csv data/hmdb51_splits.csv
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import prepare_hmdb51_csv, set_seed


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for HMDB51")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Root directory of HMDB51 (contains class subfolders)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/hmdb51_splits.csv",
        help="Where to save the CSV",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    prepare_hmdb51_csv(
        video_dir=args.video_dir,
        output_csv=args.output_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
