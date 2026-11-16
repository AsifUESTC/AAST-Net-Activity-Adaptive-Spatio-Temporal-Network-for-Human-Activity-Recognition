#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the test split and generate plots.

Example:
    python scripts/evaluate.py \
        --csv data/hmdb51_splits.csv \
        --checkpoint outputs/vit_gru/best_model.pth \
        --output_dir outputs/vit_gru
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HMDB51Dataset, get_val_transforms, collate_fn
from src.models import FocalLoss, build_model
from src.train import test
from src.utils import (
    set_seed,
    plot_confusion_matrix,
    plot_class_wise_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test videos: {len(test_df)}")

    test_dataset = HMDB51Dataset(
        test_df,
        transform=get_val_transforms(),
        frames_per_video=args.frames,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", len(test_dataset.label_to_index))
    model_name = ckpt.get("model_name", "vit_gru")

    model = build_model(model_name, num_classes=num_classes, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = FocalLoss()
    avg_loss, acc, precision, recall, f1, y_true, y_pred = test(
        model, test_loader, criterion, device
    )

    # Class names (sorted to match label_to_index)
    class_names = sorted(test_dataset.label_to_index.keys())

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        filename=os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    plot_class_wise_accuracy(
        y_true,
        y_pred,
        num_classes=num_classes,
        filename=os.path.join(args.output_dir, "class_wise_accuracy.png"),
    )

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall:    {recall:.2f}\n")
        f.write(f"F1-Score:  {f1:.2f}\n")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
