#!/usr/bin/env python3
"""
Train ViT-GRU (or ablation variants) on HMDB51.

Example:
    python scripts/train.py \
        --csv data/hmdb51_splits.csv \
        --model vit_gru \
        --epochs 10 \
        --batch_size 4 \
        --lr 1e-4 \
        --output_dir outputs/vit_gru
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    HMDB51Dataset,
    get_train_transforms,
    get_val_transforms,
    collate_fn,
)
from src.models import FocalLoss, build_model
from src.train import train_one_epoch, evaluate
from src.utils import set_seed, WarmupCosineAnnealingLR


def parse_args():
    parser = argparse.ArgumentParser(description="Train video action recognition model")
    parser.add_argument("--csv", type=str, required=True, help="Path to split CSV")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_gru",
        choices=["vit_gru", "vit_only", "gru_fe"],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--frames", type=int, default=32, help="Frames per video")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/run",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV
    df = pd.read_csv(args.csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    print(f"Train videos: {len(train_df)} | Val videos: {len(val_df)}")
    print(f"Classes: {df['label'].nunique()}")

    # Datasets & loaders
    train_dataset = HMDB51Dataset(
        train_df,
        transform=get_train_transforms(),
        frames_per_video=args.frames,
    )
    val_dataset = HMDB51Dataset(
        val_df,
        transform=get_val_transforms(),
        frames_per_video=args.frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_classes = len(train_dataset.label_to_index)
    print(f"Number of classes: {num_classes}")

    # Model
    model = build_model(args.model, num_classes=num_classes, device=device)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs
    )

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"P: {val_p:.2f} R: {val_r:.2f} F1: {val_f1:.2f} | LR: {current_lr:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "num_classes": num_classes,
                    "model_name": args.model,
                    "label_to_index": train_dataset.label_to_index,
                },
                best_path,
            )
            print(f"  → New best model saved to {best_path}")

        # Always save last checkpoint
        last_path = os.path.join(args.output_dir, "last_checkpoint.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "model_name": args.model,
                "label_to_index": train_dataset.label_to_index,
            },
            last_path,
        )

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
