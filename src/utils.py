"""
Utility functions: metrics, scheduler, data preparation, visualization.
"""

import os
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_hmdb51_csv(
    video_dir: str,
    output_csv: str = "data/hmdb51_splits.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Scan HMDB51 directory structure and create a CSV with columns:
    filename, label, split
    """
    random.seed(seed)
    filenames, labels, splits = [], [], []

    classes = sorted(
        [
            d
            for d in os.listdir(video_dir)
            if os.path.isdir(os.path.join(video_dir, d))
        ]
    )

    for class_name in classes:
        class_dir = os.path.join(video_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith(".avi")]
        random.shuffle(video_files)

        n = len(video_files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        for i, video_file in enumerate(video_files):
            path = os.path.join(class_dir, video_file)
            filenames.append(path)
            labels.append(class_name)

            if i < n_train:
                splits.append("train")
            elif i < n_train + n_val:
                splits.append("val")
            else:
                splits.append("test")

    df = pd.DataFrame({"filename": filenames, "label": labels, "split": splits})
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved split CSV to {output_csv}")
    print(df["split"].value_counts())
    return df


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_lr = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.initial_lr]

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.max_epochs - self.warmup_epochs
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [base_lr * cosine for base_lr in self.initial_lr]


def calculate_metrics(
    labels: torch.Tensor, preds: torch.Tensor, average: str = "weighted"
) -> Tuple[float, float, float]:
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    precision = precision_score(labels_np, preds_np, average=average, zero_division=0) * 100
    recall = recall_score(labels_np, preds_np, average=average, zero_division=0) * 100
    f1 = f1_score(labels_np, preds_np, average=average, zero_division=0) * 100
    return precision, recall, f1


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    filename: str = "outputs/confusion_matrix.png",
    figsize: Tuple[int, int] = (18, 16),
) -> None:
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.3,
        linecolor="gray",
    )
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)

    num_classes = conf_mat.shape[0]
    ticks = np.arange(num_classes)
    if class_names and len(class_names) == num_classes:
        plt.xticks(ticks=ticks, labels=class_names, rotation=90, fontsize=8)
        plt.yticks(ticks=ticks, labels=class_names, fontsize=8)
    else:
        plt.xticks(ticks=ticks, labels=[str(i) for i in range(num_classes)], rotation=90, fontsize=8)
        plt.yticks(ticks=ticks, labels=[str(i) for i in range(num_classes)], fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def plot_class_wise_accuracy(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    filename: str = "outputs/class_wise_accuracy.png",
) -> None:
    conf_mat = confusion_matrix(y_true, y_pred)
    class_accuracy = conf_mat.diagonal() / (conf_mat.sum(axis=1) + 1e-8) * 100

    plt.figure(figsize=(14, 8))
    plt.bar(range(num_classes), class_accuracy, color="steelblue", edgecolor="black", width=0.8)
    plt.xlabel("Classes", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Class-wise Accuracy (%)", fontsize=16)
    plt.ylim(0, 100)
    plt.xticks(range(0, num_classes, 5), rotation=90, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class-wise accuracy plot saved to {filename}")
