"""
Training and evaluation loops.
"""

from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import calculate_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    precision, recall, f1 = calculate_metrics(
        torch.tensor(all_labels), torch.tensor(all_preds)
    )
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, List[int], List[int]]:
    """Convenience wrapper that prints metrics."""
    avg_loss, acc, precision, recall, f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"Test Loss: {avg_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )
    return avg_loss, acc, precision, recall, f1, y_true, y_pred
