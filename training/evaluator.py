from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for true_label, predicted_label in zip(labels.cpu().numpy(), predictions.cpu().numpy(), strict=True):
                confusion[int(true_label), int(predicted_label)] += 1

    per_class = {}
    for digit in range(10):
        row = confusion[digit]
        support = int(row.sum())
        per_class[str(digit)] = {
            "accuracy": float(row[digit] / support) if support else 0.0,
            "correct": int(row[digit]),
            "total": support,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "loss": total_loss / total if total else 0.0,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
    }
