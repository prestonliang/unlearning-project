from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .evaluator import evaluate


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return (
        total_loss / total if total else 0.0,
        correct / total if total else 0.0,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {"epochs": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_metrics = evaluate(model, test_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "per_class_accuracy": test_metrics["per_class_accuracy"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        }
        history["epochs"].append(epoch_record)
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f}"
        )

    return history
