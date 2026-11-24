import argparse
import json
from pathlib import Path

import torch
from torch import nn

from config import ExperimentConfig
from data.mnist import create_mnist_loaders
from models.cnn import SimpleCNN
from training.evaluator import evaluate
from training.trainer import train_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.reproducibility import get_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST class-exclusion training framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an MNIST CNN")
    train_parser.add_argument("--experiment", choices=["full", "exclude"], default="full")
    train_parser.add_argument("--target-class", type=int, default=None)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--data-dir", default="./data")
    train_parser.add_argument("--output-dir", default="./")
    train_parser.add_argument("--num-workers", type=int, default=2)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data-dir", default="./data")
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--num-workers", type=int, default=2)
    eval_parser.add_argument("--output", default=None)
    return parser


def validate_experiment(config: ExperimentConfig) -> None:
    if config.experiment == "exclude":
        if config.target_class is None:
            raise ValueError("--target-class is required for --experiment exclude")
        if config.target_class not in range(10):
            raise ValueError("--target-class must be between 0 and 9")
    elif config.target_class is not None:
        raise ValueError("--target-class is only valid with --experiment exclude")


def train(config: ExperimentConfig) -> tuple[dict, Path, Path]:
    validate_experiment(config)
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    excluded_class = config.target_class if config.experiment == "exclude" else None
    training_dataset, test_dataset, train_loader, test_loader = create_mnist_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        excluded_class=excluded_class,
    )

    print(f"Training samples: {len(training_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config.epochs,
    )
    final_evaluation = evaluate(model, test_loader, criterion, device)

    checkpoint_name = (
        f"mnist_exclude_class_{excluded_class}_seed_{config.seed}.pt"
        if excluded_class is not None
        else f"mnist_full_seed_{config.seed}.pt"
    )
    result_name = checkpoint_name.replace(".pt", ".json")

    output_root = Path(config.output_dir)
    checkpoint_path = output_root / "checkpoints" / checkpoint_name
    result_path = output_root / "results" / result_name

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {"name": "SimpleCNN", "num_classes": 10},
        "training_config": config.to_dict(),
        "random_seed": config.seed,
        "training_history": history,
        "excluded_class": excluded_class,
        "final_evaluation": final_evaluation,
    }
    save_checkpoint(checkpoint_path, checkpoint)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_record = {
        "training_config": config.to_dict(),
        "random_seed": config.seed,
        "excluded_class": excluded_class,
        "training_history": history,
        "final_evaluation": final_evaluation,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_record, handle, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved results: {result_path}")
    print(f"Final test accuracy: {final_evaluation['accuracy']:.4f}")
    if excluded_class is not None:
        print(
            "Excluded class "
            f"{excluded_class} test accuracy: "
            f"{final_evaluation['per_class_accuracy'][str(excluded_class)]['accuracy']:.4f}"
        )
    return checkpoint, checkpoint_path, result_path


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    set_seed(42)
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config", {})
    num_classes = int(model_config.get("num_classes", 10))
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_dataset, _, test_loader = create_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        excluded_class=None,
    )
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    output = {
        "checkpoint": args.checkpoint,
        "test_samples": len(test_dataset),
        "excluded_class": checkpoint.get("excluded_class"),
        "evaluation": metrics,
    }
    print(json.dumps(output, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        config = ExperimentConfig(
            experiment=args.experiment,
            target_class=args.target_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
        )
        train(config)
    else:
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
