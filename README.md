# MNIST Classification and Class-Exclusion Training Framework

A reproducible PyTorch foundation for MNIST classification and future machine-unlearning experiments. The framework provides two training baselines:

- **Full training:** train on all MNIST digits (0–9).
- **Class-excluded retraining:** train from scratch while removing one digit from the training subset. The test set remains complete and the model retains all 10 output logits.

No unlearning loss or gradient-ascent behavior is implemented in this project.

## Model

`SimpleCNN` has the following structure:

- `Conv2d(1, 32, 3, padding=1)` → ReLU → `MaxPool2d(2)`
- `Conv2d(32, 64, 3, padding=1)` → ReLU → `MaxPool2d(2)`
- Flatten → `Linear(64 * 7 * 7, 128)` → ReLU → `Linear(128, 10)`

The forward method returns raw logits. Adam and `CrossEntropyLoss` are used for training.

## Setup

```bash
cd /Users/prestonliang/unlearning-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MNIST is downloaded automatically into the configured data directory.

## Train the full model

```bash
python main.py train --experiment full --epochs 5 --batch-size 64 --learning-rate 0.001 --seed 42
```

## Train with a class excluded

For example, to exclude digit 3:

```bash
python main.py train --experiment exclude --target-class 3 --epochs 5 --batch-size 64 --learning-rate 0.001 --seed 42
```

The training subset excludes the selected class, but the test loader always includes classes 0–9. Both experiments instantiate the same 10-logit architecture.

## CLI options

- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--seed`
- `--data-dir`
- `--output-dir`
- `--num-workers`

## Evaluation

Evaluation reports overall loss and accuracy, accuracy for every digit, and a 10×10 confusion matrix. It is included in the training JSON and can also be run separately:

```bash
python main.py evaluate --checkpoint checkpoints/mnist_full_seed_42.pt --data-dir ./data
python main.py evaluate --checkpoint checkpoints/mnist_exclude_class_3_seed_42.pt --data-dir ./data --output results/evaluation_exclude_3.json
```

## Outputs

- Full checkpoint: `checkpoints/mnist_full_seed_<seed>.pt`
- Excluded-class checkpoint: `checkpoints/mnist_exclude_class_<class>_seed_<seed>.pt`
- Structured results: `results/`

Each checkpoint contains:

- Model state dictionary
- Optimizer state dictionary
- Model and training configuration
- Random seed
- Per-epoch training history
- Excluded class
- Final evaluation metrics

## Reproducibility

Use the same seed and hyperparameters for a fair comparison. Python, NumPy, and PyTorch seeds are initialized, and CUDA cuDNN is configured deterministically. CPU, CUDA, and MPS are selected automatically.

## Tests

```bash
python -m pytest -q
```

The lightweight tests cover class filtering, ten-logit model output, and checkpoint save/load behavior.

## Repository structure

```text
mnist-unlearning/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── models/
├── data/
├── training/
├── utils/
├── checkpoints/
├── results/
└── tests/
```

## Known limitations

- MNIST download requires network access on the first run.
- Exact training speed depends on available CPU/CUDA/MPS hardware.
- Determinism cannot be guaranteed across all GPU libraries and versions.
- This repository establishes baselines only and does not implement machine-unlearning algorithms.
