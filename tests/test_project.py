import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.mnist import filter_mnist_dataset
from models.cnn import SimpleCNN
from utils.checkpoint import load_checkpoint, save_checkpoint


class DummyMNIST(Dataset):
    def __init__(self):
        self.labels = torch.tensor([0, 1, 2, 3, 3, 4])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), int(self.labels[index])


def test_filter_dataset_excludes_only_target_class():
    original = DummyMNIST()
    filtered = filter_mnist_dataset(original, 3)

    labels = [int(original[index][1]) for index in filtered.indices]
    assert len(filtered) == 4
    assert 3 not in labels


def test_model_returns_ten_logits():
    model = SimpleCNN(num_classes=10)
    output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)


def test_checkpoint_save_and_load():
    model = SimpleCNN(num_classes=10)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {"name": "SimpleCNN", "num_classes": 10},
            "excluded_class": 3,
        }
        save_checkpoint(path, checkpoint)
        loaded = load_checkpoint(path)

        restored = SimpleCNN(num_classes=loaded["model_config"]["num_classes"])
        restored.load_state_dict(loaded["model_state_dict"])
        assert restored(torch.zeros(1, 1, 28, 28)).shape == (1, 10)
