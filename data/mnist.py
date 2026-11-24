from typing import Optional

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def _mnist_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def filter_mnist_dataset(dataset: Dataset, target_class: int) -> Subset:
    """Return a Subset that excludes target_class without changing dataset."""
    if target_class not in range(10):
        raise ValueError("target_class must be between 0 and 9")

    indices = []
    for index in range(len(dataset)):
        _, label = dataset[index]
        if int(label) != target_class:
            indices.append(index)
    return Subset(dataset, indices)


def create_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    excluded_class: Optional[int] = None,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    transform = _mnist_transform()
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    if excluded_class is not None:
        training_dataset = filter_mnist_dataset(train_dataset, excluded_class)
    else:
        training_dataset = train_dataset

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return training_dataset, test_dataset, train_loader, test_loader
