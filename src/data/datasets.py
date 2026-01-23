from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets as tv_datasets

from .transforms import mnist_transform


def make_mnist_splits(
    data_dir: str,
    val_size: int = 10_000,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, Dataset, int, int]:
    """
    Returns:
      train_set, val_set, test_set, input_dim, num_classes

    - Uses random_split for train/val, but now with a seeded generator so the split is reproducible.
    """
    tfm = mnist_transform()

    full_train = tv_datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_set = tv_datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    if val_size <= 0:
        raise ValueError(f"val_size must be > 0, got {val_size}")

    train_size = len(full_train) - val_size
    if train_size <= 0:
        raise ValueError(
            f"val_size={val_size} is too large for MNIST train set of size {len(full_train)}"
        )

    g = torch.Generator()
    g.manual_seed(seed)

    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    input_dim = 28 * 28
    num_classes = 10
    return train_set, val_set, test_set, input_dim, num_classes
