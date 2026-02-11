"""Dataset construction utilities and reproducible train/val/test splitting."""

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets as tv_datasets

from .transforms import mnist_transform


def make_mnist_splits(
    data_dir: str,
    val_size: int = 10_000,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset, int, int]:
    """
    Build MNIST train, validation, and test splits with deterministic sampling.

    Inputs:
    - data_dir: Root directory for torchvision dataset storage.
    - val_size: Number of examples reserved from the official train set.
    - seed: RNG seed used for ``random_split`` reproducibility.

    Returns:
    - ``(train_set, val_set, test_set, input_dim, num_classes)``.

    Side effects:
    - May download MNIST files when missing.

    Assumptions:
    - ``val_size`` is strictly positive and smaller than the train split size.
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

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    input_dim = 28 * 28
    num_classes = 10
    return train_set, val_set, test_set, input_dim, num_classes
