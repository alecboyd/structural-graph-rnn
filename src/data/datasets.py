from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

@dataclass
class DatasetSpec:
    name: str
    input_dim: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader] = None

def get_dataset(
    name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
    device: str = "cpu",
    val_size: int = 10_000,
) -> DatasetSpec:
    name = name.lower()

    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

        train_size = len(full_train) - val_size
        train_set, val_set = random_split(full_train, [train_size, val_size])

        pin = (device == "cuda")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin)

        return DatasetSpec(
            name="mnist",
            input_dim=28 * 28,
            num_classes=10,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

    raise ValueError(f"Unknown dataset name: {name!r}")