from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch.utils.data import DataLoader

from .datasets import make_mnist_splits


@dataclass
class DatasetSpec:
    """
    This is kept to match the existing training code expectations.
    """
    name: str
    input_dim: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader] = None


class MNISTDataModule:
    """
    Responsible for:
    - creating dataset splits
    - creating DataLoaders with the right settings (pin_memory, shuffle)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        device: str = "cpu",
        val_size: int = 10_000,
        seed: int = 0,
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.val_size = val_size
        self.seed = seed

        self.input_dim: int = 28 * 28
        self.num_classes: int = 10

        self._train_set = None
        self._val_set = None
        self._test_set = None

    def setup(self) -> None:
        train_set, val_set, test_set, input_dim, num_classes = make_mnist_splits(
            data_dir=self.data_dir,
            val_size=self.val_size,
            seed=self.seed,
        )
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set
        self.input_dim = input_dim
        self.num_classes = num_classes

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        pin = (self.device == "cuda")
        return DataLoader(
            self._train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        pin = (self.device == "cuda")
        return DataLoader(
            self._val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        pin = (self.device == "cuda")
        return DataLoader(
            self._test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin,
        )


def get_dataset(
    name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
    device: str = "cpu",
    val_size: int = 10_000,
    seed: int = 0,
) -> DatasetSpec:
    """
    Backwards-compatible wrapper returning the same DatasetSpec your existing trainer uses.
    """
    name = name.lower()

    if name == "mnist":
        dm = MNISTDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            val_size=val_size,
            seed=seed,
        )
        dm.setup()
        return DatasetSpec(
            name="mnist",
            input_dim=dm.input_dim,
            num_classes=dm.num_classes,
            train_loader=dm.train_dataloader(),
            val_loader=dm.val_dataloader(),
            test_loader=dm.test_dataloader(),
        )

    raise ValueError(f"Unknown dataset name: {name!r}")
