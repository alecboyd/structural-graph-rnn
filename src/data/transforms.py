from __future__ import annotations

from torchvision import transforms


def mnist_transform() -> transforms.Compose:
    """
    Standard MNIST preprocessing
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
