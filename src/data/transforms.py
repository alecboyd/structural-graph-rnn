"""Input preprocessing transforms used by dataset builders."""

from __future__ import annotations

from torchvision import transforms


def mnist_transform() -> transforms.Compose:
    """
    Return the default MNIST preprocessing pipeline.

    Output:
    - ``ToTensor`` conversion followed by dataset-specific normalization.

    Assumptions:
    - Input images are grayscale MNIST images in the standard range.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
