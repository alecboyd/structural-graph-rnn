from __future__ import annotations

import torch


def default_device() -> str:
    """Return the default compute device identifier for this runtime."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_device(x, device: str):
    """
    Recursively move a nested batch structure to a target device.

    Inputs:
    - x: Tensor or container composed of tensors (list, tuple, dict).
    - device: PyTorch device string such as "cpu" or "cuda".

    Returns:
    - The same container structure with tensor leaves moved to ``device``.

    Side effects:
    - Creates new containers for lists, tuples, and dicts.

    Assumptions:
    - Non-tensor leaves are returned unchanged.
    """
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x
