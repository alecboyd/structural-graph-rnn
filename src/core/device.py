from __future__ import annotations

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_device(x, device: str):
    """
    Recursively move common batch structures to device.
    Supports Tensors, tuples/lists, and dicts.
    """
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x