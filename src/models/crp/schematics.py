from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CRPSchematic:
    name: str
    MIH: torch.Tensor
    MH: torch.Tensor
    MHL: torch.Tensor


def base(*, input_dim: int, hidden_dim: int, num_classes: int) -> CRPSchematic:
    """
    Base schematic = fully connected masks (all ones),
    matching the current model behavior.
    """
    return CRPSchematic(
        name="base",
        MIH=torch.ones(input_dim, hidden_dim),
        MH=torch.ones(hidden_dim, hidden_dim),
        MHL=torch.ones(hidden_dim, num_classes),
    )
