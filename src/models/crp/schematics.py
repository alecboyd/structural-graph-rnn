"""Structural mask builders for CRP model connectivity patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CRPSchematic:
    """
    Structural mask bundle consumed by ``CRPClassifier``.

    Invariants:
    - ``MIH`` has shape ``[input_dim, hidden_dim]``.
    - ``MH`` has shape ``[hidden_dim, hidden_dim]``.
    - ``MHL`` has shape ``[hidden_dim, num_classes]``.
    """

    name: str
    dag: bool
    MIH: torch.Tensor
    MH: torch.Tensor
    MHL: torch.Tensor
    # Optional cached edge lists; tuples of (src_index, dst_index).
    MIH_edges: Optional[torch.Tensor] = None
    MH_edges: Optional[torch.Tensor] = None
    MHL_edges: Optional[torch.Tensor] = None


def _edges_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Return edge list indices (src, dst) for nonzero entries in a mask.
    """
    if mask.dim() != 2:
        raise ValueError("mask must be 2D")
    return mask.nonzero(as_tuple=False)


def base_schematic(*, input_dim: int, hidden_dim: int, num_classes: int) -> CRPSchematic:
    """
    Build a fully connected CRP schematic using all-ones masks.

    This mirrors the dense behavior of the current CRP baseline.
    """
    MIH = torch.ones(input_dim, hidden_dim)
    MH = torch.ones(hidden_dim, hidden_dim)
    MHL = torch.ones(hidden_dim, num_classes)
    return CRPSchematic(
        name="base",
        dag=False,
        MIH=MIH,
        MH=MH,
        MHL=MHL,
        MIH_edges=_edges_from_mask(MIH),
        MH_edges=_edges_from_mask(MH),
        MHL_edges=_edges_from_mask(MHL),
    )


def feedforward_schematic(
    *, input_dim: int, hidden_dim: int, num_classes: int, num_hidden_layers: int
) -> CRPSchematic:
    """
    Build a DAG-like feedforward schematic represented as CRP masks.

    Interpretation:
    - ``hidden_dim`` is the width of each hidden layer.
    - Total hidden state size becomes ``hidden_dim * num_hidden_layers``.

    Connectivity:
    - Input connects only to layer 0.
    - Recurrent mask connects layer ``l`` to ``l + 1`` only.
    - Output reads from the last hidden layer only.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be >= 1")

    H = num_hidden_layers * hidden_dim

    # Input -> Hidden
    MIH = torch.zeros(input_dim, H)
    MIH[:, 0:hidden_dim] = 1.0

    # Hidden -> Hidden
    MH = torch.zeros(H, H)
    for l in range(num_hidden_layers - 1):
        src_start = l * hidden_dim
        src_end = (l + 1) * hidden_dim
        dst_start = (l + 1) * hidden_dim
        dst_end = (l + 2) * hidden_dim
        MH[src_start:src_end, dst_start:dst_end] = 1.0

    # Hidden -> Output
    MHL = torch.zeros(H, num_classes)
    last_start = (num_hidden_layers - 1) * hidden_dim
    last_end = num_hidden_layers * hidden_dim
    MHL[last_start:last_end, :] = 1.0

    return CRPSchematic(
        name="feedforward",
        dag=True,
        MIH=MIH,
        MH=MH,
        MHL=MHL,
        MIH_edges=_edges_from_mask(MIH),
        MH_edges=_edges_from_mask(MH),
        MHL_edges=_edges_from_mask(MHL),
    )
