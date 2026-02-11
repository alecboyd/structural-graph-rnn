"""Factory helpers for constructing MLP classifiers from typed specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import MLPClassifier
from .schematics import MLPSchematic, base as base_schematic


@dataclass(frozen=True)
class MLPSpec:
    """Construction parameters for ``MLPClassifier``."""
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    bias: bool = True


def build_mlp(
    *,
    input_dim: int,
    num_classes: int,
    spec: Optional[MLPSpec] = None,
    schematic: Optional[MLPSchematic] = None,
    init_type: str = "kaiming_uniform",
    activation: str = "leaky_relu",
    negative_slope: float = 0.05,
) -> MLPClassifier:
    """
    Build an ``MLPClassifier`` from explicit dimensions and optional spec.

    Inputs:
    - input_dim: Flattened feature size expected by the model.
    - num_classes: Number of output logits.
    - spec: Optional MLP hyperparameters (defaults when omitted).
    - schematic: Optional schematic descriptor for API symmetry with CRP.

    Notes:
    - ``schematic`` is currently not used for parameterization, but is accepted
      to keep a consistent factory signature across model families.
    """
    if spec is None:
        spec = MLPSpec()
    if schematic is None:
        schematic = base_schematic()  # unused currently

    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=spec.hidden_dim,
        num_hidden_layers=spec.num_hidden_layers,
        bias=spec.bias,
        init_type=init_type,
        activation=activation,
        negative_slope=negative_slope,
    )
