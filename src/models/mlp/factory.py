from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import MLPClassifier
from .schematics import MLPSchematic, base as base_schematic


@dataclass(frozen=True)
class MLPSpec:
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    bias: bool = True


def build_mlp(
    *,
    input_dim: int,
    num_classes: int,
    spec: Optional[MLPSpec] = None,
    schematic: Optional[MLPSchematic] = None,
) -> MLPClassifier:
    """
    Factory keeps construction policy out of model.py.
    Base schematic is unused for now.
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
    )
