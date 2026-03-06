"""Factory helpers for constructing adaptive MLP classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import MLPAdaptiveClassifier, MLPAdaptiveConfig


@dataclass(frozen=True)
class MLPAdaptiveSpec:
    """Construction parameters for ``MLPAdaptiveClassifier``."""

    hidden_dim: int = 256
    num_hidden_layers: int = 2
    bias: bool = True
    cfg: MLPAdaptiveConfig = field(default_factory=MLPAdaptiveConfig)


def build_mlp_adaptive(
    *,
    input_dim: int,
    num_classes: int,
    spec: Optional[MLPAdaptiveSpec] = None,
    init_type: str = "kaiming_uniform",
    activation: str = "leaky_relu",
    negative_slope: float = 0.05,
) -> MLPAdaptiveClassifier:
    """Build an adaptive MLP classifier from explicit dimensions and spec."""
    if spec is None:
        spec = MLPAdaptiveSpec()

    return MLPAdaptiveClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=spec.hidden_dim,
        num_hidden_layers=spec.num_hidden_layers,
        cfg=spec.cfg,
        bias=spec.bias,
        init_type=init_type,
        activation=activation,
        negative_slope=negative_slope,
    )
