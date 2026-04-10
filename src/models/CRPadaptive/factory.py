"""Factory helpers for constructing adaptive CRP classifiers from typed specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import CRPAdaptiveClassifier, CRPAdaptiveConfig
from .schematics import (
    CRPSchematic,
    base_schematic,
    feedforward_schematic,
    random_density_schematic,
)


@dataclass(frozen=True)
class CRPAdaptiveSpec:
    """
    Construction parameters for ``CRPAdaptiveClassifier``.

    ``hidden_dim`` is interpreted by schematic type:
    - ``base``: total hidden vertices.
    - ``feedforward``: per-layer width.
    """

    hidden_dim: int = 256
    bias: bool = True
    cfg: CRPAdaptiveConfig = field(default_factory=CRPAdaptiveConfig)

    schematic: str = "base"  # "base" | "feedforward" | "random_density"
    num_hidden_layers: int = 2  # only used for feedforward
    random_hh_density: float = 0.5  # only for random_density
    random_hh_seed: Optional[int] = None  # only for random_density


def build_crp_adaptive(
    *,
    input_dim: int,
    num_classes: int,
    spec: Optional[CRPAdaptiveSpec] = None,
    schematic: Optional[CRPSchematic] = None,
    init_type: str = "kaiming_uniform",
    activation: str = "leaky_relu",
    negative_slope: float = 0.05,
) -> CRPAdaptiveClassifier:
    """
    Build a ``CRPAdaptiveClassifier`` from named or explicit structural masks.
    """
    if spec is None:
        spec = CRPAdaptiveSpec()

    if schematic is None:
        name = spec.schematic.lower()
        if name == "base":
            schematic = base_schematic(
                input_dim=input_dim,
                hidden_dim=spec.hidden_dim,
                num_classes=num_classes,
            )
        elif name == "feedforward":
            schematic = feedforward_schematic(
                input_dim=input_dim,
                hidden_dim=spec.hidden_dim,
                num_classes=num_classes,
                num_hidden_layers=spec.num_hidden_layers,
            )
        elif name == "random_density":
            schematic = random_density_schematic(
                input_dim=input_dim,
                hidden_dim=spec.hidden_dim,
                num_classes=num_classes,
                hh_density=spec.random_hh_density,
                hh_seed=spec.random_hh_seed,
            )
        else:
            raise ValueError(f"Unknown CRP schematic: {spec.schematic!r}")

    return CRPAdaptiveClassifier(
        cfg=spec.cfg,
        bias=spec.bias,
        init_type=init_type,
        activation=activation,
        negative_slope=negative_slope,
        MIH=schematic.MIH,
        MH=schematic.MH,
        MHL=schematic.MHL,
        MIH_edges=schematic.MIH_edges,
        MH_edges=schematic.MH_edges,
        MHL_edges=schematic.MHL_edges,
    )
