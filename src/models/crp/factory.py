"""Factory helpers for constructing CRP classifiers from typed specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import CRPClassifier, CRPConfig
from .schematics import CRPSchematic, base_schematic, feedforward_schematic


@dataclass(frozen=True)
class CRPSpec:
    """
    Construction parameters for ``CRPClassifier``.

    ``hidden_dim`` is interpreted by schematic type:
    - ``base``: total hidden vertices.
    - ``feedforward``: per-layer width.
    """
    hidden_dim: int = 256
    bias: bool = True
    cfg: CRPConfig = field(default_factory=CRPConfig)

    schematic: str = "base"          # "base" | "feedforward"
    num_hidden_layers: int = 2       # only for feedforward

def build_crp(
    *,
    input_dim: int,
    num_classes: int,
    spec: Optional[CRPSpec] = None,
    schematic: Optional[CRPSchematic] = None,
    init_type: str = "kaiming_uniform",
    activation: str = "leaky_relu",
    negative_slope: float = 0.05,
) -> CRPClassifier:
    """
    Build a ``CRPClassifier`` using a named or explicit structural schematic.

    Inputs:
    - input_dim: Flattened feature dimension.
    - num_classes: Number of output logits.
    - spec: CRP hyperparameters and schematic selection.
    - schematic: Optional prebuilt mask bundle; overrides ``spec.schematic``.

    Returns:
    - Configured ``CRPClassifier`` with masks and dynamics config attached.
    """
    if spec is None:
        spec = CRPSpec()

    if schematic is None:
        name = spec.schematic.lower()

        if name == "base":
            # hidden_dim is TOTAL hidden vertices here, one layer is forced. 
            schematic = base_schematic(
                input_dim=input_dim,
                hidden_dim=spec.hidden_dim,
                num_classes=num_classes,
            )

        elif name == "feedforward":
            # hidden_dim is PER-LAYER width here, total hidden = hidden_dim * num_hidden_layers
            schematic = feedforward_schematic(
                input_dim=input_dim,
                hidden_dim=spec.hidden_dim,
                num_classes=num_classes,
                num_hidden_layers=spec.num_hidden_layers,
            )

        else:
            raise ValueError(f"Unknown CRP schematic: {spec.schematic!r}")

    return CRPClassifier(
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
