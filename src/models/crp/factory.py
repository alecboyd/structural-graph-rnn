from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import CRPClassifier, CRPConfig
from .schematics import CRPSchematic, base as base_schematic


@dataclass(frozen=True)
class CRPSpec:
    hidden_dim: int = 256
    bias: bool = True
    cfg: CRPConfig = field(default_factory=CRPConfig)


def build_crp(
    *,
    input_dim: int,
    num_classes: int,
    spec: Optional[CRPSpec] = None,
    schematic: Optional[CRPSchematic] = None,
) -> CRPClassifier:
    """
    Factory owns:
      - choosing schematic (base for now)
      - assembling model with masks
    """
    if spec is None:
        spec = CRPSpec()
    if schematic is None:
        schematic = base_schematic(input_dim=input_dim, hidden_dim=spec.hidden_dim, num_classes=num_classes)

    return CRPClassifier(
        input_dim=input_dim,
        hidden_dim=spec.hidden_dim,
        num_classes=num_classes,
        cfg=spec.cfg,
        bias=spec.bias,
        MIH=schematic.MIH,
        MH=schematic.MH,
        MHL=schematic.MHL,
    )
