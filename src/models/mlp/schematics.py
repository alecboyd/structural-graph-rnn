from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLPSchematic:
    """
    MLP currently has no structural schematic (no masks/adjacency).
    This exists purely for symmetry with CRP.
    """
    name: str = "base"


def base() -> MLPSchematic:
    return MLPSchematic(name="base")
