"""Schematic wrappers for the adaptive CRP package.

This module reuses the existing CRP schematic builders to keep mask semantics
identical across ``crp`` and ``crp_adaptive``.
"""

from __future__ import annotations

from src.models.crp.schematics import (
    CRPSchematic,
    base_schematic,
    feedforward_schematic,
)

__all__ = [
    "CRPSchematic",
    "base_schematic",
    "feedforward_schematic",
]
