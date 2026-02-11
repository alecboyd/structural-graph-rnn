"""Placeholder schematic definitions for the MLP model family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLPSchematic:
    """
    Lightweight schematic descriptor kept for parity with CRP factories.

    MLP currently does not use structural masks, so this object mainly
    preserves a uniform factory surface across model families.
    """
    name: str = "base"


def base() -> MLPSchematic:
    """Return the default no-op MLP schematic descriptor."""
    return MLPSchematic(name="base")
