"""Adaptive CRP package with DeepR-managed sparse matrices."""

from src.core.deepR import DeepRMaskedMatrix, StaticMaskedMatrix
from .model import CRPAdaptiveClassifier, CRPAdaptiveConfig
from .factory import CRPAdaptiveSpec, build_crp_adaptive
from .schematics import CRPSchematic, base_schematic, feedforward_schematic

__all__ = [
    "CRPAdaptiveClassifier",
    "CRPAdaptiveConfig",
    "DeepRMaskedMatrix",
    "StaticMaskedMatrix",
    "CRPAdaptiveSpec",
    "build_crp_adaptive",
    "CRPSchematic",
    "base_schematic",
    "feedforward_schematic",
]
