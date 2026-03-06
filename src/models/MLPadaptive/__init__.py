"""Adaptive MLP package with global-budget DeepR sparsity."""

from .model import MLPAdaptiveClassifier, MLPAdaptiveConfig
from .factory import MLPAdaptiveSpec, build_mlp_adaptive

__all__ = [
    "MLPAdaptiveClassifier",
    "MLPAdaptiveConfig",
    "MLPAdaptiveSpec",
    "build_mlp_adaptive",
]
