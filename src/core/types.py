"""Shared type aliases and configuration dataclasses used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

import torch

ModelID = Literal["mlp", "crp"]
AuxDict = Dict[str, Any]


@dataclass
class TrainLoopConfig:
    """
    Hyperparameters and runtime options for the training loop.

    This config is consumed by ``src.core.trainer.run_training`` to construct
    loaders, optimizer, and epoch scheduling behavior.
    """
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = None


@dataclass
class MLPModelConfig:
    """Configuration for building the MLP model variant."""
    hidden_dim: int = 256
    num_hidden_layers: int = 2


@dataclass
class CRPModelConfig:
    """Configuration for building the CRP model variant."""
    hidden_dim: int = 256

@dataclass
class CRPModelConfig:
    """
    Full configuration for CRP model construction and dynamics.

    Assumptions:
    - ``schematic`` selects the structural mask strategy used by the factory.
    - ``num_hidden_layers`` is primarily relevant for feedforward-style schematics. Other schematics lock the number of hidden layers to one. 
    """
    hidden_dim: int = 256

    schematic: str = "base"           # "base" | "feedforward" | ...
    num_hidden_layers: int = 2        # only used if schematic=="feedforward"

    kappa: float = 1.0
    c: float = 0.95
    alpha: float = 0.05
    eps: float = 1e-8
    t_max: int = 32
    use_certification: bool = True
    margin_factor: float = 2.0

    kappa: float = 1.0
    c: float = 0.95
    alpha: float = 0.05
    eps: float = 1e-8
    t_max: int = 32
    use_certification: bool = True
    margin_factor: float = 2.0



@dataclass
class ExperimentConfig:
    """
    Top-level experiment specification passed from CLI to trainer.

    It ties together dataset selection, model selection, training config, and
    optional explicit dimension overrides.
    """
    model_id: ModelID = "mlp"
    dataset: str = "mnist"
    data_dir: str = "./data"

    # IMPORTANT: must be default_factory, not TrainLoopConfig()
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)

    mlp: Optional[MLPModelConfig] = None
    crp: Optional[CRPModelConfig] = None

    # optional overrides
    input_dim: Optional[int] = None
    num_classes: Optional[int] = None

    # global model behavior
    init_type: str = "kaiming_uniform"   # "kaiming_uniform" | "linear_default"
    activation: str = "leaky_relu"       # "relu" | "leaky_relu"
    negative_slope: float = 0.05
