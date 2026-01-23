from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

import torch

ModelID = Literal["mlp", "crp"]
AuxDict = Dict[str, Any]


@dataclass
class TrainLoopConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MLPModelConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 2


@dataclass
class CRPModelConfig:
    hidden_dim: int = 256
    # CRP dynamics / certification
    kappa: float = 1.0
    c: float = 0.95
    alpha: float = 0.05
    eps: float = 1e-8
    t_max: int = 32
    use_certification: bool = True
    margin_factor: float = 2.0


@dataclass
class ExperimentConfig:
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
