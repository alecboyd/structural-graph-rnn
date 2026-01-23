from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch.nn as nn

from src.core.types import ExperimentConfig, MLPModelConfig, CRPModelConfig


# Each registry entry returns an nn.Module instance
BuildFn = Callable[[ExperimentConfig, int, int], nn.Module]


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    build: BuildFn


def _build_mlp(cfg: ExperimentConfig, input_dim: int, num_classes: int) -> nn.Module:
    from src.models.mlp.factory import build_mlp, MLPSpec

    mlp_cfg = cfg.mlp or MLPModelConfig()
    return build_mlp(
        input_dim=input_dim,
        num_classes=num_classes,
        spec=MLPSpec(
            hidden_dim=mlp_cfg.hidden_dim,
            num_hidden_layers=mlp_cfg.num_hidden_layers,
            bias=True,
        ),
    )


def _build_crp(cfg: ExperimentConfig, input_dim: int, num_classes: int) -> nn.Module:
    from src.models.crp.factory import build_crp, CRPSpec
    from src.models.crp.model import CRPConfig

    crp_cfg = cfg.crp or CRPModelConfig()
    dyn_cfg = CRPConfig(
        kappa=crp_cfg.kappa,
        c=crp_cfg.c,
        alpha=crp_cfg.alpha,
        eps=crp_cfg.eps,
        t_max=crp_cfg.t_max,
        use_certification=crp_cfg.use_certification,
        margin_factor=crp_cfg.margin_factor,
    )
    return build_crp(
        input_dim=input_dim,
        num_classes=num_classes,
        spec=CRPSpec(
            hidden_dim=crp_cfg.hidden_dim,
            bias=True,
            cfg=dyn_cfg,
        ),
    )


MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "mlp": ModelEntry(model_id="mlp", build=_build_mlp),
    "crp": ModelEntry(model_id="crp", build=_build_crp),
}


def build_model(cfg: ExperimentConfig, *, input_dim: int, num_classes: int) -> nn.Module:
    model_id = cfg.model_id.lower()
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {cfg.model_id!r}. Known: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id].build(cfg, input_dim, num_classes)
