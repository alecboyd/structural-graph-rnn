"""Model registry that maps model identifiers to construction functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch.nn as nn

from src.core.types import (
    ExperimentConfig,
    MLPModelConfig,
    CRPModelConfig,
    CRPAdaptiveModelConfig,
    MLPAdaptiveModelConfig,
)


# Each registry entry returns an nn.Module instance
BuildFn = Callable[[ExperimentConfig, int, int], nn.Module]


@dataclass(frozen=True)
class ModelEntry:
    """Registry entry containing the public model id and its builder callback."""
    model_id: str
    build: BuildFn


def _build_mlp(cfg: ExperimentConfig, input_dim: int, num_classes: int) -> nn.Module:
    """
    Internal adapter from ``ExperimentConfig`` to ``build_mlp`` arguments.

    Assumptions:
    - Missing ``cfg.mlp`` falls back to ``MLPModelConfig`` defaults.
    """
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
        init_type=cfg.init_type,
        activation=cfg.activation,
        negative_slope=cfg.negative_slope,
    )


def _build_crp(cfg: ExperimentConfig, input_dim: int, num_classes: int) -> nn.Module:
    """
    Internal adapter from ``ExperimentConfig`` to ``build_crp`` arguments.

    It translates config dataclasses into both structural specification
    (schematic settings) and CRP dynamics settings.
    """
    from src.models.crp.factory import build_crp, CRPSpec
    from src.models.crp.model import CRPConfig

    crp_cfg = cfg.crp or CRPModelConfig()
    effective_alpha = cfg.negative_slope if cfg.activation.lower() == "leaky_relu" else 0.0
    dyn_cfg = CRPConfig(
        kappa=crp_cfg.kappa,
        c=crp_cfg.c,
        eps=crp_cfg.eps,
        t_max=crp_cfg.t_max,
        use_certification=crp_cfg.use_certification,
        margin_factor=crp_cfg.margin_factor,
        recurrent_norm=crp_cfg.recurrent_norm,
        weighted_inf_iters=crp_cfg.weighted_inf_iters,
    )
    return build_crp(
        input_dim=input_dim,
        num_classes=num_classes,
        spec=CRPSpec(
            hidden_dim=crp_cfg.hidden_dim,
            schematic=crp_cfg.schematic,
            num_hidden_layers=crp_cfg.num_hidden_layers,
            random_hh_density=crp_cfg.random_hh_density,
            random_hh_seed=crp_cfg.random_hh_seed,
            bias=True,
            cfg=dyn_cfg,
        ),
        init_type=cfg.init_type,
        activation=cfg.activation,
        negative_slope=cfg.negative_slope,
    )


def _build_crp_adaptive(cfg: ExperimentConfig, input_dim: int, num_classes: int) -> nn.Module:
    """
    Internal adapter from ``ExperimentConfig`` to ``build_crp_adaptive`` arguments.
    """
    from src.models.CRPadaptive.factory import build_crp_adaptive, CRPAdaptiveSpec
    from src.models.CRPadaptive.model import CRPAdaptiveConfig

    ad_cfg = cfg.crp_adaptive or CRPAdaptiveModelConfig()
    effective_alpha = cfg.negative_slope if cfg.activation.lower() == "leaky_relu" else 0.0
    dyn_cfg = CRPAdaptiveConfig(
        kappa=ad_cfg.kappa,
        c=ad_cfg.c,
        eps=ad_cfg.eps,
        t_max=ad_cfg.t_max,
        use_certification=ad_cfg.use_certification,
        margin_factor=ad_cfg.margin_factor,
        recurrent_norm=ad_cfg.recurrent_norm,
        weighted_inf_iters=ad_cfg.weighted_inf_iters,
        deepr_ih=ad_cfg.deepr_ih,
        deepr_hh=ad_cfg.deepr_hh,
        deepr_hl=ad_cfg.deepr_hl,
        K_total=ad_cfg.K_total,
        frac_total=ad_cfg.frac_total,
        full_adjacency_allowed=ad_cfg.full_adjacency_allowed,
        deepr_drift_alpha=ad_cfg.deepr_drift_alpha,
        deepr_temperature=ad_cfg.deepr_temperature,
        deepr_debug_checks=ad_cfg.deepr_debug_checks,
    )
    return build_crp_adaptive(
        input_dim=input_dim,
        num_classes=num_classes,
        spec=CRPAdaptiveSpec(
            hidden_dim=ad_cfg.hidden_dim,
            schematic=ad_cfg.schematic,
            num_hidden_layers=ad_cfg.num_hidden_layers,
            random_hh_density=ad_cfg.random_hh_density,
            random_hh_seed=ad_cfg.random_hh_seed,
            bias=True,
            cfg=dyn_cfg,
        ),
        init_type=cfg.init_type,
        activation=cfg.activation,
        negative_slope=cfg.negative_slope,
    )


def _build_mlp_adaptive(cfg: ExperimentConfig, input_dim: int, num_classes: int) -> nn.Module:
    """
    Internal adapter from ``ExperimentConfig`` to ``build_mlp_adaptive`` arguments.
    """
    from src.models.MLPadaptive.factory import build_mlp_adaptive, MLPAdaptiveSpec
    from src.models.MLPadaptive.model import MLPAdaptiveConfig

    ad_cfg = cfg.mlp_adaptive or MLPAdaptiveModelConfig()
    dyn_cfg = MLPAdaptiveConfig(
        K_total=ad_cfg.K_total,
        frac_total=ad_cfg.frac_total,
        deepr_drift_alpha=ad_cfg.deepr_drift_alpha,
        deepr_temperature=ad_cfg.deepr_temperature,
        deepr_debug_checks=ad_cfg.deepr_debug_checks,
    )
    return build_mlp_adaptive(
        input_dim=input_dim,
        num_classes=num_classes,
        spec=MLPAdaptiveSpec(
            hidden_dim=ad_cfg.hidden_dim,
            num_hidden_layers=ad_cfg.num_hidden_layers,
            bias=True,
            cfg=dyn_cfg,
        ),
        init_type=cfg.init_type,
        activation=cfg.activation,
        negative_slope=cfg.negative_slope,
    )



MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "mlp": ModelEntry(model_id="mlp", build=_build_mlp),
    "crp": ModelEntry(model_id="crp", build=_build_crp),
    "crp_adaptive": ModelEntry(model_id="crp_adaptive", build=_build_crp_adaptive),
    "mlp_adaptive": ModelEntry(model_id="mlp_adaptive", build=_build_mlp_adaptive),
}


def build_model(cfg: ExperimentConfig, *, input_dim: int, num_classes: int) -> nn.Module:
    """
    Construct a model from registry using ``cfg.model_id``.

    Inputs:
    - cfg: Experiment-level config containing model id and model-specific fields.
    - input_dim: Flattened feature dimension expected by the model.
    - num_classes: Number of output logits.

    Returns:
    - Instantiated ``nn.Module`` implementing ``forward(..., return_aux=...)``.
    """
    model_id = cfg.model_id.lower()
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {cfg.model_id!r}. Known: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id].build(cfg, input_dim, num_classes)
