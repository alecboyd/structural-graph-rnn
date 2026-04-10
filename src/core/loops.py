"""Reusable train and evaluation loops with standardized aux metric handling."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .types import AuxDict


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute mean top-1 accuracy for a batch."""
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def _forward_with_aux(model: torch.nn.Module, x: torch.Tensor):
    """
    Call ``model`` with the standardized auxiliary-output contract.

    Returns:
    - Tuple ``(logits, aux_dict)`` where ``aux_dict`` may be empty.

    Assumptions:
    - The model implements ``forward(..., return_aux=True)`` and returns
      ``(Tensor, dict)``.
    """
    out = model(x, return_aux=True)  # standardized
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)
    return out


def _extract_aux_metrics(aux: AuxDict) -> Dict[str, float]:
    """
    Reduce optional per-sample aux tensors into scalar batch aggregates.

    Supported aux keys:
    - ``tau``: integer convergence/certification step per sample.
    - ``certified``: boolean certification indicator per sample.
    - ``recurrent_scale``: realized recurrent normalization factor.
    - ``recurrent_shrunk``: indicator that recurrent normalization shrank W_H.
    - ``recurrent_norm_m``: pre-scaling recurrent norm value m.
    - ``recurrent_norm_c``: target contraction cap c.
    - ``rho``: contraction factor used by CRP dynamics.

    Returns:
    - A dict containing sums and counts used for streaming epoch metrics.
    """
    metrics: Dict[str, float] = {}

    def add(key_out: str, value) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            t = value.detach().float()
            metrics[f"{key_out}_sum"] = float(t.sum().item())
            metrics[f"{key_out}_count"] = float(t.numel())
            return
        if isinstance(value, (bool, int, float)):
            metrics[f"{key_out}_sum"] = float(value)
            metrics[f"{key_out}_count"] = 1.0

    add("tau", aux.get("tau", None))
    add("cert", aux.get("certified", None))
    add("recurrent_scale", aux.get("recurrent_scale", None))
    add("recurrent_shrunk", aux.get("recurrent_shrunk", None))
    add("recurrent_norm_m", aux.get("recurrent_norm_m", None))
    add("recurrent_norm_c", aux.get("recurrent_norm_c", None))
    add("rho", aux.get("rho", None))

    return metrics


def _finalize_aux_epoch_metrics(sums: Dict[str, float], counts: Dict[str, float]) -> Dict[str, float]:
    """Convert accumulated aux sums/counts into stable epoch-level metrics."""
    out: Dict[str, float] = {}
    if counts.get("cert", 0.0) > 0:
        out["cert_rate"] = sums["cert"] / counts["cert"]
    if counts.get("tau", 0.0) > 0:
        out["tau_mean"] = sums["tau"] / counts["tau"]
    if counts.get("recurrent_scale", 0.0) > 0:
        out["recurrent_scale_mean"] = sums["recurrent_scale"] / counts["recurrent_scale"]
    if counts.get("recurrent_shrunk", 0.0) > 0:
        out["recurrent_shrink_rate"] = sums["recurrent_shrunk"] / counts["recurrent_shrunk"]
    if counts.get("recurrent_norm_m", 0.0) > 0:
        out["recurrent_norm_m_mean"] = sums["recurrent_norm_m"] / counts["recurrent_norm_m"]
    if counts.get("recurrent_norm_c", 0.0) > 0:
        out["recurrent_norm_c_mean"] = sums["recurrent_norm_c"] / counts["recurrent_norm_c"]
    if counts.get("rho", 0.0) > 0:
        out["rho_mean"] = sums["rho"] / counts["rho"]
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one training epoch and aggregate optional aux-derived metrics.

    Inputs:
    - model: Module that supports ``return_aux=True`` in forward (all modules are expected to do this by default).
    - loader: Supervised batches ``(x, y)``.
    - opt: Optimizer updated once per batch.
    - device: Target compute device string.

    Returns:
    - ``(mean_loss, metrics)`` where ``metrics`` may contain ``cert_rate`` and
      ``tau_mean`` depending on model aux payload.

    Side effects:
    - Mutates model and optimizer state via backpropagation.
    """
    model.train()
    total_loss = 0.0
    n = 0

    aux_sums = {
        "tau": 0.0,
        "cert": 0.0,
        "recurrent_scale": 0.0,
        "recurrent_shrunk": 0.0,
        "recurrent_norm_m": 0.0,
        "recurrent_norm_c": 0.0,
        "rho": 0.0,
    }
    aux_counts = {
        "tau": 0.0,
        "cert": 0.0,
        "recurrent_scale": 0.0,
        "recurrent_shrunk": 0.0,
        "recurrent_norm_m": 0.0,
        "recurrent_norm_c": 0.0,
        "rho": 0.0,
    }

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = _forward_with_aux(model, x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if hasattr(model, "deepr_step_all"):
            # DeepR-managed parameters are updated explicitly after optimizer step.
            lr = float(opt.param_groups[0].get("lr", 0.0))
            drift = getattr(getattr(model, "cfg", None), "deepr_drift_alpha", None)
            temp = getattr(getattr(model, "cfg", None), "deepr_temperature", None)
            kwargs = {"lr": lr}
            if drift is not None:
                kwargs["drift_alpha"] = float(drift)
            if temp is not None:
                kwargs["T"] = float(temp)
            model.deepr_step_all(**kwargs)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        n += bsz

        m = _extract_aux_metrics(aux)
        for name in aux_sums.keys():
            aux_sums[name] += m.get(f"{name}_sum", 0.0)
            aux_counts[name] += m.get(f"{name}_count", 0.0)

    metrics = _finalize_aux_epoch_metrics(aux_sums, aux_counts)

    return total_loss / max(n, 1), metrics


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Run one evaluation epoch without gradient updates.

    Returns:
    - ``(mean_loss, mean_accuracy, metrics)`` where ``metrics`` mirrors the
      aux aggregation behavior of ``train_one_epoch``.

    Side effects:
    - None on model parameters; model mode is set to eval during the call.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    aux_sums = {
        "tau": 0.0,
        "cert": 0.0,
        "recurrent_scale": 0.0,
        "recurrent_shrunk": 0.0,
        "recurrent_norm_m": 0.0,
        "recurrent_norm_c": 0.0,
        "rho": 0.0,
    }
    aux_counts = {
        "tau": 0.0,
        "cert": 0.0,
        "recurrent_scale": 0.0,
        "recurrent_shrunk": 0.0,
        "recurrent_norm_m": 0.0,
        "recurrent_norm_c": 0.0,
        "rho": 0.0,
    }

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = _forward_with_aux(model, x)
        loss = F.cross_entropy(logits, y)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        n += bsz

        m = _extract_aux_metrics(aux)
        for name in aux_sums.keys():
            aux_sums[name] += m.get(f"{name}_sum", 0.0)
            aux_counts[name] += m.get(f"{name}_count", 0.0)

    metrics = _finalize_aux_epoch_metrics(aux_sums, aux_counts)

    return total_loss / max(n, 1), total_acc / max(n, 1), metrics
