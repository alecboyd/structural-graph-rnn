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

    Returns:
    - A dict containing sums and counts used for streaming epoch metrics.
    """
    metrics: Dict[str, float] = {}

    tau = aux.get("tau", None)
    certified = aux.get("certified", None)

    if tau is not None:
        t = tau.detach().float()
        metrics["tau_sum"] = float(t.sum().item())
        metrics["tau_count"] = float(t.numel())

    if certified is not None:
        c = certified.detach().float()
        metrics["cert_sum"] = float(c.sum().item())
        metrics["cert_count"] = float(c.numel())

    return metrics


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

    tau_sum = 0.0
    tau_count = 0.0
    cert_sum = 0.0
    cert_count = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, aux = _forward_with_aux(model, x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        n += bsz

        m = _extract_aux_metrics(aux)
        tau_sum += m.get("tau_sum", 0.0)
        tau_count += m.get("tau_count", 0.0)
        cert_sum += m.get("cert_sum", 0.0)
        cert_count += m.get("cert_count", 0.0)

    metrics: Dict[str, float] = {}
    if cert_count > 0:
        metrics["cert_rate"] = cert_sum / cert_count
    if tau_count > 0:
        metrics["tau_mean"] = tau_sum / tau_count

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

    tau_sum = 0.0
    tau_count = 0.0
    cert_sum = 0.0
    cert_count = 0.0

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
        tau_sum += m.get("tau_sum", 0.0)
        tau_count += m.get("tau_count", 0.0)
        cert_sum += m.get("cert_sum", 0.0)
        cert_count += m.get("cert_count", 0.0)

    metrics: Dict[str, float] = {}
    if cert_count > 0:
        metrics["cert_rate"] = cert_sum / cert_count
    if tau_count > 0:
        metrics["tau_mean"] = tau_sum / tau_count

    return total_loss / max(n, 1), total_acc / max(n, 1), metrics
