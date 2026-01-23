from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .types import AuxDict


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def _forward_with_aux(model: torch.nn.Module, x: torch.Tensor):
    """
    Calls model(x, return_aux=True).

    Returns:
      logits, aux (aux may be empty)
    """
    out = model(x, return_aux=True)  # standardized
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)
    return out


def _extract_aux_metrics(aux: AuxDict) -> Dict[str, float]:
    """
    Standardizes optional CRP-style aux fields into scalar batch aggregates.
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
