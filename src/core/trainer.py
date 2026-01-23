from __future__ import annotations

import torch

from src.data.datamodules import get_dataset
from .loops import train_one_epoch, eval_one_epoch
from .types import ExperimentConfig, MLPModelConfig, CRPModelConfig
from src.models.registry import build_model



def _show_extra(metrics: dict[str, float], prefix: str) -> str:
    """
    Formats optional certification metrics if present.
    """
    if "cert_rate" in metrics and "tau_mean" in metrics:
        return f" | {prefix}_cert={metrics['cert_rate']:.3f} | {prefix}_tau={metrics['tau_mean']:.2f}"
    if "cert_rate" in metrics:
        return f" | {prefix}_cert={metrics['cert_rate']:.3f}"
    if "tau_mean" in metrics:
        return f" | {prefix}_tau={metrics['tau_mean']:.2f}"
    return ""


def run_training(cfg: ExperimentConfig) -> None:
    """
    Unified training entrypoint replacing train_crp.py and train_mlp.py.

    - Uses src.data.datasets.get_dataset (unchanged)
    - Uses src.models.{mlp,crp}.model (unchanged)
    - Shared train/eval loops live in core.loops
    """
    train_cfg = cfg.train

    ds = get_dataset(
        name=cfg.dataset,
        data_dir=cfg.data_dir,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        device=train_cfg.device,
    )

    input_dim = cfg.input_dim if cfg.input_dim is not None else ds.input_dim
    num_classes = cfg.num_classes if cfg.num_classes is not None else ds.num_classes

    model = build_model(cfg, input_dim=input_dim, num_classes=num_classes).to(train_cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for epoch in range(1, train_cfg.epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(model, ds.train_loader, opt, train_cfg.device)
        va_loss, va_acc, va_metrics = eval_one_epoch(model, ds.val_loader, train_cfg.device)

        extra_tr = _show_extra(tr_metrics, "train")
        extra_va = _show_extra(va_metrics, "val")

        print(
            f"epoch {epoch:03d} | train_loss={tr_loss:.4f}{extra_tr} | "
            f"val_loss={va_loss:.4f} | val_acc={va_acc:.4f}{extra_va}"
        )

    if ds.test_loader is not None:
        te_loss, te_acc, te_metrics = eval_one_epoch(model, ds.test_loader, train_cfg.device)
        extra_te = _show_extra(te_metrics, "test")
        print(f"TEST | loss={te_loss:.4f} | acc={te_acc:.4f}{extra_te}")
