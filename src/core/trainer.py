"""High-level training orchestration that wires data, models, and loop APIs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.datamodules import get_dataset
from .loops import train_one_epoch, eval_one_epoch
from .types import ExperimentConfig, MLPModelConfig, CRPModelConfig
from src.models.registry import build_model
from src.models.mlp.factory import build_mlp, MLPSpec
from src.models.crp.factory import build_crp, CRPSpec
from src.models.crp.model import CRPConfig



def _show_extra(metrics: dict[str, float], prefix: str) -> str:
    """
    Format optional aux-derived metrics for console logging.

    Inputs:
    - metrics: Dict returned by loop functions.
    - prefix: Label prefix such as ``train`` or ``val``.
    """
    if "cert_rate" in metrics and "tau_mean" in metrics:
        return f" | {prefix}_cert={metrics['cert_rate']:.3f} | {prefix}_tau={metrics['tau_mean']:.2f}"
    if "cert_rate" in metrics:
        return f" | {prefix}_cert={metrics['cert_rate']:.3f}"
    if "tau_mean" in metrics:
        return f" | {prefix}_tau={metrics['tau_mean']:.2f}"
    return ""

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _debug_compare_mlp_crp(
    cfg: ExperimentConfig,
    *,
    input_dim: int,
    num_classes: int,
    device: str,
    loader,
) -> None:
    """
    Debug-only instrumentation comparing MLP and feedforward CRP on one batch.
    """
    train_cfg = cfg.train
    mlp_cfg = cfg.mlp or MLPModelConfig()
    crp_cfg = cfg.crp or CRPModelConfig()

    # Build MLP with a deterministic init.
    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)
    mlp = build_mlp(
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
    ).to(device)

    # Build CRP feedforward with a deterministic init.
    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)
    crp = build_crp(
        input_dim=input_dim,
        num_classes=num_classes,
        spec=CRPSpec(
            hidden_dim=crp_cfg.hidden_dim,
            num_hidden_layers=crp_cfg.num_hidden_layers,
            bias=True,
            schematic=crp_cfg.schematic,
            cfg=CRPConfig(
                kappa=crp_cfg.kappa,
                c=crp_cfg.c,
                alpha=cfg.negative_slope if cfg.activation.lower() == "leaky_relu" else 0.0,
                eps=crp_cfg.eps,
                t_max=crp_cfg.t_max,
                use_certification=crp_cfg.use_certification,
                margin_factor=crp_cfg.margin_factor,
            ),
        ),
        init_type=cfg.init_type,
        activation=cfg.activation,
        negative_slope=cfg.negative_slope,
    ).to(device)

    batch = next(iter(loader))
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        mlp_logits = mlp(x)
        crp_logits = crp(x)
        max_abs = (mlp_logits - crp_logits).abs().max().item()
        mean_abs = (mlp_logits - crp_logits).abs().mean().item()

    print("DEBUG_COMPARE: parameter counts")
    print(f"DEBUG_COMPARE: MLP params = {_count_params(mlp)}")
    print(f"DEBUG_COMPARE: CRP params = {_count_params(crp)}")
    print("DEBUG_COMPARE: logits diff on one batch")
    print(f"DEBUG_COMPARE: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

    mlp.train()
    crp.train()
    mlp.zero_grad(set_to_none=True)
    crp.zero_grad(set_to_none=True)

    mlp_logits = mlp(x)
    crp_logits = crp(x)
    loss_mlp = F.cross_entropy(mlp_logits, y)
    loss_crp = F.cross_entropy(crp_logits, y)
    loss_mlp.backward()
    loss_crp.backward()

    def first_grad_norms(model: torch.nn.Module, max_items: int = 5) -> list[tuple[str, float]]:
        items = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            items.append((name, float(p.grad.norm().item())))
            if len(items) >= max_items:
                break
        return items

    print("DEBUG_COMPARE: grad norms (first few parameters)")
    for name, g in first_grad_norms(mlp):
        print(f"DEBUG_COMPARE: MLP {name} grad_norm={g:.6f}")
    for name, g in first_grad_norms(crp):
        print(f"DEBUG_COMPARE: CRP {name} grad_norm={g:.6f}")


def run_training(cfg: ExperimentConfig, *, debug_compare: bool = False) -> None:
    """
    Execute end-to-end training and evaluation for one experiment config.

    Interactions:
    - Builds dataset loaders via ``src.data.datamodules.get_dataset``.
    - Instantiates models through ``src.models.registry.build_model``.
    - Runs per-epoch train/eval logic via ``src.core.loops``.

    Side effects:
    - Allocates model/optimizer state and runs gradient updates.
    - Prints training progress and final test metrics to stdout.
    """
    train_cfg = cfg.train
    print(
        f"config: init_type={cfg.init_type} activation={cfg.activation} "
        f"negative_slope={cfg.negative_slope:.4f}"
    )

    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)

    ds = get_dataset(
        name=cfg.dataset,
        data_dir=cfg.data_dir,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        device=train_cfg.device,
        seed=train_cfg.seed,
    )

    input_dim = cfg.input_dim if cfg.input_dim is not None else ds.input_dim
    num_classes = cfg.num_classes if cfg.num_classes is not None else ds.num_classes

    if debug_compare:
        _debug_compare_mlp_crp(
            cfg,
            input_dim=input_dim,
            num_classes=num_classes,
            device=train_cfg.device,
            loader=ds.train_loader,
        )

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
