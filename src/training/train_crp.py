from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.model_crp import CRPClassifier, CRPConfig
from src.data.datasets import get_dataset


@dataclass
class TrainConfig:
    # Model size
    hidden_dim: int = 256

    # CRP dynamics / certification
    kappa: float = 1.0
    c: float = 0.95
    alpha: float = 0.05
    eps: float = 1e-8
    t_max: int = 32
    use_certification: bool = True
    margin_factor: float = 2.0

    # Optim
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Train loop
    epochs: int = 10
    batch_size: int = 128
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Returns:
      avg_loss
      metrics dict (may include certification stats if model provides aux)
    """
    model.train()
    total_loss = 0.0
    n = 0

    # Optional aux stats
    total_cert = 0.0
    total_tau = 0.0
    saw_aux = False

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # If the model supports return_aux, we can collect tau/certified.
        # Loss only depends on logits.
        try:
            out = model(x, return_aux=True)  # type: ignore[misc]
            logits, aux = out  # type: ignore[assignment]
            tau = aux.get("tau", None)
            certified = aux.get("certified", None)
            if tau is not None:
                total_tau += tau.float().sum().item()
                saw_aux = True
            if certified is not None:
                total_cert += certified.float().sum().item()
                saw_aux = True
        except TypeError:
            logits = model(x)

        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        n += bsz

    metrics: Dict[str, float] = {}
    if saw_aux and n > 0:
        metrics["cert_rate"] = total_cert / n
        metrics["tau_mean"] = total_tau / n

    return total_loss / max(n, 1), metrics


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Eval for one epoch.

    Returns:
      avg_loss, avg_acc, metrics dict (may include certification stats)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    # Optional aux stats
    total_cert = 0.0
    total_tau = 0.0
    saw_aux = False

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        try:
            logits, aux = model(x, return_aux=True)  # type: ignore[misc]
            tau = aux.get("tau", None)
            certified = aux.get("certified", None)
            if tau is not None:
                total_tau += tau.float().sum().item()
                saw_aux = True
            if certified is not None:
                total_cert += certified.float().sum().item()
                saw_aux = True
        except TypeError:
            logits = model(x)

        loss = F.cross_entropy(logits, y)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        n += bsz

    metrics: Dict[str, float] = {}
    if saw_aux and n > 0:
        metrics["cert_rate"] = total_cert / n
        metrics["tau_mean"] = total_tau / n

    return total_loss / max(n, 1), total_acc / max(n, 1), metrics

# ----------------------------
# Modular entrypoint (callable from other code)
# ----------------------------
def run_training(
    *,
    dataset: str,
    data_dir: str,
    cfg: TrainConfig,
    input_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
) -> None:
    """
    Programmatic API:
      - Build dataset loaders + dims via get_dataset(...)
      - Build CRP model + optimizer
      - Train/eval for cfg.epochs
    """
    ds = get_dataset(
        name=dataset,
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
    )

    input_dim_ = input_dim if input_dim is not None else ds.input_dim
    num_classes_ = num_classes if num_classes is not None else ds.num_classes

    crp_cfg = CRPConfig(
        kappa=cfg.kappa,
        c=cfg.c,
        alpha=cfg.alpha,
        eps=cfg.eps,
        t_max=cfg.t_max,
        use_certification=cfg.use_certification,
        margin_factor=cfg.margin_factor,
    )

    model = CRPClassifier(
        input_dim=input_dim_,
        hidden_dim=cfg.hidden_dim,
        num_classes=num_classes_,
        cfg=crp_cfg,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(model, ds.train_loader, opt, cfg.device)
        va_loss, va_acc, va_metrics = eval_one_epoch(model, ds.val_loader, cfg.device)

        extra_tr = ""
        extra_va = ""
        if "cert_rate" in tr_metrics:
            extra_tr = f" | train_cert={tr_metrics['cert_rate']:.3f} | train_tau={tr_metrics['tau_mean']:.2f}"
        if "cert_rate" in va_metrics:
            extra_va = f" | val_cert={va_metrics['cert_rate']:.3f} | val_tau={va_metrics['tau_mean']:.2f}"

        print(
            f"epoch {epoch:03d} | train_loss={tr_loss:.4f}{extra_tr} | "
            f"val_loss={va_loss:.4f} | val_acc={va_acc:.4f}{extra_va}"
        )

    if ds.test_loader is not None:
        te_loss, te_acc, te_metrics = eval_one_epoch(model, ds.test_loader, cfg.device)
        extra_te = ""
        if "cert_rate" in te_metrics:
            extra_te = f" | cert={te_metrics['cert_rate']:.3f} | tau={te_metrics['tau_mean']:.2f}"
        print(f"TEST | loss={te_loss:.4f} | acc={te_acc:.4f}{extra_te}")


# ----------------------------
# CLI wrapper (thin)
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # dataset selection
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data-dir", type=str, default="./data")

    # model size
    parser.add_argument("--hidden-dim", type=int, default=256)

    # CRP config
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--use-certification", action="store_true", default=True)
    parser.add_argument("--no-certification", action="store_false", dest="use_certification")
    parser.add_argument("--margin-factor", type=float, default=2.0)

    # optim / training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    # optional: allow overriding dims for custom datasets later
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        kappa=args.kappa,
        c=args.c,
        alpha=args.alpha,
        eps=args.eps,
        t_max=args.t_max,
        use_certification=args.use_certification,
        margin_factor=args.margin_factor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    run_training(
        dataset=args.dataset,
        data_dir=args.data_dir,
        cfg=cfg,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
