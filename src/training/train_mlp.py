from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.model_mlp import MLPClassifier
from src.data.datasets import get_dataset


@dataclass
class TrainConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 128        
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, opt, device: str) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        n += bsz

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        n += bsz

    return total_loss / max(n, 1), total_acc / max(n, 1)


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
      - Build model + optimizer
      - Train/eval for cfg.epochs

    You can import and call this function from other Python code without argparse.
    """
    ds = get_dataset(
        name=dataset,
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
    )

    # Allow manual override if you’re using a custom loader later
    input_dim_ = input_dim if input_dim is not None else ds.input_dim
    num_classes_ = num_classes if num_classes is not None else ds.num_classes

    model = MLPClassifier(  
        input_dim=input_dim_,
        num_classes=num_classes_,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, ds.train_loader, opt, cfg.device)
        va_loss, va_acc = eval_one_epoch(model, ds.val_loader, cfg.device)
        print(
            f"epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f}"
        )

    if ds.test_loader is not None:
        te_loss, te_acc = eval_one_epoch(model, ds.test_loader, cfg.device)
        print(f"TEST | loss={te_loss:.4f} | acc={te_acc:.4f}")


# ----------------------------
# CLI wrapper (thin)
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # dataset selection (replaces manual input_dim/num_classes when using built-ins)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data-dir", type=str, default="./data")

    # original training/model knobs you wanted to keep
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
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
        num_hidden_layers=args.num_hidden_layers,
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