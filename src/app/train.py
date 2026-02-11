from __future__ import annotations

import argparse
from dataclasses import replace

from src.core.types import (
    ExperimentConfig,
    TrainLoopConfig,
    MLPModelConfig,
    CRPModelConfig,
)
from src.core.trainer import run_training
from src.core.device import default_device


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for model and training configuration.

    The parser exposes shared optimization flags plus model-specific flags.
    Values are later mapped into ``ExperimentConfig`` and nested config
    dataclasses in ``src.core.types``.
    """
    parser = argparse.ArgumentParser()

    # dataset selection
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data-dir", type=str, default="./data")

    # model selection
    parser.add_argument("--model", type=str, choices=["mlp", "crp"], default="mlp")

    # training / optim (shared)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug-compare-mlp-crp", action="store_true", default=False)
    parser.add_argument("--init-type", type=str, default="kaiming_uniform")
    parser.add_argument("--activation", type=str, default="leaky_relu")
    parser.add_argument("--negative-slope", type=float, default=None)

    # optional: allow overriding dims for custom datasets later
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)

    # MLP-only
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)

    # CRP-only
    parser.add_argument("--schematic", type=str, default="base")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--use-certification", action="store_true", default=True)
    parser.add_argument("--no-certification", action="store_false", dest="use_certification")
    parser.add_argument("--margin-factor", type=float, default=2.0)

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Parse CLI arguments, build experiment config, and run training.

    Inputs:
    - argv: Optional explicit argument list for tests or embedding.

    Side effects:
    - Downloads datasets as needed.
    - Trains a model and prints epoch metrics to stdout.

    Interactions:
    - Uses ``src.core.trainer.run_training`` as the orchestration entrypoint.
    """
    args = build_arg_parser().parse_args(argv)

    train_cfg = TrainLoopConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )

    neg_slope = args.negative_slope if args.negative_slope is not None else args.alpha

    exp = ExperimentConfig(
        model_id=args.model,
        dataset=args.dataset,
        data_dir=args.data_dir,
        train=train_cfg,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        init_type=args.init_type,
        activation=args.activation,
        negative_slope=neg_slope,
    )

    if args.model == "mlp":
        exp = replace(
            exp,
            mlp=MLPModelConfig(hidden_dim=args.hidden_dim, num_hidden_layers=args.num_hidden_layers),
        )
        
    if args.model == "crp":
        exp = replace(
            exp,
            crp=CRPModelConfig(
                hidden_dim=args.hidden_dim,
                schematic=args.schematic,
                num_hidden_layers=args.num_hidden_layers,
                kappa=args.kappa,
                c=args.c,
                alpha=args.alpha,
                eps=args.eps,
                t_max=args.t_max,
                use_certification=args.use_certification,
                margin_factor=args.margin_factor,
            ),
        )

    run_training(exp, debug_compare=args.debug_compare_mlp_crp)


if __name__ == "__main__":
    main()
