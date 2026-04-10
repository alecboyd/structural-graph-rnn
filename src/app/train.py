from __future__ import annotations

import argparse
from dataclasses import replace

from src.core.types import (
    ExperimentConfig,
    TrainLoopConfig,
    MLPModelConfig,
    CRPModelConfig,
    CRPAdaptiveModelConfig,
    MLPAdaptiveModelConfig,
)
from src.core.trainer import (
    run_training,
    run_training_multiple,
    resume_training_from_state,
    run_crp_c_sensitivity_experiment,
    run_comparison_condition_experiment,
    comparison_condition_ids,
)
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
    parser.add_argument("--artifacts-dir", type=str, default="./runs")

    # model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "crp", "crp_adaptive", "mlp_adaptive"],
        default="mlp",
    )

    # training / optim (shared)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--save-state-every", type=int, default=0)
    parser.add_argument("--save-state-path", type=str, default=None)
    parser.add_argument("--resume-state", type=str, default=None)
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
    parser.add_argument("--random-hh-density", type=float, default=0.5)
    parser.add_argument("--random-hh-seed", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--use-certification", action="store_true", default=True)
    parser.add_argument("--no-certification", action="store_false", dest="use_certification")
    parser.add_argument("--margin-factor", type=float, default=2.0)
    parser.add_argument(
        "--recurrent-norm",
        type=str,
        choices=["plain_inf", "weighted_inf"],
        default="weighted_inf",
    )
    parser.add_argument("--weighted-inf-iters", type=int, default=20)

    # Adaptive-model / DeepR controls (shared by CRP-adaptive and MLP-adaptive)
    parser.add_argument("--deepr-ih", action="store_true", default=True)
    parser.add_argument("--no-deepr-ih", action="store_false", dest="deepr_ih")
    parser.add_argument("--deepr-hh", action="store_true", default=True)
    parser.add_argument("--no-deepr-hh", action="store_false", dest="deepr_hh")
    parser.add_argument("--deepr-hl", action="store_true", default=True)
    parser.add_argument("--no-deepr-hl", action="store_false", dest="deepr_hl")
    parser.add_argument("--k-total", type=int, default=None)
    parser.add_argument("--frac-total", type=float, default=1.0)
    parser.add_argument("--full-adjacency-allowed", action="store_true", default=True)
    parser.add_argument("--mask-adjacency-allowed", action="store_false", dest="full_adjacency_allowed")
    parser.add_argument("--deepr-drift-alpha", type=float, default=1e-4)
    parser.add_argument("--deepr-temperature", type=float, default=1e-6)
    parser.add_argument("--deepr-debug-checks", action="store_true", default=False)

    # one-command CRP c-sensitivity experiment
    parser.add_argument("--run-crp-c-sensitivity", action="store_true", default=False)
    parser.add_argument("--cs-k-min", type=int, default=1)
    parser.add_argument("--cs-k-max", type=int, default=10)
    parser.add_argument("--cs-trials", type=int, default=25)
    parser.add_argument("--cs-epochs", type=int, default=5)
    parser.add_argument("--cs-hidden-dim", type=int, default=128)
    parser.add_argument("--cs-hh-density", type=float, default=0.5)
    parser.add_argument("--cs-base-seed", type=int, default=12345)
    parser.add_argument("--cs-experiment-name", type=str, default=None)

    # fixed comparison-condition experiments (4 CRP variants + 2 MLP variants)
    parser.add_argument(
        "--run-comparison-condition",
        type=str,
        choices=comparison_condition_ids(),
        default=None,
    )
    parser.add_argument("--cmp-trials", type=int, default=25)
    parser.add_argument("--cmp-epochs", type=int, default=25)
    parser.add_argument("--cmp-base-seed", type=int, default=12345)
    parser.add_argument("--cmp-k-total", type=int, default=10000)
    parser.add_argument("--cmp-random-hh-density", type=float, default=0.5)
    parser.add_argument("--cmp-experiment-name", type=str, default=None)
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
    if args.save_state_every < 0:
        raise ValueError(f"--save-state-every must be >= 0, got {args.save_state_every}.")
    if args.resume_state is not None:
        resume_training_from_state(
            args.resume_state,
            save_state_path=args.save_state_path,
            save_state_every=(args.save_state_every if args.save_state_every > 0 else None),
        )
        return
    if args.num_runs < 1:
        raise ValueError(f"--num-runs must be >= 1, got {args.num_runs}.")

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
        artifacts_dir=args.artifacts_dir,
        train=train_cfg,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        init_type=args.init_type,
        activation=args.activation,
        negative_slope=neg_slope,
    )

    if args.run_crp_c_sensitivity:
        exp_for_cs = replace(
            exp,
            crp=CRPModelConfig(
                hidden_dim=args.cs_hidden_dim,
                schematic="random_density",
                num_hidden_layers=1,
                random_hh_density=args.cs_hh_density,
                random_hh_seed=None,
                kappa=args.kappa,
                c=args.c,
                alpha=args.alpha,
                eps=args.eps,
                t_max=args.t_max,
                use_certification=args.use_certification,
                margin_factor=args.margin_factor,
                recurrent_norm=args.recurrent_norm,
                weighted_inf_iters=args.weighted_inf_iters,
            ),
        )
        run_crp_c_sensitivity_experiment(
            base_cfg=exp_for_cs,
            k_min=args.cs_k_min,
            k_max=args.cs_k_max,
            trials_per_c=args.cs_trials,
            epochs_per_trial=args.cs_epochs,
            hidden_dim=args.cs_hidden_dim,
            hh_density=args.cs_hh_density,
            base_seed=args.cs_base_seed,
            experiment_name=args.cs_experiment_name,
            save_state_every=(args.save_state_every if args.save_state_every > 0 else 1),
            save_state_path=args.save_state_path,
        )
        return

    if args.run_comparison_condition is not None:
        exp_for_cmp = replace(
            exp,
            crp=CRPModelConfig(
                hidden_dim=256,
                schematic="base",
                num_hidden_layers=1,
                random_hh_density=args.cmp_random_hh_density,
                random_hh_seed=None,
                kappa=args.kappa,
                c=args.c,
                alpha=args.alpha,
                eps=args.eps,
                t_max=args.t_max,
                use_certification=args.use_certification,
                margin_factor=args.margin_factor,
                recurrent_norm=args.recurrent_norm,
                weighted_inf_iters=args.weighted_inf_iters,
            ),
            crp_adaptive=CRPAdaptiveModelConfig(
                hidden_dim=256,
                schematic="base",
                num_hidden_layers=1,
                random_hh_density=args.cmp_random_hh_density,
                random_hh_seed=None,
                kappa=args.kappa,
                c=args.c,
                alpha=args.alpha,
                eps=args.eps,
                t_max=args.t_max,
                use_certification=args.use_certification,
                margin_factor=args.margin_factor,
                recurrent_norm=args.recurrent_norm,
                weighted_inf_iters=args.weighted_inf_iters,
                deepr_ih=args.deepr_ih,
                deepr_hh=args.deepr_hh,
                deepr_hl=args.deepr_hl,
                K_total=args.cmp_k_total,
                frac_total=1.0,
                full_adjacency_allowed=args.full_adjacency_allowed,
                deepr_drift_alpha=args.deepr_drift_alpha,
                deepr_temperature=args.deepr_temperature,
                deepr_debug_checks=args.deepr_debug_checks,
            ),
            mlp_adaptive=MLPAdaptiveModelConfig(
                hidden_dim=128,
                num_hidden_layers=2,
                K_total=args.cmp_k_total,
                frac_total=1.0,
                deepr_drift_alpha=args.deepr_drift_alpha,
                deepr_temperature=args.deepr_temperature,
                deepr_debug_checks=args.deepr_debug_checks,
            ),
        )
        run_comparison_condition_experiment(
            base_cfg=exp_for_cmp,
            condition_id=args.run_comparison_condition,
            trials=args.cmp_trials,
            epochs=args.cmp_epochs,
            base_seed=args.cmp_base_seed,
            k_total=args.cmp_k_total,
            random_hh_density=args.cmp_random_hh_density,
            experiment_name=args.cmp_experiment_name,
            save_state_every=(args.save_state_every if args.save_state_every > 0 else 1),
            save_state_path=args.save_state_path,
        )
        return

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
                random_hh_density=args.random_hh_density,
                random_hh_seed=args.random_hh_seed,
                kappa=args.kappa,
                c=args.c,
                alpha=args.alpha,
                eps=args.eps,
                t_max=args.t_max,
                use_certification=args.use_certification,
                margin_factor=args.margin_factor,
                recurrent_norm=args.recurrent_norm,
                weighted_inf_iters=args.weighted_inf_iters,
            ),
        )
    if args.model == "crp_adaptive":
        exp = replace(
            exp,
            crp_adaptive=CRPAdaptiveModelConfig(
                hidden_dim=args.hidden_dim,
                schematic=args.schematic,
                num_hidden_layers=args.num_hidden_layers,
                random_hh_density=args.random_hh_density,
                random_hh_seed=args.random_hh_seed,
                kappa=args.kappa,
                c=args.c,
                alpha=args.alpha,
                eps=args.eps,
                t_max=args.t_max,
                use_certification=args.use_certification,
                margin_factor=args.margin_factor,
                recurrent_norm=args.recurrent_norm,
                weighted_inf_iters=args.weighted_inf_iters,
                deepr_ih=args.deepr_ih,
                deepr_hh=args.deepr_hh,
                deepr_hl=args.deepr_hl,
                K_total=args.k_total,
                frac_total=args.frac_total,
                full_adjacency_allowed=args.full_adjacency_allowed,
                deepr_drift_alpha=args.deepr_drift_alpha,
                deepr_temperature=args.deepr_temperature,
                deepr_debug_checks=args.deepr_debug_checks,
            ),
        )
    if args.model == "mlp_adaptive":
        exp = replace(
            exp,
            mlp_adaptive=MLPAdaptiveModelConfig(
                hidden_dim=args.hidden_dim,
                num_hidden_layers=args.num_hidden_layers,
                K_total=args.k_total,
                frac_total=args.frac_total,
                deepr_drift_alpha=args.deepr_drift_alpha,
                deepr_temperature=args.deepr_temperature,
                deepr_debug_checks=args.deepr_debug_checks,
            ),
        )

    if args.num_runs == 1:
        run_training(
            exp,
            debug_compare=args.debug_compare_mlp_crp,
            save_state_every=args.save_state_every,
            save_state_path=args.save_state_path,
        )
    else:
        run_training_multiple(
            exp,
            num_runs=args.num_runs,
            debug_compare=args.debug_compare_mlp_crp,
            save_state_every=args.save_state_every,
            save_state_path=args.save_state_path,
        )


if __name__ == "__main__":
    main()
