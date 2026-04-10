# structural-graph-rnn

PyTorch research code for training and comparing MNIST classifiers:

- `mlp`: baseline feedforward MLP
- `crp`: Contractive Recurrent Perceptron (CRP) with certification metrics
- `mlp_adaptive`: MLP with DeepR-style sparse rewiring under a global edge budget
- `crp_adaptive`: CRP with DeepR-style sparse rewiring under a global edge budget

All execution is driven by one CLI entry point:

```powershell
.\.venv\Scripts\python.exe -m src.app.train
```

## Project scope

- Supported dataset is **MNIST only** (`--dataset mnist`).
- Core training and experiment outputs are written under `./runs` by default.

## Table of contents

- [Requirements](#requirements)
- [Install dependencies](#install-dependencies)
- [Quick start](#quick-start)
- [Model variants](#model-variants)
- [Experiment modes](#experiment-modes)
- [CLI reference](#cli-reference)
- [Artifacts and checkpoints](#artifacts-and-checkpoints)
- [Validation checklist](#validation-checklist)
- [Project structure](#project-structure)
- [Development notes](#development-notes)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3
- PyTorch
- torchvision

Run commands from repository root so the `src` package imports resolve.

## Install dependencies

The repository does not include a lockfile. Install `torch` and `torchvision` in your environment first.

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

## Quick start

Show all flags:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --help
```

Run a 1-epoch MLP smoke test:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --epochs 1 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1
```

Expected behavior:

- prints per-epoch train/val metrics
- prints final `TEST` loss/accuracy

## Model variants

### `mlp`

Standard dense MLP classifier (`src/models/mlp`).

### `crp`

Contractive recurrent model (`src/models/crp`) with:

- iterative hidden-state updates up to `--t-max`
- optional earliest-winner certification metrics (enabled by default)
- recurrent normalization mode `plain_inf` or `weighted_inf`
- schematic choices including `base`, `feedforward`, and `random_density`

Example:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model crp --dataset mnist --data-dir ./data --epochs 1 --batch-size 1024 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --schematic base --kappa 1.0 --c 0.95 --t-max 8 --recurrent-norm weighted_inf
```

### `crp_adaptive`

CRP with DeepR-managed sparse matrices (`IH`, `HH`, `HL`) and one global active-edge budget.

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model crp_adaptive --dataset mnist --data-dir ./data --epochs 1 --batch-size 1024 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --schematic base --kappa 1.0 --c 0.95 --t-max 8 --recurrent-norm weighted_inf --k-total 2000
```

### `mlp_adaptive`

MLP with DeepR-managed sparse linear layers and a global edge budget.

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp_adaptive --dataset mnist --data-dir ./data --epochs 1 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --k-total 2000
```

## Experiment modes

### CRP c-sensitivity sweep

Enable with `--run-crp-c-sensitivity`.

Sweep definition:

- `c = 1 - 10^-k` for integer `k` in `[cs-k-min, cs-k-max]`
- `cs-trials` runs per `c`
- `cs-epochs` epochs per run
- CRP uses `schematic=random_density`, `num_hidden_layers=1`

Example:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --dataset mnist --data-dir ./data --artifacts-dir ./runs --device cpu --num-workers 0 --batch-size 2048 --run-crp-c-sensitivity --cs-k-min 1 --cs-k-max 3 --cs-trials 5 --cs-epochs 2 --cs-hidden-dim 128 --cs-hh-density 0.5 --cs-experiment-name cs_demo
```

### Fixed comparison-condition experiment

Enable with `--run-comparison-condition <id>`.

Supported condition IDs:

- `crp_random_sparse`
- `crp_feedforward`
- `crp_adaptive_feedforward_init`
- `crp_adaptive_full_init`
- `mlp_feedforward`
- `mlp_adaptive`

Example:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --dataset mnist --data-dir ./data --artifacts-dir ./runs --device cpu --num-workers 0 --batch-size 2048 --run-comparison-condition mlp_feedforward --cmp-trials 5 --cmp-epochs 2 --cmp-k-total 10000 --cmp-experiment-name cmp_demo
```

## CLI reference

Entry point:

```powershell
.\.venv\Scripts\python.exe -m src.app.train [OPTIONS]
```

### Shared options

- `--dataset` (default: `mnist`)
- `--data-dir` (default: `./data`)
- `--artifacts-dir` (default: `./runs`)
- `--model` one of `mlp`, `crp`, `crp_adaptive`, `mlp_adaptive`
- `--device` (default: auto-selected `cpu`/`cuda`)
- `--lr` (default: `1e-3`)
- `--weight-decay` (default: `0.0`)
- `--epochs` (default: `10`)
- `--batch-size` (default: `128`)
- `--num-workers` (default: `2`)
- `--seed` (default: unset)
- `--num-runs` (default: `1`)
- `--save-state-every` (default: `0` for normal training mode)
- `--save-state-path` (default: unset)
- `--resume-state` (default: unset)
- `--debug-compare-mlp-crp` (default: off)
- `--init-type` (default: `kaiming_uniform`)
- `--activation` (default: `leaky_relu`)
- `--negative-slope` (if unset, falls back to `--alpha`)
- `--input-dim` / `--num-classes` (optional manual overrides)
- `--hidden-dim` and `--num-hidden-layers`

### CRP / CRP-adaptive options

- `--schematic`
- `--random-hh-density`
- `--random-hh-seed`
- `--kappa` (default: `1.0`)
- `--c` (default: `0.999`)
- `--alpha` (default: `0.05`)
- `--eps` (default: `1e-8`)
- `--t-max` (default: `32`)
- `--use-certification` / `--no-certification` (default enabled)
- `--margin-factor` (default: `2.0`)
- `--recurrent-norm` one of `plain_inf`, `weighted_inf` (default: `weighted_inf`)
- `--weighted-inf-iters` (default: `20`)

### DeepR options (`crp_adaptive`, `mlp_adaptive`)

- `--deepr-ih` / `--no-deepr-ih`
- `--deepr-hh` / `--no-deepr-hh`
- `--deepr-hl` / `--no-deepr-hl`
- `--k-total` (global active-edge budget; if omitted, budget derives from `--frac-total`)
- `--frac-total` (default: `1.0`)
- `--full-adjacency-allowed` / `--mask-adjacency-allowed`
- `--deepr-drift-alpha` (default: `1e-4`)
- `--deepr-temperature` (default: `1e-6`)
- `--deepr-debug-checks` (default: off)

### CRP c-sensitivity options

- `--run-crp-c-sensitivity`
- `--cs-k-min`
- `--cs-k-max`
- `--cs-trials`
- `--cs-epochs`
- `--cs-hidden-dim`
- `--cs-hh-density`
- `--cs-base-seed`
- `--cs-experiment-name`

### Comparison-condition options

- `--run-comparison-condition` (choices listed above)
- `--cmp-trials`
- `--cmp-epochs`
- `--cmp-base-seed`
- `--cmp-k-total`
- `--cmp-random-hh-density`
- `--cmp-experiment-name`

### Schematic semantics (`crp`, `crp_adaptive`)

- `base`: `hidden_dim` is total hidden-state size
- `feedforward`: `hidden_dim` is per-layer width, total state size is `hidden_dim * num_hidden_layers`
- `random_density`: recurrent adjacency sampled by `--random-hh-density`

## Artifacts and checkpoints

### Multi-run logs (`--num-runs > 1`)

Written to:

- `runs/logs/multi_run_<model>_<dataset>_<num_runs>runs_<timestamp>.txt`

### Training-session checkpoints

Enable with:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --save-state-every 1 --save-state-path ./runs/checkpoints/session_state.pt
```

Resume with:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --resume-state ./runs/checkpoints/session_state.pt
```

### Experiment outputs

CRP c-sensitivity writes:

- `runs/experiments/<experiment_id>/epoch_metrics.csv`
- `runs/experiments/<experiment_id>/trial_metrics.csv`
- `runs/experiments/<experiment_id>/c_summary.csv`

Comparison-condition writes:

- `runs/experiments/<experiment_id>/epoch_metrics.csv`
- `runs/experiments/<experiment_id>/trial_metrics.csv`
- `runs/experiments/<experiment_id>/epoch_curve_summary.csv`
- `runs/experiments/<experiment_id>/condition_summary.csv`

By default, experiment checkpoints are saved to `runs/experiments/<experiment_id>/state.pt` unless `--save-state-path` is supplied.

### Resume behavior

`--resume-state` supports checkpoints for:

- training sessions (`kind=training_session`)
- CRP c-sensitivity experiments (`kind=crp_c_sensitivity_experiment`)
- comparison-condition experiments (`kind=comparison_condition_experiment`)

If the checkpoint is already complete, resume prints output locations and exits.

## Validation checklist

The commands below were executed successfully on this codebase state (April 10, 2026):

- syntax/import sanity:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe freeform/test_torch.py
.\.venv\Scripts\python.exe -m src.app.train --help
```

- 1-epoch smoke runs for all model families:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1
.\.venv\Scripts\python.exe -m src.app.train --model crp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 1024 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --schematic base --kappa 1.0 --c 0.95 --t-max 8 --recurrent-norm weighted_inf
.\.venv\Scripts\python.exe -m src.app.train --model mlp_adaptive --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --k-total 2000
.\.venv\Scripts\python.exe -m src.app.train --model crp_adaptive --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 1024 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --schematic base --kappa 1.0 --c 0.95 --t-max 8 --recurrent-norm weighted_inf --k-total 2000
```

- multi-run + checkpoint/resume:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --num-runs 2 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 2048 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --save-state-every 1 --save-state-path ./runs/checkpoints/validation_training_session.pt
.\.venv\Scripts\python.exe -m src.app.train --resume-state ./runs/checkpoints/validation_training_session.pt
```

- experiment modes + resume:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --dataset mnist --data-dir ./data --artifacts-dir ./runs --device cpu --num-workers 0 --batch-size 2048 --run-crp-c-sensitivity --cs-k-min 1 --cs-k-max 1 --cs-trials 1 --cs-epochs 1 --cs-hidden-dim 32 --cs-hh-density 0.3 --cs-experiment-name validation_cs --save-state-every 1 --save-state-path ./runs/checkpoints/validation_cs.pt
.\.venv\Scripts\python.exe -m src.app.train --resume-state ./runs/checkpoints/validation_cs.pt
.\.venv\Scripts\python.exe -m src.app.train --dataset mnist --data-dir ./data --artifacts-dir ./runs --device cpu --num-workers 0 --batch-size 2048 --run-comparison-condition mlp_feedforward --cmp-trials 1 --cmp-epochs 1 --cmp-experiment-name validation_cmp --save-state-every 1 --save-state-path ./runs/checkpoints/validation_cmp.pt
.\.venv\Scripts\python.exe -m src.app.train --resume-state ./runs/checkpoints/validation_cmp.pt
```

- all six comparison-condition IDs (minimal 1 trial, 1 epoch) executed successfully:
`crp_random_sparse`, `crp_feedforward`, `crp_adaptive_feedforward_init`, `crp_adaptive_full_init`, `mlp_feedforward`, `mlp_adaptive`.

## Project structure

```text
src/
  app/
    train.py
  core/
    trainer.py
    loops.py
    types.py
    deepR/
      matrices.py
  data/
    datasets.py
    datamodules.py
    transforms.py
  models/
    registry.py
    mlp/
    crp/
    MLPadaptive/
    CRPadaptive/
```

Other files:

- `freeform/test_torch.py`: small device/tensor sanity script
- `notes/main_idea`: research notes (not executable)

## Development notes

- Model `forward(..., return_aux=True)` is expected to return `(logits, aux_dict)`.
- If a model defines `deepr_step_all(...)`, the core train loop calls it after each optimizer step.
- If a model defines weighted recurrent normalization cache updates, trainer updates it once per epoch.

## Troubleshooting

- `ValueError: Unknown dataset name`: only `mnist` is implemented in `src/data/datamodules.py`.
- Very slow CRP runs: reduce `--t-max`, lower `--hidden-dim`, or increase batch size on CPU.
- Adaptive models can show low accuracy on very short runs or very small `--k-total`; increase epochs and/or edge budget.
- Resume ignoring newly supplied model/dataset flags is expected: resume loads configuration from checkpoint.
