# structural-graph-rnn

PyTorch research code for training and comparing four MNIST classifiers:

- `mlp`: baseline feedforward MLP
- `crp`: Contractive Recurrent Perceptron (CRP) with earliest certified winner logic
- `mlp_adaptive`: MLP with DeepR-style sparse rewiring under a global edge budget
- `crp_adaptive`: CRP with DeepR-style sparse rewiring under a global edge budget

All training is driven by one CLI entry point: `python -m src.app.train`.

## Project scope

Current repository state:

- Supported dataset is **MNIST only** (`--dataset mnist`).

## Table of contents

- [Requirements](#requirements)
- [Install dependencies](#install-dependencies)
- [Quick start](#quick-start)
- [Model variants](#model-variants)
- [CLI reference](#cli-reference)
- [Artifacts and checkpoints](#artifacts-and-checkpoints)
- [Project structure](#project-structure)
- [Development notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- Python 3
- PyTorch
- torchvision

The code assumes you run from the repository root so the `src` package layout resolves correctly.

## Install dependencies

This repository does not include a lockfile or pinned dependency manifest. Install `torch` and `torchvision` in your environment before running the CLI.
The commands below use the repository-local Windows virtualenv path (`.\.venv\Scripts\python.exe`); if you use another environment, replace that prefix accordingly.

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

## Quick start

Inspect all available flags:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --help
```

Run a 1-epoch MLP baseline smoke test:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --epochs 1 --batch-size 512 --num-workers 0 --device cpu --hidden-dim 128 --num-hidden-layers 2
```

Expected behavior:

- downloads MNIST automatically if missing
- prints per-epoch train/val metrics
- prints final `TEST` loss/accuracy

## Model variants

### `mlp`

Standard dense MLP classifier (`src/models/mlp`).

### `crp`

Contractive recurrent model (`src/models/crp`) with:

- iterative hidden-state updates up to `--t-max`
- optional early-stop certification logic (enabled by default)
- recurrent normalization mode `plain_inf` or `weighted_inf`

Example:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model crp --dataset mnist --data-dir ./data --epochs 1 --batch-size 512 --num-workers 0 --device cpu --hidden-dim 128 --num-hidden-layers 2 --schematic base --kappa 1.0 --c 0.95 --t-max 16 --recurrent-norm weighted_inf
```

### `crp_adaptive`

CRP with DeepR-managed sparse matrices (`IH`, `HH`, `HL`) and one global active-edge budget.

Example with an explicit sparse budget:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model crp_adaptive --dataset mnist --data-dir ./data --epochs 1 --batch-size 512 --num-workers 0 --device cpu --hidden-dim 128 --num-hidden-layers 2 --schematic base --kappa 1.0 --c 0.95 --t-max 16 --recurrent-norm weighted_inf --k-total 10000
```

### `mlp_adaptive`

MLP with DeepR-managed sparse linear layers and a global edge budget.

Example:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp_adaptive --dataset mnist --data-dir ./data --epochs 1 --batch-size 512 --num-workers 0 --device cpu --hidden-dim 128 --num-hidden-layers 2 --k-total 10000
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
- `--save-state-every` (default: `0`, disabled unless checkpointing requested)
- `--save-state-path` (default: unset)
- `--resume-state` (default: unset)
- `--debug-compare-mlp-crp` (default: off)
- `--init-type` (supported values: `kaiming_uniform`, `linear_default`)
- `--activation` (supported values: `leaky_relu`, `relu`)
- `--negative-slope` (if unset, falls back to `--alpha`)
- `--input-dim` / `--num-classes` (optional manual overrides)
- `--hidden-dim` and `--num-hidden-layers`

### CRP / CRP-adaptive options

- `--schematic` (`base` or `feedforward`)
- `--kappa` (default: `1.0`)
- `--c` (CLI default: `0.999`)
- `--alpha` (default: `0.05`)
- `--eps` (default: `1e-8`)
- `--t-max` (default: `32`)
- `--use-certification` / `--no-certification` (default enabled)
- `--margin-factor` (default: `2.0`)
- `--recurrent-norm` one of `plain_inf`, `weighted_inf` (default: `weighted_inf`)
- `--weighted-inf-iters` (default: `20`)

### DeepR options (`crp_adaptive` and `mlp_adaptive`)

- `--k-total` (global active-edge budget; if omitted, budget is derived from `--frac-total`)
- `--frac-total` (default: `1.0`)
- `--deepr-drift-alpha` (default: `1e-4`)
- `--deepr-temperature` (default: `1e-6`)
- `--deepr-debug-checks` (default: off)

CRP-adaptive-only DeepR options:

- `--deepr-ih` / `--no-deepr-ih`
- `--deepr-hh` / `--no-deepr-hh`
- `--deepr-hl` / `--no-deepr-hl`
- `--full-adjacency-allowed` / `--mask-adjacency-allowed`

### Schematic semantics (`crp`, `crp_adaptive`)

- `base`: `hidden_dim` means total hidden-state size
- `feedforward`: `hidden_dim` means per-layer width, and total hidden-state size is `hidden_dim * num_hidden_layers`

## Artifacts and checkpoints

### Multi-run logs

For `--num-runs > 1`, the trainer writes aggregate logs to:

- `runs/logs/multi_run_<model>_<dataset>_<num_runs>runs_<timestamp>.txt`

Example command:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --num-runs 2 --batch-size 1024 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1
```

### Checkpointing

Enable resumable session checkpoints:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --model mlp --dataset mnist --data-dir ./data --artifacts-dir ./runs --epochs 1 --batch-size 1024 --num-workers 0 --device cpu --hidden-dim 64 --num-hidden-layers 1 --save-state-every 1 --save-state-path ./runs/checkpoints/readme_demo_state.pt
```

Resume from a checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.app.train --resume-state ./runs/checkpoints/readme_demo_state.pt
```

Notes:

- `--resume-state` short-circuits normal CLI config construction and resumes from the serialized `ExperimentConfig` in the checkpoint.
- If a checkpoint is already complete, resume prints summary stats and exits.
- If `--save-state-path` is omitted while checkpointing, a timestamped file is created under `runs/checkpoints/`.

### Adjacency logging caveat

Adaptive models expose adjacency snapshots through `get_adjacency_matrices()`. In multi-run mode, full binary matrices are appended to the text log for each run. For larger hidden sizes, log files can become very large.

## Project structure

```text
src/
  app/
    train.py                 # CLI entry point
  core/
    trainer.py               # session orchestration, checkpointing, multi-run stats
    loops.py                 # train/eval epoch loops
    types.py                 # dataclass configs
    deepR/
      matrices.py            # DeepRMaskedMatrix and StaticMaskedMatrix
  data/
    datasets.py              # MNIST split creation
    datamodules.py           # DataLoader assembly
    transforms.py            # MNIST normalization transform
  models/
    registry.py              # model-id to builder mapping
    mlp/
    crp/
    MLPadaptive/
    CRPadaptive/
```

Other notable files:

- `freeform/test_torch.py`: tiny device/tensor sanity script
- `notes/main_idea`: research notes (not executable code)

## Development notes

- The training loop expects every model to support `forward(..., return_aux=True)` and return `(logits, aux_dict)`.
- If a model defines `deepr_step_all(...)`, the core train loop calls it after each optimizer step.
- For weighted recurrent normalization, trainer calls `update_normalization_cache()` once per epoch when present.

Quick local PyTorch sanity check:

```powershell
.\.venv\Scripts\python.exe freeform/test_torch.py
```

## Troubleshooting

- `ValueError: Unknown dataset name`: only `mnist` is implemented in `src/data/datamodules.py`.
- Very slow CRP runs: reduce `--t-max`, lower `--hidden-dim`, or use larger batch size on CPU.
- Adaptive models with poor sparsity behavior: if `--k-total` is not set, the default `--frac-total=1.0` uses a dense global budget.
- Resume not honoring new CLI model/dataset flags: resume uses checkpoint config by design; only save-path/interval overrides are accepted.

