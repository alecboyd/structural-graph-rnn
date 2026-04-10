"""High-level training orchestration that wires data, models, and loop APIs."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev, pvariance, stdev
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from src.data.datamodules import get_dataset
from .loops import train_one_epoch, eval_one_epoch
from .types import (
    ExperimentConfig,
    MLPModelConfig,
    CRPModelConfig,
    CRPAdaptiveModelConfig,
    MLPAdaptiveModelConfig,
)
from src.models.registry import build_model
from src.models.mlp.factory import build_mlp, MLPSpec
from src.models.crp.factory import build_crp, CRPSpec
from src.models.crp.model import CRPConfig


@dataclass
class TrainingRunResult:
    """Final metrics and emitted log lines for one training run."""

    final_train_loss: float
    final_val_loss: float
    final_val_acc: float
    final_train_metrics: dict[str, float]
    final_val_metrics: dict[str, float]
    test_loss: Optional[float]
    test_acc: Optional[float]
    test_metrics: dict[str, float]
    epoch_records: list[dict[str, Any]]
    adjacency_lines: list[str]
    log_lines: list[str]


CHECKPOINT_VERSION = 1
CS_EXPERIMENT_CHECKPOINT_VERSION = 1
COMPARISON_CONDITION_CHECKPOINT_VERSION = 1
COMPARISON_CONDITION_IDS: tuple[str, ...] = (
    "crp_random_sparse",
    "crp_feedforward",
    "crp_adaptive_feedforward_init",
    "crp_adaptive_full_init",
    "mlp_feedforward",
    "mlp_adaptive",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def comparison_condition_ids() -> list[str]:
    """Return stable CLI-facing identifiers for comparison-condition experiments."""
    return list(COMPARISON_CONDITION_IDS)


def _serialize_result(result: TrainingRunResult) -> dict[str, Any]:
    return {
        "final_train_loss": float(result.final_train_loss),
        "final_val_loss": float(result.final_val_loss),
        "final_val_acc": float(result.final_val_acc),
        "final_train_metrics": dict(result.final_train_metrics),
        "final_val_metrics": dict(result.final_val_metrics),
        "test_loss": None if result.test_loss is None else float(result.test_loss),
        "test_acc": None if result.test_acc is None else float(result.test_acc),
        "test_metrics": dict(result.test_metrics),
        "epoch_records": [dict(r) for r in result.epoch_records],
        "adjacency_lines": list(result.adjacency_lines),
        "log_lines": list(result.log_lines),
    }


def _deserialize_result(payload: dict[str, Any]) -> TrainingRunResult:
    return TrainingRunResult(
        final_train_loss=float(payload["final_train_loss"]),
        final_val_loss=float(payload["final_val_loss"]),
        final_val_acc=float(payload["final_val_acc"]),
        final_train_metrics=dict(payload.get("final_train_metrics", {})),
        final_val_metrics=dict(payload.get("final_val_metrics", {})),
        test_loss=None if payload.get("test_loss") is None else float(payload["test_loss"]),
        test_acc=None if payload.get("test_acc") is None else float(payload["test_acc"]),
        test_metrics=dict(payload.get("test_metrics", {})),
        epoch_records=[dict(r) for r in payload.get("epoch_records", [])],
        adjacency_lines=list(payload.get("adjacency_lines", [])),
        log_lines=list(payload.get("log_lines", [])),
    )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    if "torch_cpu" in state and state["torch_cpu"] is not None:
        torch.set_rng_state(state["torch_cpu"])
    if "python_random" in state and state["python_random"] is not None:
        random.setstate(state["python_random"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
        except Exception:
            pass


def _optimizer_to_device(opt: torch.optim.Optimizer, device: str) -> None:
    """Move optimizer state tensors onto the target device after loading."""
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")
    return obj


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _artifacts_root(cfg: ExperimentConfig) -> Path:
    """Return root directory for training artifacts (checkpoints, logs)."""
    root = getattr(cfg, "artifacts_dir", "./runs")
    return Path(root)


def _checkpoints_dir(cfg: ExperimentConfig) -> Path:
    return _artifacts_root(cfg) / "checkpoints"


def _logs_dir(cfg: ExperimentConfig) -> Path:
    return _artifacts_root(cfg) / "logs"


def _default_session_checkpoint_path(cfg: ExperimentConfig, *, num_runs: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = _checkpoints_dir(cfg)
    return subdir / f"train_state_{cfg.model_id}_{cfg.dataset}_{num_runs}runs_{ts}.pt"


def _show_extra(metrics: dict[str, float], prefix: str) -> str:
    """
    Format optional aux-derived metrics for console logging.

    Inputs:
    - metrics: Dict returned by loop functions.
    - prefix: Label prefix such as ``train`` or ``val``.
    """
    chunks: list[str] = []
    if "cert_rate" in metrics:
        chunks.append(f"{prefix}_cert={metrics['cert_rate']:.3f}")
    if "tau_mean" in metrics:
        chunks.append(f"{prefix}_tau={metrics['tau_mean']:.2f}")
    if "recurrent_scale_mean" in metrics:
        chunks.append(f"{prefix}_scale={metrics['recurrent_scale_mean']:.4f}")
    if "recurrent_shrink_rate" in metrics:
        chunks.append(f"{prefix}_shrink={metrics['recurrent_shrink_rate']:.3f}")
    if not chunks:
        return ""
    return " | " + " | ".join(chunks)

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _debug_compare_mlp_crp(
    cfg: ExperimentConfig,
    *,
    input_dim: int,
    num_classes: int,
    device: str,
    loader,
    log_fn: Callable[[str], None] = print,
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
            random_hh_density=crp_cfg.random_hh_density,
            random_hh_seed=crp_cfg.random_hh_seed,
            cfg=CRPConfig(
                kappa=crp_cfg.kappa,
                c=crp_cfg.c,
                eps=crp_cfg.eps,
                t_max=crp_cfg.t_max,
                use_certification=crp_cfg.use_certification,
                margin_factor=crp_cfg.margin_factor,
                recurrent_norm=crp_cfg.recurrent_norm,
                weighted_inf_iters=crp_cfg.weighted_inf_iters,
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

    log_fn("DEBUG_COMPARE: parameter counts")
    log_fn(f"DEBUG_COMPARE: MLP params = {_count_params(mlp)}")
    log_fn(f"DEBUG_COMPARE: CRP params = {_count_params(crp)}")
    log_fn("DEBUG_COMPARE: logits diff on one batch")
    log_fn(f"DEBUG_COMPARE: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

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

    log_fn("DEBUG_COMPARE: grad norms (first few parameters)")
    for name, g in first_grad_norms(mlp):
        log_fn(f"DEBUG_COMPARE: MLP {name} grad_norm={g:.6f}")
    for name, g in first_grad_norms(crp):
        log_fn(f"DEBUG_COMPARE: CRP {name} grad_norm={g:.6f}")


def _run_training_once(
    cfg: ExperimentConfig,
    *,
    debug_compare: bool = False,
    log_fn: Callable[[str], None] = print,
    epoch_end_callback: Optional[Callable[[], None]] = None,
    checkpoint_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    start_epoch: int = 0,
    resume_model_state: Optional[dict[str, Any]] = None,
    resume_opt_state: Optional[dict[str, Any]] = None,
    resume_run_log: Optional[list[str]] = None,
    resume_epoch_records: Optional[list[dict[str, Any]]] = None,
) -> TrainingRunResult:
    """Execute one full train/val/test cycle and return final metrics."""
    train_cfg = cfg.train
    run_log: list[str] = list(resume_run_log or [])
    epoch_records: list[dict[str, Any]] = [dict(r) for r in (resume_epoch_records or [])]

    def emit(line: str) -> None:
        run_log.append(line)
        log_fn(line)

    if not run_log:
        emit(
            f"config: init_type={cfg.init_type} activation={cfg.activation} "
            f"negative_slope={cfg.negative_slope:.4f}"
        )

    if train_cfg.seed is not None and resume_model_state is None and start_epoch == 0:
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

    if debug_compare and resume_model_state is None and start_epoch == 0:
        _debug_compare_mlp_crp(
            cfg,
            input_dim=input_dim,
            num_classes=num_classes,
            device=train_cfg.device,
            loader=ds.train_loader,
            log_fn=emit,
        )

    model = build_model(cfg, input_dim=input_dim, num_classes=num_classes).to(train_cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if resume_model_state is not None:
        model.load_state_dict(resume_model_state)
    if resume_opt_state is not None:
        opt.load_state_dict(resume_opt_state)
        _optimizer_to_device(opt, train_cfg.device)

    tr_loss = math.nan
    tr_metrics: dict[str, float] = {}
    va_loss = math.nan
    va_acc = math.nan
    va_metrics: dict[str, float] = {}

    for epoch in range(start_epoch + 1, train_cfg.epochs + 1):
        if hasattr(model, "update_normalization_cache"):
            model.update_normalization_cache()
        tr_loss, tr_metrics = train_one_epoch(model, ds.train_loader, opt, train_cfg.device)
        va_loss, va_acc, va_metrics = eval_one_epoch(model, ds.val_loader, train_cfg.device)

        extra_tr = _show_extra(tr_metrics, "train")
        extra_va = _show_extra(va_metrics, "val")

        emit(
            f"epoch {epoch:03d} | train_loss={tr_loss:.4f}{extra_tr} | "
            f"val_loss={va_loss:.4f} | val_acc={va_acc:.4f}{extra_va}"
        )
        epoch_records.append(
            {
                "epoch": int(epoch),
                "train_loss": float(tr_loss),
                "train_metrics": dict(tr_metrics),
                "val_loss": float(va_loss),
                "val_acc": float(va_acc),
                "val_metrics": dict(va_metrics),
            }
        )
        if epoch_end_callback is not None:
            epoch_end_callback()
        if checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "run_log_lines": list(run_log),
                    "epoch_records": [dict(r) for r in epoch_records],
                    "last_train_loss": float(tr_loss),
                    "last_train_metrics": dict(tr_metrics),
                    "last_val_loss": float(va_loss),
                    "last_val_acc": float(va_acc),
                    "last_val_metrics": dict(va_metrics),
                }
            )

    te_loss: Optional[float] = None
    te_acc: Optional[float] = None
    te_metrics: dict[str, float] = {}
    if ds.test_loader is not None:
        te_loss, te_acc, te_metrics = eval_one_epoch(model, ds.test_loader, train_cfg.device)
        extra_te = _show_extra(te_metrics, "test")
        emit(f"TEST | loss={te_loss:.4f} | acc={te_acc:.4f}{extra_te}")
    adjacency_lines = _format_model_adjacency_lines(model)

    return TrainingRunResult(
        final_train_loss=tr_loss,
        final_val_loss=va_loss,
        final_val_acc=va_acc,
        final_train_metrics=tr_metrics,
        final_val_metrics=va_metrics,
        test_loss=te_loss,
        test_acc=te_acc,
        test_metrics=te_metrics,
        epoch_records=epoch_records,
        adjacency_lines=adjacency_lines,
        log_lines=run_log,
    )


def _progress_bar(completed: int, total: int, *, width: int = 36) -> str:
    if total <= 0:
        return "[------------------------------------] 100.00% (0/0 epochs)"
    ratio = completed / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:6.2f}% ({completed}/{total} epochs)"


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "var": pvariance(values),
        "min": sorted_vals[0],
        "q25": _quantile(sorted_vals, 0.25),
        "median": _quantile(sorted_vals, 0.50),
        "q75": _quantile(sorted_vals, 0.75),
        "max": sorted_vals[-1],
    }


def _flatten_result_metrics(result: TrainingRunResult) -> dict[str, float]:
    flat: dict[str, float] = {
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
        "final_val_acc": result.final_val_acc,
    }
    if result.test_loss is not None:
        flat["test_loss"] = result.test_loss
    if result.test_acc is not None:
        flat["test_acc"] = result.test_acc
    for k, v in result.final_train_metrics.items():
        flat[f"final_train_{k}"] = v
    for k, v in result.final_val_metrics.items():
        flat[f"final_val_{k}"] = v
    for k, v in result.test_metrics.items():
        flat[f"test_{k}"] = v
    return flat


def _aggregate_metric_stats(results: list[TrainingRunResult]) -> dict[str, dict[str, float]]:
    by_metric: dict[str, list[float]] = {}
    for result in results:
        for metric_name, value in _flatten_result_metrics(result).items():
            by_metric.setdefault(metric_name, []).append(float(value))
    return {metric_name: _stats(values) for metric_name, values in sorted(by_metric.items())}


def _format_stats_lines(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = ["AGGREGATE METRICS (across runs):"]
    for metric_name, stats in agg.items():
        lines.append(
            f"{metric_name}: "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} var={stats['var']:.6f} "
            f"min={stats['min']:.6f} q25={stats['q25']:.6f} median={stats['median']:.6f} "
            f"q75={stats['q75']:.6f} max={stats['max']:.6f}"
        )
    return lines


def _format_model_adjacency_lines(model: torch.nn.Module) -> list[str]:
    """
    Format model adjacency snapshots as text lines for run-end logging.

    Expected model API:
    - ``get_adjacency_matrices() -> dict[str, Tensor]`` where each tensor is 2D.
    """
    if not hasattr(model, "get_adjacency_matrices"):
        return []

    try:
        mats = model.get_adjacency_matrices()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return [f"ADJACENCY_CAPTURE_ERROR: {exc!r}"]

    if not isinstance(mats, dict) or not mats:
        return []

    lines: list[str] = ["ADJACENCY SNAPSHOT (run end):"]
    for name, mat in mats.items():
        if not isinstance(mat, torch.Tensor) or mat.dim() != 2:
            lines.append(f"[{name}] invalid adjacency payload; expected 2D tensor")
            continue

        a = mat.detach().to("cpu").to(dtype=torch.int32)
        rows, cols = int(a.shape[0]), int(a.shape[1])
        active_edges = int(a.sum().item())
        lines.append(f"[{name}] shape=({rows}, {cols}) active_edges={active_edges}")
        for r in range(rows):
            row = " ".join(str(int(v)) for v in a[r].tolist())
            lines.append(row)
        lines.append("")
    return lines


def _run_training_session(
    cfg: ExperimentConfig,
    *,
    num_runs: int,
    debug_compare: bool = False,
    checkpoint_path: Optional[Path] = None,
    save_state_every: Optional[int] = None,
    resume_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Run a (possibly multi-run) training session with optional resumable checkpoints.

    Checkpoint semantics:
    - One checkpoint file per session, overwritten atomically after each epoch.
    - Stores model/optimizer state for the current run, partial analytics/logs,
      completed-run summaries, and aggregate stats on completion.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}.")

    total_epochs = num_runs * cfg.train.epochs
    show_progress_bar = (num_runs > 1)

    started_at = _now_iso()
    completed_epochs = 0
    combined_log_lines: list[str] = []
    results: list[TrainingRunResult] = []
    current_run_state: Optional[dict[str, Any]] = None
    final_agg: Optional[dict[str, dict[str, float]]] = None
    final_summary_lines: list[str] = []
    final_log_path: Optional[str] = None

    if resume_state is not None:
        if resume_state.get("kind") != "training_session":
            raise ValueError("Unsupported checkpoint kind (expected 'training_session').")
        if int(resume_state.get("version", -1)) != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {resume_state.get('version')}; "
                f"expected {CHECKPOINT_VERSION}."
            )

        completed_epochs = int(resume_state.get("completed_epochs", 0))
        combined_log_lines = list(resume_state.get("combined_log_lines", []))
        results = [
            _deserialize_result(payload)
            for payload in resume_state.get("completed_results", [])
        ]
        current_run_state = resume_state.get("current_run")
        final_agg = resume_state.get("aggregate_stats")
        final_summary_lines = list(resume_state.get("summary_lines", []))
        final_log_path = resume_state.get("final_log_path")
        started_at = str(resume_state.get("created_at", started_at))
        _restore_rng_state(resume_state.get("rng_state"))
        if save_state_every is None:
            loaded_every = int(resume_state.get("save_state_every", 1))
            save_state_every = max(1, loaded_every)

        if bool(resume_state.get("done", False)):
            if show_progress_bar:
                print(_progress_bar(completed_epochs, total_epochs))
            if final_summary_lines:
                for line in final_summary_lines:
                    print(line)
            if final_log_path:
                print(f"Saved multi-run log to {final_log_path}")
            print("Checkpoint is already complete.")
            return
    else:
        header_title = "MULTI-RUN TRAINING" if num_runs > 1 else "TRAINING SESSION"
        combined_log_lines = [
            header_title,
            f"timestamp={started_at}",
            f"num_runs={num_runs}",
            f"config={cfg}",
            "",
        ]

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if save_state_every is None:
            save_state_every = 1
        if save_state_every < 1:
            raise ValueError(f"save_state_every must be >= 1 when checkpointing is enabled, got {save_state_every}.")
        print(f"Checkpointing enabled: {checkpoint_path}")
        print(f"Checkpoint interval: every {save_state_every} epoch(s)")

    def save_session_checkpoint(*, current_run: Optional[dict[str, Any]], done: bool) -> None:
        if checkpoint_path is None:
            return
        state = {
            "kind": "training_session",
            "version": CHECKPOINT_VERSION,
            "created_at": started_at,
            "updated_at": _now_iso(),
            "cfg": cfg,
            "debug_compare": bool(debug_compare),
            "num_runs": int(num_runs),
            "save_state_every": (None if checkpoint_path is None else int(save_state_every)),
            "total_epochs": int(total_epochs),
            "completed_epochs": int(completed_epochs),
            "completed_results": [_serialize_result(r) for r in results],
            "combined_log_lines": list(combined_log_lines),
            "current_run": current_run,
            "done": bool(done),
            "aggregate_stats": final_agg,
            "summary_lines": list(final_summary_lines),
            "final_log_path": final_log_path,
            "rng_state": _capture_rng_state(),
        }
        _atomic_torch_save(state, checkpoint_path)

    if show_progress_bar:
        print(_progress_bar(completed_epochs, total_epochs), end="", flush=True)

    next_run_idx = len(results) + 1
    if current_run_state is not None:
        next_run_idx = int(current_run_state.get("run_idx", next_run_idx))

    for run_idx in range(next_run_idx, num_runs + 1):
        resuming_this_run = (
            current_run_state is not None and int(current_run_state.get("run_idx", -1)) == run_idx
        )

        if resuming_this_run:
            run_lines = list(current_run_state.get("run_log_lines", []))
            start_epoch = int(current_run_state.get("epoch_completed", 0))
            resume_model_state = current_run_state.get("model_state_dict")
            resume_opt_state = current_run_state.get("optimizer_state_dict")
            resume_epoch_records = [
                dict(r) for r in current_run_state.get("epoch_records", [])
            ]
        else:
            run_lines = [f"RUN {run_idx}/{num_runs}"] if num_runs > 1 else []
            start_epoch = 0
            resume_model_state = None
            resume_opt_state = None
            resume_epoch_records = []

        def collect(line: str) -> None:
            run_lines.append(line)
            if num_runs == 1:
                print(line)

        def on_epoch_end() -> None:
            nonlocal completed_epochs
            completed_epochs += 1
            if show_progress_bar:
                print(f"\r{_progress_bar(completed_epochs, total_epochs)}", end="", flush=True)

        def on_epoch_checkpoint(payload: dict[str, Any]) -> None:
            should_save = (
                checkpoint_path is not None
                and (
                    int(payload["epoch"]) % int(save_state_every) == 0
                    or int(payload["epoch"]) == int(cfg.train.epochs)
                )
            )
            if not should_save:
                return
            current_payload = {
                "run_idx": int(run_idx),
                "epoch_completed": int(payload["epoch"]),
                "run_log_lines": list(payload["run_log_lines"]),
                "epoch_records": [dict(r) for r in payload["epoch_records"]],
                "model_state_dict": payload["model_state_dict"],
                "optimizer_state_dict": payload["optimizer_state_dict"],
                "last_train_loss": float(payload["last_train_loss"]),
                "last_train_metrics": dict(payload["last_train_metrics"]),
                "last_val_loss": float(payload["last_val_loss"]),
                "last_val_acc": float(payload["last_val_acc"]),
                "last_val_metrics": dict(payload["last_val_metrics"]),
            }
            save_session_checkpoint(current_run=current_payload, done=False)

        result = _run_training_once(
            cfg,
            debug_compare=debug_compare,
            log_fn=collect,
            epoch_end_callback=on_epoch_end,
            checkpoint_callback=on_epoch_checkpoint,
            start_epoch=start_epoch,
            resume_model_state=resume_model_state,
            resume_opt_state=resume_opt_state,
            resume_run_log=run_lines,
            resume_epoch_records=resume_epoch_records,
        )

        results.append(result)
        if num_runs > 1:
            combined_log_lines.extend(result.log_lines)
            if result.adjacency_lines:
                combined_log_lines.extend(result.adjacency_lines)
            combined_log_lines.append("")
        else:
            combined_log_lines = list(result.log_lines)

        current_run_state = None
        save_session_checkpoint(current_run=None, done=False)

    if show_progress_bar:
        print()

    if num_runs > 1:
        final_agg = _aggregate_metric_stats(results)
        final_summary_lines = _format_stats_lines(final_agg)
        for line in final_summary_lines:
            print(line)
        combined_log_lines.extend(final_summary_lines)

        log_dir = _logs_dir(cfg)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"multi_run_{cfg.model_id}_{cfg.dataset}_{num_runs}runs_{ts}.txt"
        log_path.write_text("\n".join(combined_log_lines) + "\n", encoding="utf-8")
        final_log_path = str(log_path)
        print(f"Saved multi-run log to {log_path}")
    elif results:
        # Keep single-run console behavior unchanged; still store final metrics in checkpoint state.
        final_agg = _aggregate_metric_stats(results)
        final_summary_lines = _format_stats_lines(final_agg)

    save_session_checkpoint(current_run=None, done=True)


CS_EPOCH_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "run_index",
    "k",
    "c_value",
    "trial_index",
    "run_seed",
    "mask_seed",
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "train_cert_rate",
    "train_tau_mean",
    "train_recurrent_scale_mean",
    "train_recurrent_shrink_rate",
    "train_recurrent_norm_m_mean",
    "train_rho_mean",
    "val_cert_rate",
    "val_tau_mean",
    "val_recurrent_scale_mean",
    "val_recurrent_shrink_rate",
    "val_recurrent_norm_m_mean",
    "val_rho_mean",
    "hh_density_target",
    "hh_active_edges",
    "hh_total_edges",
    "hh_density_realized",
]


CS_TRIAL_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "run_index",
    "k",
    "c_value",
    "trial_index",
    "run_seed",
    "mask_seed",
    "runtime_seconds",
    "epochs_completed",
    "final_train_loss",
    "final_val_loss",
    "final_val_acc",
    "test_loss",
    "test_acc",
    "final_train_cert_rate",
    "final_train_tau_mean",
    "final_train_recurrent_scale_mean",
    "final_train_recurrent_shrink_rate",
    "final_train_recurrent_norm_m_mean",
    "final_val_cert_rate",
    "final_val_tau_mean",
    "final_val_recurrent_scale_mean",
    "final_val_recurrent_shrink_rate",
    "final_val_recurrent_norm_m_mean",
    "test_cert_rate",
    "test_tau_mean",
    "test_recurrent_scale_mean",
    "test_recurrent_shrink_rate",
    "test_recurrent_norm_m_mean",
    "mean_epoch_train_recurrent_scale",
    "mean_epoch_train_recurrent_shrink_rate",
    "mean_epoch_val_recurrent_scale",
    "mean_epoch_val_recurrent_shrink_rate",
    "hh_density_target",
    "hh_active_edges",
    "hh_total_edges",
    "hh_density_realized",
]


CS_SUMMARY_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "k",
    "c_value",
    "completed_trials",
    "expected_trials",
    "val_acc_mean",
    "val_acc_std",
    "val_acc_se",
    "val_acc_ci95_low",
    "val_acc_ci95_high",
    "val_acc_min",
    "val_acc_max",
    "test_acc_mean",
    "test_acc_std",
    "test_acc_se",
    "test_acc_ci95_low",
    "test_acc_ci95_high",
    "test_acc_min",
    "test_acc_max",
    "final_val_recurrent_scale_mean",
    "final_val_recurrent_scale_std",
    "final_val_recurrent_shrink_rate_mean",
    "final_val_recurrent_shrink_rate_std",
    "mean_epoch_val_recurrent_scale_mean",
    "mean_epoch_val_recurrent_scale_std",
    "mean_epoch_val_recurrent_shrink_rate_mean",
    "mean_epoch_val_recurrent_shrink_rate_std",
]


def _default_cs_checkpoint_path(cfg: ExperimentConfig, experiment_id: str) -> Path:
    return _artifacts_root(cfg) / "experiments" / experiment_id / "state.pt"


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def _write_csv_table(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {k: _csv_cell(row.get(k, "")) for k in fieldnames}
            writer.writerow(payload)


def _safe_metric(metrics: dict[str, float], key: str) -> float:
    if key not in metrics:
        return math.nan
    try:
        return float(metrics[key])
    except Exception:
        return math.nan


def _mean_epoch_metric(result: TrainingRunResult, split: str, key: str) -> float:
    values: list[float] = []
    metric_key = f"{split}_metrics"
    for rec in result.epoch_records:
        m = rec.get(metric_key, {})
        if not isinstance(m, dict) or key not in m:
            continue
        try:
            v = float(m[key])
        except Exception:
            continue
        if math.isfinite(v):
            values.append(v)
    if not values:
        return math.nan
    return float(mean(values))


def _finite_values(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _stats_with_ci95(values: list[float]) -> dict[str, float]:
    clean = _finite_values(values)
    if not clean:
        return {
            "n": 0.0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    n = len(clean)
    m = float(mean(clean))
    sd = float(stdev(clean)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else math.nan
    ci = 1.96 * se if math.isfinite(se) else math.nan
    return {
        "n": float(n),
        "mean": m,
        "std": sd,
        "se": se,
        "ci95_low": (m - ci) if math.isfinite(ci) else math.nan,
        "ci95_high": (m + ci) if math.isfinite(ci) else math.nan,
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def _build_cs_run_plan(*, k_min: int, k_max: int, trials_per_c: int, base_seed: int) -> list[dict[str, Any]]:
    planner = random.Random(int(base_seed))
    run_plan: list[dict[str, Any]] = []
    for k in range(int(k_min), int(k_max) + 1):
        c_value = 1.0 - (10.0 ** (-k))
        for trial_idx in range(1, int(trials_per_c) + 1):
            run_plan.append(
                {
                    "run_index": len(run_plan) + 1,
                    "k": int(k),
                    "c_value": float(c_value),
                    "trial_index": int(trial_idx),
                    "run_seed": int(planner.randrange(0, 2**31 - 1)),
                    "mask_seed": int(planner.randrange(0, 2**31 - 1)),
                }
            )
    return run_plan


def _ordered_ck_values(run_plan: list[dict[str, Any]]) -> list[tuple[int, float]]:
    by_k: dict[int, float] = {}
    for item in run_plan:
        k = int(item["k"])
        by_k[k] = float(item["c_value"])
    return [(k, by_k[k]) for k in sorted(by_k.keys())]


def _build_cs_epoch_row(
    *,
    experiment_id: str,
    run_meta: dict[str, Any],
    record: dict[str, Any],
    hh_density: float,
    hh_active_edges: int,
    hh_total_edges: int,
) -> dict[str, Any]:
    train_metrics = dict(record.get("train_metrics", {}))
    val_metrics = dict(record.get("val_metrics", {}))
    return {
        "timestamp": _now_iso(),
        "experiment_id": experiment_id,
        "run_index": int(run_meta["run_index"]),
        "k": int(run_meta["k"]),
        "c_value": float(run_meta["c_value"]),
        "trial_index": int(run_meta["trial_index"]),
        "run_seed": int(run_meta["run_seed"]),
        "mask_seed": int(run_meta["mask_seed"]),
        "epoch": int(record["epoch"]),
        "train_loss": float(record["train_loss"]),
        "val_loss": float(record["val_loss"]),
        "val_acc": float(record["val_acc"]),
        "train_cert_rate": _safe_metric(train_metrics, "cert_rate"),
        "train_tau_mean": _safe_metric(train_metrics, "tau_mean"),
        "train_recurrent_scale_mean": _safe_metric(train_metrics, "recurrent_scale_mean"),
        "train_recurrent_shrink_rate": _safe_metric(train_metrics, "recurrent_shrink_rate"),
        "train_recurrent_norm_m_mean": _safe_metric(train_metrics, "recurrent_norm_m_mean"),
        "train_rho_mean": _safe_metric(train_metrics, "rho_mean"),
        "val_cert_rate": _safe_metric(val_metrics, "cert_rate"),
        "val_tau_mean": _safe_metric(val_metrics, "tau_mean"),
        "val_recurrent_scale_mean": _safe_metric(val_metrics, "recurrent_scale_mean"),
        "val_recurrent_shrink_rate": _safe_metric(val_metrics, "recurrent_shrink_rate"),
        "val_recurrent_norm_m_mean": _safe_metric(val_metrics, "recurrent_norm_m_mean"),
        "val_rho_mean": _safe_metric(val_metrics, "rho_mean"),
        "hh_density_target": float(hh_density),
        "hh_active_edges": int(hh_active_edges),
        "hh_total_edges": int(hh_total_edges),
        "hh_density_realized": (float(hh_active_edges) / float(hh_total_edges)) if hh_total_edges > 0 else math.nan,
    }


def _build_cs_trial_row(
    *,
    experiment_id: str,
    run_meta: dict[str, Any],
    result: TrainingRunResult,
    runtime_seconds: float,
    hh_density: float,
    hh_active_edges: int,
    hh_total_edges: int,
) -> dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "experiment_id": experiment_id,
        "run_index": int(run_meta["run_index"]),
        "k": int(run_meta["k"]),
        "c_value": float(run_meta["c_value"]),
        "trial_index": int(run_meta["trial_index"]),
        "run_seed": int(run_meta["run_seed"]),
        "mask_seed": int(run_meta["mask_seed"]),
        "runtime_seconds": float(runtime_seconds),
        "epochs_completed": int(len(result.epoch_records)),
        "final_train_loss": float(result.final_train_loss),
        "final_val_loss": float(result.final_val_loss),
        "final_val_acc": float(result.final_val_acc),
        "test_loss": (math.nan if result.test_loss is None else float(result.test_loss)),
        "test_acc": (math.nan if result.test_acc is None else float(result.test_acc)),
        "final_train_cert_rate": _safe_metric(result.final_train_metrics, "cert_rate"),
        "final_train_tau_mean": _safe_metric(result.final_train_metrics, "tau_mean"),
        "final_train_recurrent_scale_mean": _safe_metric(result.final_train_metrics, "recurrent_scale_mean"),
        "final_train_recurrent_shrink_rate": _safe_metric(result.final_train_metrics, "recurrent_shrink_rate"),
        "final_train_recurrent_norm_m_mean": _safe_metric(result.final_train_metrics, "recurrent_norm_m_mean"),
        "final_val_cert_rate": _safe_metric(result.final_val_metrics, "cert_rate"),
        "final_val_tau_mean": _safe_metric(result.final_val_metrics, "tau_mean"),
        "final_val_recurrent_scale_mean": _safe_metric(result.final_val_metrics, "recurrent_scale_mean"),
        "final_val_recurrent_shrink_rate": _safe_metric(result.final_val_metrics, "recurrent_shrink_rate"),
        "final_val_recurrent_norm_m_mean": _safe_metric(result.final_val_metrics, "recurrent_norm_m_mean"),
        "test_cert_rate": _safe_metric(result.test_metrics, "cert_rate"),
        "test_tau_mean": _safe_metric(result.test_metrics, "tau_mean"),
        "test_recurrent_scale_mean": _safe_metric(result.test_metrics, "recurrent_scale_mean"),
        "test_recurrent_shrink_rate": _safe_metric(result.test_metrics, "recurrent_shrink_rate"),
        "test_recurrent_norm_m_mean": _safe_metric(result.test_metrics, "recurrent_norm_m_mean"),
        "mean_epoch_train_recurrent_scale": _mean_epoch_metric(result, "train", "recurrent_scale_mean"),
        "mean_epoch_train_recurrent_shrink_rate": _mean_epoch_metric(result, "train", "recurrent_shrink_rate"),
        "mean_epoch_val_recurrent_scale": _mean_epoch_metric(result, "val", "recurrent_scale_mean"),
        "mean_epoch_val_recurrent_shrink_rate": _mean_epoch_metric(result, "val", "recurrent_shrink_rate"),
        "hh_density_target": float(hh_density),
        "hh_active_edges": int(hh_active_edges),
        "hh_total_edges": int(hh_total_edges),
        "hh_density_realized": (float(hh_active_edges) / float(hh_total_edges)) if hh_total_edges > 0 else math.nan,
    }


def _build_cs_summary_rows(
    *,
    trial_rows: list[dict[str, Any]],
    run_plan: list[dict[str, Any]],
    trials_per_c: int,
    experiment_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k, c_value in _ordered_ck_values(run_plan):
        subset = [r for r in trial_rows if int(r["k"]) == int(k)]
        val_stats = _stats_with_ci95([float(r["final_val_acc"]) for r in subset])
        test_stats = _stats_with_ci95([float(r["test_acc"]) for r in subset])
        scale_stats = _stats_with_ci95([float(r["final_val_recurrent_scale_mean"]) for r in subset])
        shrink_stats = _stats_with_ci95([float(r["final_val_recurrent_shrink_rate"]) for r in subset])
        epoch_scale_stats = _stats_with_ci95([float(r["mean_epoch_val_recurrent_scale"]) for r in subset])
        epoch_shrink_stats = _stats_with_ci95([float(r["mean_epoch_val_recurrent_shrink_rate"]) for r in subset])
        rows.append(
            {
                "timestamp": _now_iso(),
                "experiment_id": experiment_id,
                "k": int(k),
                "c_value": float(c_value),
                "completed_trials": int(val_stats["n"]),
                "expected_trials": int(trials_per_c),
                "val_acc_mean": val_stats["mean"],
                "val_acc_std": val_stats["std"],
                "val_acc_se": val_stats["se"],
                "val_acc_ci95_low": val_stats["ci95_low"],
                "val_acc_ci95_high": val_stats["ci95_high"],
                "val_acc_min": val_stats["min"],
                "val_acc_max": val_stats["max"],
                "test_acc_mean": test_stats["mean"],
                "test_acc_std": test_stats["std"],
                "test_acc_se": test_stats["se"],
                "test_acc_ci95_low": test_stats["ci95_low"],
                "test_acc_ci95_high": test_stats["ci95_high"],
                "test_acc_min": test_stats["min"],
                "test_acc_max": test_stats["max"],
                "final_val_recurrent_scale_mean": scale_stats["mean"],
                "final_val_recurrent_scale_std": scale_stats["std"],
                "final_val_recurrent_shrink_rate_mean": shrink_stats["mean"],
                "final_val_recurrent_shrink_rate_std": shrink_stats["std"],
                "mean_epoch_val_recurrent_scale_mean": epoch_scale_stats["mean"],
                "mean_epoch_val_recurrent_scale_std": epoch_scale_stats["std"],
                "mean_epoch_val_recurrent_shrink_rate_mean": epoch_shrink_stats["mean"],
                "mean_epoch_val_recurrent_shrink_rate_std": epoch_shrink_stats["std"],
            }
        )
    return rows


def run_crp_c_sensitivity_experiment(
    *,
    base_cfg: ExperimentConfig,
    k_min: int = 1,
    k_max: int = 10,
    trials_per_c: int = 25,
    epochs_per_trial: int = 5,
    hidden_dim: int = 128,
    hh_density: float = 0.5,
    base_seed: int = 12345,
    experiment_name: Optional[str] = None,
    save_state_every: Optional[int] = 1,
    save_state_path: Optional[str] = None,
    resume_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Run or resume the full CRP c-sensitivity sweep with CSV analytics and checkpointing.

    Sweep definition:
    - c in {1 - 10^-k | k in [k_min, k_max]}
    - trials_per_c independent runs per c
    - epochs_per_trial epochs per run
    - CRP random-density schematic:
      MIH=1, MHL=1, MH sampled at hh_density and fixed per run.
    """
    if resume_state is None:
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}.")
        if k_max < k_min:
            raise ValueError(f"k_max must be >= k_min, got k_min={k_min}, k_max={k_max}.")
        if trials_per_c < 1:
            raise ValueError(f"trials_per_c must be >= 1, got {trials_per_c}.")
        if epochs_per_trial < 1:
            raise ValueError(f"epochs_per_trial must be >= 1, got {epochs_per_trial}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        if not (0.0 <= float(hh_density) <= 1.0):
            raise ValueError(f"hh_density must be in [0, 1], got {hh_density}.")
        if save_state_every is None or save_state_every < 1:
            raise ValueError(f"save_state_every must be >= 1, got {save_state_every}.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (experiment_name or "crp_c_sensitivity").strip().replace(" ", "_")
        experiment_id = f"{slug}_{ts}"

        train_cfg = replace(base_cfg.train, epochs=int(epochs_per_trial), seed=None)
        crp_cfg = replace(
            (base_cfg.crp or CRPModelConfig()),
            hidden_dim=int(hidden_dim),
            schematic="random_density",
            num_hidden_layers=1,
            random_hh_density=float(hh_density),
            random_hh_seed=None,
        )
        cfg = replace(
            base_cfg,
            model_id="crp",
            dataset="mnist",
            train=train_cfg,
            mlp=None,
            crp=crp_cfg,
            crp_adaptive=None,
            mlp_adaptive=None,
            input_dim=None,
            num_classes=None,
        )

        run_plan = _build_cs_run_plan(
            k_min=int(k_min),
            k_max=int(k_max),
            trials_per_c=int(trials_per_c),
            base_seed=int(base_seed),
        )
        total_runs = len(run_plan)
        total_epochs = total_runs * int(epochs_per_trial)
        started_at = _now_iso()

        experiment_dir = _artifacts_root(cfg) / "experiments" / experiment_id
        checkpoint_path = (
            Path(save_state_path)
            if save_state_path is not None
            else _default_cs_checkpoint_path(cfg, experiment_id)
        )
        epoch_csv_path = experiment_dir / "epoch_metrics.csv"
        trial_csv_path = experiment_dir / "trial_metrics.csv"
        summary_csv_path = experiment_dir / "c_summary.csv"

        epoch_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        completed_epochs = 0
        current_run_state: Optional[dict[str, Any]] = None
        next_plan_index = 0
    else:
        if resume_state.get("kind") != "crp_c_sensitivity_experiment":
            raise ValueError("Unsupported checkpoint kind for CRP c-sensitivity resume.")
        if int(resume_state.get("version", -1)) != CS_EXPERIMENT_CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported CRP c-sensitivity checkpoint version {resume_state.get('version')}; "
                f"expected {CS_EXPERIMENT_CHECKPOINT_VERSION}."
            )
        cfg = resume_state.get("cfg")
        if not isinstance(cfg, ExperimentConfig):
            raise ValueError("Checkpoint does not contain a valid ExperimentConfig.")

        experiment_id = str(resume_state["experiment_id"])
        started_at = str(resume_state["created_at"])
        run_plan = [dict(item) for item in resume_state.get("run_plan", [])]
        if not run_plan:
            raise ValueError("Checkpoint run_plan is empty.")

        total_runs = int(resume_state.get("total_runs", len(run_plan)))
        total_epochs = int(resume_state.get("total_epochs", total_runs * int(cfg.train.epochs)))
        trials_per_c = int(resume_state.get("trials_per_c", trials_per_c))
        hidden_dim = int(
            resume_state.get(
                "hidden_dim",
                (cfg.crp.hidden_dim if cfg.crp is not None else hidden_dim),
            )
        )
        hh_density = float(resume_state.get("hh_density", hh_density))

        epoch_rows = [dict(r) for r in resume_state.get("epoch_rows", [])]
        trial_rows = [dict(r) for r in resume_state.get("trial_rows", [])]
        summary_rows = [dict(r) for r in resume_state.get("summary_rows", [])]

        completed_epochs = int(resume_state.get("completed_epochs", 0))
        next_plan_index = int(resume_state.get("next_plan_index", 0))
        current_run_state = resume_state.get("current_run")

        epoch_csv_path = Path(str(resume_state["epoch_csv_path"]))
        trial_csv_path = Path(str(resume_state["trial_csv_path"]))
        summary_csv_path = Path(str(resume_state["summary_csv_path"]))
        checkpoint_path = (
            Path(save_state_path)
            if save_state_path is not None
            else Path(str(resume_state["checkpoint_path"]))
        )

        if save_state_every is None:
            save_state_every = int(resume_state.get("save_state_every", 1))
        if save_state_every < 1:
            raise ValueError(f"save_state_every must be >= 1, got {save_state_every}.")

        _restore_rng_state(resume_state.get("rng_state"))
        if bool(resume_state.get("done", False)):
            _write_csv_table(epoch_csv_path, CS_EPOCH_FIELDNAMES, epoch_rows)
            _write_csv_table(trial_csv_path, CS_TRIAL_FIELDNAMES, trial_rows)
            _write_csv_table(summary_csv_path, CS_SUMMARY_FIELDNAMES, summary_rows)
            print(f"CRP c-sensitivity checkpoint already complete: {checkpoint_path}")
            print(f"Epoch CSV:   {epoch_csv_path}")
            print(f"Trial CSV:   {trial_csv_path}")
            print(f"Summary CSV: {summary_csv_path}")
            return

    hh_total_edges = int(hidden_dim) * int(hidden_dim)
    hh_active_edges = int(round(float(hh_density) * float(hh_total_edges)))
    hh_active_edges = max(0, min(hh_active_edges, hh_total_edges))

    def persist_csv_outputs() -> None:
        _write_csv_table(epoch_csv_path, CS_EPOCH_FIELDNAMES, epoch_rows)
        _write_csv_table(trial_csv_path, CS_TRIAL_FIELDNAMES, trial_rows)
        _write_csv_table(summary_csv_path, CS_SUMMARY_FIELDNAMES, summary_rows)

    def save_experiment_checkpoint(*, current_run: Optional[dict[str, Any]], done: bool) -> None:
        state = {
            "kind": "crp_c_sensitivity_experiment",
            "version": CS_EXPERIMENT_CHECKPOINT_VERSION,
            "created_at": started_at,
            "updated_at": _now_iso(),
            "cfg": cfg,
            "experiment_id": experiment_id,
            "checkpoint_path": str(checkpoint_path),
            "epoch_csv_path": str(epoch_csv_path),
            "trial_csv_path": str(trial_csv_path),
            "summary_csv_path": str(summary_csv_path),
            "run_plan": [dict(x) for x in run_plan],
            "total_runs": int(total_runs),
            "total_epochs": int(total_epochs),
            "completed_epochs": int(completed_epochs),
            "next_plan_index": int(next_plan_index),
            "trials_per_c": int(trials_per_c),
            "hidden_dim": int(hidden_dim),
            "hh_density": float(hh_density),
            "save_state_every": int(save_state_every),
            "epoch_rows": [dict(r) for r in epoch_rows],
            "trial_rows": [dict(r) for r in trial_rows],
            "summary_rows": [dict(r) for r in summary_rows],
            "current_run": current_run,
            "done": bool(done),
            "rng_state": _capture_rng_state(),
        }
        _atomic_torch_save(state, checkpoint_path)

    print(f"CRP c-sensitivity experiment: {experiment_id}")
    print(f"Output checkpoint: {checkpoint_path}")
    print(f"Epoch CSV:   {epoch_csv_path}")
    print(f"Trial CSV:   {trial_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(
        f"Grid: {len(_ordered_ck_values(run_plan))} c-values x {trials_per_c} trials "
        f"x {cfg.train.epochs} epochs = {total_epochs} epochs"
    )

    persist_csv_outputs()
    save_experiment_checkpoint(current_run=current_run_state, done=False)

    print(_progress_bar(completed_epochs, total_epochs), end="", flush=True)

    plan_index = int(next_plan_index)
    if current_run_state is not None:
        plan_index = int(current_run_state.get("plan_index", plan_index))

    while plan_index < len(run_plan):
        run_meta = dict(run_plan[plan_index])
        resuming_this_run = (
            current_run_state is not None
            and int(current_run_state.get("plan_index", -1)) == int(plan_index)
        )

        run_cfg = replace(
            cfg,
            train=replace(cfg.train, seed=int(run_meta["run_seed"])),
            crp=replace(
                (cfg.crp or CRPModelConfig()),
                c=float(run_meta["c_value"]),
                random_hh_seed=int(run_meta["mask_seed"]),
            ),
        )

        if resuming_this_run:
            start_epoch = int(current_run_state.get("epoch_completed", 0))
            resume_model_state = current_run_state.get("model_state_dict")
            resume_opt_state = current_run_state.get("optimizer_state_dict")
            resume_run_log = list(current_run_state.get("run_log_lines", []))
            resume_epoch_records = [dict(r) for r in current_run_state.get("epoch_records", [])]
            resume_elapsed_seconds = float(current_run_state.get("elapsed_seconds", 0.0))
        else:
            start_epoch = 0
            resume_model_state = None
            resume_opt_state = None
            resume_run_log = []
            resume_epoch_records = []
            resume_elapsed_seconds = 0.0

        run_started_wall = datetime.now()
        print(
            (
                f"\nrun {int(run_meta['run_index'])}/{total_runs} | "
                f"k={int(run_meta['k'])} c={float(run_meta['c_value']):.10f} | "
                f"trial={int(run_meta['trial_index'])}/{trials_per_c} | "
                f"seed={int(run_meta['run_seed'])} mask_seed={int(run_meta['mask_seed'])}"
            )
        )

        def collect(_line: str) -> None:
            # Keep per-run lines in checkpoint state without flooding stdout.
            return

        def on_epoch_end() -> None:
            nonlocal completed_epochs
            completed_epochs += 1
            print(f"\r{_progress_bar(completed_epochs, total_epochs)}", end="", flush=True)

        def on_epoch_checkpoint(payload: dict[str, Any]) -> None:
            rec = dict(payload["epoch_records"][-1])
            epoch_rows.append(
                _build_cs_epoch_row(
                    experiment_id=experiment_id,
                    run_meta=run_meta,
                    record=rec,
                    hh_density=float(hh_density),
                    hh_active_edges=int(hh_active_edges),
                    hh_total_edges=int(hh_total_edges),
                )
            )
            persist_csv_outputs()

            should_save = (
                int(payload["epoch"]) % int(save_state_every) == 0
                or int(payload["epoch"]) == int(run_cfg.train.epochs)
            )
            if not should_save:
                return
            current_payload = {
                "plan_index": int(plan_index),
                "run_meta": dict(run_meta),
                "epoch_completed": int(payload["epoch"]),
                "elapsed_seconds": float(
                    resume_elapsed_seconds + (datetime.now() - run_started_wall).total_seconds()
                ),
                "run_log_lines": list(payload["run_log_lines"]),
                "epoch_records": [dict(r) for r in payload["epoch_records"]],
                "model_state_dict": payload["model_state_dict"],
                "optimizer_state_dict": payload["optimizer_state_dict"],
            }
            save_experiment_checkpoint(current_run=current_payload, done=False)

        result = _run_training_once(
            run_cfg,
            debug_compare=False,
            log_fn=collect,
            epoch_end_callback=on_epoch_end,
            checkpoint_callback=on_epoch_checkpoint,
            start_epoch=start_epoch,
            resume_model_state=resume_model_state,
            resume_opt_state=resume_opt_state,
            resume_run_log=resume_run_log,
            resume_epoch_records=resume_epoch_records,
        )

        runtime_seconds = resume_elapsed_seconds + (datetime.now() - run_started_wall).total_seconds()
        trial_rows.append(
            _build_cs_trial_row(
                experiment_id=experiment_id,
                run_meta=run_meta,
                result=result,
                runtime_seconds=float(runtime_seconds),
                hh_density=float(hh_density),
                hh_active_edges=int(hh_active_edges),
                hh_total_edges=int(hh_total_edges),
            )
        )
        summary_rows = _build_cs_summary_rows(
            trial_rows=trial_rows,
            run_plan=run_plan,
            trials_per_c=int(trials_per_c),
            experiment_id=experiment_id,
        )
        persist_csv_outputs()

        print(
            (
                f"\ncompleted run {int(run_meta['run_index'])}/{total_runs}: "
                f"val_acc={float(result.final_val_acc):.4f} "
                f"test_acc={(float(result.test_acc) if result.test_acc is not None else math.nan):.4f} "
                f"val_scale={_safe_metric(result.final_val_metrics, 'recurrent_scale_mean'):.6f} "
                f"val_shrink={_safe_metric(result.final_val_metrics, 'recurrent_shrink_rate'):.6f}"
            )
        )

        current_run_state = None
        plan_index += 1
        next_plan_index = plan_index
        save_experiment_checkpoint(current_run=None, done=False)

    print()
    summary_rows = _build_cs_summary_rows(
        trial_rows=trial_rows,
        run_plan=run_plan,
        trials_per_c=int(trials_per_c),
        experiment_id=experiment_id,
    )
    persist_csv_outputs()
    save_experiment_checkpoint(current_run=None, done=True)

    print("CRP c-sensitivity experiment complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch CSV:   {epoch_csv_path}")
    print(f"Trial CSV:   {trial_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")


CMP_EPOCH_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "condition_id",
    "run_index",
    "trial_index",
    "run_seed",
    "mask_seed",
    "model_id",
    "schematic",
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "train_cert_rate",
    "train_tau_mean",
    "train_recurrent_scale_mean",
    "train_recurrent_shrink_rate",
    "train_rho_mean",
    "val_cert_rate",
    "val_tau_mean",
    "val_recurrent_scale_mean",
    "val_recurrent_shrink_rate",
    "val_rho_mean",
    "k_total_target",
    "hh_density_target",
]


CMP_TRIAL_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "condition_id",
    "run_index",
    "trial_index",
    "run_seed",
    "mask_seed",
    "model_id",
    "schematic",
    "runtime_seconds",
    "epochs_completed",
    "final_train_loss",
    "final_val_loss",
    "final_val_acc",
    "test_loss",
    "test_acc",
    "final_train_cert_rate",
    "final_train_tau_mean",
    "final_train_recurrent_scale_mean",
    "final_train_recurrent_shrink_rate",
    "final_val_cert_rate",
    "final_val_tau_mean",
    "final_val_recurrent_scale_mean",
    "final_val_recurrent_shrink_rate",
    "test_cert_rate",
    "test_tau_mean",
    "test_recurrent_scale_mean",
    "test_recurrent_shrink_rate",
    "mean_epoch_train_recurrent_scale",
    "mean_epoch_val_recurrent_scale",
    "k_total_target",
    "hh_density_target",
]


CMP_EPOCH_CURVE_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "condition_id",
    "model_id",
    "schematic",
    "epoch",
    "completed_trials",
    "expected_trials",
    "train_loss_mean",
    "train_loss_std",
    "val_loss_mean",
    "val_loss_std",
    "val_acc_mean",
    "val_acc_std",
    "val_acc_se",
    "val_acc_ci95_low",
    "val_acc_ci95_high",
    "train_recurrent_scale_mean",
    "train_recurrent_scale_std",
    "val_recurrent_scale_mean",
    "val_recurrent_scale_std",
    "train_recurrent_shrink_rate_mean",
    "train_recurrent_shrink_rate_std",
    "val_recurrent_shrink_rate_mean",
    "val_recurrent_shrink_rate_std",
]


CMP_SUMMARY_FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "condition_id",
    "model_id",
    "schematic",
    "completed_trials",
    "expected_trials",
    "val_acc_mean",
    "val_acc_std",
    "val_acc_se",
    "val_acc_ci95_low",
    "val_acc_ci95_high",
    "val_acc_min",
    "val_acc_max",
    "test_acc_mean",
    "test_acc_std",
    "test_acc_se",
    "test_acc_ci95_low",
    "test_acc_ci95_high",
    "test_acc_min",
    "test_acc_max",
    "final_val_recurrent_scale_mean",
    "final_val_recurrent_scale_std",
    "final_val_recurrent_shrink_rate_mean",
    "final_val_recurrent_shrink_rate_std",
    "k_total_target",
    "hh_density_target",
]


def _default_comparison_checkpoint_path(cfg: ExperimentConfig, experiment_id: str) -> Path:
    return _artifacts_root(cfg) / "experiments" / experiment_id / "state.pt"


def _comparison_condition_meta(
    *,
    condition_id: str,
    k_total: int,
    random_hh_density: float,
) -> dict[str, Any]:
    if condition_id == "crp_random_sparse":
        return {
            "model_id": "crp",
            "schematic": "random_density",
            "k_total_target": math.nan,
            "hh_density_target": float(random_hh_density),
        }
    if condition_id == "crp_feedforward":
        return {
            "model_id": "crp",
            "schematic": "feedforward",
            "k_total_target": math.nan,
            "hh_density_target": math.nan,
        }
    if condition_id == "crp_adaptive_feedforward_init":
        return {
            "model_id": "crp_adaptive",
            "schematic": "feedforward",
            "k_total_target": float(k_total),
            "hh_density_target": math.nan,
        }
    if condition_id == "crp_adaptive_full_init":
        return {
            "model_id": "crp_adaptive",
            "schematic": "base",
            "k_total_target": float(k_total),
            "hh_density_target": math.nan,
        }
    if condition_id == "mlp_feedforward":
        return {
            "model_id": "mlp",
            "schematic": "feedforward",
            "k_total_target": math.nan,
            "hh_density_target": math.nan,
        }
    if condition_id == "mlp_adaptive":
        return {
            "model_id": "mlp_adaptive",
            "schematic": "feedforward",
            "k_total_target": float(k_total),
            "hh_density_target": math.nan,
        }
    raise ValueError(
        f"Unknown comparison condition {condition_id!r}. "
        f"Expected one of {list(COMPARISON_CONDITION_IDS)}."
    )


def _build_comparison_run_plan(*, trials: int, base_seed: int) -> list[dict[str, Any]]:
    planner = random.Random(int(base_seed))
    out: list[dict[str, Any]] = []
    for trial_idx in range(1, int(trials) + 1):
        out.append(
            {
                "run_index": trial_idx,
                "trial_index": trial_idx,
                "run_seed": int(planner.randrange(0, 2**31 - 1)),
                "mask_seed": int(planner.randrange(0, 2**31 - 1)),
            }
        )
    return out


def _build_comparison_run_cfg(
    *,
    base_cfg: ExperimentConfig,
    condition_id: str,
    epochs: int,
    run_seed: int,
    mask_seed: int,
    k_total: int,
    random_hh_density: float,
) -> ExperimentConfig:
    if condition_id not in COMPARISON_CONDITION_IDS:
        raise ValueError(
            f"Unknown comparison condition {condition_id!r}. "
            f"Expected one of {list(COMPARISON_CONDITION_IDS)}."
        )

    train_cfg = replace(base_cfg.train, epochs=int(epochs), seed=int(run_seed))
    cfg = replace(
        base_cfg,
        dataset="mnist",
        model_id="mlp",
        train=train_cfg,
        init_type="kaiming_uniform",
        activation="leaky_relu",
        input_dim=None,
        num_classes=None,
        mlp=None,
        crp=None,
        crp_adaptive=None,
        mlp_adaptive=None,
    )

    base_crp = base_cfg.crp or CRPModelConfig()
    base_crp_ad = base_cfg.crp_adaptive or CRPAdaptiveModelConfig()
    base_mlp_ad = base_cfg.mlp_adaptive or MLPAdaptiveModelConfig()

    if condition_id == "crp_random_sparse":
        crp_cfg = replace(
            base_crp,
            hidden_dim=256,
            schematic="random_density",
            num_hidden_layers=1,
            random_hh_density=float(random_hh_density),
            random_hh_seed=int(mask_seed),
        )
        return replace(cfg, model_id="crp", crp=crp_cfg)

    if condition_id == "crp_feedforward":
        crp_cfg = replace(
            base_crp,
            hidden_dim=128,
            schematic="feedforward",
            num_hidden_layers=2,
            random_hh_seed=None,
        )
        return replace(cfg, model_id="crp", crp=crp_cfg)

    if condition_id == "crp_adaptive_feedforward_init":
        ad_cfg = replace(
            base_crp_ad,
            hidden_dim=128,
            schematic="feedforward",
            num_hidden_layers=2,
            random_hh_seed=None,
            K_total=int(k_total),
            frac_total=1.0,
            full_adjacency_allowed=True,
            deepr_ih=True,
            deepr_hh=True,
            deepr_hl=True,
        )
        return replace(cfg, model_id="crp_adaptive", crp_adaptive=ad_cfg)

    if condition_id == "crp_adaptive_full_init":
        ad_cfg = replace(
            base_crp_ad,
            hidden_dim=256,
            schematic="base",
            num_hidden_layers=1,
            random_hh_seed=None,
            K_total=int(k_total),
            frac_total=1.0,
            full_adjacency_allowed=True,
            deepr_ih=True,
            deepr_hh=True,
            deepr_hl=True,
        )
        return replace(cfg, model_id="crp_adaptive", crp_adaptive=ad_cfg)

    if condition_id == "mlp_feedforward":
        return replace(
            cfg,
            model_id="mlp",
            mlp=MLPModelConfig(hidden_dim=128, num_hidden_layers=2),
        )

    if condition_id == "mlp_adaptive":
        mlp_ad_cfg = replace(
            base_mlp_ad,
            hidden_dim=128,
            num_hidden_layers=2,
            K_total=int(k_total),
            frac_total=1.0,
        )
        return replace(
            cfg,
            model_id="mlp_adaptive",
            mlp_adaptive=mlp_ad_cfg,
        )

    raise AssertionError("unreachable condition branch")


def _build_comparison_epoch_row(
    *,
    experiment_id: str,
    condition_id: str,
    run_meta: dict[str, Any],
    meta: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    train_metrics = dict(record.get("train_metrics", {}))
    val_metrics = dict(record.get("val_metrics", {}))
    return {
        "timestamp": _now_iso(),
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "run_index": int(run_meta["run_index"]),
        "trial_index": int(run_meta["trial_index"]),
        "run_seed": int(run_meta["run_seed"]),
        "mask_seed": int(run_meta["mask_seed"]),
        "model_id": str(meta["model_id"]),
        "schematic": str(meta["schematic"]),
        "epoch": int(record["epoch"]),
        "train_loss": float(record["train_loss"]),
        "val_loss": float(record["val_loss"]),
        "val_acc": float(record["val_acc"]),
        "train_cert_rate": _safe_metric(train_metrics, "cert_rate"),
        "train_tau_mean": _safe_metric(train_metrics, "tau_mean"),
        "train_recurrent_scale_mean": _safe_metric(train_metrics, "recurrent_scale_mean"),
        "train_recurrent_shrink_rate": _safe_metric(train_metrics, "recurrent_shrink_rate"),
        "train_rho_mean": _safe_metric(train_metrics, "rho_mean"),
        "val_cert_rate": _safe_metric(val_metrics, "cert_rate"),
        "val_tau_mean": _safe_metric(val_metrics, "tau_mean"),
        "val_recurrent_scale_mean": _safe_metric(val_metrics, "recurrent_scale_mean"),
        "val_recurrent_shrink_rate": _safe_metric(val_metrics, "recurrent_shrink_rate"),
        "val_rho_mean": _safe_metric(val_metrics, "rho_mean"),
        "k_total_target": float(meta["k_total_target"]),
        "hh_density_target": float(meta["hh_density_target"]),
    }


def _build_comparison_trial_row(
    *,
    experiment_id: str,
    condition_id: str,
    run_meta: dict[str, Any],
    meta: dict[str, Any],
    result: TrainingRunResult,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "timestamp": _now_iso(),
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "run_index": int(run_meta["run_index"]),
        "trial_index": int(run_meta["trial_index"]),
        "run_seed": int(run_meta["run_seed"]),
        "mask_seed": int(run_meta["mask_seed"]),
        "model_id": str(meta["model_id"]),
        "schematic": str(meta["schematic"]),
        "runtime_seconds": float(runtime_seconds),
        "epochs_completed": int(len(result.epoch_records)),
        "final_train_loss": float(result.final_train_loss),
        "final_val_loss": float(result.final_val_loss),
        "final_val_acc": float(result.final_val_acc),
        "test_loss": (math.nan if result.test_loss is None else float(result.test_loss)),
        "test_acc": (math.nan if result.test_acc is None else float(result.test_acc)),
        "final_train_cert_rate": _safe_metric(result.final_train_metrics, "cert_rate"),
        "final_train_tau_mean": _safe_metric(result.final_train_metrics, "tau_mean"),
        "final_train_recurrent_scale_mean": _safe_metric(result.final_train_metrics, "recurrent_scale_mean"),
        "final_train_recurrent_shrink_rate": _safe_metric(result.final_train_metrics, "recurrent_shrink_rate"),
        "final_val_cert_rate": _safe_metric(result.final_val_metrics, "cert_rate"),
        "final_val_tau_mean": _safe_metric(result.final_val_metrics, "tau_mean"),
        "final_val_recurrent_scale_mean": _safe_metric(result.final_val_metrics, "recurrent_scale_mean"),
        "final_val_recurrent_shrink_rate": _safe_metric(result.final_val_metrics, "recurrent_shrink_rate"),
        "test_cert_rate": _safe_metric(result.test_metrics, "cert_rate"),
        "test_tau_mean": _safe_metric(result.test_metrics, "tau_mean"),
        "test_recurrent_scale_mean": _safe_metric(result.test_metrics, "recurrent_scale_mean"),
        "test_recurrent_shrink_rate": _safe_metric(result.test_metrics, "recurrent_shrink_rate"),
        "mean_epoch_train_recurrent_scale": _mean_epoch_metric(result, "train", "recurrent_scale_mean"),
        "mean_epoch_val_recurrent_scale": _mean_epoch_metric(result, "val", "recurrent_scale_mean"),
        "k_total_target": float(meta["k_total_target"]),
        "hh_density_target": float(meta["hh_density_target"]),
    }


def _build_comparison_curve_rows(
    *,
    epoch_rows: list[dict[str, Any]],
    epochs: int,
    expected_trials: int,
    experiment_id: str,
    condition_id: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch_idx in range(1, int(epochs) + 1):
        subset = [r for r in epoch_rows if int(r["epoch"]) == int(epoch_idx)]
        tr_loss = _stats_with_ci95([float(r["train_loss"]) for r in subset])
        va_loss = _stats_with_ci95([float(r["val_loss"]) for r in subset])
        va_acc = _stats_with_ci95([float(r["val_acc"]) for r in subset])
        tr_scale = _stats_with_ci95([float(r["train_recurrent_scale_mean"]) for r in subset])
        va_scale = _stats_with_ci95([float(r["val_recurrent_scale_mean"]) for r in subset])
        tr_shrink = _stats_with_ci95([float(r["train_recurrent_shrink_rate"]) for r in subset])
        va_shrink = _stats_with_ci95([float(r["val_recurrent_shrink_rate"]) for r in subset])
        rows.append(
            {
                "timestamp": _now_iso(),
                "experiment_id": experiment_id,
                "condition_id": condition_id,
                "model_id": str(meta["model_id"]),
                "schematic": str(meta["schematic"]),
                "epoch": int(epoch_idx),
                "completed_trials": int(va_acc["n"]),
                "expected_trials": int(expected_trials),
                "train_loss_mean": tr_loss["mean"],
                "train_loss_std": tr_loss["std"],
                "val_loss_mean": va_loss["mean"],
                "val_loss_std": va_loss["std"],
                "val_acc_mean": va_acc["mean"],
                "val_acc_std": va_acc["std"],
                "val_acc_se": va_acc["se"],
                "val_acc_ci95_low": va_acc["ci95_low"],
                "val_acc_ci95_high": va_acc["ci95_high"],
                "train_recurrent_scale_mean": tr_scale["mean"],
                "train_recurrent_scale_std": tr_scale["std"],
                "val_recurrent_scale_mean": va_scale["mean"],
                "val_recurrent_scale_std": va_scale["std"],
                "train_recurrent_shrink_rate_mean": tr_shrink["mean"],
                "train_recurrent_shrink_rate_std": tr_shrink["std"],
                "val_recurrent_shrink_rate_mean": va_shrink["mean"],
                "val_recurrent_shrink_rate_std": va_shrink["std"],
            }
        )
    return rows


def _build_comparison_summary_rows(
    *,
    trial_rows: list[dict[str, Any]],
    expected_trials: int,
    experiment_id: str,
    condition_id: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    val_stats = _stats_with_ci95([float(r["final_val_acc"]) for r in trial_rows])
    test_stats = _stats_with_ci95([float(r["test_acc"]) for r in trial_rows])
    scale_stats = _stats_with_ci95([float(r["final_val_recurrent_scale_mean"]) for r in trial_rows])
    shrink_stats = _stats_with_ci95([float(r["final_val_recurrent_shrink_rate"]) for r in trial_rows])
    return [
        {
            "timestamp": _now_iso(),
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "model_id": str(meta["model_id"]),
            "schematic": str(meta["schematic"]),
            "completed_trials": int(val_stats["n"]),
            "expected_trials": int(expected_trials),
            "val_acc_mean": val_stats["mean"],
            "val_acc_std": val_stats["std"],
            "val_acc_se": val_stats["se"],
            "val_acc_ci95_low": val_stats["ci95_low"],
            "val_acc_ci95_high": val_stats["ci95_high"],
            "val_acc_min": val_stats["min"],
            "val_acc_max": val_stats["max"],
            "test_acc_mean": test_stats["mean"],
            "test_acc_std": test_stats["std"],
            "test_acc_se": test_stats["se"],
            "test_acc_ci95_low": test_stats["ci95_low"],
            "test_acc_ci95_high": test_stats["ci95_high"],
            "test_acc_min": test_stats["min"],
            "test_acc_max": test_stats["max"],
            "final_val_recurrent_scale_mean": scale_stats["mean"],
            "final_val_recurrent_scale_std": scale_stats["std"],
            "final_val_recurrent_shrink_rate_mean": shrink_stats["mean"],
            "final_val_recurrent_shrink_rate_std": shrink_stats["std"],
            "k_total_target": float(meta["k_total_target"]),
            "hh_density_target": float(meta["hh_density_target"]),
        }
    ]


def run_comparison_condition_experiment(
    *,
    base_cfg: ExperimentConfig,
    condition_id: str,
    trials: int = 25,
    epochs: int = 25,
    base_seed: int = 12345,
    k_total: int = 10_000,
    random_hh_density: float = 0.5,
    experiment_name: Optional[str] = None,
    save_state_every: Optional[int] = 1,
    save_state_path: Optional[str] = None,
    resume_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Run or resume one fixed comparison condition with multi-trial CSV analytics.
    """
    if resume_state is None:
        if condition_id not in COMPARISON_CONDITION_IDS:
            raise ValueError(
                f"Unknown comparison condition {condition_id!r}. "
                f"Expected one of {list(COMPARISON_CONDITION_IDS)}."
            )
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}.")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}.")
        if k_total < 1:
            raise ValueError(f"k_total must be >= 1, got {k_total}.")
        if not (0.0 <= float(random_hh_density) <= 1.0):
            raise ValueError(
                f"random_hh_density must be in [0, 1], got {random_hh_density}."
            )
        if save_state_every is None or save_state_every < 1:
            raise ValueError(f"save_state_every must be >= 1, got {save_state_every}.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = (experiment_name or "comparison_condition").strip().replace(" ", "_")
        experiment_id = f"{prefix}_{condition_id}_{ts}"

        template_cfg = replace(
            base_cfg,
            dataset="mnist",
            init_type="kaiming_uniform",
            activation="leaky_relu",
            input_dim=None,
            num_classes=None,
            train=replace(base_cfg.train, epochs=int(epochs), seed=None),
        )
        run_plan = _build_comparison_run_plan(trials=int(trials), base_seed=int(base_seed))
        total_runs = len(run_plan)
        total_epochs = total_runs * int(epochs)
        started_at = _now_iso()

        meta = _comparison_condition_meta(
            condition_id=condition_id,
            k_total=int(k_total),
            random_hh_density=float(random_hh_density),
        )

        experiment_dir = _artifacts_root(template_cfg) / "experiments" / experiment_id
        checkpoint_path = (
            Path(save_state_path)
            if save_state_path is not None
            else _default_comparison_checkpoint_path(template_cfg, experiment_id)
        )
        epoch_csv_path = experiment_dir / "epoch_metrics.csv"
        trial_csv_path = experiment_dir / "trial_metrics.csv"
        curve_csv_path = experiment_dir / "epoch_curve_summary.csv"
        summary_csv_path = experiment_dir / "condition_summary.csv"

        epoch_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        completed_epochs = 0
        next_plan_index = 0
        current_run_state: Optional[dict[str, Any]] = None
    else:
        if resume_state.get("kind") != "comparison_condition_experiment":
            raise ValueError("Unsupported checkpoint kind for comparison-condition resume.")
        if int(resume_state.get("version", -1)) != COMPARISON_CONDITION_CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported comparison-condition checkpoint version {resume_state.get('version')}; "
                f"expected {COMPARISON_CONDITION_CHECKPOINT_VERSION}."
            )
        template_cfg = resume_state.get("template_cfg")
        if not isinstance(template_cfg, ExperimentConfig):
            raise ValueError("Checkpoint does not contain a valid ExperimentConfig.")

        condition_id = str(resume_state.get("condition_id"))
        if condition_id not in COMPARISON_CONDITION_IDS:
            raise ValueError(
                f"Checkpoint contains unknown condition_id {condition_id!r}."
            )
        trials = int(resume_state.get("trials", trials))
        epochs = int(resume_state.get("epochs", epochs))
        k_total = int(resume_state.get("k_total", k_total))
        random_hh_density = float(resume_state.get("random_hh_density", random_hh_density))

        meta = dict(
            resume_state.get(
                "meta",
                _comparison_condition_meta(
                    condition_id=condition_id,
                    k_total=int(k_total),
                    random_hh_density=float(random_hh_density),
                ),
            )
        )

        experiment_id = str(resume_state["experiment_id"])
        started_at = str(resume_state["created_at"])
        run_plan = [dict(x) for x in resume_state.get("run_plan", [])]
        if not run_plan:
            raise ValueError("Checkpoint run_plan is empty.")
        total_runs = int(resume_state.get("total_runs", len(run_plan)))
        total_epochs = int(resume_state.get("total_epochs", total_runs * int(epochs)))

        epoch_rows = [dict(r) for r in resume_state.get("epoch_rows", [])]
        trial_rows = [dict(r) for r in resume_state.get("trial_rows", [])]
        curve_rows = [dict(r) for r in resume_state.get("curve_rows", [])]
        summary_rows = [dict(r) for r in resume_state.get("summary_rows", [])]

        completed_epochs = int(resume_state.get("completed_epochs", 0))
        next_plan_index = int(resume_state.get("next_plan_index", 0))
        current_run_state = resume_state.get("current_run")

        epoch_csv_path = Path(str(resume_state["epoch_csv_path"]))
        trial_csv_path = Path(str(resume_state["trial_csv_path"]))
        curve_csv_path = Path(str(resume_state["curve_csv_path"]))
        summary_csv_path = Path(str(resume_state["summary_csv_path"]))
        checkpoint_path = (
            Path(save_state_path)
            if save_state_path is not None
            else Path(str(resume_state["checkpoint_path"]))
        )

        if save_state_every is None:
            save_state_every = int(resume_state.get("save_state_every", 1))
        if save_state_every < 1:
            raise ValueError(f"save_state_every must be >= 1, got {save_state_every}.")

        _restore_rng_state(resume_state.get("rng_state"))
        if bool(resume_state.get("done", False)):
            _write_csv_table(epoch_csv_path, CMP_EPOCH_FIELDNAMES, epoch_rows)
            _write_csv_table(trial_csv_path, CMP_TRIAL_FIELDNAMES, trial_rows)
            _write_csv_table(curve_csv_path, CMP_EPOCH_CURVE_FIELDNAMES, curve_rows)
            _write_csv_table(summary_csv_path, CMP_SUMMARY_FIELDNAMES, summary_rows)
            print(f"Comparison-condition checkpoint already complete: {checkpoint_path}")
            print(f"Epoch CSV:   {epoch_csv_path}")
            print(f"Trial CSV:   {trial_csv_path}")
            print(f"Curve CSV:   {curve_csv_path}")
            print(f"Summary CSV: {summary_csv_path}")
            return

    def persist_csv_outputs() -> None:
        _write_csv_table(epoch_csv_path, CMP_EPOCH_FIELDNAMES, epoch_rows)
        _write_csv_table(trial_csv_path, CMP_TRIAL_FIELDNAMES, trial_rows)
        _write_csv_table(curve_csv_path, CMP_EPOCH_CURVE_FIELDNAMES, curve_rows)
        _write_csv_table(summary_csv_path, CMP_SUMMARY_FIELDNAMES, summary_rows)

    def save_checkpoint(*, current_run: Optional[dict[str, Any]], done: bool) -> None:
        state = {
            "kind": "comparison_condition_experiment",
            "version": COMPARISON_CONDITION_CHECKPOINT_VERSION,
            "created_at": started_at,
            "updated_at": _now_iso(),
            "template_cfg": template_cfg,
            "condition_id": condition_id,
            "trials": int(trials),
            "epochs": int(epochs),
            "k_total": int(k_total),
            "random_hh_density": float(random_hh_density),
            "meta": dict(meta),
            "experiment_id": experiment_id,
            "checkpoint_path": str(checkpoint_path),
            "epoch_csv_path": str(epoch_csv_path),
            "trial_csv_path": str(trial_csv_path),
            "curve_csv_path": str(curve_csv_path),
            "summary_csv_path": str(summary_csv_path),
            "run_plan": [dict(x) for x in run_plan],
            "total_runs": int(total_runs),
            "total_epochs": int(total_epochs),
            "completed_epochs": int(completed_epochs),
            "next_plan_index": int(next_plan_index),
            "save_state_every": int(save_state_every),
            "epoch_rows": [dict(r) for r in epoch_rows],
            "trial_rows": [dict(r) for r in trial_rows],
            "curve_rows": [dict(r) for r in curve_rows],
            "summary_rows": [dict(r) for r in summary_rows],
            "current_run": current_run,
            "done": bool(done),
            "rng_state": _capture_rng_state(),
        }
        _atomic_torch_save(state, checkpoint_path)

    print(f"Comparison condition experiment: {experiment_id}")
    print(f"Condition: {condition_id}")
    print(f"Output checkpoint: {checkpoint_path}")
    print(f"Epoch CSV:   {epoch_csv_path}")
    print(f"Trial CSV:   {trial_csv_path}")
    print(f"Curve CSV:   {curve_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Trials={trials}, epochs={epochs}, total_epochs={total_epochs}")

    persist_csv_outputs()
    save_checkpoint(current_run=current_run_state, done=False)
    print(_progress_bar(completed_epochs, total_epochs), end="", flush=True)

    plan_index = int(next_plan_index)
    if current_run_state is not None:
        plan_index = int(current_run_state.get("plan_index", plan_index))

    while plan_index < len(run_plan):
        run_meta = dict(run_plan[plan_index])
        resuming_this_run = (
            current_run_state is not None
            and int(current_run_state.get("plan_index", -1)) == int(plan_index)
        )

        run_cfg = _build_comparison_run_cfg(
            base_cfg=template_cfg,
            condition_id=condition_id,
            epochs=int(epochs),
            run_seed=int(run_meta["run_seed"]),
            mask_seed=int(run_meta["mask_seed"]),
            k_total=int(k_total),
            random_hh_density=float(random_hh_density),
        )

        if resuming_this_run:
            start_epoch = int(current_run_state.get("epoch_completed", 0))
            resume_model_state = current_run_state.get("model_state_dict")
            resume_opt_state = current_run_state.get("optimizer_state_dict")
            resume_run_log = list(current_run_state.get("run_log_lines", []))
            resume_epoch_records = [dict(r) for r in current_run_state.get("epoch_records", [])]
            resume_elapsed_seconds = float(current_run_state.get("elapsed_seconds", 0.0))
        else:
            start_epoch = 0
            resume_model_state = None
            resume_opt_state = None
            resume_run_log = []
            resume_epoch_records = []
            resume_elapsed_seconds = 0.0

        run_started_wall = datetime.now()
        print(
            (
                f"\nrun {int(run_meta['run_index'])}/{total_runs} | "
                f"trial={int(run_meta['trial_index'])}/{trials} | "
                f"seed={int(run_meta['run_seed'])} aux_seed={int(run_meta['mask_seed'])}"
            )
        )

        def collect(_line: str) -> None:
            return

        def on_epoch_end() -> None:
            nonlocal completed_epochs
            completed_epochs += 1
            print(f"\r{_progress_bar(completed_epochs, total_epochs)}", end="", flush=True)

        def on_epoch_checkpoint(payload: dict[str, Any]) -> None:
            rec = dict(payload["epoch_records"][-1])
            epoch_rows.append(
                _build_comparison_epoch_row(
                    experiment_id=experiment_id,
                    condition_id=condition_id,
                    run_meta=run_meta,
                    meta=meta,
                    record=rec,
                )
            )
            persist_csv_outputs()

            should_save = (
                int(payload["epoch"]) % int(save_state_every) == 0
                or int(payload["epoch"]) == int(run_cfg.train.epochs)
            )
            if not should_save:
                return
            current_payload = {
                "plan_index": int(plan_index),
                "run_meta": dict(run_meta),
                "epoch_completed": int(payload["epoch"]),
                "elapsed_seconds": float(
                    resume_elapsed_seconds + (datetime.now() - run_started_wall).total_seconds()
                ),
                "run_log_lines": list(payload["run_log_lines"]),
                "epoch_records": [dict(r) for r in payload["epoch_records"]],
                "model_state_dict": payload["model_state_dict"],
                "optimizer_state_dict": payload["optimizer_state_dict"],
            }
            save_checkpoint(current_run=current_payload, done=False)

        result = _run_training_once(
            run_cfg,
            debug_compare=False,
            log_fn=collect,
            epoch_end_callback=on_epoch_end,
            checkpoint_callback=on_epoch_checkpoint,
            start_epoch=start_epoch,
            resume_model_state=resume_model_state,
            resume_opt_state=resume_opt_state,
            resume_run_log=resume_run_log,
            resume_epoch_records=resume_epoch_records,
        )

        runtime_seconds = resume_elapsed_seconds + (datetime.now() - run_started_wall).total_seconds()
        trial_rows.append(
            _build_comparison_trial_row(
                experiment_id=experiment_id,
                condition_id=condition_id,
                run_meta=run_meta,
                meta=meta,
                result=result,
                runtime_seconds=float(runtime_seconds),
            )
        )
        curve_rows = _build_comparison_curve_rows(
            epoch_rows=epoch_rows,
            epochs=int(epochs),
            expected_trials=int(trials),
            experiment_id=experiment_id,
            condition_id=condition_id,
            meta=meta,
        )
        summary_rows = _build_comparison_summary_rows(
            trial_rows=trial_rows,
            expected_trials=int(trials),
            experiment_id=experiment_id,
            condition_id=condition_id,
            meta=meta,
        )
        persist_csv_outputs()

        print(
            (
                f"\ncompleted run {int(run_meta['run_index'])}/{total_runs}: "
                f"val_acc={float(result.final_val_acc):.4f} "
                f"test_acc={(float(result.test_acc) if result.test_acc is not None else math.nan):.4f}"
            )
        )

        current_run_state = None
        plan_index += 1
        next_plan_index = plan_index
        save_checkpoint(current_run=None, done=False)

    print()
    curve_rows = _build_comparison_curve_rows(
        epoch_rows=epoch_rows,
        epochs=int(epochs),
        expected_trials=int(trials),
        experiment_id=experiment_id,
        condition_id=condition_id,
        meta=meta,
    )
    summary_rows = _build_comparison_summary_rows(
        trial_rows=trial_rows,
        expected_trials=int(trials),
        experiment_id=experiment_id,
        condition_id=condition_id,
        meta=meta,
    )
    persist_csv_outputs()
    save_checkpoint(current_run=None, done=True)

    print("Comparison-condition experiment complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch CSV:   {epoch_csv_path}")
    print(f"Trial CSV:   {trial_csv_path}")
    print(f"Curve CSV:   {curve_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")


def resume_training_from_state(
    state_path: str,
    *,
    save_state_path: Optional[str] = None,
    save_state_every: Optional[int] = None,
) -> None:
    """
    Resume a checkpointed training session from disk.

    By default, continued checkpoints overwrite the same file that was loaded.
    """
    source_path = Path(state_path)
    state = _torch_load_checkpoint(source_path)
    kind = str(state.get("kind", ""))
    target_path = Path(save_state_path) if save_state_path is not None else source_path
    print(f"Resuming from checkpoint: {source_path}")
    if target_path != source_path:
        print(f"Continuing checkpoints will be written to: {target_path}")

    if kind == "training_session":
        cfg = state.get("cfg")
        if not isinstance(cfg, ExperimentConfig):
            raise ValueError("Checkpoint does not contain a valid ExperimentConfig.")
        num_runs = int(state.get("num_runs", 1))
        debug_compare = bool(state.get("debug_compare", False))
        _run_training_session(
            cfg,
            num_runs=num_runs,
            debug_compare=debug_compare,
            checkpoint_path=target_path,
            save_state_every=save_state_every,
            resume_state=state,
        )
        return

    if kind == "crp_c_sensitivity_experiment":
        cfg = state.get("cfg")
        if not isinstance(cfg, ExperimentConfig):
            raise ValueError("Checkpoint does not contain a valid ExperimentConfig.")
        run_crp_c_sensitivity_experiment(
            base_cfg=cfg,
            save_state_every=save_state_every,
            save_state_path=str(target_path),
            resume_state=state,
        )
        return

    if kind == "comparison_condition_experiment":
        cfg = state.get("template_cfg")
        if not isinstance(cfg, ExperimentConfig):
            raise ValueError("Checkpoint does not contain a valid ExperimentConfig.")
        condition_id = str(state.get("condition_id", ""))
        run_comparison_condition_experiment(
            base_cfg=cfg,
            condition_id=condition_id,
            save_state_every=save_state_every,
            save_state_path=str(target_path),
            resume_state=state,
        )
        return

    raise ValueError(
        f"Unsupported checkpoint kind {kind!r}. "
        "Expected 'training_session', 'crp_c_sensitivity_experiment', "
        "or 'comparison_condition_experiment'."
    )


def run_training(
    cfg: ExperimentConfig,
    *,
    debug_compare: bool = False,
    save_state_every: int = 0,
    save_state_path: Optional[str] = None,
) -> None:
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
    if save_state_every < 0:
        raise ValueError(f"save_state_every must be >= 0, got {save_state_every}.")
    if save_state_every > 0 or save_state_path is not None:
        checkpoint_path = (
            Path(save_state_path)
            if save_state_path is not None
            else _default_session_checkpoint_path(cfg, num_runs=1)
        )
        _run_training_session(
            cfg,
            num_runs=1,
            debug_compare=debug_compare,
            checkpoint_path=checkpoint_path,
            save_state_every=(save_state_every if save_state_every > 0 else 1),
        )
        return

    _run_training_once(cfg, debug_compare=debug_compare, log_fn=print)


def run_training_multiple(
    cfg: ExperimentConfig,
    *,
    num_runs: int,
    debug_compare: bool = False,
    save_state_every: int = 0,
    save_state_path: Optional[str] = None,
) -> None:
    """
    Execute multiple independent runs with aggregated progress and summary stats.

    For each run:
    - A new dataset/model/optimizer is instantiated.
    - Model parameters and initial state are freshly initialized.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}.")
    if save_state_every < 0:
        raise ValueError(f"save_state_every must be >= 0, got {save_state_every}.")
    if save_state_every > 0 or save_state_path is not None:
        checkpoint_path = (
            Path(save_state_path)
            if save_state_path is not None
            else _default_session_checkpoint_path(cfg, num_runs=num_runs)
        )
        _run_training_session(
            cfg,
            num_runs=num_runs,
            debug_compare=debug_compare,
            checkpoint_path=checkpoint_path,
            save_state_every=(save_state_every if save_state_every > 0 else 1),
        )
        return
    if num_runs == 1:
        run_training(cfg, debug_compare=debug_compare)
        return

    total_epochs = num_runs * cfg.train.epochs
    completed_epochs = 0
    results: list[TrainingRunResult] = []
    combined_log_lines: list[str] = [
        f"MULTI-RUN TRAINING",
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"num_runs={num_runs}",
        f"config={cfg}",
        "",
    ]

    print(_progress_bar(0, total_epochs), end="", flush=True)
    for run_idx in range(1, num_runs + 1):
        run_lines: list[str] = [f"RUN {run_idx}/{num_runs}"]

        def collect(line: str) -> None:
            run_lines.append(line)

        def on_epoch_end() -> None:
            nonlocal completed_epochs
            completed_epochs += 1
            print(f"\r{_progress_bar(completed_epochs, total_epochs)}", end="", flush=True)

        result = _run_training_once(
            cfg,
            debug_compare=debug_compare,
            log_fn=collect,
            epoch_end_callback=on_epoch_end,
        )
        results.append(result)
        combined_log_lines.extend(result.log_lines)
        if result.adjacency_lines:
            combined_log_lines.extend(result.adjacency_lines)
        combined_log_lines.append("")

    print()

    agg = _aggregate_metric_stats(results)
    summary_lines = _format_stats_lines(agg)
    for line in summary_lines:
        print(line)
    combined_log_lines.extend(summary_lines)

    log_dir = _logs_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"multi_run_{cfg.model_id}_{cfg.dataset}_{num_runs}runs_{ts}.txt"
    log_path.write_text("\n".join(combined_log_lines) + "\n", encoding="utf-8")
    print(f"Saved multi-run log to {log_path}")
