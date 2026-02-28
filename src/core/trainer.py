"""High-level training orchestration that wires data, models, and loop APIs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev, pvariance
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from src.data.datamodules import get_dataset
from .loops import train_one_epoch, eval_one_epoch
from .types import ExperimentConfig, MLPModelConfig, CRPModelConfig
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
    log_lines: list[str]


CHECKPOINT_VERSION = 1


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def _default_session_checkpoint_path(cfg: ExperimentConfig, *, num_runs: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = Path(cfg.data_dir) / "checkpoints"
    return subdir / f"train_state_{cfg.model_id}_{cfg.dataset}_{num_runs}runs_{ts}.pt"


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
            cfg=CRPConfig(
                kappa=crp_cfg.kappa,
                c=crp_cfg.c,
                alpha=cfg.negative_slope if cfg.activation.lower() == "leaky_relu" else 0.0,
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
        if cfg.crp is not None:
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

        data_dir = Path(cfg.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = data_dir / f"multi_run_{cfg.model_id}_{cfg.dataset}_{num_runs}runs_{ts}.txt"
        log_path.write_text("\n".join(combined_log_lines) + "\n", encoding="utf-8")
        final_log_path = str(log_path)
        print(f"Saved multi-run log to {log_path}")
    elif results:
        # Keep single-run console behavior unchanged; still store final metrics in checkpoint state.
        final_agg = _aggregate_metric_stats(results)
        final_summary_lines = _format_stats_lines(final_agg)

    save_session_checkpoint(current_run=None, done=True)


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

    cfg = state.get("cfg")
    if not isinstance(cfg, ExperimentConfig):
        raise ValueError("Checkpoint does not contain a valid ExperimentConfig.")
    num_runs = int(state.get("num_runs", 1))
    debug_compare = bool(state.get("debug_compare", False))

    target_path = Path(save_state_path) if save_state_path is not None else source_path
    print(f"Resuming from checkpoint: {source_path}")
    if target_path != source_path:
        print(f"Continuing checkpoints will be written to: {target_path}")

    _run_training_session(
        cfg,
        num_runs=num_runs,
        debug_compare=debug_compare,
        checkpoint_path=target_path,
        save_state_every=save_state_every,
        resume_state=state,
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
        combined_log_lines.extend(run_lines)
        combined_log_lines.append("")

    print()

    agg = _aggregate_metric_stats(results)
    summary_lines = _format_stats_lines(agg)
    for line in summary_lines:
        print(line)
    combined_log_lines.extend(summary_lines)

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = data_dir / f"multi_run_{cfg.model_id}_{cfg.dataset}_{num_runs}runs_{ts}.txt"
    log_path.write_text("\n".join(combined_log_lines) + "\n", encoding="utf-8")
    print(f"Saved multi-run log to {log_path}")
