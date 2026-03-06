"""Adaptive MLP with global-budget DeepR over feedforward edges.

README:
- Use this model exactly like other classifiers in training loops.
- After ``loss.backward()``, call ``model.deepr_step_all(...)`` to apply
  explicit DeepR updates (the core loop already does this when available).
- The DeepR budget is global across all feedforward linear layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from src.core.deepR import DeepRMaskedMatrix


@dataclass
class MLPAdaptiveConfig:
    """DeepR controls for the adaptive MLP."""

    K_total: Optional[int] = None
    frac_total: float = 1.0
    deepr_drift_alpha: float = 1e-4
    deepr_temperature: float = 1e-6
    deepr_debug_checks: bool = False


class MLPAdaptiveClassifier(nn.Module):
    """
    Feedforward MLP with DeepR-managed sparse linear layers.

    DeepR policy:
    - Allowed edges are the full feedforward adjacency for each layer pair.
    - A single global budget is enforced across all layers.
    """

    MODEL_ID = "mlp_adaptive"

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_hidden_layers: int,
        cfg: Optional[MLPAdaptiveConfig] = None,
        bias: bool = True,
        init_type: str = "kaiming_uniform",
        activation: str = "leaky_relu",
        negative_slope: float = 0.05,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.bias = bool(bias)

        self.cfg = cfg if cfg is not None else MLPAdaptiveConfig()
        self.init_type = init_type
        self.activation = activation
        self.negative_slope = float(negative_slope)

        # Adjacent-layer dimensions for a standard feedforward MLP.
        self.layer_dims = self._build_layer_dims()
        self.num_linear_layers = len(self.layer_dims) - 1

        self.act = self._make_activation()
        self.layer_mats = nn.ModuleList()
        self.biases = nn.ParameterList()

        # Build one DeepR matrix per linear layer using full allowed adjacency.
        for i in range(self.num_linear_layers):
            in_d = self.layer_dims[i]
            out_d = self.layer_dims[i + 1]
            init_dense = self._make_init_dense(in_d, out_d)
            mat = DeepRMaskedMatrix(
                allowed_mask=torch.ones(in_d, out_d, dtype=torch.bool),
                K=0,  # set by global allocator below
                init_dense=init_dense,
                debug_checks=bool(self.cfg.deepr_debug_checks),
            )
            self.layer_mats.append(mat)
            if self.bias:
                self.biases.append(nn.Parameter(torch.zeros(out_d)))

        # One global budget across every layer edge.
        self.global_allowed_count = sum(mat.allowed_count for mat in self.layer_mats)
        self.global_active_target_count = self._resolve_global_budget()
        self._initialize_global_active_set()

        if self.bias:
            self._init_biases()

    def _build_layer_dims(self) -> list[int]:
        if self.num_hidden_layers == 0:
            return [self.input_dim, self.num_classes]
        dims = [self.input_dim]
        dims.extend([self.hidden_dim] * self.num_hidden_layers)
        dims.append(self.num_classes)
        return dims

    def _make_activation(self) -> nn.Module:
        name = self.activation.lower()
        if name == "relu":
            return nn.ReLU(inplace=False)
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)
        raise ValueError(f"Unknown activation: {self.activation!r}")

    def _make_init_dense(self, in_d: int, out_d: int) -> torch.Tensor:
        w = torch.empty(in_d, out_d)
        init = self.init_type.lower()
        if init == "linear_default":
            nn.init.kaiming_uniform_(w.t(), a=math.sqrt(5))
            return w
        if init == "kaiming_uniform":
            a = self.negative_slope if self.activation.lower() == "leaky_relu" else 0.0
            nn.init.kaiming_uniform_(w.t(), a=a)
            return w
        raise ValueError(f"Unknown init_type: {self.init_type!r}")

    def _init_biases(self) -> None:
        init = self.init_type.lower()
        for i, b in enumerate(self.biases):
            if init == "linear_default":
                fan_in = self.layer_dims[i]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(b, -bound, bound)
            elif init == "kaiming_uniform":
                nn.init.zeros_(b)
            else:
                raise ValueError(f"Unknown init_type: {self.init_type!r}")

    def _resolve_global_budget(self) -> int:
        total_allowed = int(self.global_allowed_count)
        if total_allowed <= 0:
            return 0
        if self.cfg.K_total is not None:
            return max(0, min(int(self.cfg.K_total), total_allowed))
        return max(0, min(int(round(float(self.cfg.frac_total) * total_allowed)), total_allowed))

    @torch.no_grad()
    def _initialize_global_active_set(self) -> None:
        for mat in self.layer_mats:
            mat.set_all_inactive()

        target = int(self.global_active_target_count)
        total = int(self.global_allowed_count)
        if target <= 0:
            return
        if target >= total:
            for mat in self.layer_mats:
                mat.set_all_active_from_allowed(theta_mode="seed")
            return

        # Uniformly sample active edges from the union of all layer edge sets.
        sample = torch.randperm(total)[:target]
        offset = 0
        for mat in self.layer_mats:
            count = int(mat.allowed_count)
            sel = (sample >= offset) & (sample < offset + count)
            if sel.any():
                local_pos = sample[sel] - offset
                local_flat = mat.allowed_flat_idx[local_pos]
                mat.activate_flat_indices(local_flat, theta_mode="seed")
            offset += count

        if self.cfg.deepr_debug_checks:
            self._assert_global_budget()

    @torch.no_grad()
    def deepr_step_all(
        self,
        *,
        lr: float,
        drift_alpha: Optional[float] = None,
        T: Optional[float] = None,
    ) -> None:
        """
        Apply one DeepR update step with global budget conservation.

        1) Update/prune active theta in each layer.
        2) Refill globally from dormant edges to keep the global target.
        """
        drift = float(self.cfg.deepr_drift_alpha if drift_alpha is None else drift_alpha)
        temp = float(self.cfg.deepr_temperature if T is None else T)

        if self.global_active_target_count >= self.global_allowed_count:
            for mat in self.layer_mats:
                mat.set_all_active_from_allowed(theta_mode="keep")
                mat.update_active(lr=lr, drift_alpha=drift, T=temp, prune_negative=False)
                mat.clear_grad()
            return

        for mat in self.layer_mats:
            mat.update_active(lr=lr, drift_alpha=drift, T=temp, prune_negative=True)

        current_active = sum(mat.active_count() for mat in self.layer_mats)
        target = int(self.global_active_target_count)

        if current_active > target:
            self._prune_global_excess(current_active - target)
        elif current_active < target:
            self._rewire_global_deficit(target - current_active)

        for mat in self.layer_mats:
            mat.clear_grad()

        if self.cfg.deepr_debug_checks:
            self._assert_global_budget()

    @torch.no_grad()
    def _prune_global_excess(self, excess: int) -> None:
        if excess <= 0:
            return
        active_lists = [mat.active_flat_indices() for mat in self.layer_mats]
        total_active = sum(int(idx.numel()) for idx in active_lists)
        if total_active <= 0:
            return

        take = min(int(excess), total_active)
        sample = torch.randperm(total_active)[:take]
        offset = 0
        for mat, active_idx in zip(self.layer_mats, active_lists):
            count = int(active_idx.numel())
            if count > 0:
                sel = (sample >= offset) & (sample < offset + count)
                if sel.any():
                    local_pos = sample[sel] - offset
                    mat.deactivate_flat_indices(active_idx[local_pos])
            offset += count

    @torch.no_grad()
    def _rewire_global_deficit(self, need: int) -> None:
        if need <= 0:
            return
        dormant_lists = [mat.dormant_flat_indices() for mat in self.layer_mats]
        total_dormant = sum(int(idx.numel()) for idx in dormant_lists)
        if total_dormant <= 0:
            return

        take = min(int(need), total_dormant)
        sample = torch.randperm(total_dormant)[:take]
        offset = 0
        for mat, dormant_idx in zip(self.layer_mats, dormant_lists):
            count = int(dormant_idx.numel())
            if count > 0:
                sel = (sample >= offset) & (sample < offset + count)
                if sel.any():
                    local_pos = sample[sel] - offset
                    mat.activate_flat_indices(dormant_idx[local_pos], theta_mode="zero")
            offset += count

    def deepr_parameters(self) -> Iterator[nn.Parameter]:
        for mat in self.layer_mats:
            yield mat.theta

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Exclude DeepR theta from standard optimizer traversal."""
        theta_ids = {id(p) for p in self.deepr_parameters()}
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            if id(param) in theta_ids:
                continue
            yield name, param

    @torch.no_grad()
    def _assert_global_budget(self) -> None:
        for mat in self.layer_mats:
            mat.assert_invariants()
        active_total = sum(mat.active_count() for mat in self.layer_mats)
        target = int(self.global_active_target_count)
        if active_total != target:
            raise RuntimeError(f"Global budget mismatch: active_total={active_total}, target={target}.")

    def get_adjacency_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Export run-end adjacency matrices for each linear layer.

        Keys are ``L0``, ``L1``, ... in feedforward order.
        """
        out: Dict[str, torch.Tensor] = {}
        for i, mat in enumerate(self.layer_mats):
            out[f"L{i}"] = (mat.A & mat.allowed_mask).detach().to(dtype=torch.int8)
        return out

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected x with shape [B, {self.input_dim}], got {tuple(x.shape)}")

        h = x
        for i, mat in enumerate(self.layer_mats):
            W = mat.build_weight()
            h = h @ W
            if self.bias:
                h = h + self.biases[i]
            if i < self.num_linear_layers - 1:
                h = self.act(h)

        if not return_aux:
            return h
        aux: Dict[str, Any] = {
            "model_id": self.MODEL_ID,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
        }
        return h, aux
