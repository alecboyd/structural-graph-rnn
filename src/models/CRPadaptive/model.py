"""Adaptive CRP classifier backed by reusable DeepR matrix primitives.

README:
- During training, run ``loss.backward()`` first.
- Then call ``model.deepr_step_all(lr=..., drift_alpha=..., T=...)`` to apply
  explicit DeepR updates for IH/HH/HL in one global-budget step.
- When ``cfg.recurrent_norm == "weighted_inf"``, call
  ``model.update_w_inf(iters=..., eps=...)`` (or ``update_normalization_cache``)
  once per epoch before forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from src.core.deepR import DeepRMaskedMatrix, StaticMaskedMatrix
from src.models.crp.model import CRPClassifier, CRPConfig


@dataclass
class CRPAdaptiveConfig(CRPConfig):
    """CRP config plus adaptive DeepR toggles and a single global budget."""

    deepr_ih: bool = True
    deepr_hh: bool = True
    deepr_hl: bool = True

    # Global DeepR budget across all enabled matrices.
    K_total: Optional[int] = None
    frac_total: float = 1.0
    full_adjacency_allowed: bool = True

    # DeepR update hyperparameters.
    deepr_drift_alpha: float = 1e-4
    deepr_temperature: float = 1e-6
    deepr_debug_checks: bool = False


class CRPAdaptiveClassifier(CRPClassifier):
    """
    CRP variant with DeepR-managed sparse weights on IH/HH/HL.

    This class extends ``CRPClassifier`` and reuses its forward dynamics and
    certification logic, overriding only weight construction and DeepR updates.
    """

    MODEL_ID = "crp_adaptive"

    def __init__(
        self,
        *,
        cfg: Optional[CRPAdaptiveConfig] = None,
        bias: bool = True,
        init_type: str = "kaiming_uniform",
        activation: str = "leaky_relu",
        negative_slope: float = 0.05,
        MIH: torch.Tensor,
        MH: torch.Tensor,
        MHL: torch.Tensor,
        MIH_edges: Optional[torch.Tensor] = None,
        MH_edges: Optional[torch.Tensor] = None,
        MHL_edges: Optional[torch.Tensor] = None,
    ) -> None:
        adaptive_cfg = cfg if cfg is not None else CRPAdaptiveConfig()
        init_seed_MIH = (MIH > 0)
        init_seed_MH = (MH > 0)
        init_seed_MHL = (MHL > 0)

        # Initialize base CRP state, dimensions, masks, biases, and dense init.
        super().__init__(
            cfg=adaptive_cfg,
            bias=bias,
            init_type=init_type,
            activation=activation,
            negative_slope=negative_slope,
            MIH=MIH,
            MH=MH,
            MHL=MHL,
            MIH_edges=MIH_edges,
            MH_edges=MH_edges,
            MHL_edges=MHL_edges,
        )

        # Capture initialized dense tensors, then replace with matrix modules.
        init_RIH = self.RIH.detach().clone()
        init_RH = self.RH.detach().clone()
        init_RHL = self.RHL.detach().clone()

        self.register_parameter("RIH", None)
        self.register_parameter("RH", None)
        self.register_parameter("RHL", None)

        if adaptive_cfg.full_adjacency_allowed:
            allowed_IH = torch.ones_like(self.MIH, dtype=torch.bool)
            allowed_HH = torch.ones_like(self.MH, dtype=torch.bool)
            allowed_HL = torch.ones_like(self.MHL, dtype=torch.bool)
        else:
            allowed_IH = (self.MIH > 0)
            allowed_HH = (self.MH > 0)
            allowed_HL = (self.MHL > 0)

        self.IH_mat = self._build_matrix(
            allowed_mask=allowed_IH,
            init_dense=init_RIH,
            use_deepr=adaptive_cfg.deepr_ih,
        )
        self.HH_mat = self._build_matrix(
            allowed_mask=allowed_HH,
            init_dense=init_RH,
            use_deepr=adaptive_cfg.deepr_hh,
        )
        self.HL_mat = self._build_matrix(
            allowed_mask=allowed_HL,
            init_dense=init_RHL,
            use_deepr=adaptive_cfg.deepr_hl,
        )

        self.register_buffer("seed_MIH", init_seed_MIH)
        self.register_buffer("seed_MH", init_seed_MH)
        self.register_buffer("seed_MHL", init_seed_MHL)

        self.global_allowed_count = sum(mat.allowed_count for mat in self._deepr_matrices())
        self.global_active_target_count = self._resolve_global_budget()
        self._initialize_global_active_set()

    def _build_matrix(
        self,
        *,
        allowed_mask: torch.Tensor,
        init_dense: torch.Tensor,
        use_deepr: bool,
    ) -> nn.Module:
        if use_deepr:
            # K=0 here; global budget initialization sets active edges jointly.
            return DeepRMaskedMatrix(
                allowed_mask=allowed_mask,
                K=0,
                init_dense=init_dense,
                debug_checks=bool(getattr(self.cfg, "deepr_debug_checks", False)),
            )
        return StaticMaskedMatrix(
            allowed_mask=allowed_mask,
            init_dense=init_dense,
        )

    def _deepr_matrices(self) -> Tuple[DeepRMaskedMatrix, ...]:
        mats: list[DeepRMaskedMatrix] = []
        for mat in (self.IH_mat, self.HH_mat, self.HL_mat):
            if isinstance(mat, DeepRMaskedMatrix):
                mats.append(mat)
        return tuple(mats)

    def _resolve_global_budget(self) -> int:
        total_allowed = int(self.global_allowed_count)
        if total_allowed <= 0:
            return 0

        cfg = self.cfg
        K_total = getattr(cfg, "K_total", None)
        if K_total is not None:
            return max(0, min(int(K_total), total_allowed))

        frac_total = float(getattr(cfg, "frac_total", 1.0))
        return max(0, min(int(round(frac_total * total_allowed)), total_allowed))

    @torch.no_grad()
    def _initialize_global_active_set(self) -> None:
        """
        Initialize one global active set:
        - Seed from user-provided masks first.
        - If seed > budget, subsample seed uniformly.
        - Fill remaining slots from dormant allowed edges uniformly.
        """
        mats = self._deepr_matrices()
        if not mats:
            return

        for mat in mats:
            mat.set_all_inactive()

        target = int(self.global_active_target_count)
        total = int(self.global_allowed_count)
        if target <= 0:
            return

        if target >= total:
            for mat in mats:
                mat.set_all_active_from_allowed(theta_mode="seed")
            return

        seed_lists: list[torch.Tensor] = []
        for mat, seed_mask in zip(mats, (self.seed_MIH, self.seed_MH, self.seed_MHL)):
            local_seed_flat = seed_mask.to(device=mat.A.device).view(-1).nonzero(as_tuple=False).squeeze(1)
            if local_seed_flat.numel() > 0:
                allowed_flat = mat.allowed_mask.view(-1)
                local_seed_flat = local_seed_flat[allowed_flat[local_seed_flat]]
            seed_lists.append(local_seed_flat)

        seed_total = sum(int(idx.numel()) for idx in seed_lists)
        if seed_total > 0:
            if seed_total <= target:
                for mat, local_seed_flat in zip(mats, seed_lists):
                    if local_seed_flat.numel() > 0:
                        mat.activate_flat_indices(local_seed_flat, theta_mode="seed")
            else:
                self._activate_global_sample(
                    mats=mats,
                    local_index_lists=seed_lists,
                    take=target,
                    theta_mode="seed",
                )

        remaining = target - sum(mat.active_count() for mat in mats)
        if remaining > 0:
            dormant_lists = [mat.dormant_flat_indices() for mat in mats]
            self._activate_global_sample(
                mats=mats,
                local_index_lists=dormant_lists,
                take=remaining,
                theta_mode="zero",
            )

        if bool(getattr(self.cfg, "deepr_debug_checks", False)):
            self._assert_global_budget()

    def _build_weights(self):
        """
        Build effective IH/HH/HL matrices from adaptive matrix modules.

        Reuses base normalization rule and straight-through correction for HH.
        """
        W_IH = self.IH_mat.build_weight()
        W_HL = self.HL_mat.build_weight()

        W_H_raw = self.HH_mat.build_weight()
        if self.cfg.recurrent_norm == "weighted_inf":
            if torch.any(self.w_inf <= 0):
                self.update_w_inf(self.cfg.weighted_inf_iters, self.cfg.eps)
        W_H_norm = self._normalize_recurrent(W_H_raw)
        W_H = W_H_norm + (W_H_raw - W_H_raw.detach())
        return W_IH, W_H, W_HL

    @torch.no_grad()
    def update_w_inf(self, iters: int, eps: float) -> None:
        """Power iteration cache update over current adaptive recurrent matrix."""
        W_H_raw = self.HH_mat.build_weight().detach()
        A = W_H_raw.abs().t()
        w = self.w_inf.clamp_min(float(eps))
        for _ in range(max(1, int(iters))):
            w = A @ w
            w = w / w.max().clamp_min(float(eps))
            w = w.clamp_min(float(eps))
        self.w_inf.copy_(w)

    @torch.no_grad()
    def deepr_step_all(
        self,
        *,
        lr: float,
        drift_alpha: Optional[float] = None,
        T: Optional[float] = None,
    ) -> None:
        """
        Apply one global-budget DeepR step across all enabled matrices.

        1) Update active theta and prune negatives.
        2) Refill globally from dormant allowed edges to maintain global budget.
        """
        mats = self._deepr_matrices()
        if not mats:
            return

        drift = float(self.cfg.deepr_drift_alpha if drift_alpha is None else drift_alpha)
        temp = float(self.cfg.deepr_temperature if T is None else T)

        # No dormant pool exists when budget covers all allowed edges.
        if self.global_active_target_count >= self.global_allowed_count:
            for mat in mats:
                mat.set_all_active_from_allowed(theta_mode="keep")
                mat.update_active(
                    lr=lr,
                    drift_alpha=drift,
                    T=temp,
                    prune_negative=False,
                )
                mat.clear_grad()
            return

        for mat in mats:
            mat.update_active(
                lr=lr,
                drift_alpha=drift,
                T=temp,
                prune_negative=True,
            )

        current_active = sum(mat.active_count() for mat in mats)
        target = int(self.global_active_target_count)

        if current_active > target:
            self._prune_global_excess(current_active - target)
        elif current_active < target:
            self._rewire_global_deficit(target - current_active)

        for mat in mats:
            mat.clear_grad()

        if bool(getattr(self.cfg, "deepr_debug_checks", False)):
            self._assert_global_budget()

    @torch.no_grad()
    def _prune_global_excess(self, excess: int) -> None:
        if excess <= 0:
            return
        mats = self._deepr_matrices()
        active_lists = [mat.active_flat_indices() for mat in mats]
        total_active = sum(int(idx.numel()) for idx in active_lists)
        if total_active <= 0:
            return

        take = min(int(excess), total_active)
        sample = torch.randperm(total_active, device=self.MIH.device)[:take]
        offset = 0
        for mat, active_idx in zip(mats, active_lists):
            count = int(active_idx.numel())
            if count <= 0:
                continue
            sel = (sample >= offset) & (sample < offset + count)
            if sel.any():
                local_pos = sample[sel] - offset
                mat.deactivate_flat_indices(active_idx[local_pos])
            offset += count

    @torch.no_grad()
    def _rewire_global_deficit(self, need: int) -> None:
        if need <= 0:
            return
        mats = self._deepr_matrices()
        dormant_lists = [mat.dormant_flat_indices() for mat in mats]
        self._activate_global_sample(
            mats=mats,
            local_index_lists=dormant_lists,
            take=need,
            theta_mode="zero",
        )

    @torch.no_grad()
    def _activate_global_sample(
        self,
        *,
        mats: Tuple[DeepRMaskedMatrix, ...],
        local_index_lists: list[torch.Tensor],
        take: int,
        theta_mode: str,
    ) -> None:
        total = sum(int(idx.numel()) for idx in local_index_lists)
        if take <= 0 or total <= 0:
            return
        n_take = min(int(take), int(total))
        sample = torch.randperm(total, device=self.MIH.device)[:n_take]
        offset = 0
        for mat, local_idx in zip(mats, local_index_lists):
            count = int(local_idx.numel())
            if count <= 0:
                continue
            sel = (sample >= offset) & (sample < offset + count)
            if sel.any():
                local_pos = sample[sel] - offset
                mat.activate_flat_indices(local_idx[local_pos], theta_mode=theta_mode)
            offset += count

    def deepr_parameters(self) -> Iterator[nn.Parameter]:
        """Expose DeepR theta parameters explicitly."""
        for mat in self._deepr_matrices():
            yield mat.theta

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """
        Exclude DeepR theta from default optimizer traversal.

        DeepR theta should be updated only via explicit ``deepr_step_all``.
        """
        theta_ids = {id(p) for p in self.deepr_parameters()}
        for name, param in super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):
            if id(param) in theta_ids:
                continue
            yield name, param

    @torch.no_grad()
    def _assert_global_budget(self) -> None:
        mats = self._deepr_matrices()
        for mat in mats:
            mat.assert_invariants()
        active_total = sum(mat.active_count() for mat in mats)
        target = int(self.global_active_target_count)
        if active_total != target:
            raise RuntimeError(f"Global budget mismatch: active_total={active_total}, target={target}.")

    def _adjacency_from_matrix(self, mat: nn.Module) -> torch.Tensor:
        """
        Return a binary adjacency tensor for a matrix module.

        DeepR matrices expose explicit active masks; static matrices use their
        structural allowed mask.
        """
        if isinstance(mat, DeepRMaskedMatrix):
            return (mat.A & mat.allowed_mask).detach().to(dtype=torch.int8)
        if isinstance(mat, StaticMaskedMatrix):
            return mat.allowed_mask.detach().to(dtype=torch.int8)
        raise TypeError(f"Unsupported matrix module type for adjacency export: {type(mat)!r}")

    def get_adjacency_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Export run-end adjacency matrices for logging/inspection.

        Keys:
        - ``IH``: input-to-hidden active adjacency
        - ``HH``: hidden-to-hidden active adjacency
        - ``HL``: hidden-to-logit active adjacency
        """
        return {
            "IH": self._adjacency_from_matrix(self.IH_mat),
            "HH": self._adjacency_from_matrix(self.HH_mat),
            "HL": self._adjacency_from_matrix(self.HL_mat),
        }
