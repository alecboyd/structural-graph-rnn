"""Reusable masked matrix components for explicit DeepR-style sparsity."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


def _random_sign(num: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample i.i.d. signs in {-1, +1}."""
    if num <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.where(
        torch.rand(num, device=device) < 0.5,
        -torch.ones(num, device=device, dtype=dtype),
        torch.ones(num, device=device, dtype=dtype),
    )


class StaticMaskedMatrix(nn.Module):
    """Dense trainable matrix constrained by an immutable allowed-edge mask."""

    def __init__(
        self,
        *,
        allowed_mask: torch.Tensor,
        init_dense: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if allowed_mask.dim() != 2:
            raise ValueError("allowed_mask must be 2D")

        allowed = allowed_mask.bool()
        self.register_buffer("allowed_mask", allowed)

        rows, cols = allowed.shape
        if init_dense is not None and tuple(init_dense.shape) != (rows, cols):
            raise ValueError(
                f"init_dense shape mismatch: expected {(rows, cols)}, got {tuple(init_dense.shape)}"
            )
        init = init_dense if init_dense is not None else torch.zeros(rows, cols, device=allowed.device)
        self.weight = nn.Parameter(init.clone())

    def build_weight(self) -> torch.Tensor:
        """Return masked dense matrix."""
        return self.weight * self.allowed_mask.to(dtype=self.weight.dtype)


class DeepRMaskedMatrix(nn.Module):
    """
    DeepR-managed sparse masked matrix.

    State:
    - allowed_mask: immutable allowed-edge mask.
    - A: active-edge mask.
    - S: sign buffer in {-1, 0, +1}.
    - theta: trainable non-negative magnitude state on active entries.
    """

    def __init__(
        self,
        *,
        allowed_mask: torch.Tensor,
        K: int,
        init_dense: Optional[torch.Tensor] = None,
        debug_checks: bool = False,
    ) -> None:
        super().__init__()
        if allowed_mask.dim() != 2:
            raise ValueError("allowed_mask must be 2D")

        allowed = allowed_mask.bool()
        self.register_buffer("allowed_mask", allowed)

        rows, cols = allowed.shape
        if init_dense is not None and tuple(init_dense.shape) != (rows, cols):
            raise ValueError(
                f"init_dense shape mismatch: expected {(rows, cols)}, got {tuple(init_dense.shape)}"
            )

        seed = init_dense.abs() if init_dense is not None else torch.zeros(rows, cols, device=allowed.device)
        self.register_buffer("theta_seed", seed.to(dtype=torch.float32))

        allowed_idx = allowed.nonzero(as_tuple=False)
        self.register_buffer("allowed_idx", allowed_idx)
        allowed_flat_idx = allowed_idx[:, 0] * cols + allowed_idx[:, 1]
        self.register_buffer("allowed_flat_idx", allowed_flat_idx)

        self.allowed_count = int(allowed_flat_idx.numel())
        self.active_target_count = min(max(int(K), 0), self.allowed_count)
        self.debug_checks = bool(debug_checks)

        A = torch.zeros_like(allowed, dtype=torch.bool)
        if self.allowed_count > 0 and self.active_target_count > 0:
            if self.active_target_count >= self.allowed_count:
                A.copy_(allowed)
            else:
                perm = torch.randperm(self.allowed_count, device=allowed.device)[: self.active_target_count]
                chosen = allowed_flat_idx[perm]
                A.view(-1)[chosen] = True
        self.register_buffer("A", A)

        S = torch.zeros(rows, cols, device=allowed.device, dtype=torch.float32)
        if self.active_target_count > 0:
            active_flat = A.view(-1).nonzero(as_tuple=False).squeeze(1)
            S.view(-1)[active_flat] = _random_sign(
                int(active_flat.numel()),
                device=S.device,
                dtype=S.dtype,
            )
        self.register_buffer("S", S)

        theta_init = torch.zeros(rows, cols, device=allowed.device, dtype=torch.float32)
        if self.active_target_count > 0:
            theta_init[A] = self.theta_seed[A]
        self.theta = nn.Parameter(theta_init)

        self._sanitize_state()
        if self.debug_checks:
            self.assert_invariants(expected_count=self.active_target_count)

    def build_weight(self) -> torch.Tensor:
        """
        Build dense effective matrix:
            W = (A & allowed) * (S * clamp_min(theta, 0)).
        """
        active = self.A & self.allowed_mask
        mag = self.theta.clamp_min(0.0)
        return active.to(dtype=mag.dtype) * (self.S.to(dtype=mag.dtype) * mag)

    def active_count(self) -> int:
        """Return current number of active edges."""
        return int(self.A.sum().item())

    def active_flat_indices(self) -> torch.Tensor:
        """Return flattened active indices."""
        return self.A.view(-1).nonzero(as_tuple=False).squeeze(1)

    def dormant_flat_indices(self) -> torch.Tensor:
        """Return flattened dormant-allowed indices."""
        allowed_flat = self.allowed_mask.view(-1)
        active_flat = self.A.view(-1)
        return (allowed_flat & ~active_flat).nonzero(as_tuple=False).squeeze(1)

    @torch.no_grad()
    def set_all_inactive(self) -> None:
        """Clear active set and values."""
        self.A.zero_()
        self.S.zero_()
        self.theta.data.zero_()

    @torch.no_grad()
    def set_all_active_from_allowed(self, *, theta_mode: Literal["seed", "zero", "keep"] = "seed") -> None:
        """Activate all allowed edges."""
        prev_A = self.A.clone()
        self.A.copy_(self.allowed_mask)
        missing_sign = self.A & (self.S == 0)
        if missing_sign.any():
            self.S[missing_sign] = _random_sign(
                int(missing_sign.sum().item()),
                device=self.S.device,
                dtype=self.S.dtype,
            )
        if theta_mode == "seed":
            self.theta.data[self.A] = self.theta_seed[self.A]
        elif theta_mode == "zero":
            self.theta.data[self.A] = 0.0
        elif theta_mode == "keep":
            newly = self.A & (~prev_A)
            if newly.any():
                self.theta.data[newly] = self.theta_seed[newly]
        else:
            raise ValueError(f"Unknown theta_mode: {theta_mode!r}")
        self._sanitize_state()

    @torch.no_grad()
    def activate_flat_indices(
        self,
        flat_idx: torch.Tensor,
        *,
        theta_mode: Literal["seed", "zero"] = "zero",
    ) -> None:
        """Activate selected flat indices and assign fresh random signs."""
        if flat_idx.numel() == 0:
            return
        idx = torch.unique(flat_idx.to(device=self.A.device, dtype=torch.long))
        allowed_flat = self.allowed_mask.view(-1)
        idx = idx[allowed_flat[idx]]
        if idx.numel() == 0:
            return

        A_flat = self.A.view(-1)
        newly = ~A_flat[idx]
        if not newly.any():
            return
        new_idx = idx[newly]
        A_flat[new_idx] = True

        S_flat = self.S.view(-1)
        theta_flat = self.theta.data.view(-1)
        seed_flat = self.theta_seed.view(-1)

        S_flat[new_idx] = _random_sign(
            int(new_idx.numel()),
            device=S_flat.device,
            dtype=S_flat.dtype,
        )
        if theta_mode == "seed":
            theta_flat[new_idx] = seed_flat[new_idx]
        elif theta_mode == "zero":
            theta_flat[new_idx] = 0.0
        else:
            raise ValueError(f"Unknown theta_mode: {theta_mode!r}")

        self._sanitize_state()

    @torch.no_grad()
    def deactivate_flat_indices(self, flat_idx: torch.Tensor) -> None:
        """Deactivate selected flat indices and clear their state."""
        if flat_idx.numel() == 0:
            return
        idx = torch.unique(flat_idx.to(device=self.A.device, dtype=torch.long))
        A_flat = self.A.view(-1)
        S_flat = self.S.view(-1)
        theta_flat = self.theta.data.view(-1)
        A_flat[idx] = False
        S_flat[idx] = 0.0
        theta_flat[idx] = 0.0
        self._sanitize_state()

    @torch.no_grad()
    def update_active(
        self,
        *,
        lr: float,
        drift_alpha: float,
        T: float,
        prune_negative: bool,
    ) -> int:
        """Apply gradient/noise/drift to active theta and optionally prune negatives."""
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}.")
        if drift_alpha < 0:
            raise ValueError(f"drift_alpha must be >= 0, got {drift_alpha}.")
        if T < 0:
            raise ValueError(f"T must be >= 0, got {T}.")

        grad = self.theta.grad
        grad_data = grad if grad is not None else torch.zeros_like(self.theta)

        active_idx = self.active_flat_indices()
        if active_idx.numel() > 0:
            theta_flat = self.theta.data.view(-1)
            grad_flat = grad_data.view(-1)
            upd = -float(lr) * grad_flat[active_idx] - float(lr) * float(drift_alpha)
            if lr > 0.0 and T > 0.0:
                noise_std = math.sqrt(2.0 * float(lr) * float(T))
                upd = upd + noise_std * torch.randn_like(upd)
            theta_flat[active_idx] += upd

        pruned = 0
        if prune_negative:
            theta_flat = self.theta.data.view(-1)
            active_idx = self.active_flat_indices()
            if active_idx.numel() > 0:
                prune_idx = active_idx[theta_flat[active_idx] < 0.0]
                pruned = int(prune_idx.numel())
                if pruned > 0:
                    self.deactivate_flat_indices(prune_idx)
        else:
            self.theta.data.clamp_min_(0.0)
            self._sanitize_state()

        return pruned

    @torch.no_grad()
    def deepr_step(self, lr: float, drift_alpha: float, T: float) -> None:
        """
        Local DeepR update using this matrix's target count.

        If ``allowed_count <= target``, active set is fixed to allowed edges and
        no rewiring is performed.
        """
        if self.allowed_count <= self.active_target_count:
            self.set_all_active_from_allowed(theta_mode="keep")
            self.update_active(
                lr=lr,
                drift_alpha=drift_alpha,
                T=T,
                prune_negative=False,
            )
            self.clear_grad()
            if self.debug_checks:
                self.assert_invariants(expected_count=self.allowed_count)
            return

        self.update_active(
            lr=lr,
            drift_alpha=drift_alpha,
            T=T,
            prune_negative=True,
        )

        need = self.active_target_count - self.active_count()
        if need > 0:
            dormant = self.dormant_flat_indices()
            if dormant.numel() > 0:
                take = min(int(need), int(dormant.numel()))
                perm = torch.randperm(int(dormant.numel()), device=dormant.device)[:take]
                self.activate_flat_indices(dormant[perm], theta_mode="zero")

        self.clear_grad()
        if self.debug_checks:
            self.assert_invariants(expected_count=self.active_target_count)

    @torch.no_grad()
    def assert_invariants(self, *, expected_count: Optional[int] = None) -> None:
        """Validate internal state consistency."""
        if not torch.all(self.A <= self.allowed_mask):
            raise RuntimeError("Invariant failed: active mask is not a subset of allowed mask.")
        if not torch.all(self.S[~self.A] == 0):
            raise RuntimeError("Invariant failed: sign must be 0 on inactive edges.")
        if not torch.all(self.theta.data[~self.A] == 0):
            raise RuntimeError("Invariant failed: theta must be 0 on inactive edges.")
        if expected_count is not None:
            count = self.active_count()
            if count != int(expected_count):
                raise RuntimeError(
                    f"Invariant failed: active count mismatch (got {count}, expected {int(expected_count)})."
                )

    def clear_grad(self) -> None:
        """Clear theta gradient buffer."""
        if self.theta.grad is not None:
            self.theta.grad.zero_()

    def _sanitize_state(self) -> None:
        """Enforce subset and zero-on-inactive invariants."""
        allowed_flat = self.allowed_mask.view(-1)
        A_flat = self.A.view(-1)
        S_flat = self.S.view(-1)
        theta_flat = self.theta.data.view(-1)
        A_flat &= allowed_flat
        S_flat[~A_flat] = 0.0
        theta_flat[~A_flat] = 0.0
        theta_flat[~allowed_flat] = 0.0
