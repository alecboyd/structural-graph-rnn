from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CRPConfig:
    # Contraction / dynamics
    kappa: float = 1.0
    c: float = 0.95
    alpha: float = 0.05
    eps: float = 1e-8

    # Inference budget
    t_max: int = 32

    # Certification
    use_certification: bool = True
    margin_factor: float = 2.0


class CRPClassifier(nn.Module):
    """
    Contractive Recurrent Perceptron classifier.

    Standardized interface:
      forward(x, return_aux=False) -> logits
      forward(x, return_aux=True)  -> (logits, aux)

    aux always contains:
      - model_id: "crp"
    """

    MODEL_ID = "crp"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        cfg: Optional[CRPConfig] = None,
        bias: bool = True,
        MIH: torch.Tensor,
        MH: torch.Tensor,
        MHL: torch.Tensor,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.cfg = cfg if cfg is not None else CRPConfig()

        # Validate schematic-provided masks
        if MIH.shape != (self.input_dim, self.hidden_dim):
            raise ValueError(f"MIH shape must be {(self.input_dim, self.hidden_dim)}, got {tuple(MIH.shape)}")
        if MH.shape != (self.hidden_dim, self.hidden_dim):
            raise ValueError(f"MH shape must be {(self.hidden_dim, self.hidden_dim)}, got {tuple(MH.shape)}")
        if MHL.shape != (self.hidden_dim, self.num_classes):
            raise ValueError(f"MHL shape must be {(self.hidden_dim, self.num_classes)}, got {tuple(MHL.shape)}")

        self.register_buffer("MIH", MIH.float())
        self.register_buffer("MH", MH.float())
        self.register_buffer("MHL", MHL.float())

        # Raw weights (trainable)
        self.RIH = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.RH = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.RHL = nn.Parameter(torch.empty(self.hidden_dim, self.num_classes))

        # Biases
        if bias:
            self.BH = nn.Parameter(torch.zeros(self.hidden_dim))
            self.BL = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.register_parameter("BH", None)
            self.register_parameter("BL", None)

        self.act = nn.LeakyReLU(negative_slope=float(self.cfg.alpha), inplace=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.RIH, a=self.cfg.alpha)
        nn.init.kaiming_uniform_(self.RH, a=self.cfg.alpha)
        nn.init.kaiming_uniform_(self.RHL, a=self.cfg.alpha)
        if self.BH is not None:
            nn.init.zeros_(self.BH)
        if self.BL is not None:
            nn.init.zeros_(self.BL)

    @torch.no_grad()
    def _normalize_recurrent(self, RH_masked: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.eps)
        c = float(self.cfg.c)

        # ||W_H^T||_inf = max column L1 sum
        col_l1 = RH_masked.abs().sum(dim=0)
        max_col_l1 = col_l1.max().clamp_min(eps)

        scale = c / max_col_l1
        return RH_masked * scale

    def _build_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_IH = self.RIH * self.MIH
        W_HL = self.RHL * self.MHL

        RH_masked = self.RH * self.MH
        W_H = self._normalize_recurrent(RH_masked)

        # Straight-through gradient trick: forward uses normalized, backward sees identity-ish
        W_H = W_H + (RH_masked - RH_masked.detach())
        return W_IH, W_H, W_HL

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected x with shape [B, {self.input_dim}], got {tuple(x.shape)}")

        B = x.size(0)
        device = x.device
        dtype = x.dtype

        W_IH, W_H, W_HL = self._build_weights()

        BH = self.BH if self.BH is not None else torch.zeros(self.hidden_dim, device=device, dtype=dtype)
        BL = self.BL if self.BL is not None else torch.zeros(self.num_classes, device=device, dtype=dtype)

        kappa = float(self.cfg.kappa)
        c = float(self.cfg.c)
        t_max = int(self.cfg.t_max)

        if not (0.0 <= kappa <= 1.0):
            raise ValueError("cfg.kappa must be in [0, 1]")
        if not (0.0 <= c < 1.0):
            raise ValueError("cfg.c must be in [0, 1)")

        rho = (1.0 - kappa) + kappa * c
        denom = max(1e-12, 1.0 - rho)

        # ||W_HL^T||_inf = max column L1 sum
        W_HL_T_inf = W_HL.abs().sum(dim=0).max()

        H = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
        active = torch.ones(B, device=device, dtype=torch.bool)

        tau = torch.full((B,), t_max, device=device, dtype=torch.long)
        certified = torch.zeros(B, device=device, dtype=torch.bool)
        logits_tau = torch.zeros(B, self.num_classes, device=device, dtype=dtype)

        H_prev = H

        for t in range(1, t_max + 1):
            if active.any():
                pre = H @ W_H + x @ W_IH + BH
                H_cand = (1.0 - kappa) * H + kappa * self.act(pre)
                H_next = torch.where(active.unsqueeze(1), H_cand, H)
            else:
                H_next = H

            logits = H_next @ W_HL + BL

            if self.cfg.use_certification:
                dH = (H_next - H_prev).abs().max(dim=1).values
                Gamma = W_HL_T_inf * (rho / denom) * dH

                top2 = torch.topk(logits, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                newly_cert = active & (margin > float(self.cfg.margin_factor) * Gamma)
            else:
                newly_cert = torch.zeros(B, device=device, dtype=torch.bool)

            if newly_cert.any():
                tau = torch.where(newly_cert, torch.tensor(t, device=device, dtype=tau.dtype), tau)
                certified = certified | newly_cert
                logits_tau = torch.where(newly_cert.unsqueeze(1), logits, logits_tau)
                active = active & (~newly_cert)

            H_prev = H_next
            H = H_next

        # If never certified, output final logits
        never = ~certified
        if never.any():
            logits_last = H @ W_HL + BL
            logits_tau = torch.where(never.unsqueeze(1), logits_last, logits_tau)

        if not return_aux:
            return logits_tau

        aux: Dict[str, Any] = {
            "model_id": self.MODEL_ID,
            "tau": tau,
            "certified": certified,
            "steps_used": t_max,
            "rho": torch.tensor(rho, device=device, dtype=dtype),
        }
        return logits_tau, aux
