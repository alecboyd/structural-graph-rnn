"""Contractive Recurrent Perceptron classifier and dynamics configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CRPConfig:
    """Hyperparameters controlling CRP dynamics, budget, and certification."""
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

    IMPORTANT: dimensions are inferred from masks:
      MIH: (input_dim, hidden_dim)
      MH:  (hidden_dim, hidden_dim)
      MHL: (hidden_dim, num_classes)

    Standardized interface:
      forward(x, return_aux=False) -> logits
      forward(x, return_aux=True)  -> (logits, aux)

    aux always contains:
      - model_id: "crp"
    """

    MODEL_ID = "crp"

    def __init__(
        self,
        *,
        cfg: Optional[CRPConfig] = None,
        bias: bool = True,
        dag: bool = False,
        init_type: str = "kaiming_uniform",
        activation: str = "leaky_relu",
        negative_slope: float = 0.05,
        MIH: torch.Tensor,
        MH: torch.Tensor,
        MHL: torch.Tensor,
    ) -> None:
        """
        Initialize a CRP classifier from structural masks and dynamic settings.

        Inputs:
        - cfg: Dynamics and certification configuration.
        - bias: Whether hidden/output bias terms are trainable.
        - dag: When true, do not enforce contraction normalization.
        - init_type: Global weight initialization policy.
        - activation: Global activation selection.
        - negative_slope: LeakyReLU slope when activation is leaky.
        - MIH, MH, MHL: Binary or weighted masks defining graph structure.

        Invariants:
        - Mask dimensions must be 2D and mutually consistent on hidden size.
        - ``MH`` must be square.
        """
        super().__init__()

        if MIH.dim() != 2 or MH.dim() != 2 or MHL.dim() != 2:
            raise ValueError("MIH, MH, MHL must all be 2D tensors")

        # Infer dimensions from masks
        input_dim = int(MIH.shape[0])
        hidden_from_MIH = int(MIH.shape[1])
        hidden_from_MH0 = int(MH.shape[0])
        hidden_from_MH1 = int(MH.shape[1])
        hidden_from_MHL = int(MHL.shape[0])
        num_classes = int(MHL.shape[1])

        # Validate basic dims
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_from_MH0 <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")

        # Validate consistency across masks
        if hidden_from_MH0 != hidden_from_MH1:
            raise ValueError(f"MH must be square, got {tuple(MH.shape)}")
        if hidden_from_MIH != hidden_from_MH0:
            raise ValueError(
                f"Mask mismatch: MIH hidden={hidden_from_MIH}, MH hidden={hidden_from_MH0}"
            )
        if hidden_from_MHL != hidden_from_MH0:
            raise ValueError(
                f"Mask mismatch: MHL hidden={hidden_from_MHL}, MH hidden={hidden_from_MH0}"
            )

        self.input_dim = input_dim
        # NOTE: now this is TOTAL hidden vertices (graph nodes), not "per layer"
        self.hidden_dim = hidden_from_MH0
        self.num_classes = num_classes

        self.cfg = cfg if cfg is not None else CRPConfig()
        self.dag = bool(dag)
        self.init_type = init_type
        self.activation = activation
        self.negative_slope = float(negative_slope)
        if self.dag:
            warnings.warn(
                "CRPClassifier: schematic marked dag=True; skipping contraction normalization."
            )

        # Store masks as buffers so they move with .to(device) and are saved in state_dict
        self.register_buffer("MIH", MIH.float())
        self.register_buffer("MH", MH.float())
        self.register_buffer("MHL", MHL.float())

        # Trainable raw weights (masked later)
        self.RIH = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.RH = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.RHL = nn.Parameter(torch.empty(self.hidden_dim, self.num_classes))

        if bias:
            self.BH = nn.Parameter(torch.zeros(self.hidden_dim))
            self.BL = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.register_parameter("BH", None)
            self.register_parameter("BL", None)

        self.act = self._make_activation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable raw weights and optional biases."""
        init = self.init_type.lower()
        if init == "linear_default":
            self._init_linear_default(self.RIH, self.BH)
            self._init_linear_default(self.RH, self.BH)
            self._init_linear_default(self.RHL, self.BL)
            if self.BH is not None:
                pass
            if self.BL is not None:
                pass
            return

        if init == "kaiming_uniform":
            a = self.negative_slope if self.activation.lower() == "leaky_relu" else 0.0
            nn.init.kaiming_uniform_(self.RIH.t(), a=a)
            nn.init.kaiming_uniform_(self.RH.t(), a=a)
            nn.init.kaiming_uniform_(self.RHL.t(), a=a)
            if self.BH is not None:
                nn.init.zeros_(self.BH)
            if self.BL is not None:
                nn.init.zeros_(self.BL)

        else:
            raise ValueError(f"Unknown init_type: {self.init_type!r}")
    
    def _make_activation(self) -> nn.Module:
        name = self.activation.lower()
        if name == "relu":
            return nn.ReLU(inplace=False)
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)
        raise ValueError(f"Unknown activation: {self.activation!r}")

    def _init_linear_default(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        nn.init.kaiming_uniform_(weight.t(), a=math.sqrt(5))
        if bias is None:
            return
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight.t())
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    @torch.no_grad()
    def _normalize_recurrent(self, RH_masked: torch.Tensor) -> torch.Tensor:
        """
        Scale masked recurrent weights so the induced infinity norm is bounded.

        Assumption:
        - ``cfg.c`` is in ``[0, 1)`` to preserve contraction behavior.
        - Only used when ``dag`` is false.
        """
        eps = float(self.cfg.eps)
        c = float(self.cfg.c)

        # ||W_H^T||_inf = max column L1 sum
        col_l1 = RH_masked.abs().sum(dim=0)
        max_col_l1 = col_l1.max().clamp_min(eps)

        scale = c / max_col_l1
        return RH_masked * scale

    def _build_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masks and recurrent normalization to produce effective weights.

        Returns:
        - ``(W_IH, W_H, W_HL)`` used by forward dynamics.

        Notes:
        - Uses a straight-through style correction so forward uses normalized
          recurrent weights while gradients flow through masked raw weights.
        """
        W_IH = self.RIH * self.MIH
        W_HL = self.RHL * self.MHL

        RH_masked = self.RH * self.MH
        if self.dag:
            W_H = RH_masked
        else:
            W_H = self._normalize_recurrent(RH_masked)
            # Straight-through gradient trick: forward uses normalized, backward sees RH_masked gradient
            W_H = W_H + (RH_masked - RH_masked.detach())
        return W_IH, W_H, W_HL

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run CRP iterative dynamics and return logits or ``(logits, aux)``.

        Inputs:
        - x: Batch tensor of shape ``[B, input_dim]`` or higher-rank images.
        - return_aux: Include diagnostics such as ``tau`` and ``certified``.

        Returns:
        - Final logits for each sample.
        - Optional aux payload with model id and certification diagnostics.

        Side effects:
        - None outside standard module forward computation.

        Assumptions:
        - ``cfg.kappa`` is in ``[0, 1]`` and ``cfg.c`` is in ``[0, 1)``.
        """
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
        if not self.dag:
            if not (0.0 <= c < 1.0):
                raise ValueError("cfg.c must be in [0, 1) when dag=False")

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
