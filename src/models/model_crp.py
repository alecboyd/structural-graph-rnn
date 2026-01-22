from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

@dataclass
class CRPConfig:
    # Contraction / dynamics
    kappa: float = 1.0          # κ in [0,1]
    c: float = 0.95             # contraction cap for ||W_H^T||_infty
    alpha: float = 0.05         # leaky ReLU slope (<=1 for 1-Lipschitz)
    eps: float = 1e-8           # ε for normalization stability

    # Inference budget
    t_max: int = 32             # global unroll cap for batching

    # Certification
    use_certification: bool = True
    margin_factor: float = 2.0  # the "2" in m(t) > 2*Gamma(t)

class CRPClassifier(nn.Module):
    """
    Contractive Recurrent Perceptron (CRP) classifier.

    Shapes (PyTorch convention):
      x: [B, input_dim]
      H: [B, hidden_dim]
      logits: [B, num_classes]

    Masking:
      MIH: [input_dim, hidden_dim]
      MH:  [hidden_dim, hidden_dim]
      MHL: [hidden_dim, num_classes]

    Contractive recurrence:
      H_{t+1} = (1-κ) H_t + κ σ( H_t W_H + x W_IH + B_H )

    Logits are memoryless:
      L_t = H_t W_HL + B_L

    Normalization:
      W_H = Normalize( R_H o M_H ) so that ||W_H^T||_infty <= c
      (equivalently: max column-L1 sum of |W_H| <= c)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        cfg: Optional[CRPConfig] = None,
        *,
        bias: bool = True,
        # Optional masks (if None -> fully connected)
        MIH: Optional[torch.Tensor] = None,
        MH: Optional[torch.Tensor] = None,
        MHL: Optional[torch.Tensor] = None,
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
        

        self.cfg = cfg if cfg is not None else CRPConfig() # Defaults

        # --- Masks (buffers) ---
        # Default to all-ones (fully connected)
        if MIH is None:
            MIH = torch.ones(self.input_dim, self.hidden_dim)
        if MH is None:
            MH = torch.ones(self.hidden_dim, self.hidden_dim)
        if MHL is None:
            MHL = torch.ones(self.hidden_dim, self.num_classes)

        # Validate shapes
        if MIH.shape != (self.input_dim, self.hidden_dim):
            raise ValueError(f"MIH shape must be {(self.input_dim, self.hidden_dim)}, got {tuple(MIH.shape)}")
        if MH.shape != (self.hidden_dim, self.hidden_dim):
            raise ValueError(f"MH shape must be {(self.hidden_dim, self.hidden_dim)}, got {tuple(MH.shape)}")
        if MHL.shape != (self.hidden_dim, self.num_classes):
            raise ValueError(f"MHL shape must be {(self.hidden_dim, self.num_classes)}, got {tuple(MHL.shape)}")

        # Registering (with floats to prevent type issues in multiplication later)
        self.register_buffer("MIH", MIH.float())
        self.register_buffer("MH", MH.float())
        self.register_buffer("MHL", MHL.float())

        # --- Raw trainable weights (parameters) ---
        # These are masked each forward; RH additionally normalized.
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

        # Nonlinearity (1-Lipschitz if alpha <= 1)
        self.act = nn.LeakyReLU(negative_slope=float(self.cfg.alpha), inplace=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Reasonable defaults, can be changed later
        nn.init.kaiming_uniform_(self.RIH, a=self.cfg.alpha)
        nn.init.kaiming_uniform_(self.RH, a=self.cfg.alpha)
        nn.init.kaiming_uniform_(self.RHL, a=self.cfg.alpha)

        if self.BH is not None:
            nn.init.zeros_(self.BH)
        if self.BL is not None:
            nn.init.zeros_(self.BL)

    @torch.no_grad()
    def _normalize_recurrent(self, RH_masked: torch.Tensor) -> torch.Tensor:
        """
        Scale RH_masked so max column-wise L1 norm of |W_H| is <= c.
        That enforces ||W_H^T||_infty <= c.

        RH_masked: [H, H]
        returns W_H: [H, H]
        """
        eps = float(self.cfg.eps)
        c = float(self.cfg.c)

        # column sums of abs: [H]
        col_l1 = RH_masked.abs().sum(dim=0)
        max_col_l1 = col_l1.max().clamp_min(eps) # clamps to avoid division by 0 

        # Scale so that max column sum becomes c (or less)
        scale = c / max_col_l1
        return RH_masked * scale

    def _build_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          W_IH: [I, H]
          W_H:  [H, H]  (normalized for contraction)
          W_HL: [H, L]
        """
        W_IH = self.RIH * self.MIH
        W_HL = self.RHL * self.MHL

        RH_masked = self.RH * self.MH

        # normalize recurrent (no grad through normalization itself in this version)
        W_H = self._normalize_recurrent(RH_masked)

        # Important: W_H is treated as a constant w.r.t RH in this version.
        # That matches the "normalize each batch" intent but with stop-grad on the scaling.
        # We can make it differentiable later if necessary 

        W_H = W_H + (RH_masked - RH_masked.detach())  # to pass gradient backwards through to RH_masked and then to RH

        return W_IH, W_H, W_HL

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Batched CRP forward.

        Returns:
          logits_at_tau: [B, num_classes]
        Optionally returns aux dict with:
          tau: [B] (int64)
          certified: [B] (bool)
          steps_used: int (T_max)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected x with dim=({x.size(0)}, {self.input_dim}), got {tuple(x.shape)}")

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
            raise ValueError("cfg.kappa must be in [0,1]")
        if not (0.0 <= c < 1.0):
            raise ValueError("cfg.c must be in [0,1)")
            
        # Contraction factor rho = (1-kappa) + kappa*c
        rho = (1.0 - kappa) + kappa * c
        # avoid division by tiny numbers if rho close to 1
        denom = max(1e-12, 1.0 - rho)

        # ||W_HL^T||_infty = max column-wise L1 sum of |W_HL|
        # W_HL shape [H, L] so columns correspond to classes
        W_HL_T_inf = W_HL.abs().sum(dim=0).max()

        # Hidden init
        H = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)

        # Per-sample activity + outputs
        active = torch.ones(B, device=device, dtype=torch.bool)

        tau = torch.full((B,), t_max, device=device, dtype=torch.long) # Defaults to t_max, overwritten per sample
        certified = torch.zeros(B, device=device, dtype=torch.bool) # Metadata, not relevant for execution directly. Tells what % certify

        logits_tau = torch.zeros(B, self.num_classes, device=device, dtype=dtype) # Logit matrix, updated per sample

        H_prev = H

        for t in range(1, t_max + 1):
            

            # Freeze inactive samples (identity dynamics)
            if active.any():
                # Compute candidate update for everyone (since it needs to be done if at least one active)
                pre = H @ W_H + x @ W_IH + BH  # [B, H]
                H_cand = (1.0 - kappa) * H + kappa * self.act(pre)
                a = active.unsqueeze(1)  # [B,1]
                H_next = torch.where(a, H_cand, H)
            else:
                H_next = H  # nothing active, but still finish loop cheaply

            # Logits from H_next 
            logits = H_next @ W_HL + BL  # [B, L]

            if self.cfg.use_certification:
                # DeltaH(t) = H(t) - H(t-1) in infty norm per sample
                dH = (H_next - H_prev).abs().max(dim=1).values  # [B]

                # Gamma(t) = ||W_HL^T||_infty * rho/(1-rho) * ||DeltaH||_infty
                Gamma = W_HL_T_inf * (rho / denom) * dH  # [B]

                # margin m(t): top1 - top2
                top2 = torch.topk(logits, k=2, dim=1).values  # [B,2]
                margin = top2[:, 0] - top2[:, 1]            # [B]

                newly_cert = active & (margin > float(self.cfg.margin_factor) * Gamma)
            else:
                newly_cert = torch.zeros(B, device=device, dtype=torch.bool)

            # Record first certification time + logits
            if newly_cert.any():
                tau = torch.where(newly_cert, torch.tensor(t, device=device, dtype=tau.dtype), tau) # just updates the tau values
                certified = certified | newly_cert
                logits_tau = torch.where(newly_cert.unsqueeze(1), logits, logits_tau) # updates the logits only on the certified samples

                # deactivate those
                active = active & (~newly_cert)

            # Update history for next iteration
            H_prev = H_next
            H = H_next

        # For samples that never certified, use logits at T_max
        never = ~certified
        if never.any():
            logits_last = H @ W_HL + BL
            logits_tau = torch.where(never.unsqueeze(1), logits_last, logits_tau)

        if return_aux:
            aux = {
                "tau": tau,
                "certified": certified,
                "steps_used": t_max,
                "rho": torch.tensor(rho, device=device, dtype=dtype),
            }
            return logits_tau, aux
        return logits_tau