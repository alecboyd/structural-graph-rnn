"""Baseline MLP classifier implementation with standardized aux interface."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Baseline feedforward classifier.

    Standardized interface:
      forward(x, return_aux=False) -> logits
      forward(x, return_aux=True)  -> (logits, aux)

    aux always contains:
      - model_id: "mlp"
    """

    MODEL_ID = "mlp"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_hidden_layers: int,
        bias: bool = True,
        init_type: str = "kaiming_uniform",
        activation: str = "leaky_relu",
        negative_slope: float = 0.05,
    ) -> None:
        """
        Initialize an MLP with configurable hidden depth and width.

        Assumptions:
        - Inputs are flattened to ``[batch, input_dim]`` during ``forward``.
        - ``num_hidden_layers=0`` produces a single linear classifier.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        self.init_type = init_type
        self.activation = activation
        self.negative_slope = float(negative_slope)

        layers: list[nn.Module] = []
        act = self._make_activation()

        if num_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, num_classes, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            layers.append(act)

            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(self._make_activation())

            layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))

        self.net = nn.Sequential(*layers)

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.bias = bool(bias)
        self._apply_init()

    def _make_activation(self) -> nn.Module:
        name = self.activation.lower()
        if name == "relu":
            return nn.ReLU(inplace=False)
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)
        raise ValueError(f"Unknown activation: {self.activation!r}")

    def _apply_init(self) -> None:
        init = self.init_type.lower()
        if init == "linear_default":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)
            return
        if init == "kaiming_uniform":
            a = self.negative_slope if self.activation.lower() == "leaky_relu" else 0.0
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=a)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"Unknown init_type: {self.init_type!r}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute logits and optionally return lightweight model metadata.

        Inputs:
        - x: Batch tensor of shape ``[B, input_dim]`` or higher-rank images.
        - return_aux: When true, return ``(logits, aux_dict)``.

        Returns:
        - Logits tensor when ``return_aux`` is false.
        - Tuple ``(logits, aux)`` when true, where aux includes ``model_id``.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        logits = self.net(x)

        if not return_aux:
            return logits

        aux: Dict[str, Any] = {
            "model_id": self.MODEL_ID,
            # Optional metadata (cheap, sometimes useful)
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
        }
        return logits, aux
