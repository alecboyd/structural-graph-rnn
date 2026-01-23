from __future__ import annotations

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

        layers: list[nn.Module] = []

        if num_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, num_classes, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU(inplace=False))

            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(nn.ReLU(inplace=False))

            layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))

        self.net = nn.Sequential(*layers)

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.bias = bool(bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
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
