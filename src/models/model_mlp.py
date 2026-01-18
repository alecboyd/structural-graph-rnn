from __future__ import annotations

import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    Baseline feedforward classifier.

    Architecture:
      x -> [Linear -> ReLU] * (num_hidden_layers) -> Linear -> logits

    Notes:
      - No activation on the final layer (logits).
      - Use torch.nn.functional.cross_entropy(logits, y) for training (no softmax needed).
    """

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
            # Direct linear classifier
            layers.append(nn.Linear(input_dim, num_classes, bias=bias))
        else:
            # First hidden layer (because it has input_dim dimensions)
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU(inplace=False))

            # Additional hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(nn.ReLU(inplace=False))

            # Output layer (logits)
            layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        returns logits: [B, num_classes]
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)