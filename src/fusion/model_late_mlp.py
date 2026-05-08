from __future__ import annotations

import torch
from torch import nn


class LateFusionMLP(nn.Module):
    def __init__(self, input_dim: int = 773, hidden1: int = 512, hidden2: int = 256, num_classes: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
