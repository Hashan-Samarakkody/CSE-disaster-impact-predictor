"""Shallow multi-task MLP with weighted joint loss."""

from __future__ import annotations

import torch
from torch import nn


class ShallowMultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_aspi = nn.Linear(hidden_dim, 1)
        self.head_volume = nn.Linear(hidden_dim, 1)
        self.head_recovery = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        y1 = self.head_aspi(h)
        y2 = self.head_volume(h)
        y3 = self.head_recovery(h)
        return torch.cat([y1, y2, y3], dim=1)


class WeightedMultiTaskMSELoss(nn.Module):
    def __init__(self, weights: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_target = ((preds - targets) ** 2).mean(dim=0)
        return (per_target * self.weights).sum()
