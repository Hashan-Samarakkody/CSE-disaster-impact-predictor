"""Shallow multi-task MLP with weighted joint loss."""

from __future__ import annotations

import torch
from torch import nn


class ShallowMultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5,
                 n_targets: int = 3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # One head per target, sized at construction. Hardcoded heads meant a fourth target
        # raised "size of tensor a (3) must match tensor b (5)" deep in the training loop.
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


class WeightedMultiTaskMSELoss(nn.Module):
    def __init__(self, weights) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_target = ((preds - targets) ** 2).mean(dim=0)
        return (per_target * self.weights).sum()
