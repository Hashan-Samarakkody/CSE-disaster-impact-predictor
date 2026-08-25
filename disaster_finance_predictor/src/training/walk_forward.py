"""Chronological walk-forward validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class WalkForwardSplit:
    train_index: np.ndarray
    test_index: np.ndarray


def generate_walk_forward_splits(
    n_samples: int,
    train_window: int,
    test_window: int,
    step: int,
) -> Iterator[WalkForwardSplit]:
    """Yield rolling train/test windows for time-series cross-validation."""
    start = 0
    while start + train_window + test_window <= n_samples:
        train_idx = np.arange(start, start + train_window)
        test_idx = np.arange(start + train_window, start + train_window + test_window)
        yield WalkForwardSplit(train_idx, test_idx)
        start += step
