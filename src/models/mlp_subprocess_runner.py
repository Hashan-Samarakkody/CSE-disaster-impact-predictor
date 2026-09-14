"""Trains `ShallowMultiTaskMLP` in a fresh process and writes predictions to disk.

On Windows, `import torch` after scikit-learn/XGBoost/SHAP in the same process raises
`OSError: [WinError 1114] ... c10.dll`; importing it first in a clean process never does.

Usage: `python -m src.models.mlp_subprocess_runner <input_npz> <output_npz>`, where the
input holds already-scaled `X_train`, `y_train`, `X_test` and `pred` is written back."""

from __future__ import annotations

import sys

import numpy as np


def main(input_path: str, output_path: str) -> None:
    import torch

    from src.models.shallow_mlp import ShallowMultiTaskMLP, WeightedMultiTaskMSELoss

    data = np.load(input_path)
    X_train, y_train, X_test = data["X_train"], data["y_train"], data["X_test"]

    torch.manual_seed(42)
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    Xte_t = torch.tensor(X_test, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)

    n_targets = y_train.shape[1]
    model = ShallowMultiTaskMLP(input_dim=X_train.shape[1], hidden_dim=64, dropout=0.5,
                                n_targets=n_targets)
    # Y1/Y2/Y3 keep their original relative weights; any further target (the cumulative
    # event-window returns added 2026-09-12) is weighted like Y1, since CAR is the same
    # quantity measured over a longer window.
    base_weights = (1.0, 0.1, 0.5)
    weights = tuple(base_weights[i] if i < len(base_weights) else 1.0
                    for i in range(n_targets))
    loss_fn = WeightedMultiTaskMSELoss(weights=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(200):
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(Xte_t).numpy()

    np.savez(output_path, pred=pred_test)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
