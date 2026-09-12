"""Standalone entry point that trains `ShallowMultiTaskMLP` in a fresh Python
process and writes predictions back to disk.

Why this exists: on the Windows machine this project was developed on,
`import torch` reliably raises ``OSError: [WinError 1114] ... c10.dll``
whenever it happens inside the *same* Python process that has already
imported scikit-learn/XGBoost/SHAP earlier (as the notebook's pipeline does)
-- but never when torch is imported first in a fresh process. This is a real,
reproducible environment issue (isolated and confirmed during this project's
own development, not assumed), not a code bug in this repo or in torch
itself. Running the MLP training in its own subprocess sidesteps it
entirely, at the cost of a small amount of IPC overhead per walk-forward
fold -- negligible given this project's N (a few dozen events per fold).

Usage: ``python -m src.models.mlp_subprocess_runner <input_npz> <output_npz>``
``input_npz`` must contain arrays ``X_train``, ``y_train``, ``X_test`` (already
scaled/imputed by the caller); writes ``pred`` to ``output_npz``.
"""

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
