"""Episode-level permutation test for ASPI direction classifiers.

For each of the 10 (horizon, model) combinations, runs B permutations where
y_true labels are shuffled at the episode level (preserving within-episode
assignment) and the pooled ROC AUC is recomputed. The p-value is

    p = (1 + count(perm_auc >= observed_auc)) / (1 + B)

The primary comparison is horizon=10, model=logistic (the pre-declared
confirmatory classifier). All 10 comparisons are saved so Holm correction
can be applied.

Output: artifacts/tables/aspi_direction_permutation.parquet

Usage (repo root):
    python scripts/run_direction_permutation_test.py          # B=5000
    python scripts/run_direction_permutation_test.py --b 1000 # quick test
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame


def _pooled_auc(df: pd.DataFrame) -> float:
    """ROC AUC on the concatenated (y_true, y_prob) across all folds."""
    return float(roc_auc_score(df["y_true"], df["y_prob"]))


def _permuted_auc(df: pd.DataFrame, episode_ids: np.ndarray,
                  rng: np.random.Generator) -> float:
    """
    Episode-level permutation: shuffle the y_true values across episodes
    (keeping within-episode label assignments intact) and recompute AUC.

    Within each episode, every row gets the same permuted label drawn from
    the original label for that episode position.  Here episodes are
    individual events (each event appears exactly once in the test set), so
    this simplifies to permuting the y_true array by episode.
    """
    unique_eps = np.unique(episode_ids)
    # Map each episode to its rows
    ep_to_rows: dict[int, list[int]] = {}
    for idx, ep in enumerate(episode_ids):
        ep_to_rows.setdefault(int(ep), []).append(idx)

    # Build the original y_true per episode (single value per episode here)
    ep_labels = {ep: df["y_true"].iloc[rows[0]] for ep, rows in ep_to_rows.items()}

    # Permute: shuffle episode label assignments
    eps_arr = np.array(unique_eps)
    perm_labels = rng.permutation(np.array([ep_labels[e] for e in eps_arr]))
    ep_perm = dict(zip(eps_arr, perm_labels))

    # Assign permuted labels back
    perm_y_true = np.array([ep_perm[int(e)] for e in episode_ids])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(roc_auc_score(perm_y_true, df["y_prob"].to_numpy()))
        except ValueError:
            # All one class after permutation — return 0.5 (chance)
            return 0.5


def _run_one(group_key: tuple, group: pd.DataFrame, B: int, seed: int) -> dict:
    horizon, model = group_key
    obs_auc = _pooled_auc(group)
    episode_ids = group["episode_id"].to_numpy()

    rng = np.random.default_rng(seed)
    perm_aucs = np.array(
        Parallel(n_jobs=-1, backend="loky")(
            delayed(lambda s=rng.integers(0, 2**31): _permuted_auc(
                group, episode_ids, np.random.default_rng(s)
            ))()
            for _ in range(B)
        )
    )

    p_value = float((1 + np.sum(perm_aucs >= obs_auc)) / (1 + B))
    return {
        "horizon": int(horizon),
        "model": str(model),
        "n_test": len(group),
        "n_permutations": B,
        "observed_auc": obs_auc,
        "perm_auc_mean": float(perm_aucs.mean()),
        "perm_auc_sd": float(perm_aucs.std()),
        "perm_auc_p95": float(np.percentile(perm_aucs, 95)),
        "p_value": p_value,
        "primary": bool(horizon == 10 and model == "logistic"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=5000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    B = args.b
    seed = args.seed

    preds = pd.read_parquet(artifact_file("aspi_direction_predictions.parquet"))

    rows = []
    groups = list(preds.groupby(["horizon", "model"]))
    print(f"Running {B} permutations for {len(groups)} (horizon, model) combinations …")
    for key, group in sorted(groups):
        horizon, model = key
        print(f"  horizon={horizon}, model={model}, n={len(group)} ", end="", flush=True)
        result = _run_one(key, group.reset_index(drop=True), B, seed + hash(key) % 10000)
        marker = " <-- PRIMARY" if result["primary"] else ""
        print(f"AUC={result['observed_auc']:.4f}  p={result['p_value']:.4f}{marker}")
        rows.append(result)

    out = pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    save_frame(out, "aspi_direction_permutation",
               note=(
                   f"Episode-level permutation test, B={B}, seed={seed}. "
                   "y_true permuted at episode level (one event per episode). "
                   "p = (1 + count(perm_auc >= obs_auc)) / (1 + B). "
                   "primary=True marks the pre-declared horizon=10 / logistic comparison."
               ))
    print(f"\nSaved: artifacts/tables/aspi_direction_permutation.parquet")

    primary = out[out["primary"]]
    if not primary.empty:
        r = primary.iloc[0]
        print(f"\nPrimary (h=10, logistic): AUC={r.observed_auc:.4f}  "
              f"p={r.p_value:.4f}  B={B}")


if __name__ == "__main__":
    main()
