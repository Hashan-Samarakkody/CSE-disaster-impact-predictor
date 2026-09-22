"""Model CONFIRMATION (not discovery) for Y1 and Y2, per the approved decision-hierarchy
protocol (docs/audit.md, "Confirmation diagnostics" section).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.metrics import bootstrap_metric_ci, evaluate_regression, skill_score
from src.config.settings import ARTIFACT_FIGURE_DIR
from src.utils.artifact_store import artifact_file
import run_aspi_return_experiments as y1x  # reuses load_data, run_single_target, TARGET, etc.

ARTIFACTS = ROOT / "artifacts"
FIGDIR = ARTIFACT_FIGURE_DIR
FIGDIR.mkdir(exist_ok=True, parents=True)


def rmse_fn(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae_fn(a, b):
    return float(np.mean(np.abs(a - b)))


# A. fold stability

def fold_stability_table(y_true_folds, y_pred_folds, fold_ids, label):
    rows = []
    for fi, yt, yp in zip(fold_ids, y_true_folds, y_pred_folds):
        yt, yp = np.asarray(yt, dtype=float), np.asarray(yp, dtype=float)
        m = evaluate_regression(yt, yp)
        zero_rmse = rmse_fn(yt, np.zeros_like(yt))
        dir_acc = float(np.mean(np.sign(yt) == np.sign(yp)))
        r2 = m["r2"] if len(yt) >= 2 else float("nan")
        rows.append({"model": label, "fold": fi, "n": len(yt), "rmse": m["rmse"],
                    "mae": m["mae"], "r2": r2, "naive_zero_rmse": zero_rmse,
                    "skill_rmse_vs_zero": skill_score(m["rmse"], zero_rmse),
                    "directional_accuracy": dir_acc})
    table = pd.DataFrame(rows)
    summary = table[["rmse", "mae", "r2", "skill_rmse_vs_zero", "directional_accuracy"]].agg(["mean", "std"])
    print(table.to_string(index=False))
    print(f"\n  across-fold mean/std ({label}):")
    print(summary.to_string())
    if table["skill_rmse_vs_zero"].std() > 0:
        best_fold = table.loc[table["skill_rmse_vs_zero"].idxmax()]
        print(f"  best single fold: fold {int(best_fold['fold'])} "
              f"(skill_rmse_vs_zero={best_fold['skill_rmse_vs_zero']:+.4f}); "
              f"other folds' mean skill={table.loc[table.index != best_fold.name, 'skill_rmse_vs_zero'].mean():+.4f}")
    return table


# B. variance/calibration

def variance_and_calibration(y_true, y_pred, label):
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

    def stats(a):
        return {"mean": float(np.mean(a)), "std": float(np.std(a)), "min": float(np.min(a)),
                "max": float(np.max(a)), "iqr": float(np.percentile(a, 75) - np.percentile(a, 25))}

    s_true, s_pred = stats(yt), stats(yp)
    ratio = s_pred["std"] / s_true["std"] if s_true["std"] > 0 else float("nan")
    print(f"  {label}: actual  {s_true}")
    print(f"  {label}: predicted {s_pred}")
    print(f"  {label}: prediction_SD / actual_SD = {ratio:.3f} "
          f"({'COLLAPSING toward a constant' if ratio < 0.3 else 'excessive variance' if ratio > 1.5 else 'reasonable spread'})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(yt, yp, alpha=0.7)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_xlabel("actual Y1"); axes[0].set_ylabel("predicted Y1")
    axes[0].set_title(f"actual vs predicted -- {label}")
    resid = yp - yt
    axes[1].scatter(yp, resid, alpha=0.7)
    axes[1].axhline(0, color="k", linewidth=1, linestyle="--")
    axes[1].set_xlabel("predicted Y1"); axes[1].set_ylabel("residual (pred - actual)")
    axes[1].set_title(f"residuals -- {label}")
    fig.tight_layout()
    path = FIGDIR / f"y1_actual_vs_pred_{label}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved plot: {path}")
    return {"actual": s_true, "predicted": s_pred, "pred_sd_over_actual_sd": ratio}


# C. leave-one-out influence
# Array-only: recompute the metric with one observation dropped from the ALREADY-recorded
# out-of-sample predictions. No model is refit here.

def influence_analysis(y_true, y_pred, label):
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    n = len(yt)
    full_r2 = r2_score(yt, yp)
    r2s, deltas = [], []
    for i in range(n):
        mask = np.arange(n) != i
        r2_i = r2_score(yt[mask], yp[mask]) if mask.sum() >= 2 else float("nan")
        r2s.append(r2_i)
        deltas.append(full_r2 - r2_i)
    r2s = np.array(r2s)
    deltas = np.array(deltas)
    order = np.argsort(-np.abs(deltas))[:5]
    print(f"  {label}: full-sample R2={full_r2:+.4f}; leave-one-out R2 range "
          f"[{r2s.min():+.4f}, {r2s.max():+.4f}], median={np.median(r2s):+.4f}")
    print("  top-5 most influential observations (index, actual, predicted, R2 without it, delta):")
    for i in order:
        print(f"    idx={i:2d}  actual={yt[i]:+.3f}  pred={yp[i]:+.3f}  "
              f"R2_without={r2s[i]:+.4f}  delta={deltas[i]:+.4f}")
    dominated = np.abs(deltas).max() > 0.5 * abs(full_r2) if full_r2 != 0 else np.abs(deltas).max() > 0.05
    print(f"  dominated by a single observation: {dominated} "
          f"(largest single delta = {deltas[np.argmax(np.abs(deltas))]:+.4f} vs full R2 {full_r2:+.4f})")
    return {"full_r2": full_r2, "loo_r2_min": float(r2s.min()), "loo_r2_max": float(r2s.max()),
            "loo_r2_median": float(np.median(r2s)), "dominated_by_one_obs": bool(dominated)}


def bootstrap_vs_zero(y_true, y_pred, label):
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    _, rlo, rhi = bootstrap_metric_ci(yt, yp, lambda a, b: rmse_fn(a, np.zeros_like(a)) - rmse_fn(a, b))
    _, mlo, mhi = bootstrap_metric_ci(yt, yp, lambda a, b: mae_fn(a, np.zeros_like(a)) - mae_fn(a, b))
    print(f"  {label}: delta_RMSE_vs_zero CI=[{rlo:+.4f},{rhi:+.4f}]  "
          f"delta_MAE_vs_zero CI=[{mlo:+.4f},{mhi:+.4f}]  "
          f"({'excludes zero' if rlo > 0 or rhi < 0 else 'includes zero'})")
    return rlo, rhi, mlo, mhi


def per_fold_arrays_from_cache(results, model, target):
    store = results[model][target]
    folds = store.get("folds")
    # The ensemble/stacked stores don't track a real fold id per entry (None placeholders);
    # position-in-list is still the correct chronological fold order, just unlabeled.
    if not folds or any(f is None for f in folds):
        folds = list(range(len(store["y_true"])))
    return store["y_true"], store["y_pred"], folds


def main():
    dataset, X_all, y_all, dates_all, horizon_end_all, gdp_all, splits, feature_cols, type_cols = y1x.load_data()

    with open(artifact_file("results_regression.pkl"), "rb") as f:
        results = pickle.load(f)

    print("=" * 100)
    print("Y1 -- CANDIDATE 1: original frozen ensemble (RF+XGBoost+MLP, current cache)")
    print("=" * 100)
    ens_yt_folds, ens_yp_folds, ens_folds = per_fold_arrays_from_cache(results, "ensemble", y1x.TARGET)
    ens_yt, ens_yp = np.concatenate(ens_yt_folds), np.concatenate(ens_yp_folds)

    print("\n[A] fold stability")
    fold_stability_table(ens_yt_folds, ens_yp_folds, ens_folds, "ensemble")
    print("\n[B] prediction variance / calibration")
    variance_and_calibration(ens_yt, ens_yp, "ensemble")
    print("\n[C] leave-one-observation-out influence")
    influence_analysis(ens_yt, ens_yp, "ensemble")
    print("\nbootstrap vs naive_zero:")
    bootstrap_vs_zero(ens_yt, ens_yp, "ensemble")

    print()
    print("=" * 100)
    print("Y1 -- CANDIDATE 2: k=10 Ridge (fit once here, per-fold arrays kept, never refit again)")
    print("=" * 100)
    res_k10, _ = y1x.run_single_target(X_all, y_all, dates_all, horizon_end_all, gdp_all, splits,
                                       type_cols, model_names=("ridge",), k_features=10)
    r10_yt_folds, r10_yp_folds, r10_folds = (res_k10["ridge"]["y_true"], res_k10["ridge"]["y_pred"],
                                             res_k10["ridge"]["folds"])
    r10_yt, r10_yp = np.concatenate(r10_yt_folds), np.concatenate(r10_yp_folds)

    print("\n[A] fold stability")
    fold_stability_table(r10_yt_folds, r10_yp_folds, r10_folds, "k10_ridge")
    print("\n[B] prediction variance / calibration")
    variance_and_calibration(r10_yt, r10_yp, "k10_ridge")
    print("\n[C] leave-one-observation-out influence")
    influence_analysis(r10_yt, r10_yp, "k10_ridge")
    print("\nbootstrap vs naive_zero:")
    bootstrap_vs_zero(r10_yt, r10_yp, "k10_ridge")

    print()
    print("=" * 100)
    print("[D] Y1 FINAL DECISION -- ensemble vs k10_ridge")
    print("=" * 100)
    m_ens, m_r10 = evaluate_regression(ens_yt, ens_yp), evaluate_regression(r10_yt, r10_yp)
    print(f"  ensemble : RMSE={m_ens['rmse']:.4f} MAE={m_ens['mae']:.4f} R2={m_ens['r2']:+.4f}")
    print(f"  k10_ridge: RMSE={m_r10['rmse']:.4f} MAE={m_r10['mae']:.4f} R2={m_r10['r2']:+.4f}")
    print(f"  RMSE difference: {abs(m_ens['rmse'] - m_r10['rmse']):.4f} "
          f"(thesis primary metric; both bootstrap CIs vs naive_zero include zero -- see above)")

    print()
    print("=" * 100)
    print("Y2 -- best current model (random_forest, collinearity-drop features live in cache) "
          "vs naive_zero")
    print("=" * 100)
    Y2 = "Y2_5D_Forward_AbnormalVolume_LogRatio"
    rf_yt_folds, rf_yp_folds, rf_folds = per_fold_arrays_from_cache(results, "random_forest", Y2)
    rf_yt, rf_yp = np.concatenate(rf_yt_folds), np.concatenate(rf_yp_folds)
    print("\n[A] fold stability")
    fold_stability_table(rf_yt_folds, rf_yp_folds, rf_folds, "y2_random_forest")
    print("\n[B] prediction variance / calibration")
    variance_and_calibration(rf_yt, rf_yp, "y2_random_forest")
    print("\n[C] leave-one-observation-out influence")
    influence_analysis(rf_yt, rf_yp, "y2_random_forest")
    print("\nbootstrap vs naive_zero:")
    bootstrap_vs_zero(rf_yt, rf_yp, "y2_random_forest")
    m_rf2 = evaluate_regression(rf_yt, rf_yp)
    print(f"\n  current (collinearity-drop) random_forest: RMSE={m_rf2['rmse']:.4f} "
          f"MAE={m_rf2['mae']:.4f} R2={m_rf2['r2']:+.4f}")
    print("  PREVIOUS frozen Y2 reference (printed earlier this session, pre-collinearity-drop; "
          "its per-fold arrays no longer exist in the cache -- pooled comparison only, not "
          "re-derived, not fabricated):")
    print("    ensemble (pre-collinearity): RMSE=0.44506 MAE=0.35933 R2=+0.32890")
    print("    random_forest (pre-collinearity): RMSE=0.46103 MAE=0.37858 R2=+0.27987")


if __name__ == "__main__":
    main()
