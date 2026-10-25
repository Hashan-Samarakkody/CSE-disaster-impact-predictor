"""Build minimum_detectable_effect.parquet — prospective power/MDE calculation."""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.artifact_store import artifact_file, save_frame


def main():
    with open(artifact_file("splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    with open(artifact_file("results_regression.pkl"), "rb") as f:
        results = pickle.load(f)

    alpha = 0.05
    power = 0.80
    z_alpha = stats.norm.ppf(1 - alpha / 2)   # 1.96
    z_power = stats.norm.ppf(power)           # 0.842

    # n_held_out
    n_held_out_total = sum(len(s.test_index) for s in splits)  # 40

    target_map = {
        "Y1_ASPI_5D_Forward_LogReturn_Pct": "Y1_5D",
        "Y2_5D_Forward_AbnormalVolume_LogRatio": "Y2_5D",
        "Y3_ASPI_Recovery_Time": "Y3",
        "Y1_ASPI_10D_Forward_LogReturn_Pct": "Y1_10D",
    }

    rows = []
    for target_col, target_label in target_map.items():
        if "naive_train_mean" not in results or target_col not in results["naive_train_mean"]:
            continue

        baseline_store = results["naive_train_mean"][target_col]
        yt_list = baseline_store["y_true"]
        yp_list = baseline_store["y_pred"]

        # Concatenate across folds
        y_true = np.concatenate(yt_list)
        y_pred_base = np.concatenate(yp_list)

        # Drop NaN pairs
        mask = np.isfinite(y_true) & np.isfinite(y_pred_base)
        y_true = y_true[mask]
        y_pred_base = y_pred_base[mask]

        n = len(y_true)
        if n == 0:
            continue

        # Residuals from baseline
        residuals = y_true - y_pred_base
        sd_residual = float(np.std(residuals, ddof=1))

        # MDE for paired t-test on RMSE differences
        # sd_delta is the std of per-observation squared-error differences
        # For a paired test: mde = (z_alpha/2 + z_power) * sd_delta / sqrt(n)
        # sd_delta ≈ sqrt(2) * sd_residual (approximation for paired squared errors)
        # Full formula: mde_rmse = sqrt(2) * sd_residual * (z_alpha/2 + z_power) / sqrt(n)
        mde_rmse = float(np.sqrt(2) * sd_residual * (z_alpha + z_power) / np.sqrt(n))

        # MDE in R² units: R² = 1 - RMSE²/Var(y_true)
        var_y = float(np.var(y_true, ddof=1))
        if var_y > 0:
            mde_r2 = float(1 - (sd_residual - mde_rmse / 2) ** 2 / var_y - (1 - sd_residual ** 2 / var_y))
            # Simpler: delta_r2 corresponding to delta_rmse
            rmse_base = float(np.sqrt(np.mean(residuals ** 2)))
            rmse_alt = max(0.0, rmse_base - mde_rmse)
            r2_base = 1 - rmse_base ** 2 / var_y
            r2_alt = 1 - rmse_alt ** 2 / var_y
            mde_r2 = float(r2_alt - r2_base)
        else:
            mde_r2 = float("nan")

        rows.append({
            "target": target_label,
            "n_held_out": n,
            "sd_residual": sd_residual,
            "alpha": alpha,
            "power": power,
            "mde_rmse": mde_rmse,
            "mde_r2": mde_r2,
            "dependence_assumption": (
                "episode-clustered, effective n reduced by cluster structure"
            ),
        })

    df = pd.DataFrame(rows)
    save_frame(df, "minimum_detectable_effect",
               note="Prospective MDE only. Computed from naive_train_mean baseline residuals. "
                    "Formula: sqrt(2) * sd_residual * (z_alpha/2 + z_power) / sqrt(n). "
                    "Observed power not computed.")


if __name__ == "__main__":
    main()
