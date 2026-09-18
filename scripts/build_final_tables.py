"""Protocol Part 7,  the required final Y1 / Y3 / summary tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifact_store import artifact_file
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "thesis_materials"
OUT.mkdir(parents=True, exist_ok=True)


def build_y1_table() -> pd.DataFrame:
    """One row per (horizon, info_set, K, model), carrying every column Part 7 asks for."""
    metrics = pd.read_parquet(artifact_file("aspi_grid_metrics.parquet"))
    verdicts = pd.read_parquet(artifact_file("aspi_grid_verdicts.parquet"))

    base = (metrics[metrics.info_set == "baseline"]
            .pivot(index="horizon", columns="model", values="rmse"))
    # `metrics` already carries a `skill_vs_zero`; drop it so the three skill columns
    # below are all computed the same way, from the same baseline RMSE table.
    rows = metrics[metrics.info_set != "baseline"].drop(columns=["skill_vs_zero"]).copy()
    for b in ("naive_zero", "naive_train_mean", "market_only_expected"):
        rows[f"skill_vs_{b}"] = 1.0 - rows["rmse"] / rows["horizon"].map(base[b])

    primary = verdicts[verdicts.baseline == "market_only_expected"].set_index(
        ["horizon", "info_set", "k", "model"])
    rows = rows.set_index(["horizon", "info_set", "k", "model"])
    rows["bootstrap_delta"] = primary["delta_rmse"]
    rows["CI_low"] = primary["ci_low"]
    rows["CI_high"] = primary["ci_high"]
    rows["Holm_p"] = primary["p_holm"]
    rows["significant_single_test"] = primary["boot_beats"]
    rows["significant_after_adjustment"] = primary["holm_significant"]
    rows["verdict"] = primary["verdict"]
    return rows.reset_index().rename(columns={
        "rmse": "RMSE", "mae": "MAE", "pooled_r2": "R2",
        "skill_vs_naive_zero": "skill_vs_zero",
        "skill_vs_naive_train_mean": "skill_vs_train_mean",
        "skill_vs_market_only_expected": "skill_vs_market_only",
    })[["horizon", "info_set", "k", "model", "n", "RMSE", "MAE", "R2",
        "skill_vs_zero", "skill_vs_train_mean", "skill_vs_market_only",
        "bootstrap_delta", "CI_low", "CI_high", "Holm_p",
        "significant_single_test", "significant_after_adjustment", "verdict"]]


def build_y3_table() -> pd.DataFrame:
    m = pd.read_parquet(artifact_file("recovery_grid_metrics.parquet"))
    cal = pd.read_parquet(artifact_file("recovery_probability_calibration.parquet"))
    worst_gap = (cal.assign(g=cal["calibration_gap"].abs())
                    .groupby("model")["g"].max().rename("max_abs_calibration_gap"))
    m = m.merge(worst_gap, left_on="model", right_index=True, how="left")
    m["C_index_CI"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi
                       in zip(m["c_index_ci_low"], m["c_index_ci_high"])]
    return m.rename(columns={
        "c_index": "C_index", "integrated_brier_score": "Integrated_Brier_Score",
        "median_abs_error_uncensored": "Median_abs_error_uncensored",
    })[["model", "n", "n_recovered", "n_censored", "C_index", "C_index_CI",
        "Integrated_Brier_Score", "Median_abs_error_uncensored", "mae_uncensored",
        "rmse_uncensored", "max_abs_calibration_gap", "p_holm", "holm_significant",
        "verdict"]]


def _md(df, floats=4):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floats)
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in d.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main():
    y1 = build_y1_table()
    y1.to_csv(OUT / "final_table_aspi.csv", index=False)
    y3 = build_y3_table()
    y3.to_csv(OUT / "final_table_recovery.csv", index=False)

    # Headline extract: the best configuration (lowest RMSE) per horizon x information set.
    best = (y1.sort_values("RMSE").groupby(["horizon", "info_set"], as_index=False).first()
              .sort_values(["horizon", "info_set"]))
    direction = pd.read_parquet(artifact_file("aspi_direction_metrics.parquet"))
    cats = pd.read_parquet(artifact_file("recovery_category_metrics.parquet")).sort_values(
        "roc_auc", ascending=False)

    text = [
        "## Y1 final table (best configuration per horizon x information set)\n",
        _md(best.drop(columns=["verdict"])),
        "\n\nFull 240-row grid: `docs/thesis_materials/final_table_aspi.csv`.\n",
        "\n## Y1 direction analysis (secondary, a different question)\n",
        _md(direction),
        "\n\n## Y3 final table\n",
        _md(y3),
        "\n\n## Y3 recovery category `recovery <= 20 trading days`\n",
        _md(cats),
    ]
    # The markdown is printed rather than written to a file: docs/results.md already
    # carries these tables inline, and a second copy on disk would drift from it.
    print(chr(10).join(text))
    print(f"wrote {OUT/'final_table_aspi.csv'} ({len(y1)} rows), "
          f"{OUT/'final_table_recovery.csv'} ({len(y3)} rows)")
    print(f"Y1 configurations significant on a single test: "
          f"{int(y1['significant_single_test'].sum())} / {len(y1)}")
    print(f"Y3 models with a C-index CI excluding 0.5: "
          f"{int((pd.read_parquet(artifact_file('recovery_grid_metrics.parquet'))['c_index_beats_chance']).sum())}")


if __name__ == "__main__":
    main()
