"""Build missingness_class_table.parquet — manuscript Table 13.

For each feature in the dataset, reports: n_observed, pct_missing, block
(predictor block), availability_class (real_time / ex_post), and a composite
Class label used in the manuscript.

Correct Class labels for notable features:
  gdp_growth_pct, inflation_cpi_pct  → macro / ex_post
  garch_cond_vol                      → market_state / real_time
  vol_ratio_*, log_vol_change_1, etc. → market_state / real_time
  date_macro                          → excluded (join key, not a feature)
  event_name                          → excluded (identifier, not a feature)

event_name is never included; it is an event identifier, not a predictor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    fa = pd.read_parquet(artifact_file("feature_availability.parquet"))

    n_total = len(dataset)

    rows = []
    for _, fa_row in fa.iterrows():
        feat = fa_row["feature"]
        block = fa_row["block"]
        info_set = fa_row["info_set"]

        if feat not in dataset.columns:
            continue

        col = dataset[feat]
        n_observed = int(col.notna().sum())
        n_missing = n_total - n_observed
        pct_missing = float(n_missing / n_total * 100)

        # Composite class label for the manuscript table
        class_label = f"{block} / {info_set}"

        rows.append({
            "feature": feat,
            "block": block,
            "availability_class": info_set,
            "class_label": class_label,
            "n_total": n_total,
            "n_observed": n_observed,
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 1),
        })

    df = pd.DataFrame(rows).sort_values(["block", "availability_class", "feature"])
    df = df.reset_index(drop=True)

    # Verify the specific cases the examiner flagged
    _check(df, "gdp_growth_pct", "macro / ex_post")
    _check(df, "inflation_cpi_pct", "macro / ex_post")
    _check(df, "garch_cond_vol", "market_state / real_time")
    _check(df, "log_vol_change_1", "market_state / real_time")
    # date_macro and event_name must NOT appear
    for excluded in ("date_macro", "event_name"):
        assert excluded not in df["feature"].values, (
            f"{excluded} must not appear in the missingness table (not a feature)"
        )

    save_frame(
        df,
        "missingness_class_table",
        note=(
            "Per-feature missingness counts and class labels (manuscript Table 13). "
            "block and availability_class from feature_availability.parquet. "
            "date_macro and event_name excluded (not features). "
            "class_label = block / availability_class."
        ),
    )
    print(f"wrote {len(df)} rows to artifacts/tables/missingness_class_table.parquet")
    # Print sample to confirm labels
    for feat in ("gdp_growth_pct", "inflation_cpi_pct", "garch_cond_vol", "log_vol_change_1"):
        row = df[df["feature"] == feat]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {feat}: class='{r.class_label}', pct_missing={r.pct_missing}%")


def _check(df: pd.DataFrame, feat: str, expected_class: str) -> None:
    rows = df[df["feature"] == feat]
    if rows.empty:
        print(f"WARNING: {feat} not in dataset")
        return
    actual = rows.iloc[0]["class_label"]
    assert actual == expected_class, (
        f"{feat}: expected class '{expected_class}', got '{actual}'"
    )


if __name__ == "__main__":
    main()
