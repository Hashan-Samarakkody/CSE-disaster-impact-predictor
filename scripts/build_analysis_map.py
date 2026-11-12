"""Build analysis_map.parquet — one row per comparison actually performed."""

from __future__ import annotations

import pandas as pd

from src.utils.artifact_store import artifact_file, save_frame


def build_analysis_map(vt: pd.DataFrame, grid: pd.DataFrame, dm: pd.DataFrame,
                       cs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # --- Primary, pre-declared information-set comparison ---
    for _, row in grid.iterrows():
        target = f"Y1_ASPI_{row['horizon']}D_Forward_LogReturn_Pct"
        family_id = f"{target}|horizon={row['horizon']}|k={row['k']}"
        rows.append({
            "target": target,
            "target_role": "primary",
            "model": row["model"],
            "tier": "confirmatory" if row["confirmatory"] else "exploratory",
            "info_set": row["info_set"],
            "k": row["k"],
            "benchmark": row["baseline"],
            "n": row["n"],
            "family_id": family_id,
            "p_raw": row["p_one_sided"],
            "p_holm": row["p_holm"],
            "holm_significant": row["holm_significant"],
            "source_artifact": "aspi_grid_verdicts.parquet",
        })

    # --- Older full-feature comparisons, retained as an exploratory inventory ---
    for _, row in vt.iterrows():
        target = row["target"]

        if target == "Y1_ASPI_5D_Forward_LogReturn_Pct":
            target_role = "primary"
        elif target == "Y1_ASPI_10D_Forward_LogReturn_Pct":
            target_role = "secondary"
        elif target.startswith("Y2"):
            target_role = "primary"
        elif target.startswith("Y3"):
            target_role = "primary"
        else:
            target_role = "secondary"

        info_set = "combined"
        k = 20
        family_id = f"{target}|{info_set}|k={k}|secondary_inventory"

        rows.append({
            "target": target,
            "target_role": target_role,
            "model": row["model"],
            "tier": "exploratory",
            "info_set": info_set,
            "k": k,
            "benchmark": row["baseline"],
            "n": row["n"],
            "family_id": family_id,
            "p_raw": row["p_model_worse"],
            "p_holm": row["p_holm"],
            "holm_significant": row["holm_significant"],
            "source_artifact": "verdict_table.parquet",
        })

    # --- Direction/classification rows from aspi_direction_metrics ---
    for _, row in dm.iterrows():
        target_name = f"direction_h{row['horizon']}"
        family_id = "direction|combined|k=10"
        rows.append({
            "target": target_name,
            "target_role": "secondary",
            "model": row["model"],
            "tier": "exploratory",
            "info_set": "combined",
            "k": 10,
            "benchmark": "majority_baseline",
            "n": row["n"],
            "family_id": family_id,
            "p_raw": row["p_one_sided"],
            "p_holm": row["p_holm"],
            "holm_significant": row["holm_significant"],
            "source_artifact": "aspi_direction_metrics.parquet",
        })

    # --- Classification rows from classification_summary ---
    # Exclude majority_baseline which is the benchmark, not a model
    model_cs = cs[cs["model"] != "majority_baseline"].copy()
    for _, row in model_cs.iterrows():
        label = row["label"]
        family_id = f"{label}|combined|k=20"
        rows.append({
            "target": label,
            "target_role": "secondary",
            "model": row["model"],
            "tier": "exploratory",
            "info_set": "combined",
            "k": 20,
            "benchmark": "majority_baseline",
            "n": row["n"],
            "family_id": family_id,
            "p_raw": float("nan"),
            "p_holm": float("nan"),
            "holm_significant": False,
            "source_artifact": "classification_summary.parquet",
        })

    df = pd.DataFrame(rows)

    # Add family_size
    family_sizes = df.groupby("family_id")["family_id"].transform("count")
    df["family_size"] = family_sizes

    # Reorder columns
    df = df[[
        "target", "target_role", "model", "tier", "info_set", "k",
        "benchmark", "n", "family_id", "family_size",
        "p_raw", "p_holm", "holm_significant", "source_artifact",
    ]]

    return df


def main():
    vt = pd.read_parquet(artifact_file("verdict_table.parquet"))
    grid = pd.read_parquet(artifact_file("aspi_grid_verdicts.parquet"))
    dm = pd.read_parquet(artifact_file("aspi_direction_metrics.parquet"))
    cs = pd.read_parquet(artifact_file("classification_summary.parquet"))

    df = build_analysis_map(vt, grid, dm, cs)
    save_frame(df, "analysis_map",
               note="The pre-declared information-set comparison is primary and sourced "
                    "from aspi_grid_verdicts. Older full-feature results, direction models "
                    "(combined features, k=10), and classification models (combined "
                    "features, k=20) are exploratory inventories.")


if __name__ == "__main__":
    main()
