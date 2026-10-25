"""Build analysis_map.parquet — one row per comparison actually performed."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.artifact_store import artifact_file, save_frame


def main():
    vt = pd.read_parquet(artifact_file("verdict_table.parquet"))
    dm = pd.read_parquet(artifact_file("aspi_direction_metrics.parquet"))
    cs = pd.read_parquet(artifact_file("classification_summary.parquet"))

    rows = []

    # --- Regression rows from verdict_table ---
    for _, row in vt.iterrows():
        target = row["target"]

        if target == "Y1_ASPI_5D_Forward_LogReturn_Pct":
            target_role = "primary"
            info_set = "ex_post"
            k = 20
        elif target == "Y1_ASPI_10D_Forward_LogReturn_Pct":
            target_role = "secondary"
            info_set = "ex_post"
            k = 20
        elif target.startswith("Y2"):
            target_role = "primary"
            info_set = "ex_post"
            k = 20
        elif target.startswith("Y3"):
            target_role = "primary"
            info_set = "ex_post"
            k = 20
        else:
            target_role = "secondary"
            info_set = "ex_post"
            k = 20

        tier = "confirmatory" if row["confirmatory"] else "exploratory"
        # family_id: target + info_set (matches Holm grouping)
        family_id = f"{target}|{info_set}"

        rows.append({
            "target": target,
            "target_role": target_role,
            "model": row["model"],
            "tier": tier,
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
        family_id = f"{target_name}|real_time"
        rows.append({
            "target": target_name,
            "target_role": "secondary",
            "model": row["model"],
            "tier": "confirmatory" if row.get("confirmatory", False) else "exploratory",
            "info_set": "real_time",
            "k": 5,   # real_time feature set
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
        family_id = f"{label}|real_time"
        rows.append({
            "target": label,
            "target_role": "secondary",
            "model": row["model"],
            "tier": "exploratory",
            "info_set": "real_time",
            "k": 5,
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

    save_frame(df, "analysis_map",
               note="One row per comparison actually performed. Regression rows from "
                    "verdict_table; direction rows from aspi_direction_metrics; "
                    "classification rows from classification_summary.")


if __name__ == "__main__":
    main()
