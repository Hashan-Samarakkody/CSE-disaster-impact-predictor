"""Patch verdict_table.parquet: reclassify MLP as confirmatory (T8 pre-registration).

The notebook 06_evaluation.ipynb omitted mlp from CONFIRMATORY_MODELS; the grid
script run_aspi_return_grid.py includes it (T8, pre-registered in docs/audit.md 8.4).
This script applies the correction and reapplies per-target Holm so analysis_map and
downstream artifacts see a consistent tier for MLP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifact_store import artifact_file, save_frame

vt = pd.read_parquet(artifact_file("verdict_table.parquet"))

before = int(vt["confirmatory"].sum())
vt.loc[vt["model"] == "mlp", "confirmatory"] = True
after = int(vt["confirmatory"].sum())
print(f"confirmatory rows: {before} → {after} (+{after-before} MLP rows now confirmatory)")

# Reapply per-target Holm (same logic as verification.py)
vt["p_holm"] = np.nan
for target, block in vt.groupby("target"):
    family = block["confirmatory"].to_numpy(bool)
    if not family.any():
        continue
    rows_idx = block.index[family]
    pvals = vt.loc[rows_idx, "p_model_worse"].fillna(1.0).to_numpy()
    _, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    vt.loc[rows_idx, "p_holm"] = p_holm

confirmatory = vt["confirmatory"].to_numpy(bool)
vt["holm_significant"] = (vt["p_holm"] < 0.05) & vt["boot_beats"] & confirmatory

save_frame(vt, "verdict_table",
           "Per-(model, target, baseline) verdict with per-target Holm. "
           "MLP reclassified as confirmatory (T8, pre-registered in docs/audit.md 8.4).")

print(f"holm_significant after patch: {int(vt['holm_significant'].sum())} rows")
print(vt[vt["model"] == "mlp"][["target", "model", "baseline", "confirmatory",
                                  "p_holm", "holm_significant"]].to_string())
