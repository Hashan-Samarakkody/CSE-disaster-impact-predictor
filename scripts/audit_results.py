"""Full metric audit across every model and every target."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import evaluate_regression  # noqa: E402
from src.utils.artifact_store import artifact_file

ART = ROOT / "artifacts"
pd.set_option("display.width", 250)

# Pre-change baseline: N=64, 31 features, 3 folds, 30 pooled test points.
# Source: git show 85590a8:...notebooks/04_modeling_regression.ipynb executed output.
BASELINE_R2 = {
    ("ridge", "Y1_ASPI_5D_Forward_LogReturn_Pct"): -0.5338,
    ("ridge", "Y2_5D_Forward_AbnormalVolume_LogRatio"): 0.1307,
    ("ridge", "Y3_ASPI_Recovery_Time"): -0.1853,
    ("random_forest", "Y1_ASPI_5D_Forward_LogReturn_Pct"): -0.3151,
    ("random_forest", "Y2_5D_Forward_AbnormalVolume_LogRatio"): -0.1221,
    ("random_forest", "Y3_ASPI_Recovery_Time"): -0.1608,
    ("xgboost", "Y1_ASPI_5D_Forward_LogReturn_Pct"): -1.7660,
    ("xgboost", "Y2_5D_Forward_AbnormalVolume_LogRatio"): -0.4630,
    ("xgboost", "Y3_ASPI_Recovery_Time"): -0.1693,
}
# Source: git show 85590a8:...notebooks/05_modeling_classification.ipynb executed output.
BASELINE_C2 = pd.DataFrame([
    {"model": "logistic", "n": 30, "balanced_accuracy": 0.6986, "precision": 0.5294,
     "recall": 0.8182, "f1": 0.6429, "auc": 0.7943, "auc_boot_lo": 0.5933,
     "auc_boot_hi": 0.9569, "beats_baseline": True},
    {"model": "rf_clf", "n": 30, "balanced_accuracy": 0.7273, "precision": 1.0000,
     "recall": 0.4545, "f1": 0.6250, "auc": 0.7895, "auc_boot_lo": 0.5932,
     "auc_boot_hi": 0.9426, "beats_baseline": True},
    {"model": "xgb_clf", "n": 30, "balanced_accuracy": 0.5383, "precision": 0.5000,
     "recall": 0.1818, "f1": 0.2667, "auc": 0.7081, "auc_boot_lo": 0.4785,
     "auc_boot_hi": 0.9187, "beats_baseline": False},
])


def rule(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def full_metric_table():
    """Every model x target: RMSE, MAE, pooled R2, and skill against both nulls."""
    import pickle

    results = pickle.load(open(artifact_file("results_regression.pkl"), "rb"))
    rows = []
    for model, targets in results.items():
        for target, store in targets.items():
            if not store.get("y_true"):
                continue
            yt = np.concatenate(store["y_true"])
            yp = np.concatenate(store["y_pred"])
            m = evaluate_regression(yt, yp)
            zero_rmse = float(np.sqrt(np.mean(yt ** 2)))
            rows.append({
                "target": target, "model": model, "n": len(yt),
                "RMSE": m["rmse"], "MAE": m["mae"], "pooled_R2": m["r2"],
                "RMSE_null_zero": zero_rmse,
                "skill_vs_zero": 1.0 - m["rmse"] / zero_rmse if zero_rmse else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["target", "pooled_R2"], ascending=[True, False])


def main():
    rule("1. EVERY MODEL x EVERY TARGET -- regression metrics (pooled out-of-fold)")
    table = full_metric_table()
    for target in sorted(table.target.unique()):
        sub = table[table.target == target]
        print(f"\n--- {target} (n={int(sub.n.iloc[0])} pooled test points) ---")
        print(sub.drop(columns="target").round(5).to_string(index=False))

    rule("2. DID IT BEAT A NAIVE BASELINE? (paired event bootstrap + Diebold-Mariano)")
    v = pd.read_parquet(artifact_file("verdict_table.parquet"))
    print(v[["target", "model", "baseline", "n", "delta_rmse", "ci_low", "ci_high",
             "dm_p", "verdict"]].round(5).to_string(index=False))
    print(f"\n>>> comparisons whose CI excludes zero: {int(v.boot_beats.sum())} of {len(v)}")

    rule("3. CLASSIFICATION -- every label x every model")
    c = pd.read_parquet(artifact_file("classification_summary.parquet"))
    cols = ["label", "model", "n", "n_pos", "prevalence", "accuracy", "balanced_accuracy",
            "mcc", "precision", "recall", "f1", "pr_auc", "auc", "auc_boot_lo",
            "auc_boot_hi", "beats_baseline"]
    print(c[[x for x in cols if x in c.columns]].round(3).to_string(index=False))
    print(f"\n>>> model/label pairs clearing BOTH the majority rule and chance: "
          f"{int(c.beats_baseline.sum())}")

    rule("4. BEFORE vs AFTER -- pooled R2 against the pre-change baseline (commit 85590a8)")
    print("baseline: N=64, 31 features, 30 test points | now: N=74, 54 features, "
          "40 test points (34 for Y2)\n")
    rows = []
    for (model, target), old in BASELINE_R2.items():
        hit = table[(table.model == model) & (table.target == target)]
        if hit.empty:
            continue
        new = float(hit.pooled_R2.iloc[0])
        rows.append({"target": target, "model": model, "R2_before": old,
                     "R2_after": new, "change": new - old,
                     "direction": "better" if new > old else "worse"})
    cmp = pd.DataFrame(rows).sort_values(["target", "model"])
    print(cmp.round(4).to_string(index=False))
    print(f"\n>>> improved: {(cmp.direction == 'better').sum()} of {len(cmp)} model/target pairs")

    rule("5. DECOMPOSITION -- was it the extra events, or the extra features?")
    print("The two changes landed together. The ablation separates them: `no_external` is")
    print("N=74 with the ORIGINAL 31 features, so comparing it to the N=64 baseline")
    print("isolates the sample extension, and comparing it to `full` isolates the data.\n")
    ab = pd.read_parquet(artifact_file("ablation_blocks.parquet"))
    piv = ab.pivot_table(index=["target", "model"], columns="config", values="pooled_r2")
    dec = []
    for (target, model), row in piv.iterrows():
        key = (model, target)
        if key not in BASELINE_R2:
            continue
        dec.append({
            "target": target, "model": model,
            "A_before (N=64, 31f)": BASELINE_R2[key],
            "B_more_events (N=74, 31f)": row.get("no_external", np.nan),
            "C_plus_external (N=74, 54f)": row.get("full", np.nan),
            "events_effect": row.get("no_external", np.nan) - BASELINE_R2[key],
            "external_effect": row.get("full", np.nan) - row.get("no_external", np.nan),
        })
    d = pd.DataFrame(dec).sort_values(["target", "model"])
    print(d.round(4).to_string(index=False))
    print(f"\n>>> sample extension helped: {(d.events_effect > 0).sum()}/{len(d)}")
    print(f">>> external features helped: {(d.external_effect > 0).sum()}/{len(d)}")
    print("\nCAVEAT, and it matters for how much weight the two columns carry:")
    print("  `external_effect` (C - B) is a clean comparison -- identical events, identical")
    print("  folds, identical test points, only the feature set differs. It is the column")
    print("  the block ablation puts confidence intervals on.")
    print("  `events_effect` (B - A) is NOT clean. A and B are scored on DIFFERENT test")
    print("  sets (30 vs 40 points, different events), so an R2 difference there confounds")
    print("  'the model got better' with 'the test set got easier or harder'. R2 is")
    print("  normalised by the target's own variance, which changes with the sample.")
    print("  Read it as directional only; no CI is computed for it and none should be.")

    rule("6. WHICH EXTERNAL BLOCK EARNED ITS PLACE? (CI excluding zero)")
    contrib = pd.read_parquet(artifact_file("ablation_block_contrib.parquet"))
    sig = contrib[contrib.significant]
    print(contrib.pivot_table(index=["target", "block"], columns="model",
                              values="delta_rmse").round(5).to_string())
    print(f"\n>>> significant: {len(sig)} of {len(contrib)}")
    if len(sig):
        print(sig[["target", "model", "block", "delta_rmse", "lo", "hi"]]
              .round(5).to_string(index=False))

    rule("7. CLASSIFICATION BEFORE vs AFTER -- C2_volume_spike, the one that works")
    now = c[(c.label == "C2_volume_spike") & (c.model != "majority_baseline")]
    keep = ["model", "n", "balanced_accuracy", "precision", "recall", "f1", "auc",
            "auc_boot_lo", "auc_boot_hi", "beats_baseline"]
    print("BEFORE (N=64, 30 test points -- and 0 of them had missing volume):")
    print(BASELINE_C2[keep].round(3).to_string(index=False))
    print("\nAFTER (N=74, 34 test points; 6 events with no volume data now correctly dropped):")
    print(now[keep].round(3).to_string(index=False))

    rule("8. SECTOR PANEL")
    s = pd.read_parquet(artifact_file("sector_panel_verdict.parquet"))
    print(s.round(4).to_string(index=False))
    print(f"\n>>> sector comparisons beating their null: {int(s.significant.sum())} of {len(s)}")

    rule("9. Y3 HURDLE")
    print(pd.read_parquet(artifact_file("hurdle_table.parquet")).round(3).to_string(index=False))

    rule("10. DATA COVERAGE -- what the external sources actually bought")
    ds = pd.read_parquet(artifact_file("dataset.parquet"))
    ext = [c for c in ds.columns
           if c.startswith(("hz_", "di_", "fx_")) or c.startswith(("days_to_election",
                                                                   "election_within"))]
    print(f"events: {len(ds)} | features: {ds.shape[1]} | external features: {len(ext)}")
    print(f"external feature coverage: "
          f"{ds[ext].notna().mean().min() * 100:.1f}%-{ds[ext].notna().mean().max() * 100:.1f}%")
    print(f"financial_damage real (the variable they were added to replace): "
          f"{int((ds.financial_damage > 0).sum())}/{len(ds)}")
    print(f"DesInventar matched: {int(ds.di_available.sum())}/{len(ds)}")
    for t in ("Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_5D_Forward_AbnormalVolume_LogRatio", "Y3_ASPI_Recovery_Time"):
        print(f"  {t}: {int(ds[t].notna().sum())}/{len(ds)} observed")


if __name__ == "__main__":
    main()
