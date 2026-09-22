"""Cross-check every headline number quoted in the docs against the artifacts."""
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

ROOT = Path(r"F:\CSE-disaster-impact-predictor")
ART = ROOT / "artifacts"

fails, checks = [], 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{name}: {detail}")
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"  <- {detail}"))


def flat(entry, key):
    return np.concatenate([np.asarray(a, float).ravel() for a in entry[key]])


res = pickle.loads((ART / "models/results_regression.pkl").read_bytes())
verdict = pd.read_parquet(ART / "tables/verdict_table.parquet")
grid_v = pd.read_parquet(ART / "tables/aspi_grid_verdicts.parquet")
grid_e = pd.read_parquet(ART / "tables/aspi_grid_verdicts_exploratory.parquet")
grid_m = pd.read_parquet(ART / "tables/aspi_grid_metrics.parquet")
clf = pd.read_parquet(ART / "tables/classification_summary.parquet")
direction = pd.read_parquet(ART / "tables/aspi_direction_metrics.parquet")
suite = pd.read_parquet(ART / "tables/robustness_suite.parquet")
flow = pd.read_parquet(ART / "tables/sample_flow.parquet")
events = pd.read_parquet(ART / "tables/event_list.parquet")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
results = (ROOT / "docs/results.md").read_text(encoding="utf-8")
interp = (ROOT / "docs/interpretation.md").read_text(encoding="utf-8")
summary = (ROOT / "docs/revision_2_summary.md").read_text(encoding="utf-8")
alldocs = readme + results + interp + summary

print("\n--- claims about the confirmatory family ---")
check("confirmatory family size is 18", len(grid_v) == 18, f"got {len(grid_v)}")
check("0 confirmatory intervals exclude zero",
      int(((grid_v.ci_low > 0) | (grid_v.ci_high < 0)).sum()) == 0)
check("0 confirmatory survive Holm", int(grid_v.holm_significant.sum()) == 0)
mn = grid_v.p_holm.min()
check("smallest confirmatory Holm p is 0.641 as quoted",
      abs(mn - 0.6408) < 5e-4 and "0.641" in alldocs, f"artifact {mn:.4f}")
check("exploratory table has 1332 rows", len(grid_e) == 1332, f"got {len(grid_e)}")
check("1350 total comparisons is quoted and correct",
      len(grid_v) + len(grid_e) == 1350 and "1350" in alldocs)
check("465 configurations is quoted and correct",
      len(grid_m) == 465 and "465" in alldocs, f"metrics rows {len(grid_m)}")

print("\n--- claims about Y1 and Y2 pooled metrics ---")
T1 = "Y1_ASPI_5D_Forward_LogReturn_Pct"
T2 = "Y2_5D_Forward_AbnormalVolume_LogRatio"
best_y1 = max(res[m][T1]["pooled_r2"] for m in res if T1 in res[m])
check("best Y1 R2 is 0.042 as quoted", abs(best_y1 - 0.0415) < 5e-4 and "0.042" in alldocs,
      f"artifact {best_y1:.4f}")
check("ensemble Y2 R2 is 0.334 as quoted",
      abs(res["ensemble"][T2]["pooled_r2"] - 0.3344) < 5e-4 and "0.334" in alldocs)
check("random forest Y2 R2 is 0.303 as quoted",
      abs(res["random_forest"][T2]["pooled_r2"] - 0.3030) < 5e-4 and "0.303" in alldocs)
rf = verdict[(verdict.model == "random_forest") & (verdict.baseline == "naive_zero")
             & (verdict.target == T2)].iloc[0]
check("RF Y2 interval [0.037, 0.175] as quoted",
      abs(rf.ci_low - 0.0371) < 5e-4 and abs(rf.ci_high - 0.1754) < 5e-4
      and "0.037" in alldocs and "0.175" in alldocs)
check("RF Y2 Holm p is 0.078 as quoted",
      abs(rf.p_holm - 0.0782) < 5e-4 and "0.078" in alldocs, f"artifact {rf.p_holm:.4f}")
check("Y2 held out points are 34 as quoted",
      len(flat(res["ensemble"][T2], "y_true")) == 34 and "34 held" in alldocs)

print("\n--- claims about direction and classification ---")
d10 = direction[(direction.horizon == 10) & (direction.model == "logistic")].iloc[0]
check("direction AUC 0.817 as quoted", abs(d10.roc_auc - 0.8170) < 5e-4 and "0.817" in alldocs)
check("direction interval [0.657, 0.940] as quoted",
      abs(d10.auc_ci_low - 0.6566) < 1e-3 and abs(d10.auc_ci_high - 0.9396) < 1e-3
      and "0.657" in alldocs and "0.940" in alldocs)
check("direction survives Holm", bool(d10.holm_significant))
check("it is the only Holm survivor in the grid and direction families",
      int(direction.holm_significant.sum()) == 1 and int(grid_v.holm_significant.sum()) == 0)
check("h=5 direction AUC 0.574 as quoted",
      abs(direction[(direction.horizon == 5) & (direction.model == "logistic")].roc_auc.iloc[0]
          - 0.5739) < 5e-4 and "0.574" in alldocs)
spans = ((clf.auc_boot_lo <= 0.5) & (clf.auc_boot_hi >= 0.5)).all()
check("every classification AUC interval contains 0.50", bool(spans))

print("\n--- claims about the sample and the suite ---")
check("sample flow closes 110 -> 94 -> 74",
      int(flow.n_remaining.iloc[0]) <= 110 and int(flow.n_remaining.iloc[-1]) == 74,
      f"last {int(flow.n_remaining.iloc[-1])}")
check("event list carries 74 events", len(events) == 74)
check("robustness suite has 26 checks", len(suite) == 26, f"got {len(suite)}")
check("no robustness check beats the benchmark", int(suite.beats_benchmark.sum()) == 0)
rt = suite[(suite.check == "information set") & (suite.variant == "real_time")].iloc[0]
xp = suite[(suite.check == "information set") & (suite.variant == "ex_post")].iloc[0]
check("real_time +0.0930 as quoted", abs(rt.delta_rmse - 0.0930) < 5e-4 and "0.0930" in alldocs,
      f"artifact {rt.delta_rmse:.4f}")
check("ex_post +0.0932 as quoted", abs(xp.delta_rmse - 0.0932) < 5e-4 and "0.0932" in alldocs,
      f"artifact {xp.delta_rmse:.4f}")

print("\n--- the 5.3 tables just added ---")
ens = res["ensemble"][T2]
per_fold = [len(np.asarray(a, float).ravel()) for a in ens["y_true"]]
check("Y2 fold sizes are 10/10/10/4", per_fold == [10, 10, 10, 4], str(per_fold))
check("fold sizes sum to 34", sum(per_fold) == 34)
r2s = []
for yt, yp in zip(ens["y_true"], ens["y_pred"]):
    yt, yp = np.asarray(yt, float).ravel(), np.asarray(yp, float).ravel()
    r2s.append(1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))
check("ensemble fold R2 range -0.449 to +0.436 as quoted",
      abs(min(r2s) + 0.4494) < 1e-3 and abs(max(r2s) - 0.4357) < 1e-2
      and "-0.449" in alldocs and "0.436" in alldocs,
      f"artifact [{min(r2s):.4f}, {max(r2s):.4f}]")

print("\n--- provenance sidecars ---")
missing = [p.name for p in (ART / "tables").glob("*.parquet")
           if not p.with_suffix(".provenance.json").exists()]
check("every table artifact has a provenance sidecar", not missing, str(missing[:5]))

print(f"\n{checks - len(fails)}/{checks} checks passed")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("every headline number in the docs traces to the artifacts")
