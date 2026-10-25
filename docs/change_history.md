# Analysis Decision History

This table records every methodological decision, its date, and whether it was made before,
during, or after examining model results. It replaces the term 'pre-registered' which
appeared in earlier drafts.

| Decision | Date | Stage | Evidence |
|---|---|---|---|
| Three target definitions (Y1, Y2, Y3) chosen | Before 2026-07-17 | Before initial analysis | Initial thesis submission |
| Walk-forward validation design (30 train / 10 test) | Before 2026-07-17 | Before initial analysis | Initial thesis submission |
| Confirmatory models (ridge, random forest, MLP) selected | Before 2026-07-17 | Before initial analysis | Initial thesis submission |
| Leakage in volume baseline discovered and fixed | 2026-09-11 | During revision | audit.md Part 6 |
| Sample widened from 76 to 94 eligible, target reformulated | 2026-09-12 | During revision | audit.md Part 7 |
| Feature reclassification (date_is_exact, election features → ex_post) | 2026-09-15 | During revision | feature_availability.parquet |
| Target definitions frozen | 2026-09-19 | Before final run | commit 928a255, TARGET_DEFINITION_PROTOCOL.md |
| Confirmatory family defined (18 comparisons, Holm α=0.05; enlarged to 24 by adding naive_train_median baseline during examiner revision) | 2026-09-19 | Before final run | audit.md Part 8.4 |
| SMOGN ablation pre-declared | 2026-09-19 | Before final run | audit.md Part 8.7 |
| Robustness suite (26 checks) pre-declared | 2026-09-19 | Before final run | audit.md Part 8.7 |
| MLP reclassified as confirmatory (consistent with main text) | 2026-09-21 | Before final run | scripts/patch_mlp_tier.py |
| Single final pipeline run | 2026-09-22 | Final run | audit.md Part 8.11 |

The protocol was frozen at commit 928a255d on 2026-09-19, before any model performance under
the current target definitions was observed. However, the investigator had prior exposure to
data and earlier model results under different target definitions during the revision period
(2026-09-01 to 2026-09-19). This protocol freeze is useful but is not equivalent to
prospective registration before examining relevant results.
