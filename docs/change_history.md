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
| Feature reclassification (date_is_exact, election features → ex_post) | 2026-10-25 | During revision | feature_availability.parquet |
| Target definitions frozen | 2026-09-19 | Before final run | commit 928a255, TARGET_DEFINITION_PROTOCOL.md |
| Confirmatory family defined (18 comparisons, Holm α=0.05; enlarged to 24 by adding naive_train_median baseline during examiner revision) | 2026-09-19 | Before final run | audit.md Part 8.4 |
| SMOGN ablation pre-declared | 2026-09-19 | Before final run | audit.md Part 8.7 |
| Robustness suite (26 checks) pre-declared | 2026-09-19 | Before final run | audit.md Part 8.7 |
| MLP reclassified as confirmatory (consistent with main text) | 2026-09-21 | Before final run | scripts/patch_mlp_tier.py |
| Single final pipeline run | 2026-09-22 | Final run | audit.md Part 8.11 |
| R1 examiner revision: n_purged split into n_unevaluated_tail and true n_purged | 2026-11-16 | Post-examination | scripts/build_sample_accounting.py, commit 7912c50 |
| R1 examiner revision: docs replaced "pre-registered" with "protocol-frozen" | 2026-11-16 | Post-examination | docs/results.md, interpretation.md, audit.md, commit 7912c50 |
| R1 examiner revision: confirmatory family corrected to 24 comparisons, Holm p=0.024 | 2026-11-16 | Post-examination | README.md, commit 7912c50 |
| R1 examiner revision: MDE dependence_assumption corrected to i.i.d. formula | 2026-11-16 | Post-examination | scripts/build_mde.py, commit e316634 |
| R1 examiner revision: episode-level permutation test added for direction classifiers | 2026-11-16 | Post-examination | scripts/run_direction_permutation_test.py, artifacts/tables/aspi_direction_permutation.parquet |
| R1 examiner revision: volume sensitivity analysis (1/5/10/20 session windows) | 2026-11-16 | Post-examination | artifacts/volume_sensitivity_horizons.py, artifacts/tables/volume_sensitivity_horizons.parquet |
| R2 examiner revision: data sources retrieval dates corrected (4 sources had post-Aug-13 coverage) | 2026-11-18 | Post-examination | docs/data_sources_retrieval.csv, docs/data_sources_reconciliation.md, commit a88def4 |
| R2 examiner revision: statistical_testing_guide updated (18→24 comparisons, 0.078→0.024 Holm p) | 2026-11-18 | Post-examination | docs/statistical_testing_guide.md, docs/CURRENT_NUMBERS.md, commit a88def4 |
| R2 examiner revision: missingness class table regenerated with correct block/availability labels | 2026-11-18 | Post-examination | scripts/build_missingness_class_table.py, artifacts/tables/missingness_class_table.parquet, commit 0d715b6 |
| R2 examiner revision: KP cross-correlation estimator replaced with actual pairwise sample correlations | 2026-11-18 | Post-examination | src/evaluation/event_study.py, docs/event_study_dependence.md, commit 36d6341 |
| R2 examiner revision: full-refit permutation mode added for direction primary (h=10, logistic) | 2026-11-18 | Post-examination | scripts/run_direction_permutation_test.py --refit, artifacts/tables/aspi_direction_permutation_refit.parquet, commit 0b4c00c |
| R2 examiner revision: manuscript_numbers.json generated from artifacts (88 number records) | 2026-11-18 | Post-examination | scripts/build_manuscript_numbers.py, docs/manuscript_numbers.json, commit 7cf297f |
| R2 examiner revision: full-refit permutation artifact generated (B=5000, h=10 logistic, L2 C=1.0, k=10, combined, seed 42); primary refit p=0.0004 Holm_p=0.0040 — consistent with fixed-prediction result | 2026-11-18 | Post-examination | scripts/run_direction_permutation_test.py --refit, artifacts/tables/aspi_direction_permutation_refit.parquet |
| R2-revision6 A1: selected hyperparameters logged per fold (Table 14a); scripts/log_selected_hyperparameters.py committed; artifacts/tables/selected_hyperparameters.parquet (24 rows, seed=42); results_to_artifact_map.csv extended | 2026-11-20 | Post-examination | scripts/log_selected_hyperparameters.py, artifacts/tables/selected_hyperparameters.parquet, commits 8d2cefd |
| R2-revision6 A2: Kolari–Pynnonen statistic replaced by randomization inference (B=5000 placebo sets, calendar-matched pools); all 8 cells p_rand>0.30 (returns: max 0.898; volume: max 0.981); docs/event_study_dependence.md written | 2026-11-20 | Post-examination | scripts/placebo_randomization_test.py, artifacts/tables/event_study_randomization.parquet, commit 758abd4 |
| R2-revision6 A3: stale ridge RMSE 2.8478 in docs/results.md (exploratory/verdict table) corrected to 2.9729 (matches aspi_grid_metrics.parquet); docs/CURRENT_NUMBERS.md updated; note added that 2.8478 in ablation_blocks.parquet is current and correct | 2026-11-20 | Post-examination | docs/results.md, docs/CURRENT_NUMBERS.md, commit 54ff6c9 |
| R2-revision6 A4: no-DOI references verified via Crossref/JSTOR/publisher; 16 confirmed DOIs added to references_apa.txt; 10 plausible DOIs annotated; 3 not found (Corrado 1989, MacKinlay 1997, Worthington & Valadkhani 2004); reference_check.csv extended to 35 rows | 2026-11-20 | Post-examination | docs/references_apa.txt, docs/reference_check.csv, docs/references_notes.md, commit d80ce17 |
| R2-revision6 A5: docs/cse_market_structure.md written; dated CSE structural changes 2000–2025 (ATS 1997/2014, COVID closure Mar–May 2020, trading hours, circuit breakers, T+2 settlement 2024, S&P SL 20 launch, MPI discontinuation) | 2026-11-20 | Post-examination | docs/cse_market_structure.md, commit d70b3ed |

The protocol was frozen at commit 928a255d on 2026-09-19, before any model performance under
the current target definitions was observed. However, the investigator had prior exposure to
data and earlier model results under different target definitions during the revision period
(2026-09-01 to 2026-09-19). This protocol freeze is useful but is not equivalent to
prospective registration before examining relevant results.
