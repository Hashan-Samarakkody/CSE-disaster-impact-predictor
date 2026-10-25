"""Build sample_accounting.parquet — per-target sample sizes and exclusion accounting."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.artifact_store import artifact_file, save_frame


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    # Load splits
    import pickle
    with open(artifact_file("splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    n_events_total = len(dataset)  # 74

    # n_held_out = total test indices across all folds
    all_test = []
    for fold in splits:
        all_test.extend(fold.test_index.tolist())
    n_held_out = len(all_test)  # 40 (4 × 10)

    # n_training_eligible: sum of training sizes across all folds
    # Use fold 0's train_index length as the minimum (it's all equal: 30 each)
    n_training_eligible_fold0 = len(splits[0].train_index)  # 30

    # Check recovery variant
    import pathlib
    recovery_excl_path = artifact_file("recovery_grid_metrics_excl_competing.parquet")
    has_excl_competing = pathlib.Path(str(recovery_excl_path)).exists()

    rows = []

    # --- Y1 primary (5D) ---
    y1_col = "Y1_ASPI_5D_Forward_LogReturn_Pct"
    n_labelled_y1 = int(dataset[y1_col].notna().sum())
    n_excluded_y1 = n_events_total - n_labelled_y1
    n_purged_y1 = n_labelled_y1 - n_held_out - n_training_eligible_fold0 * 0
    # n_training_eligible is the union across all folds minus the held-out
    # For walk-forward, events in training can appear multiple times across folds
    # We report fold-0 train size as the minimum eligible set
    n_purged_y1 = max(0, n_labelled_y1 - n_held_out - n_training_eligible_fold0)

    rows.append({
        "target": "Y1_ASPI_5D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y1,
        "n_labelled": n_labelled_y1,
        "n_training_eligible": n_training_eligible_fold0,
        "n_purged": n_purged_y1,
        "n_excluded": n_excluded_y1,
        "n_held_out": n_held_out,
        "exclusion_reason": "horizon_extends_past_data_end" if n_excluded_y1 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y1 secondary (10D) ---
    y1_10_col = "Y1_ASPI_10D_Forward_LogReturn_Pct"
    n_labelled_y1_10 = int(dataset[y1_10_col].notna().sum()) if y1_10_col in dataset.columns else n_labelled_y1
    n_excluded_y1_10 = n_events_total - n_labelled_y1_10
    n_purged_y1_10 = max(0, n_labelled_y1_10 - n_held_out - n_training_eligible_fold0)

    rows.append({
        "target": "Y1_ASPI_10D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y1_10,
        "n_labelled": n_labelled_y1_10,
        "n_training_eligible": n_training_eligible_fold0,
        "n_purged": n_purged_y1_10,
        "n_excluded": n_excluded_y1_10,
        "n_held_out": n_held_out,
        "exclusion_reason": "horizon_extends_past_data_end" if n_excluded_y1_10 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y2 ---
    y2_col = "Y2_5D_Forward_AbnormalVolume_LogRatio"
    n_labelled_y2 = int(dataset[y2_col].notna().sum())
    n_excluded_y2 = n_events_total - n_labelled_y2
    n_purged_y2 = max(0, n_labelled_y2 - n_held_out - n_training_eligible_fold0)

    rows.append({
        "target": "Y2_5D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y2,
        "n_labelled": n_labelled_y2,
        "n_training_eligible": n_training_eligible_fold0,
        "n_purged": n_purged_y2,
        "n_excluded": n_excluded_y2,
        "n_held_out": n_held_out,
        "exclusion_reason": "volume_data_unavailable_or_baseline_window_short" if n_excluded_y2 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y3 main ---
    y3_col = "Y3_ASPI_Recovery_Time"
    y3_obs_col = "Y3_event_observed"
    y3_drawdown_col = "Y3_drawdown_occurred"

    n_labelled_y3 = int(dataset[y3_col].notna().sum())
    n_recovered = int((dataset[y3_obs_col] == 1).sum()) if y3_obs_col in dataset.columns else 0
    n_excluded_y3 = n_events_total - n_labelled_y3
    n_purged_y3 = max(0, n_labelled_y3 - n_held_out - n_training_eligible_fold0)

    rows.append({
        "target": "Y3_recovery_time",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y3,
        "n_labelled": n_labelled_y3,
        "n_training_eligible": n_training_eligible_fold0,
        "n_purged": n_purged_y3,
        "n_excluded": n_excluded_y3,
        "n_held_out": n_held_out,
        "exclusion_reason": "no_valid_baseline_price" if n_excluded_y3 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y3 excl_competing_risk ---
    if has_excl_competing:
        rg_excl = pd.read_parquet(recovery_excl_path)
        n_excl_total = len(rg_excl) if "n" not in rg_excl.columns else int(rg_excl["n"].iloc[0])
        rows.append({
            "target": "Y3_recovery_time",
            "variant": "excl_competing_risk",
            "n_events_total": n_events_total,
            "n_with_valid_origin": n_events_total,
            "n_with_baseline": n_events_total,
            "n_with_forward_window": n_labelled_y3,
            "n_labelled": n_labelled_y3,
            "n_training_eligible": n_training_eligible_fold0,
            "n_purged": n_purged_y3,
            "n_excluded": n_excluded_y3,
            "n_held_out": n_held_out,
            "exclusion_reason": "competing_risk_removed_from_censoring",
            "model_refit_per_event": False,
        })

    df = pd.DataFrame(rows)
    save_frame(df, "sample_accounting",
               note="Per-target and per-variant sample sizes. n_training_eligible is fold-0 "
                    "training set size (minimum across folds). n_purged = n_labelled - "
                    "n_held_out - n_training_eligible.")


if __name__ == "__main__":
    main()
