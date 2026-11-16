"""Build sample_accounting.parquet — per-target sample sizes and exclusion accounting."""

from __future__ import annotations

import pandas as pd
from src.utils.artifact_store import artifact_file, save_frame
from src.training.walk_forward import purge_horizon_overlap


def _sample_counts(dataset: pd.DataFrame, target_col: str, splits,
                   horizon_end_col: str | None = None) -> tuple[int, int, int, int, int]:
    """Count target-valid training and held-out events over the recorded folds.

    Returns (n_labelled, n_training_eligible, n_held_out,
             n_unevaluated_tail, n_purged).

    n_unevaluated_tail: labelled events that never appear in any test fold
                        (the final events beyond the last fold boundary).
    n_purged: labelled events dropped from training by purge_horizon_overlap.
              Previously these two were collapsed into a single n_purged column
              that equalled only the unevaluated tail count.
    """
    valid = dataset[target_col].notna()
    n_labelled = int(valid.sum())
    n_training_eligible = min(
        int(valid.iloc[fold.train_index].sum()) for fold in splits)
    n_held_out = sum(
        int(valid.iloc[fold.test_index].sum()) for fold in splits)

    # Unevaluated tail: labelled events beyond the last test fold
    all_test = set()
    for s in splits:
        all_test.update(s.test_index.tolist())
    max_test = max(all_test)
    tail_mask = pd.Series(False, index=range(len(dataset)))
    tail_mask[max_test + 1:] = True
    n_unevaluated_tail = int((tail_mask & valid).sum())

    # True purged: labelled events removed from training by horizon overlap
    n_purged = 0
    if horizon_end_col is not None and horizon_end_col in dataset.columns:
        horizon_end = dataset[horizon_end_col]
        event_dates = dataset["event_date"]
        purged_indices: set[int] = set()
        for s in splits:
            s_p = purge_horizon_overlap(s, event_dates, horizon_end)
            purged_indices.update(set(s.train_index.tolist()) - set(s_p.train_index.tolist()))
        valid_indices = set(dataset.index[valid].tolist())
        n_purged = len(purged_indices & valid_indices)

    return n_labelled, n_training_eligible, n_held_out, n_unevaluated_tail, n_purged


def build_sample_accounting(dataset: pd.DataFrame, splits,
                            has_excl_competing: bool = False,
                            recovery_excl_path=None) -> pd.DataFrame:
    n_events_total = len(dataset)

    rows = []

    # --- Y1 primary (5D) ---
    y1_col = "Y1_ASPI_5D_Forward_LogReturn_Pct"
    (n_labelled_y1, n_training_y1, n_held_out_y1,
     n_tail_y1, n_purged_y1) = _sample_counts(
        dataset, y1_col, splits, horizon_end_col="Y1_horizon_end_date")
    n_excluded_y1 = n_events_total - n_labelled_y1

    rows.append({
        "target": "Y1_ASPI_5D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y1,
        "n_labelled": n_labelled_y1,
        "n_training_eligible": n_training_y1,
        "n_unevaluated_tail": n_tail_y1,
        "n_purged": n_purged_y1,
        "n_excluded": n_excluded_y1,
        "n_held_out": n_held_out_y1,
        "exclusion_reason": "horizon_extends_past_data_end" if n_excluded_y1 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y1 secondary (10D) ---
    y1_10_col = "Y1_ASPI_10D_Forward_LogReturn_Pct"
    (n_labelled_y1_10, n_training_y1_10, n_held_out_y1_10,
     n_tail_y1_10, n_purged_y1_10) = _sample_counts(
        dataset, y1_10_col, splits, horizon_end_col="Y1_10D_horizon_end_date")
    n_excluded_y1_10 = n_events_total - n_labelled_y1_10

    rows.append({
        "target": "Y1_ASPI_10D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y1_10,
        "n_labelled": n_labelled_y1_10,
        "n_training_eligible": n_training_y1_10,
        "n_unevaluated_tail": n_tail_y1_10,
        "n_purged": n_purged_y1_10,
        "n_excluded": n_excluded_y1_10,
        "n_held_out": n_held_out_y1_10,
        "exclusion_reason": "horizon_extends_past_data_end" if n_excluded_y1_10 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y2 ---
    y2_col = "Y2_5D_Forward_AbnormalVolume_LogRatio"
    (n_labelled_y2, n_training_y2, n_held_out_y2,
     n_tail_y2, n_purged_y2) = _sample_counts(
        dataset, y2_col, splits, horizon_end_col="Y2_horizon_end_date")
    n_with_baseline_y2 = int(dataset["Y2_V_base"].notna().sum())
    n_with_forward_window_y2 = int(dataset["Y2_V_future5"].notna().sum())
    n_excluded_y2 = n_events_total - n_labelled_y2

    rows.append({
        "target": "Y2_5D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_with_baseline_y2,
        "n_with_forward_window": n_with_forward_window_y2,
        "n_labelled": n_labelled_y2,
        "n_training_eligible": n_training_y2,
        "n_unevaluated_tail": n_tail_y2,
        "n_purged": n_purged_y2,
        "n_excluded": n_excluded_y2,
        "n_held_out": n_held_out_y2,
        "exclusion_reason": "volume_data_unavailable_or_baseline_window_short" if n_excluded_y2 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y3 main ---
    y3_col = "Y3_ASPI_Recovery_Time"

    (n_labelled_y3, n_training_y3, n_held_out_y3,
     n_tail_y3, n_purged_y3) = _sample_counts(
        dataset, y3_col, splits, horizon_end_col="Y3_label_end_date")
    n_excluded_y3 = n_events_total - n_labelled_y3

    rows.append({
        "target": "Y3_recovery_time",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y3,
        "n_labelled": n_labelled_y3,
        "n_training_eligible": n_training_y3,
        "n_unevaluated_tail": n_tail_y3,
        "n_purged": n_purged_y3,
        "n_excluded": n_excluded_y3,
        "n_held_out": n_held_out_y3,
        "exclusion_reason": "no_valid_baseline_price" if n_excluded_y3 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y3 excl_competing_risk ---
    if has_excl_competing:
        excl_metrics = pd.read_parquet(str(recovery_excl_path))
        n_held_out_excl = int(excl_metrics["n"].iloc[0])
        n_competing_removed = n_held_out_y3 - n_held_out_excl
        rows.append({
            "target": "Y3_recovery_time",
            "variant": "excl_competing_risk",
            "n_events_total": n_events_total,
            "n_with_valid_origin": n_events_total,
            "n_with_baseline": n_events_total,
            "n_with_forward_window": n_labelled_y3,
            "n_labelled": n_labelled_y3,
            "n_training_eligible": n_training_y3,
            "n_unevaluated_tail": n_tail_y3,
            "n_purged": n_purged_y3,
            "n_excluded": n_competing_removed,
            "n_held_out": n_held_out_excl,
            "exclusion_reason": "competing_risk_removed_from_censoring",
            "model_refit_per_event": False,
        })

    return pd.DataFrame(rows)


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    import pickle
    with open(artifact_file("splits.pkl"), "rb") as f:
        splits = pickle.load(f)

    import pathlib
    recovery_excl_path = artifact_file("recovery_grid_metrics_excl_competing.parquet")
    has_excl_competing = pathlib.Path(str(recovery_excl_path)).exists()

    df = build_sample_accounting(dataset, splits, has_excl_competing, recovery_excl_path)
    save_frame(df, "sample_accounting",
               note="Per-target and per-variant sample sizes. Training and held-out counts "
                    "include only non-missing labels for that target; training eligibility "
                    "is the minimum target-valid training count across folds.")


if __name__ == "__main__":
    main()
