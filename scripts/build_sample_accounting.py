"""Build sample_accounting.parquet — per-target sample sizes and exclusion accounting."""

from __future__ import annotations

import pandas as pd
from src.utils.artifact_store import artifact_file, save_frame


def _sample_counts(dataset: pd.DataFrame, target_col: str, splits) -> tuple[int, int, int]:
    """Count target-valid training and held-out events over the recorded folds."""
    valid = dataset[target_col].notna()
    n_labelled = int(valid.sum())
    n_training_eligible = min(
        int(valid.iloc[fold.train_index].sum()) for fold in splits)
    n_held_out = sum(
        int(valid.iloc[fold.test_index].sum()) for fold in splits)
    return n_labelled, n_training_eligible, n_held_out


def build_sample_accounting(dataset: pd.DataFrame, splits,
                            has_excl_competing: bool = False,
                            recovery_excl_path=None) -> pd.DataFrame:
    n_events_total = len(dataset)

    rows = []

    # --- Y1 primary (5D) ---
    y1_col = "Y1_ASPI_5D_Forward_LogReturn_Pct"
    n_labelled_y1, n_training_y1, n_held_out_y1 = _sample_counts(
        dataset, y1_col, splits)
    n_excluded_y1 = n_events_total - n_labelled_y1
    n_purged_y1 = max(0, n_labelled_y1 - n_held_out_y1 - n_training_y1)

    rows.append({
        "target": "Y1_ASPI_5D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y1,
        "n_labelled": n_labelled_y1,
        "n_training_eligible": n_training_y1,
        "n_purged": n_purged_y1,
        "n_excluded": n_excluded_y1,
        "n_held_out": n_held_out_y1,
        "exclusion_reason": "horizon_extends_past_data_end" if n_excluded_y1 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y1 secondary (10D) ---
    y1_10_col = "Y1_ASPI_10D_Forward_LogReturn_Pct"
    n_labelled_y1_10, n_training_y1_10, n_held_out_y1_10 = _sample_counts(
        dataset, y1_10_col, splits)
    n_excluded_y1_10 = n_events_total - n_labelled_y1_10
    n_purged_y1_10 = max(
        0, n_labelled_y1_10 - n_held_out_y1_10 - n_training_y1_10)

    rows.append({
        "target": "Y1_ASPI_10D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y1_10,
        "n_labelled": n_labelled_y1_10,
        "n_training_eligible": n_training_y1_10,
        "n_purged": n_purged_y1_10,
        "n_excluded": n_excluded_y1_10,
        "n_held_out": n_held_out_y1_10,
        "exclusion_reason": "horizon_extends_past_data_end" if n_excluded_y1_10 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y2 ---
    y2_col = "Y2_5D_Forward_AbnormalVolume_LogRatio"
    n_labelled_y2, n_training_y2, n_held_out_y2 = _sample_counts(
        dataset, y2_col, splits)
    n_with_baseline_y2 = int(dataset["Y2_V_base"].notna().sum())
    n_with_forward_window_y2 = int(dataset["Y2_V_future5"].notna().sum())
    n_excluded_y2 = n_events_total - n_labelled_y2
    n_purged_y2 = max(0, n_labelled_y2 - n_held_out_y2 - n_training_y2)

    rows.append({
        "target": "Y2_5D",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_with_baseline_y2,
        "n_with_forward_window": n_with_forward_window_y2,
        "n_labelled": n_labelled_y2,
        "n_training_eligible": n_training_y2,
        "n_purged": n_purged_y2,
        "n_excluded": n_excluded_y2,
        "n_held_out": n_held_out_y2,
        "exclusion_reason": "volume_data_unavailable_or_baseline_window_short" if n_excluded_y2 > 0 else "",
        "model_refit_per_event": False,
    })

    # --- Y3 main ---
    y3_col = "Y3_ASPI_Recovery_Time"

    n_labelled_y3, n_training_y3, n_held_out_y3 = _sample_counts(
        dataset, y3_col, splits)
    n_excluded_y3 = n_events_total - n_labelled_y3
    n_purged_y3 = max(0, n_labelled_y3 - n_held_out_y3 - n_training_y3)

    rows.append({
        "target": "Y3_recovery_time",
        "variant": "main",
        "n_events_total": n_events_total,
        "n_with_valid_origin": n_events_total,
        "n_with_baseline": n_events_total,
        "n_with_forward_window": n_labelled_y3,
        "n_labelled": n_labelled_y3,
        "n_training_eligible": n_training_y3,
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
