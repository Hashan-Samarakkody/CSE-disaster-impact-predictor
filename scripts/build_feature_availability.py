"""Build feature_availability.parquet — per-feature availability metadata."""

from __future__ import annotations

import json
import re
import pandas as pd

from src.utils.artifact_store import artifact_file, save_frame


# Block assignment rules: (pattern, block)
BLOCK_RULES = [
    # Disaster severity
    (r"^financial_damage", "disaster_severity"),
    (r"^log_financial_damage$", "disaster_severity"),
    (r"^population_affected$", "disaster_severity"),
    (r"^log_population_affected$", "disaster_severity"),
    (r"^total_deaths$", "disaster_severity"),
    (r"^no_homeless$", "disaster_severity"),
    (r"^mag_", "disaster_severity"),
    (r"^damage_to_gdp$", "disaster_severity"),
    (r"^log_damage_x_flood$", "disaster_severity"),
    (r"^di_affected_log$", "disaster_severity"),
    (r"^di_districts_hit$", "disaster_severity"),
    (r"^di_houses_destroyed_log$", "disaster_severity"),
    (r"^di_houses_damaged_log$", "disaster_severity"),
    (r"^di_deaths_log$", "disaster_severity"),
    # Availability indicators
    (r"_available$", "availability_indicators"),
    (r"^di_records$", "availability_indicators"),
    # Market state
    (r"^log_return$", "market_state"),
    (r"^lag_return_", "market_state"),
    (r"^price_to_sma_", "market_state"),
    (r"^price_to_ema_", "market_state"),
    (r"^sma_", "market_state"),
    (r"^ema_", "market_state"),
    (r"^rolling_std_", "market_state"),
    (r"^squared_return$", "market_state"),
    (r"^garch_cond_vol$", "market_state"),
    (r"^vol_ratio_", "market_state"),
    (r"^vol_cv_30$", "market_state"),
    (r"^log_vol_change_1$", "market_state"),
    # Macro
    (r"^gdp_growth_pct$", "macro"),
    (r"^inflation_cpi_pct$", "macro"),
    (r"^gdp_current_usd$", "macro"),
    # Market factor
    (r"^sp500_log_return$", "market_factor"),
    (r"^fx_logret", "market_factor"),
    (r"^fx_vol_", "market_factor"),
    # Hazard
    (r"^hz_", "hazard"),
    # Election
    (r"^days_to_election$", "election"),
    (r"^election_within_5d$", "election"),
    # Disaster context
    (r"^disaster_", "disaster_context"),
    (r"^days_since_last_disaster$", "disaster_context"),
    (r"^disasters_trailing_365d$", "disaster_context"),
    (r"^date_is_exact$", "disaster_context"),
    # DI records (catch-all for di_ prefixed)
    (r"^di_", "disaster_severity"),
]


def assign_block(feature: str) -> str:
    for pattern, block in BLOCK_RULES:
        if re.match(pattern, feature):
            return block
    return "other"


MARKET_STATE_FEATURES = {
    "log_return", "lag_return_t-1", "lag_return_t-2", "lag_return_t-3", "lag_return_t-5",
    "price_to_sma_5", "price_to_ema_5", "price_to_sma_10", "price_to_ema_10",
    "price_to_sma_20", "price_to_ema_20",
    "rolling_std_5", "rolling_std_10", "rolling_std_20", "rolling_std_30",
    "squared_return", "garch_cond_vol",
    "vol_ratio_1_30", "vol_ratio_5_30", "vol_ratio_10_30", "vol_cv_30", "log_vol_change_1",
}


def main():
    spec = json.loads((artifact_file("feature_spec.json")).read_text(encoding="utf-8"))
    avail_class = spec["AVAILABILITY_CLASS"]

    rows = []
    for feature, info_set in avail_class.items():
        block = assign_block(feature)
        notes_parts = []

        # Apply reclassifications
        final_info_set = info_set

        if feature == "date_is_exact":
            final_info_set = "ex_post"
            notes_parts.append("reclassified ex_post: database quality, unknown at origin")

        if feature == "days_to_election":
            final_info_set = "ex_post"
            notes_parts.append("reclassified ex_post: gazette dates unavailable per-event")

        if feature == "election_within_5d":
            final_info_set = "ex_post"
            notes_parts.append("reclassified ex_post: gazette dates unavailable per-event")

        if feature in ("population_affected", "log_population_affected", "di_affected_log"):
            final_info_set = "ex_post"
            notes_parts.append("ex_post: finalised totals from 1,000-affected threshold")

        if feature == "log_return":
            notes_parts.append("return realised at P0, observation contemporaneous")

        # Observation/publication rules
        if final_info_set == "real_time":
            obs_rule = "available at event origin (P0)"
            pub_rule = "same as observation"
            rev_policy = "not revised"
            usable_from = "P0"
        else:
            obs_rule = "available after event resolution"
            pub_rule = "delayed; final values used"
            rev_policy = "may be revised post-publication"
            usable_from = "ex_post"

        shifted_to_P0 = feature in MARKET_STATE_FEATURES

        rows.append({
            "feature": feature,
            "block": block,
            "observation_date_rule": obs_rule,
            "publication_date_rule": pub_rule,
            "revision_policy": rev_policy,
            "usable_from_rule": usable_from,
            "info_set": final_info_set,
            "shifted_to_P0": shifted_to_P0,
            "notes": "; ".join(notes_parts) if notes_parts else "",
        })

    df = pd.DataFrame(rows)
    save_frame(df, "feature_availability",
               note="One row per feature with availability class, block assignment, and "
                    "reclassifications applied for date_is_exact, days_to_election, "
                    "election_within_5d, and 1000-affected threshold features.")


if __name__ == "__main__":
    main()
