"""Feature dictionary: every column in FEATURE_COLS, grouped, with why/internal-external/
how-it-helps/literature. Pulled from this repo's own declared rationale
(docs/EXTERNAL_DATA_PRE_DECLARATION.md, docs/METHODOLOGY_AUDIT.md, src/data_pipeline/
feature_eng.py docstrings) -- no citation is invented for a feature the repo never cited.
Where no external literature backs a feature, "literature" says so plainly.

    python scripts/generate_feature_dictionary.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (feature, category, internal/external, why_used, how_it_helps, literature)
ROWS = [
    # ---------------------------------------------------------------- returns/momentum
    ("log_return", "Returns/Momentum", "Internal",
     "Raw day-over-day ASPI log return; base signal every lag/ratio below derives from.",
     "Captures the market's most recent directional move going into the event.",
     "Standard event-study return definition (continuously compounded return); no single external citation, foundational finance convention."),
    ("lag_return_t-1", "Returns/Momentum", "Internal",
     "1-day-lagged return, strictly pre-event (shift applied before any rolling calc).",
     "Short-horizon momentum/reversal signal available at prediction time.",
     "Standard autoregressive lag feature in return-prediction literature; no specific external citation used."),
    ("lag_return_t-2", "Returns/Momentum", "Internal", "2-day-lagged return.",
     "Extends short-horizon momentum/reversal context by one more day.",
     "Same convention as lag_return_t-1."),
    ("lag_return_t-3", "Returns/Momentum", "Internal", "3-day-lagged return.",
     "Extends short-horizon momentum/reversal context.",
     "Same convention as lag_return_t-1."),
    ("lag_return_t-5", "Returns/Momentum", "Internal", "5-day-lagged return.",
     "Captures whether the pre-event week was trending up or down.",
     "Same convention as lag_return_t-1."),
    ("price_to_sma_5", "Returns/Momentum", "Internal",
     "Price relative to 5-day simple moving average (stationary; raw SMA level is trending across the 2000-2026 sample and would break chronological folds).",
     "Momentum/deviation-from-trend signal without the non-stationarity of a raw price level.",
     "Standard technical-momentum feature (price-to-MA ratio); no external citation, engineering fix for stationarity, documented in feature_eng.py."),
    ("price_to_ema_5", "Returns/Momentum", "Internal", "Price relative to 5-day EMA (same stationarity fix, more weight on recent days).",
     "Momentum signal that reacts faster to the most recent few days than the SMA version.",
     "Same rationale as price_to_sma_5."),
    ("price_to_sma_10", "Returns/Momentum", "Internal", "Price relative to 10-day SMA.",
     "Medium-horizon momentum/deviation-from-trend signal.", "Same rationale as price_to_sma_5."),
    ("price_to_ema_10", "Returns/Momentum", "Internal", "Price relative to 10-day EMA.",
     "Medium-horizon momentum signal, EMA-weighted.", "Same rationale as price_to_sma_5."),
    ("price_to_sma_20", "Returns/Momentum", "Internal", "Price relative to 20-day SMA.",
     "Longer-horizon trend-deviation signal (roughly one trading month).", "Same rationale as price_to_sma_5."),
    ("price_to_ema_20", "Returns/Momentum", "Internal", "Price relative to 20-day EMA.",
     "Longer-horizon trend signal, EMA-weighted.", "Same rationale as price_to_sma_5."),

    # ---------------------------------------------------------------- volatility
    ("rolling_std_5", "Volatility", "Internal", "5-day realized (backward-looking) return volatility, shift(1)-guarded.",
     "Short-horizon market-calm/turbulence proxy going into the event.",
     "Standard realized-volatility feature; no specific external citation."),
    ("rolling_std_10", "Volatility", "Internal", "10-day realized volatility.",
     "Medium-horizon turbulence proxy.", "Same rationale as rolling_std_5."),
    ("rolling_std_20", "Volatility", "Internal", "20-day realized volatility.",
     "Longer-horizon turbulence proxy (~1 trading month).", "Same rationale as rolling_std_5."),
    ("rolling_std_30", "Volatility", "Internal", "30-day realized volatility; mirrors the 30-day baseline window used to build Y2 (abnormal volume).",
     "'Panic proxy' -- distinguishes an already-jittery pre-event market from a calm one.",
     "Matches the target's own 30-day baseline construction; no specific external citation."),
    ("squared_return", "Volatility", "Internal", "Squared same-day log return; a cheap, well-known realized-variance proxy.",
     "Nonlinear volatility signal complementary to the rolling-std features.",
     "Standard ARCH-family proxy for instantaneous variance; foundational to Engle (1982) ARCH literature."),
    ("garch_cond_vol", "Volatility", "Internal", "One-step-ahead GARCH(1,1) conditional-volatility forecast (annual-refit, causal by construction). Added this session as a candidate feature for Y1.",
     "A genuinely different signal from realized rolling_std: a forward-looking forecast that weights recent shocks more and mean-reverts, rather than a backward-looking average.",
     "Davidescu et al. (2025), 'Evaluating Sectoral Vulnerability to Natural Disasters in the US Stock Market... DCC-GARCH Models'; 'GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets' (ACM, 2024). MEASURED RESULT: null -- never selected by the per-fold feature selector in any fold/target this session (docs/METHODOLOGY_AUDIT.md)."),

    # ---------------------------------------------------------------- volume
    ("vol_ratio_1_30", "Volume", "Internal", "Day-0 trading volume vs its own 30-day pre-event baseline.",
     "Direct 'is trading unusually heavy today' signal -- close cousin of the Y2 target's own construction, used as a feature only, never leaking the target's own value.",
     "Standard abnormal-volume convention in event-study literature; no specific external citation."),
    ("vol_ratio_5_30", "Volume", "Internal", "5-day average volume vs 30-day baseline.",
     "Smooths day-0 noise, captures a sustained (not one-day) volume shift.", "Same convention as vol_ratio_1_30."),
    ("vol_ratio_10_30", "Volume", "Internal", "10-day average volume vs 30-day baseline.",
     "Longer-horizon volume-regime signal.", "Same convention as vol_ratio_1_30."),
    ("vol_cv_30", "Volume", "Internal", "Coefficient of variation of volume over the trailing 30 days.",
     "Captures whether recent trading activity has been erratic vs steady, independent of its average level.",
     "Standard dispersion-normalisation technique; no specific external citation."),
    ("log_vol_change_1", "Volume", "Internal", "Log change in trading volume, 1 day.",
     "Short-horizon volume-momentum signal.", "Standard log-difference convention; no specific external citation."),

    # ---------------------------------------------------------------- macro/global
    ("gdp_growth_pct", "Macro/Global", "External (World Bank)", "Sri Lanka annual GDP growth.",
     "Captures whether the disaster hit during broad economic expansion or contraction -- a confound the event-study design must control for.",
     "Standard macro control in event-study/disaster-finance literature; no single specific citation, general practice."),
    ("inflation_cpi_pct", "Macro/Global", "External (World Bank)", "Sri Lanka annual CPI inflation.",
     "Controls for macro/economic-crisis conditions (e.g. the 2022 crisis) independent of the disaster itself.",
     "Standard macro control; no specific external citation."),
    ("sp500_log_return", "Macro/Global", "External (market data)", "S&P 500 day-of log return.",
     "Global-market-stress proxy -- distinguishes a local disaster reaction from a worldwide selloff happening to coincide with it.",
     "Cross-border equity spillover argued in disaster-finance literature (e.g. Fiordelisi et al., Zhou -- cited in this session's Y1 improvement plan); this repo has only S&P 500, not a full spillover battery (India Sensex/MSCI Frontier/VIX flagged as a literature-suggested but NOT implemented addition)."),
    ("fx_logret_1", "Macro/Global", "External (FRED, DEXSLUS)", "1-day USD/LKR log return, computed strictly at/before t-1.",
     "Currency-crisis/instability proxy at daily (not annual) resolution -- replaces an annual macro series that was indefensibly matched to day-0 events.",
     "docs/EXTERNAL_DATA_PRE_DECLARATION.md Block C: 'an annual macro control against a day-0 event is indefensible, and replacing it is a correctness fix, not a performance lever.' Pre-declared expectation: 'expected to do little on its own.'"),
    ("fx_logret_5", "Macro/Global", "External (FRED, DEXSLUS)", "5-day USD/LKR log return.",
     "Slightly longer FX-instability window than the 1-day version.", "Same pre-declaration as fx_logret_1."),
    ("fx_vol_30", "Macro/Global", "External (FRED, DEXSLUS)", "30-day USD/LKR return volatility.",
     "Broader currency-instability regime indicator (vs a single day's move).", "Same pre-declaration as fx_logret_1."),

    # ---------------------------------------------------------------- hazard (measured)
    ("hz_precip_max3d", "Hazard (measured)", "External (NASA POWER)", "Max 3-day rainfall accumulation across 7 district points.",
     "Flood-generating window is standard meteorologically; max (not mean) because flooding is localised and an island average dilutes it.",
     "docs/EXTERNAL_DATA_PRE_DECLARATION.md Block A: measured by instrument, not assessed by a reporter -- no missingness, no reporting bias, available same-day (admissible even in the strictest ex-ante specification, unlike financial_damage)."),
    ("hz_precip_mean3d", "Hazard (measured)", "External (NASA POWER)", "Mean 3-day rainfall accumulation across districts.",
     "Island-wide intensity, matching the index-level (not sector-level) target.", "Same Block A rationale as hz_precip_max3d."),
    ("hz_precip_spread3d", "Hazard (measured)", "External (NASA POWER)", "Std-dev of 3-day rainfall across districts.",
     "Distinguishes one severely-hit district from uniform island-wide rain at equal mean intensity.", "Same Block A rationale."),
    ("hz_districts_wet", "Hazard (measured)", "External (NASA POWER)", "Count of districts exceeding 50mm/3-day rainfall.",
     "50mm/3-day is the Sri Lanka Dept. of Meteorology's own published heavy-rain advisory threshold -- an operational standard, not a value swept for fit.",
     "Sri Lanka Department of Meteorology operational threshold (cited in docs/EXTERNAL_DATA_PRE_DECLARATION.md Block A)."),
    ("hz_wind_max3d", "Hazard (measured)", "External (NASA POWER)", "Max 3-day 10m wind speed across districts.",
     "Storm-intensity analogue for the Storm-type events, where wind (not rain) is the damaging mechanism.", "Same Block A rationale."),
    ("hz_precip_anom", "Hazard (measured)", "External (NASA POWER)", "3-day rainfall vs the location's own trailing-30-day baseline (ends t-3, no overlap with event window).",
     "Same functional form as the Y2 target (V/V_bar_30 - 1) applied to rainfall -- rain relative to that location's own recent normal, not an absolute threshold.",
     "Deliberate structural parallel to this thesis's own Y2 definition (docs/EXTERNAL_DATA_PRE_DECLARATION.md Block A)."),

    # ---------------------------------------------------------------- physical severity (DesInventar)
    ("di_districts_hit", "Physical Severity", "External (DesInventar/UNDRR)", "Distinct districts with >=1 disaster record in a [-7,+14]-day window around the event.",
     "Geographic breadth of the shock -- an exposure dimension the missing-73%-of-the-time financial_damage figure cannot supply.",
     "docs/EXTERNAL_DATA_PRE_DECLARATION.md Block B: added because financial_damage is missing (zero-filled) for 47 of 64 events, non-randomly (EM-DAT under-reports smaller/older events)."),
    ("di_affected_log", "Physical Severity", "External (DesInventar/UNDRR)", "log1p(total people affected) in the matched window.",
     "Headcount severity proxy where monetary damage is unavailable.", "Same Block B rationale."),
    ("di_houses_destroyed_log", "Physical Severity", "External (DesInventar/UNDRR)", "log1p(total houses destroyed).",
     "Closest available physical proxy to the missing monetary damage figure (capital destruction).", "Same Block B rationale."),
    ("di_houses_damaged_log", "Physical Severity", "External (DesInventar/UNDRR)", "log1p(total houses damaged, partial loss).",
     "Partial-loss counterpart to houses_destroyed.", "Same Block B rationale."),
    ("di_deaths_log", "Physical Severity", "External (DesInventar/UNDRR)", "log1p(total deaths) in the matched window.",
     "Human-severity proxy; the DesInventar field most consistently recorded.", "Same Block B rationale."),
    ("di_records", "Physical Severity", "External (DesInventar/UNDRR)", "Count of DesInventar records in the matched window.",
     "Explicitly declared as a REPORTING-INTENSITY measure, not a severity one -- so if it dominates a model that is stated, not silently misread as 'the disaster was worse'.",
     "docs/EXTERNAL_DATA_PRE_DECLARATION.md Block B, explicit interpretive caveat."),
    ("di_available", "Physical Severity", "External (DesInventar/UNDRR)", "1 if the event matched any DesInventar record, else 0 (DesInventar coverage ends 2020-12-20).",
     "Missingness indicator -- without it, a zero in the other di_* columns is indistinguishable between 'confirmed no damage' and 'outside DesInventar's date coverage'.",
     "docs/EXTERNAL_DATA_PRE_DECLARATION.md Block B, audit item P2-7 (mandatory missingness flag)."),

    # ---------------------------------------------------------------- disaster severity (EM-DAT)
    ("financial_damage", "Disaster Severity", "External (EM-DAT)", "Reported USD economic damage.",
     "The thesis's headline severity variable, in money terms -- directly answers 'how big was the disaster financially'.",
     "EM-DAT is the standard disaster-impact database in this literature; CAVEAT (own audit): real for only 18/76 events, zero-filled elsewhere, and missingness is non-random (favours large/recent/international events) -- Category C, known only after post-event assessment, not real-time."),
    ("population_affected", "Disaster Severity", "External (EM-DAT)", "Reported people affected.",
     "Headcount severity from the same authoritative disaster database.",
     "EM-DAT standard field; same missingness caveat as financial_damage, though less sparse."),
    ("log_financial_damage", "Disaster Severity", "External (EM-DAT)", "log1p(financial_damage).",
     "Compresses the multi-order-of-magnitude range of damage figures for modeling.", "Standard log-transform; same EM-DAT caveats as financial_damage."),
    ("log_population_affected", "Disaster Severity", "External (EM-DAT)", "log1p(population_affected).",
     "Compresses the range of affected-population figures.", "Standard log-transform; same EM-DAT caveats."),
    ("damage_to_gdp", "Disaster Severity", "External (EM-DAT + World Bank)", "financial_damage normalised by national GDP.",
     "Sizes the disaster relative to the economy it hit, rather than in absolute (currency-inflating, economy-growing) dollar terms.",
     "Standard disaster-economics normalisation; no single specific citation."),
    ("log_damage_x_flood", "Disaster Severity", "External (EM-DAT, derived)", "log_financial_damage interacted with the Flood disaster-type flag.",
     "Tests whether the damage-return relationship is flood-specific rather than uniform across disaster types.",
     "Interaction-term convention from the thesis's own severity-by-type hypothesis; no external citation."),

    # ---------------------------------------------------------------- disaster type
    ("disaster_Drought", "Disaster Type", "External (EM-DAT)", "One-hot: disaster type is Drought.",
     "Disaster-type fixed effect -- different hazard types plausibly have different market transmission mechanisms.", "Standard categorical control; no specific citation."),
    ("disaster_Flood", "Disaster Type", "External (EM-DAT)", "One-hot: disaster type is Flood.",
     "Disaster-type fixed effect for the most common event type in this sample.", "Same rationale as disaster_Drought."),
    ("disaster_Other", "Disaster Type", "External (EM-DAT)", "One-hot: disaster type pooled into 'Other' (types with <3 events, to avoid a one-hot that is a memorisation key).",
     "Prevents rare-type one-hots from acting as an identifier for a single training row.", "Pooling-rare-categories is standard practice for small-N categorical features; no specific citation."),
    ("disaster_Storm", "Disaster Type", "External (EM-DAT)", "One-hot: disaster type is Storm.",
     "Disaster-type fixed effect; also the type hz_wind_max3d specifically targets.", "Same rationale as disaster_Drought."),

    # ---------------------------------------------------------------- recency
    ("days_since_last_disaster", "Recency", "Internal (derived from EM-DAT dates)", "Days since the previous qualifying disaster event.",
     "Captures 'disaster fatigue' or compounding-shock effects -- a market that already absorbed a recent shock may react differently.",
     "General compounding-shock rationale in disaster literature; no specific citation."),
    ("disasters_trailing_365d", "Recency", "Internal (derived from EM-DAT dates)", "Count of qualifying disasters in the trailing 365 days.",
     "Captures a high-frequency-disaster regime vs an isolated single event.", "Same rationale as days_since_last_disaster."),

    # ---------------------------------------------------------------- confounder
    ("days_to_election", "Confounder", "External (Wikidata)", "Signed days to nearest national election.",
     "Controls for a specific known contamination: the 2005-11-17 election falls 4 days before the 2005-11-21 event, which carries the single largest observed Y1 drop.",
     "docs/EXTERNAL_DATA_PRE_DECLARATION.md Block D, motivated by a named, pre-identified contamination case."),
    ("election_within_5d", "Confounder", "External (Wikidata)", "1 if |days_to_election| <= 5.",
     "+-5 days matches the event-window contamination screen already specified in the thesis audit.", "Same Block D rationale."),

    # ---------------------------------------------------------------- other/metadata-derived
    ("date_is_exact", "Other/Metadata", "External (EM-DAT)", "Whether EM-DAT records an exact (not estimated) event date.",
     "Data-quality flag -- an imprecise event date weakens every date-aligned feature/target built from it.", "No external citation; internal data-quality control."),
    ("total_deaths", "Other/Metadata", "External (EM-DAT)", "Reported deaths (EM-DAT's own field, distinct from di_deaths_log's DesInventar count).",
     "Human-severity measure from the primary disaster database.", "EM-DAT standard field."),
    ("deaths_available", "Other/Metadata", "External (EM-DAT)", "Missingness flag for total_deaths.",
     "Prevents a zero-death value from being confused with an unreported one.", "Standard missingness-flag practice, same principle as di_available."),
    ("no_homeless", "Other/Metadata", "External (EM-DAT)", "Reported number left homeless.",
     "Additional severity dimension (housing displacement) from EM-DAT.", "EM-DAT standard field."),
    ("homeless_available", "Other/Metadata", "External (EM-DAT)", "Missingness flag for no_homeless.",
     "Prevents a zero from being confused with unreported.", "Standard missingness-flag practice."),
    ("mag_area_km2", "Other/Metadata", "External (EM-DAT)", "Reported disaster-affected area (km^2).",
     "Physical-extent severity measure, independent of population/monetary figures.", "EM-DAT standard field."),
    ("mag_wind_kph", "Other/Metadata", "External (EM-DAT)", "Reported peak wind speed (km/h), EM-DAT's own field (distinct from hz_wind_max3d's NASA POWER measurement).",
     "Storm-intensity measure directly from the disaster database, as a cross-check against the independently-measured hz_wind_max3d.", "EM-DAT standard field."),
    ("mag_area_available", "Other/Metadata", "External (EM-DAT)", "Missingness flag for mag_area_km2.",
     "Prevents a zero from being confused with unreported.", "Standard missingness-flag practice."),
    ("mag_wind_available", "Other/Metadata", "External (EM-DAT)", "Missingness flag for mag_wind_kph.",
     "Prevents a zero from being confused with unreported.", "Standard missingness-flag practice."),
]


def main():
    spec = json.loads((ROOT / "artifacts" / "feature_spec.json").read_text(encoding="utf-8"))
    feature_cols = set(spec["FEATURE_COLS"])
    documented = {r[0] for r in ROWS}

    missing = feature_cols - documented
    extra = documented - feature_cols
    if missing:
        print(f"WARNING: {len(missing)} FEATURE_COLS not documented: {sorted(missing)}")
    if extra:
        print(f"WARNING: {len(extra)} documented rows are not in current FEATURE_COLS "
              f"(stale entry, or feature renamed): {sorted(extra)}")

    out_path = ROOT / "docs" / "feature_dictionary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "feature", "internal_or_external", "why_used", "how_it_helps", "literature_backup"])
        for feature, category, kind, why, how, lit in ROWS:
            w.writerow([category, feature, kind, why, how, lit])
    print(f"wrote {len(ROWS)} rows to {out_path}")


if __name__ == "__main__":
    main()
