"""Demo web app: replay a real disaster through the fitted models, with editable
severity inputs. See `src/inference.py` for what this can and cannot honestly claim.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inference import LABEL_DESCRIPTIONS, TARGET_LABELS, get_bundle  # noqa: E402

st.set_page_config(page_title="CSE Disaster Impact Predictor", layout="wide")

bundle = get_bundle()

st.title("Colombo Stock Exchange -- Disaster Impact Predictor")
st.caption(
    "Demo for the BSc thesis *A Machine Learning Approach to Predicting the Impact of "
    "Natural Disasters on the Colombo Stock Exchange*. Replays a real historical "
    "disaster through the pipeline's fitted models."
)

st.warning(
    "**Scope.** This is an *ex-post impact-attribution* demo, not a forecaster. "
    "Editing severity below answers \"what would the model have said about a more/less "
    "severe version of this real event\" -- the market, macro, hazard, FX and election "
    "features stay fixed at the real conditions observed on that event's date, because "
    "those are only knowable in hindsight. It cannot predict tomorrow's disaster.",
    icon="⚠️",
)

# ---------------------------------------------------------------- event picker

events = bundle.list_events()
events["choice"] = events["event_date"] + "  --  " + events["disaster_type"]

with st.sidebar:
    st.header("1. Choose a real event")
    choice = st.selectbox("Event", events["choice"], index=0)
    event_id = int(events.loc[events["choice"] == choice, "event_id"].iloc[0])
    real_row = bundle.event_row(event_id)

    st.header("2. Adjust severity (optional)")
    st.caption("Defaults are the real recorded values for this event.")

    disaster_type = st.selectbox(
        "Disaster type", ["Drought", "Flood", "Other", "Storm"],
        index=["Drought", "Flood", "Other", "Storm"].index(
            next((t for t in ["Drought", "Flood", "Other", "Storm"]
                  if real_row.get(f"disaster_{t}") == 1.0), "Other")),
    )
    financial_damage = st.number_input(
        "Financial damage (US$)", min_value=0.0,
        value=float(real_row["financial_damage"]), step=1_000_000.0, format="%.0f")
    population_affected = st.number_input(
        "Population affected", min_value=0.0,
        value=float(real_row["population_affected"]), step=1000.0, format="%.0f")
    total_deaths = st.number_input(
        "Deaths", min_value=0.0,
        value=float(real_row["total_deaths"]) if pd.notna(real_row["total_deaths"]) else 0.0)
    no_homeless = st.number_input(
        "Left homeless", min_value=0.0,
        value=float(real_row["no_homeless"]) if pd.notna(real_row["no_homeless"]) else 0.0)
    mag_area_km2 = st.number_input(
        "Flood area (km^2, 0 if not a flood)", min_value=0.0,
        value=float(real_row["mag_area_km2"]) if pd.notna(real_row["mag_area_km2"]) else 0.0)
    mag_wind_kph = st.number_input(
        "Storm wind speed (kph, 0 if not a storm)", min_value=0.0,
        value=float(real_row["mag_wind_kph"]) if pd.notna(real_row["mag_wind_kph"]) else 0.0)

overrides = {
    "disaster_type": disaster_type, "financial_damage": financial_damage,
    "population_affected": population_affected, "total_deaths": total_deaths,
    "no_homeless": no_homeless, "mag_area_km2": mag_area_km2, "mag_wind_kph": mag_wind_kph,
}
feature_row = bundle.build_feature_row(event_id, overrides)

# ---------------------------------------------------------------- event summary

st.subheader(f"Event: {real_row['event_date'].date()} -- {real_row['disaster_type']}")
cols = st.columns(4)
cols[0].metric("Population affected (real)", f"{int(real_row['population_affected']):,}")
cols[1].metric("Financial damage (real)", f"US$ {real_row['financial_damage']:,.0f}")
cols[2].metric("Deaths (real)",
              f"{int(real_row['total_deaths'])}" if pd.notna(real_row["total_deaths"]) else "not recorded")
cols[3].metric("Damage source",
              str(real_row.get("damage_source", "n/a")))

# ---------------------------------------------------------------- regression

st.header("Regression targets")
st.caption(
    "Every target here was tested against two naive baselines -- a constant-zero "
    "\"no measurable effect\" null and the training-fold mean -- with a paired bootstrap "
    "and a Diebold-Mariano test. See `docs/RESULTS_AUDIT.txt` for the full verdict."
)

reg_pred = bundle.predict_regression(feature_row)
rows = []
for target, label in TARGET_LABELS.items():
    baselines = bundle.naive_baselines(target)
    actual = real_row[target]
    rows.append({
        "Target": label,
        "Model prediction": round(reg_pred[target], 5),
        "Naive: constant zero": round(baselines["constant_zero"], 5),
        "Naive: training mean": round(baselines["training_mean"], 5),
        "Actual (real event)": round(float(actual), 5) if pd.notna(actual) else "n/a",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "None of the three original targets (day-0 return, abnormal volume, recovery days) "
    "beat both naive baselines in the walk-forward audit. Treat 'model prediction' as "
    "the pipeline's best point estimate, not as evidence it is accurate."
)

st.subheader("Y3 recovery days: two competing models")
c1, c2, c3 = st.columns(3)
c1.metric("Single-stage RF (above table)", f"{reg_pred['Y3_recovery_days']:.1f} days")
c2.metric("Two-stage hurdle model", f"{bundle.predict_hurdle(feature_row):.1f} days")
actual_y3 = real_row["Y3_recovery_days"]
c3.metric("Actual (real event)",
         f"{actual_y3:.0f} days" if pd.notna(actual_y3) else "n/a")
st.caption(
    "The hurdle model was built to fix the 90-day censoring problem, but the recorded "
    "audit shows it does NOT outperform a naive training-fold mean on MAE. Shown here "
    "for completeness, not as the recommended model."
)

# ---------------------------------------------------------------- classification

st.header("Classification labels")
st.caption(
    "Success criterion: balanced accuracy above 0.5 **and** an AUC interval excluding "
    "0.5 -- not raw accuracy, which an imbalanced label can score highly by always "
    "predicting the majority class."
)

clf_pred = bundle.predict_classification(feature_row)
clf_rows = []
for name, info in clf_pred.items():
    clf_rows.append({
        "Label": LABEL_DESCRIPTIONS[name],
        "Predicted": "YES" if info["predicted_positive"] else "no",
        "Probability": round(info["probability"], 3),
        "Historical AUC": round(info["historical_auc"], 3),
        "Beats baseline?": "✓ yes" if info["beats_baseline"] else "✗ no",
    })
clf_df = pd.DataFrame(clf_rows)


def _highlight(row):
    colour = "background-color: #d4f4dd" if row["Beats baseline?"].startswith("✓") else \
             "background-color: #f8d7da"
    return [colour] * len(row)


st.dataframe(clf_df.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)
st.caption(
    "Only the two green rows (`C2_volume_spike`, `C4_car5_negative`) clear both the "
    "majority rule and chance in the recorded walk-forward audit. The red rows are "
    "shown for completeness -- their predictions here are no more trustworthy than a "
    "coin flip, regardless of how confident the number looks."
)

st.divider()
st.caption(
    "Source: this repository's stage 04/05 walk-forward pipeline. Models shown are "
    "refit on all 76 real events (see `scripts/train_final_models.py`), matching the "
    "pattern stage 04 already uses for `final_rf_models.pkl`. No number in this app is "
    "a new result -- every historical metric quoted here is copied from "
    "`docs/RESULTS_AUDIT.txt`."
)
