"""Demo web app: replay a real disaster through the fitted models, with editable
severity inputs. See `src/models/inference.py` for what this can and cannot honestly claim.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.inference import LABEL_DESCRIPTIONS, TARGET_LABELS, get_bundle  # noqa: E402

st.set_page_config(page_title="CSE Disaster Impact Predictor", layout="wide")

# ---------------------------------------------------------------- theme (MUI v6 look)

INK = INK2 = INK3 = "#000000"
PRIMARY, GOOD, WARN, BAD = "#1565C0", "#2E7D32", "#ED6C02", "#D32F2F"
BORDER, PAGE_BG = "#E0E3E7", "#F4F6F8"

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{ font-family: 'Roboto', 'Segoe UI', sans-serif; color: {INK}; }}
.stApp {{ background: {PAGE_BG}; }}
.material-symbols-outlined {{ vertical-align: middle; font-size: 1.2em; margin-right: 6px; }}
h1, h2, h3 {{ font-weight: 700 !important; color: {INK} !important; }}
p, span, label {{ color: {INK2}; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color: {INK} !important; }}
.appbar {{
    background: {PRIMARY}; color: #fff; padding: 14px 22px; border-radius: 8px;
    display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}}
.appbar h1 {{ color: #fff !important; font-size: 1.4rem !important; margin: 0; }}
.appbar .material-symbols-outlined {{ font-size: 1.6em; }}
.mui-card {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 18px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 14px; color: {INK};
}}
.mui-alert {{
    background: #FFF4E5; border-left: 4px solid {WARN}; border-radius: 4px;
    padding: 14px 18px; margin-bottom: 18px; color: {INK};
}}
.tile-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }}
.tile {{
    flex: 1 1 220px; background: #FFFFFF; border: 1px solid {BORDER};
    border-radius: 8px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.tile .label {{
    font-size: 0.72rem; color: {INK3}; text-transform: uppercase; letter-spacing: 0.06em;
}}
.tile .value {{ font-size: 1.5rem; font-weight: 700; color: {INK}; font-variant-numeric: tabular-nums; }}
.chip {{
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
}}
.chip-good {{ background: #E8F5E9; color: {GOOD}; }}
.chip-bad {{ background: #FDECEA; color: {BAD}; }}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{
    border-radius: 8px; overflow: hidden; background: #FFFFFF;
    border: 1px solid {BORDER}; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 6px;
}}
[data-testid="stExpander"] {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 8px; }}
button[kind] {{ border-radius: 6px !important; }}
[data-testid="stSelectbox"] div[data-baseweb="select"] * {{ color: #FFFFFF !important; }}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{ background: {PRIMARY} !important; }}
[data-testid="stNumberInput"] input {{ background: {PRIMARY} !important; color: #FFFFFF !important; border-radius: 8px !important; }}
[data-testid="stNumberInput"] button {{ background: {PRIMARY} !important; color: #FFFFFF !important; }}
[data-testid="stNumberInput"] svg {{ fill: #FFFFFF !important; }}
</style>
""", unsafe_allow_html=True)


def icon(name: str) -> str:
    return f'<span class="material-symbols-outlined">{name}</span>'


def mui_tiles(items: list[tuple[str, str, str]]) -> None:
    """items = (icon_name, label, value)"""
    html = '<div class="tile-row">'
    for ic, label, value in items:
        html += (f'<div class="tile">{icon(ic)}<div class="label">{label}</div>'
                  f'<div class="value">{value}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def grouped_bar(labels: list[str], series: dict[str, list[float]], colors: dict[str, str],
                title: str) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Bar(name=name, x=labels, y=values, marker_color=colors.get(name)))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#000000"), x=0.02, xanchor="left"),
        barmode="group", template="plotly_white",
        font=dict(family="Roboto", color="#000000"), margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", y=1.2, x=1, xanchor="right", font=dict(color="#000000")),
        height=380,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(color="#000000", tickfont=dict(color="#000000")),
        yaxis=dict(color="#000000", tickfont=dict(color="#000000"), gridcolor="#E5E7EB"),
    )
    return fig


@st.cache_data
def load_market() -> pd.DataFrame:
    m = pd.read_parquet(ROOT / "artifacts" / "market.parquet")
    m["date"] = pd.to_datetime(m["date"])
    return m


def aspi_window_chart(event_date: pd.Timestamp, window: int = 30) -> go.Figure:
    market = load_market()
    lo, hi = event_date - pd.Timedelta(days=window * 1.6), event_date + pd.Timedelta(days=window * 1.6)
    seg = market[(market["date"] >= lo) & (market["date"] <= hi)]
    fig = go.Figure(go.Scatter(
        x=seg["date"], y=seg["aspi_close"], mode="lines", line=dict(color=PRIMARY, width=2.5),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
    ))
    event_x = event_date.isoformat()
    fig.add_shape(type="line", x0=event_x, x1=event_x, y0=0, y1=1, yref="paper",
                  line=dict(color=BAD, dash="dash"))
    fig.add_annotation(x=event_x, y=1, yref="paper", text="Event date", showarrow=False,
                        yshift=10, font=dict(color=BAD))
    fig.update_layout(
        title=dict(text="ASPI close, ~30 trading days either side of the event",
                    font=dict(color="#000000"), x=0.02, xanchor="left"),
        template="plotly_white", font=dict(family="Roboto", color="#000000"),
        margin=dict(l=10, r=10, t=50, b=10), height=340,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(color="#000000", tickfont=dict(color="#000000"), title="Date"),
        yaxis=dict(color="#000000", tickfont=dict(color="#000000"), gridcolor="#E5E7EB",
                   title="ASPI close"),
    )
    return fig


def importance_chart(model, feats: list[str], top_n: int = 10) -> go.Figure:
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PRIMARY))
    fig.update_layout(
        title=dict(text="What drives the Y1 model's prediction (RF feature importance)",
                    font=dict(color="#000000"), x=0.02, xanchor="left"),
        template="plotly_white", font=dict(family="Roboto", color="#000000"),
        margin=dict(l=10, r=10, t=50, b=10), height=340,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(color="#000000", tickfont=dict(color="#000000"), title="Importance"),
        yaxis=dict(color="#000000", tickfont=dict(color="#000000")),
    )
    return fig


bundle = get_bundle()

st.markdown(
    f"<div class='appbar'>{icon('finance_chip')}"
    "<h1>Colombo Stock Exchange — Disaster Impact Predictor</h1></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Demo for the BSc thesis *A Machine Learning Approach to Predicting the Impact of "
    "Natural Disasters on the Colombo Stock Exchange*. Replays a real historical "
    "disaster through the pipeline's fitted models."
)

st.markdown(
    f'<div class="mui-alert">{icon("warning")}<b>Scope.</b> This is an <i>ex-post '
    "impact-attribution</i> demo, not a forecaster. Editing severity below answers "
    '"what would the model have said about a more/less severe version of this real '
    "event\" — the market, macro, hazard, FX and election features stay fixed at the "
    "real conditions observed on that event's date, because those are only knowable "
    "in hindsight. It cannot predict tomorrow's disaster.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- event picker

events = bundle.list_events()
events["choice"] = events["event_date"] + "  --  " + events["disaster_type"]

with st.sidebar:
    st.markdown(f"### {icon('event')}1. Choose a real event", unsafe_allow_html=True)
    choice = st.selectbox("Event", events["choice"], index=0)
    event_id = int(events.loc[events["choice"] == choice, "event_id"].iloc[0])
    real_row = bundle.event_row(event_id)

    st.markdown(f"### {icon('tune')}2. Adjust severity (optional)", unsafe_allow_html=True)
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

    with st.expander("How to use this demo", expanded=False):
        st.markdown(
            "1. **Pick a real event** from the dropdown above.\n"
            "2. Its recorded severity (damage, deaths, population affected...) auto-fills.\n"
            "3. Optionally **drag the severity up or down** to ask *'what if this event "
            "had been worse/milder?'* — everything else (market, macro, FX) stays fixed "
            "at what really happened that day.\n"
            "4. Read the charts below: bars close to **Actual** are a good sign, but check "
            "the caption under each chart — most targets here do **not** beat a naive "
            "baseline, so treat predictions as point estimates, not proven forecasts.\n"
            "5. Green pills/rows in the classification table are the only labels "
            "statistically confirmed to beat chance."
        )

overrides = {
    "disaster_type": disaster_type, "financial_damage": financial_damage,
    "population_affected": population_affected, "total_deaths": total_deaths,
    "no_homeless": no_homeless, "mag_area_km2": mag_area_km2, "mag_wind_kph": mag_wind_kph,
}
feature_row = bundle.build_feature_row(event_id, overrides)

# ---------------------------------------------------------------- event summary

st.markdown(
    f"### {icon('calendar_today')}Event: {real_row['event_date'].date()} — {real_row['disaster_type']}",
    unsafe_allow_html=True,
)
mui_tiles([
    ("groups", "Population affected (real)", f"{int(real_row['population_affected']):,}"),
    ("payments", "Financial damage (real)", f"US$ {real_row['financial_damage']:,.0f}"),
    ("emergency", "Deaths (real)",
     f"{int(real_row['total_deaths'])}" if pd.notna(real_row["total_deaths"]) else "not recorded"),
    ("database", "Damage source", str(real_row.get("damage_source", "n/a"))),
])
st.plotly_chart(aspi_window_chart(pd.to_datetime(real_row["event_date"])), use_container_width=True)

# ---------------------------------------------------------------- regression

st.markdown(f"## {icon('trending_up')}Regression targets", unsafe_allow_html=True)
st.caption(
    "Every target here was tested against two naive baselines -- a constant-zero "
    "\"no measurable effect\" null and the training-fold mean -- with a paired bootstrap "
    "and a Diebold-Mariano test. See `docs/results.md` for the full verdict."
)

reg_pred = bundle.predict_regression(feature_row)
rows, chart_labels = [], []
model_vals, zero_vals, mean_vals, actual_vals = [], [], [], []
for target, label in TARGET_LABELS.items():
    baselines = bundle.naive_baselines(target)
    actual = real_row[target]
    rows.append({
        "Target": label,
        "Model prediction": round(reg_pred[target], 5),
        "Naive: constant zero": round(baselines["constant_zero"], 5),
        "Naive: training mean": round(baselines["training_mean"], 5),
        "Actual (real event)": f"{float(actual):.5f}" if pd.notna(actual) else "n/a",
    })
    chart_labels.append(label)
    model_vals.append(reg_pred[target])
    zero_vals.append(baselines["constant_zero"])
    mean_vals.append(baselines["training_mean"])
    actual_vals.append(float(actual) if pd.notna(actual) else 0.0)

st.plotly_chart(grouped_bar(
    chart_labels,
    {"Model": model_vals, "Naive zero": zero_vals, "Naive mean": mean_vals, "Actual": actual_vals},
    {"Model": PRIMARY, "Naive zero": INK3, "Naive mean": "#B7C0C7", "Actual": GOOD},
    "Model vs. baselines vs. actual",
), use_container_width=True)

st.markdown('<div class="mui-card">', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
st.caption(
    "None of the three original targets (5-day forward return, abnormal volume, recovery days) "
    "beat both naive baselines in the walk-forward audit. Treat 'model prediction' as "
    "the pipeline's best point estimate, not as evidence it is accurate."
)

st.plotly_chart(
    importance_chart(bundle.regressors["Y1_ASPI_5D_Forward_LogReturn_Pct"],
                      bundle.selected_features["Y1_ASPI_5D_Forward_LogReturn_Pct"]),
    use_container_width=True,
)
st.caption(
    "Real `feature_importances_` from the fitted RandomForest -- what the model actually "
    "weighs, not a claim about statistical causation."
)

st.markdown(f"### {icon('healing')}Y3 recovery days: two competing models", unsafe_allow_html=True)
actual_y3 = real_row["Y3_ASPI_Recovery_Time"]
hurdle_pred = bundle.predict_hurdle(feature_row)
mui_tiles([
    ("model_training", "Single-stage RF (above table)", f"{reg_pred['Y3_ASPI_Recovery_Time']:.1f} days"),
    ("layers", "Two-stage hurdle model", f"{hurdle_pred:.1f} days"),
    ("fact_check", "Actual (real event)",
     f"{actual_y3:.0f} days" if pd.notna(actual_y3) else "n/a"),
])
st.plotly_chart(grouped_bar(
    ["Y3 recovery days"],
    {"Single-stage RF": [reg_pred["Y3_ASPI_Recovery_Time"]], "Hurdle model": [hurdle_pred],
     "Actual": [float(actual_y3) if pd.notna(actual_y3) else 0.0]},
    {"Single-stage RF": PRIMARY, "Hurdle model": WARN, "Actual": GOOD},
    "Recovery-day estimates",
), use_container_width=True)
st.caption(
    "The hurdle model was built to fix the 90-day censoring problem, but the recorded "
    "audit shows it does NOT outperform a naive training-fold mean on MAE. Shown here "
    "for completeness, not as the recommended model."
)

# ---------------------------------------------------------------- classification

st.markdown(f"## {icon('checklist')}Classification labels", unsafe_allow_html=True)
st.caption(
    "Success criterion: balanced accuracy above 0.5 **and** an AUC interval excluding "
    "0.5 -- not raw accuracy, which an imbalanced label can score highly by always "
    "predicting the majority class."
)

clf_pred = bundle.predict_classification(feature_row)
clf_rows, bar_labels, bar_probs, bar_colors = [], [], [], []
for name, info in clf_pred.items():
    beats = info["beats_baseline"]
    clf_rows.append({
        "Label": LABEL_DESCRIPTIONS[name],
        "Predicted": "YES" if info["predicted_positive"] else "no",
        "Probability": round(info["probability"], 3),
        "Historical AUC": round(info["historical_auc"], 3),
        "Beats baseline?": "yes" if beats else "no",
    })
    bar_labels.append(LABEL_DESCRIPTIONS[name])
    bar_probs.append(info["probability"])
    bar_colors.append(GOOD if beats else BAD)

fig = go.Figure(go.Bar(
    x=bar_probs, y=bar_labels, orientation="h", marker_color=bar_colors,
    text=[f"{p:.0%}" for p in bar_probs], textposition="outside",
))
fig.add_vline(x=0.5, line_dash="dash", line_color="#888")
fig.update_layout(
    title=dict(
        text="Predicted probability (green = confirmed beats chance, red = no better than a coin flip)",
        font=dict(color="#000000"),
    ),
    template="plotly_white", font=dict(family="Roboto", color="#000000"),
    margin=dict(l=10, r=10, t=40, b=10), height=340, xaxis_range=[0, 1],
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    xaxis=dict(color="#000000", tickfont=dict(color="#000000"), gridcolor="#E5E7EB"),
    yaxis=dict(color="#000000", tickfont=dict(color="#000000")),
)
st.plotly_chart(fig, use_container_width=True)

clf_df = pd.DataFrame(clf_rows)


def _highlight(row):
    colour = "background-color: rgba(30,158,107,0.15)" if row["Beats baseline?"] == "yes" \
             else "background-color: rgba(214,69,69,0.12)"
    return [colour] * len(row)


st.markdown('<div class="mui-card">', unsafe_allow_html=True)
st.dataframe(clf_df.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
st.caption(
    "Only the green row (`C2_volume_spike`) clears both the majority rule and chance "
    "in the recorded walk-forward audit. The red rows are shown for completeness -- "
    "their predictions here are no more trustworthy than a coin flip, regardless of "
    "how confident the number looks."
)

st.divider()
st.caption(
    "Source: this repository's stage 04/05 walk-forward pipeline. Models shown are "
    "refit on all 76 real events (see `scripts/train_final_models.py`), matching the "
    "pattern stage 04 already uses for `final_rf_models.pkl`. No number in this app is "
    "a new result -- every historical metric quoted here is copied from "
    "`docs/results.md`."
)
