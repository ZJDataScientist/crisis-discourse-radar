"""
pages/1_Overview.py
--------------------
Overview page: corpus summary, KPI metrics, world map, model snapshot.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Overview", page_icon="🌍", layout="wide")

# ── data path ─────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent.parent / "dashboard_data"

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_topic_assignments():
    df = pd.read_csv(DATA / "topic_document_assignments_trimmed.csv")
    print(f"[Overview] topic_document_assignments: {df.shape}")
    return df

@st.cache_data
def load_model_comparison():
    df = pd.read_csv(DATA / "model_comparison.csv")
    print(f"[Overview] model_comparison: {df.shape}")
    return df

@st.cache_data
def load_sentiment_country():
    df = pd.read_csv(DATA / "sentiment_by_country.csv")
    print(f"[Overview] sentiment_by_country: {df.shape}")
    return df

assignments   = load_topic_assignments()
model_comp    = load_model_comparison()
sentiment_cty = load_sentiment_country()

# ── colour palette (consistent across pages) ─────────────────────────────────
REGIME_COLORS = {"Democratic": "#2196F3", "Authoritarian": "#E53935"}
DOMAIN_COLORS = {
    "health_salience":        "#43A047",
    "security_salience":      "#E53935",
    "energy_salience":        "#FB8C00",
    "economic_risk_salience": "#8E24AA",
}

# ── sidebar filters (shared across pages via session state) ───────────────────
st.sidebar.header("Global Filters")
all_countries = sorted(assignments["country"].unique())
all_regimes   = sorted(assignments["regime"].unique())
all_h_phases  = sorted(assignments["health_crisis_phase"].unique())
all_s_phases  = sorted(assignments["security_crisis_phase"].unique())

sel_countries = st.sidebar.multiselect("Country",             all_countries, default=all_countries)
sel_regimes   = st.sidebar.multiselect("Regime",              all_regimes,   default=all_regimes)
sel_h_phases  = st.sidebar.multiselect("Health Crisis Phase", all_h_phases,  default=all_h_phases)
sel_s_phases  = st.sidebar.multiselect("Security Crisis Phase", all_s_phases, default=all_s_phases)

filtered = assignments[
    assignments["country"].isin(sel_countries) &
    assignments["regime"].isin(sel_regimes) &
    assignments["health_crisis_phase"].isin(sel_h_phases) &
    assignments["security_crisis_phase"].isin(sel_s_phases)
]

# ── page header ───────────────────────────────────────────────────────────────
st.title("Crisis Discourse Dashboard")
st.subheader("Executive Communication Across Regimes and Crisis Phases")
st.markdown(
    """
    This dashboard analyzes how executive political discourse is restructured
    across crisis types — public health (COVID-19) and geopolitical security
    (Ukraine war) — and whether these shifts are universal or conditioned by
    regime type.  The corpus covers **14 615 speeches** from the United States,
    United Kingdom, Russia, and China between 2017 and 2024.
    """
)
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

speeches_n  = len(filtered)
countries_n = filtered["country"].nunique()
regimes_n   = filtered["regime"].nunique()
leaders_n   = filtered["leader"].nunique() if "leader" in filtered.columns else "—"

# Time span from date column
if "date" in filtered.columns:
    dates = pd.to_datetime(filtered["date"], errors="coerce").dropna()
    time_span = f"{dates.min().year}–{dates.max().year}" if len(dates) else "—"
else:
    time_span = "2017–2024"

k1.metric("Speeches",  f"{speeches_n:,}")
k2.metric("Countries", countries_n)
k3.metric("Regimes",   regimes_n)
k4.metric("Leaders",   leaders_n)
k5.metric("Period",    time_span)

st.divider()

# ── row 1: world map + speeches per country ───────────────────────────────────
col_map, col_bar = st.columns([3, 2], gap="large")

with col_map:
    st.subheader("🌍 Corpus Coverage by Country")

    # Country-level summary for the map
    country_summary = (
        filtered.groupby(["country", "regime"])
        .size()
        .reset_index(name="speeches")
    )

    # ISO-3 codes for Plotly choropleth
    iso_map = {
        "United States":  "USA",
        "United Kingdom": "GBR",
        "Russia":         "RUS",
        "China":          "CHN",
    }
    country_summary["iso3"] = country_summary["country"].map(iso_map)

    fig_map = px.choropleth(
        country_summary,
        locations="iso3",
        color="regime",
        hover_name="country",
        hover_data={"speeches": True, "regime": True, "iso3": False},
        color_discrete_map=REGIME_COLORS,
        title="Regime Classification and Speech Volume",
        projection="natural earth",
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Regime",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgray",
            showland=True,
            landcolor="#F5F5F5",
            showocean=True,
            oceancolor="#EEF4FB",
            showcountries=True,
            countrycolor="#CCCCCC",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_bar:
    st.subheader("📊 Speeches per Country")

    speech_counts = (
        filtered.groupby(["country", "regime"])
        .size()
        .reset_index(name="speeches")
        .sort_values("speeches", ascending=True)
    )

    fig_bar = px.bar(
        speech_counts,
        x="speeches",
        y="country",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        orientation="h",
        text="speeches",
        title="Total Speeches per Country",
        labels={"speeches": "Number of Speeches", "country": ""},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=True,
        legend_title_text="Regime",
        margin=dict(l=0, r=20, t=40, b=0),
        yaxis=dict(tickfont=dict(size=13)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── row 2: speech distribution by phase ───────────────────────────────────────
col_h, col_s = st.columns(2, gap="large")

with col_h:
    st.subheader("Health Crisis Phases")
    hp_counts = (
        filtered.groupby(["health_crisis_phase", "regime"])
        .size()
        .reset_index(name="speeches")
    )
    phase_order_h = ["pre_health", "during_health", "post_health"]
    hp_counts["health_crisis_phase"] = pd.Categorical(
        hp_counts["health_crisis_phase"], categories=phase_order_h, ordered=True
    )
    hp_counts = hp_counts.sort_values("health_crisis_phase")

    fig_hp = px.bar(
        hp_counts,
        x="health_crisis_phase",
        y="speeches",
        color="regime",
        barmode="group",
        color_discrete_map=REGIME_COLORS,
        labels={"health_crisis_phase": "Phase", "speeches": "Speeches"},
        title="Speeches by Health Crisis Phase",
    )
    fig_hp.update_layout(legend_title_text="Regime", margin=dict(t=40))
    st.plotly_chart(fig_hp, use_container_width=True)

with col_s:
    st.subheader("Security Crisis Phases")
    sp_counts = (
        filtered.groupby(["security_crisis_phase", "regime"])
        .size()
        .reset_index(name="speeches")
    )
    phase_order_s = ["pre_security", "during_security"]
    sp_counts["security_crisis_phase"] = pd.Categorical(
        sp_counts["security_crisis_phase"], categories=phase_order_s, ordered=True
    )
    sp_counts = sp_counts.sort_values("security_crisis_phase")

    fig_sp = px.bar(
        sp_counts,
        x="security_crisis_phase",
        y="speeches",
        color="regime",
        barmode="group",
        color_discrete_map=REGIME_COLORS,
        labels={"security_crisis_phase": "Phase", "speeches": "Speeches"},
        title="Speeches by Security Crisis Phase",
    )
    fig_sp.update_layout(legend_title_text="Regime", margin=dict(t=40))
    st.plotly_chart(fig_sp, use_container_width=True)

st.divider()

# ── row 3: model accuracy summary ─────────────────────────────────────────────
st.subheader("🤖 Model Performance Snapshot")
st.caption(
    "Overall accuracy of three supervised models trained on the corpus. "
    "See RQ5 for full class-level evaluation."
)

# Pull overall accuracy rows only (no epoch rows)
model_acc = (
    model_comp[
        (model_comp["metric"] == "accuracy") &
        (model_comp["class"] == "overall") &
        (model_comp["epoch"].isna())
    ]
    .copy()
    .sort_values("value", ascending=False)
)

mc1, mc2, mc3 = st.columns(3)
cards = [mc1, mc2, mc3]

model_icons = {
    "Logistic Regression": "📈",
    "Linear SVM":          "⚡",
    "LSTM":                "🧠",
}
model_desc = {
    "Logistic Regression": "Classical baseline",
    "Linear SVM":          "Best classical model",
    "LSTM":                "Deep learning model",
}

for i, (_, row) in enumerate(model_acc.iterrows()):
    if i >= 3:
        break
    icon = model_icons.get(row["model"], "📊")
    desc = model_desc.get(row["model"], "")
    cards[i].metric(
        label=f"{icon} {row['model']}",
        value=f"{row['value']:.1%}",
        help=desc,
    )

# Accuracy bar chart
fig_acc = px.bar(
    model_acc,
    x="model",
    y="value",
    color="model",
    text=model_acc["value"].apply(lambda v: f"{v:.1%}"),
    color_discrete_sequence=["#2196F3", "#43A047", "#FF7043"],
    title="Overall Accuracy by Model",
    labels={"value": "Accuracy", "model": "Model"},
    range_y=[0, 1],
)
fig_acc.update_traces(textposition="outside")
fig_acc.update_layout(
    showlegend=False,
    margin=dict(t=40, b=0),
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig_acc, use_container_width=True)
