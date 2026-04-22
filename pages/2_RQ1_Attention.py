"""
pages/2_RQ1_Attention.py
-------------------------
RQ1: What topics receive attention?
Covers issue salience (Layer 1) and topic modeling (Layer 1/2).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="RQ1 – Attention", page_icon="📡", layout="wide")

DATA = Path(__file__).parent.parent / "dashboard_data"

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_salience_health():
    return pd.read_csv(DATA / "salience_by_health_phase.csv")

@st.cache_data
def load_salience_security():
    return pd.read_csv(DATA / "salience_by_security_phase.csv")

@st.cache_data
def load_salience_regime_health():
    return pd.read_csv(DATA / "salience_by_regime_health_phase.csv")

@st.cache_data
def load_salience_regime_security():
    return pd.read_csv(DATA / "salience_by_regime_security_phase.csv")

@st.cache_data
def load_topic_regime():
    return pd.read_csv(DATA / "topic_share_by_regime.csv")

@st.cache_data
def load_topic_country():
    return pd.read_csv(DATA / "topic_share_by_country.csv")

@st.cache_data
def load_topic_words():
    return pd.read_csv(DATA / "topic_top_words.csv")

@st.cache_data
def load_topic_health():
    return pd.read_csv(DATA / "topic_share_by_health_phase.csv")

@st.cache_data
def load_topic_security():
    return pd.read_csv(DATA / "topic_share_by_security_phase.csv")

sal_health    = load_salience_health()
sal_security  = load_salience_security()
sal_rh        = load_salience_regime_health()
sal_rs        = load_salience_regime_security()
topic_regime  = load_topic_regime()
topic_country = load_topic_country()
topic_words   = load_topic_words()
topic_health  = load_topic_health()
topic_security= load_topic_security()

# ── colour maps ───────────────────────────────────────────────────────────────
DOMAIN_COLORS = {
    "Health":        "#43A047",
    "Security":      "#E53935",
    "Energy":        "#FB8C00",
    "Economic Risk": "#8E24AA",
}
REGIME_COLORS = {"Democratic": "#2196F3", "Authoritarian": "#E53935"}

TOPIC_COLORS = {
    "Global Development and Multilateral Cooperation":              "#1565C0",
    "Centralized Governance and Domestic State Authority":          "#C62828",
    "National Economic Identity and Labor Framing":                 "#2E7D32",
    "Personalized Political Rhetoric and Executive Communication":  "#F57F17",
    "Geopolitical Security and Strategic Relations":                "#6A1B9A",
}

SALIENCE_COLS = [
    "health_salience",
    "security_salience",
    "energy_salience",
    "economic_risk_salience",
]
SALIENCE_LABELS = {
    "health_salience":        "Health",
    "security_salience":      "Security",
    "energy_salience":        "Energy",
    "economic_risk_salience": "Economic Risk",
}

# ── phase ordering ────────────────────────────────────────────────────────────
HEALTH_ORDER    = ["pre_health", "during_health", "post_health"]
SECURITY_ORDER  = ["pre_security", "during_security"]

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("RQ1 Filters")
show_regimes = st.sidebar.multiselect(
    "Regime (salience × regime charts)",
    ["Democratic", "Authoritarian"],
    default=["Democratic", "Authoritarian"],
)
show_domains = st.sidebar.multiselect(
    "Salience Domains",
    list(SALIENCE_LABELS.values()),
    default=list(SALIENCE_LABELS.values()),
)

# map label → column name for filter
selected_cols = [
    col for col, lbl in SALIENCE_LABELS.items() if lbl in show_domains
]

# ── page header ───────────────────────────────────────────────────────────────
st.title("📡 RQ1: What Topics Receive Attention?")
st.markdown(
    """
    **Research Question 1** examines whether and how executive attention shifts
    across policy domains during crises. Issue salience is measured through
    curated domain dictionaries, normalised by speech length to prevent volume
    distortion. Topic structure is captured through LDA-based topic modelling
    and SVM topic classification.

    > *Crisis-driven attention reallocation operates as a systemic structural
    > mechanism — health crises produce large punctuated surges while security
    > crises amplify an already-prominent domain.*
    """
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – ISSUE SALIENCE
# ══════════════════════════════════════════════════════════════════════════════
st.header("Layer 1 — Issue Salience")
st.caption(
    "Normalised domain keyword density per speech, averaged by crisis phase. "
    "Higher values indicate greater proportion of domain-relevant tokens."
)

# ── 1a: salience across health phases ─────────────────────────────────────────
tab_h, tab_s = st.tabs(["Health Crisis Phases", "Security Crisis Phases"])

with tab_h:
    st.subheader("Domain Salience Across Health Crisis Phases")

    # Melt to long format for line chart
    sal_h_long = (
        sal_health[["health_crisis_phase"] + selected_cols]
        .melt(id_vars="health_crisis_phase", var_name="domain", value_name="salience")
    )
    sal_h_long["domain"] = sal_h_long["domain"].map(SALIENCE_LABELS)
    sal_h_long["health_crisis_phase"] = pd.Categorical(
        sal_h_long["health_crisis_phase"], categories=HEALTH_ORDER, ordered=True
    )
    sal_h_long = sal_h_long.sort_values("health_crisis_phase")

    fig_sal_h = px.line(
        sal_h_long,
        x="health_crisis_phase",
        y="salience",
        color="domain",
        markers=True,
        color_discrete_map=DOMAIN_COLORS,
        title="Mean Domain Salience by Health Crisis Phase",
        labels={"health_crisis_phase": "Phase", "salience": "Normalised Salience", "domain": "Domain"},
    )
    fig_sal_h.update_traces(line=dict(width=2.5), marker=dict(size=9))
    fig_sal_h.update_layout(
        xaxis=dict(
            ticktext=["Pre-Health", "During Health (COVID)", "Post-Health"],
            tickvals=HEALTH_ORDER,
        ),
        legend_title_text="Domain",
        margin=dict(t=50),
    )
    st.plotly_chart(fig_sal_h, use_container_width=True)

    col_interp, col_table = st.columns([2, 1])
    with col_interp:
        st.info(
            "**Key finding:** Health salience spikes sharply during the COVID-19 "
            "phase (×5 relative to pre-health), confirming punctuated agenda "
            "reordering. Security salience rises in the post-health period, "
            "indicating a substitution effect as geopolitical tensions escalate."
        )
    with col_table:
        st.dataframe(
            sal_health[["health_crisis_phase"] + selected_cols]
            .rename(columns=SALIENCE_LABELS)
            .set_index("health_crisis_phase")
            .style.format("{:.5f}"),
            use_container_width=True,
        )

with tab_s:
    st.subheader("Domain Salience Across Security Crisis Phases")

    sal_s_long = (
        sal_security[["security_crisis_phase"] + selected_cols]
        .melt(id_vars="security_crisis_phase", var_name="domain", value_name="salience")
    )
    sal_s_long["domain"] = sal_s_long["domain"].map(SALIENCE_LABELS)
    sal_s_long["security_crisis_phase"] = pd.Categorical(
        sal_s_long["security_crisis_phase"], categories=SECURITY_ORDER, ordered=True
    )
    sal_s_long = sal_s_long.sort_values("security_crisis_phase")

    fig_sal_s = px.line(
        sal_s_long,
        x="security_crisis_phase",
        y="salience",
        color="domain",
        markers=True,
        color_discrete_map=DOMAIN_COLORS,
        title="Mean Domain Salience by Security Crisis Phase",
        labels={"security_crisis_phase": "Phase", "salience": "Normalised Salience", "domain": "Domain"},
    )
    fig_sal_s.update_traces(line=dict(width=2.5), marker=dict(size=9))
    fig_sal_s.update_layout(
        xaxis=dict(
            ticktext=["Pre-Security", "During Security (Ukraine)"],
            tickvals=SECURITY_ORDER,
        ),
        legend_title_text="Domain",
        margin=dict(t=50),
    )
    st.plotly_chart(fig_sal_s, use_container_width=True)

    st.info(
        "**Key finding:** Security salience intensifies ~27 % during the Ukraine "
        "crisis while health salience declines, consistent with domain substitution. "
        "The security domain is already institutionally prominent prior to the "
        "conflict — the crisis amplifies rather than activates it."
    )

st.divider()

# ── 1b: salience × regime interaction ─────────────────────────────────────────
st.header("Layer 1 — Regime × Salience Interaction")
st.caption(
    "Mean domain salience by regime and phase. "
    "Compares whether crisis-driven attention shifts are universal or regime-conditioned."
)

tab_rh, tab_rs = st.tabs(["Regime × Health Phase", "Regime × Security Phase"])

with tab_rh:
    rh_filtered = sal_rh[sal_rh["regime"].isin(show_regimes)]
    rh_long = (
        rh_filtered[["regime", "health_crisis_phase"] + selected_cols]
        .melt(id_vars=["regime", "health_crisis_phase"], var_name="domain", value_name="salience")
    )
    rh_long["domain"]               = rh_long["domain"].map(SALIENCE_LABELS)
    rh_long["health_crisis_phase"]  = pd.Categorical(
        rh_long["health_crisis_phase"], categories=HEALTH_ORDER, ordered=True
    )
    rh_long = rh_long.sort_values("health_crisis_phase")

    # One subplot per domain
    domain_sel = [d for d in list(DOMAIN_COLORS.keys()) if d in rh_long["domain"].unique()]

    fig_rh = px.line(
        rh_long,
        x="health_crisis_phase",
        y="salience",
        color="regime",
        facet_col="domain",
        facet_col_wrap=2,
        markers=True,
        color_discrete_map=REGIME_COLORS,
        title="Regime × Health Phase: Domain Salience",
        labels={"health_crisis_phase": "Phase", "salience": "Salience", "regime": "Regime"},
    )
    fig_rh.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig_rh.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_rh.update_xaxes(
        ticktext=["Pre", "During", "Post"],
        tickvals=HEALTH_ORDER,
        tickangle=-30,
    )
    fig_rh.update_layout(margin=dict(t=60), legend_title_text="Regime")
    st.plotly_chart(fig_rh, use_container_width=True)

    st.info(
        "**Regime interaction:** Democratic regimes exhibit a sharper "
        "proportional health salience spike (×8) relative to authoritarian regimes "
        "(×4), indicating regime-conditioned responsiveness to public health crises. "
        "This aligns with the paper's significant Crisis × Regime interaction (health, p < .05)."
    )

with tab_rs:
    rs_filtered = sal_rs[sal_rs["regime"].isin(show_regimes)]
    rs_long = (
        rs_filtered[["regime", "security_crisis_phase"] + selected_cols]
        .melt(id_vars=["regime", "security_crisis_phase"], var_name="domain", value_name="salience")
    )
    rs_long["domain"]                 = rs_long["domain"].map(SALIENCE_LABELS)
    rs_long["security_crisis_phase"]  = pd.Categorical(
        rs_long["security_crisis_phase"], categories=SECURITY_ORDER, ordered=True
    )
    rs_long = rs_long.sort_values("security_crisis_phase")

    fig_rs = px.line(
        rs_long,
        x="security_crisis_phase",
        y="salience",
        color="regime",
        facet_col="domain",
        facet_col_wrap=2,
        markers=True,
        color_discrete_map=REGIME_COLORS,
        title="Regime × Security Phase: Domain Salience",
        labels={"security_crisis_phase": "Phase", "salience": "Salience", "regime": "Regime"},
    )
    fig_rs.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig_rs.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_rs.update_xaxes(
        ticktext=["Pre", "During"],
        tickvals=SECURITY_ORDER,
        tickangle=-30,
    )
    fig_rs.update_layout(margin=dict(t=60), legend_title_text="Regime")
    st.plotly_chart(fig_rs, use_container_width=True)

    st.info(
        "**Regime interaction:** Security salience increases in both regimes "
        "during the Ukraine crisis. The interaction term is not statistically "
        "significant — security reallocation operates as a systemic structural "
        "mechanism rather than a regime-contingent amplification process."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – TOPIC MODELING
# ══════════════════════════════════════════════════════════════════════════════
st.header("Layer 1/2 — Topic Structure (LDA + SVM)")
st.caption(
    "5-topic LDA model trained on the full corpus. "
    "Topic shares show the proportion of speeches dominated by each topic, "
    "aggregated by regime, country, and crisis phase."
)

# ── 2a: topic share by regime (stacked bar) ────────────────────────────────────
col_t1, col_t2 = st.columns(2, gap="large")

with col_t1:
    st.subheader("Topic Share by Regime")

    tr_filtered = topic_regime[topic_regime["regime"].isin(show_regimes)]
    fig_tr = px.bar(
        tr_filtered,
        x="regime",
        y="proportion",
        color="topic_name",
        text=tr_filtered["proportion"].apply(lambda v: f"{v:.0%}"),
        color_discrete_map=TOPIC_COLORS,
        title="Proportional Topic Distribution by Regime",
        labels={"proportion": "Proportion", "regime": "Regime", "topic_name": "Topic"},
    )
    fig_tr.update_traces(textposition="inside", textfont_size=10)
    fig_tr.update_layout(
        barmode="stack",
        yaxis_tickformat=".0%",
        legend_title_text="Topic",
        legend=dict(font=dict(size=10)),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_tr, use_container_width=True)

with col_t2:
    st.subheader("Topic Share by Country")

    fig_tc = px.bar(
        topic_country,
        x="country",
        y="proportion",
        color="topic_name",
        text=topic_country["proportion"].apply(lambda v: f"{v:.0%}"),
        color_discrete_map=TOPIC_COLORS,
        title="Proportional Topic Distribution by Country",
        labels={"proportion": "Proportion", "country": "Country", "topic_name": "Topic"},
    )
    fig_tc.update_traces(textposition="inside", textfont_size=10)
    fig_tc.update_layout(
        barmode="stack",
        yaxis_tickformat=".0%",
        legend_title_text="Topic",
        legend=dict(font=dict(size=10)),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_tc, use_container_width=True)

# ── 2b: topic share by phase (heatmap) ────────────────────────────────────────
st.subheader("Topic Share Across Crisis Phases")

tab_th, tab_ts = st.tabs(["Health Phase × Topic", "Security Phase × Topic"])

with tab_th:
    # Pivot to matrix for heatmap
    th_pivot = (
        topic_health.pivot_table(
            index="health_crisis_phase",
            columns="topic_name",
            values="proportion",
            fill_value=0,
        )
        .reindex(HEALTH_ORDER)
    )
    fig_th = px.imshow(
        th_pivot,
        text_auto=".2%",
        color_continuous_scale="Blues",
        title="Topic Proportion Heatmap: Health Crisis Phases",
        labels=dict(x="Topic", y="Phase", color="Proportion"),
        aspect="auto",
    )
    fig_th.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        coloraxis_colorbar_tickformat=".0%",
        margin=dict(t=50, b=100),
    )
    st.plotly_chart(fig_th, use_container_width=True)

with tab_ts:
    ts_pivot = (
        topic_security.pivot_table(
            index="security_crisis_phase",
            columns="topic_name",
            values="proportion",
            fill_value=0,
        )
        .reindex(SECURITY_ORDER)
    )
    fig_ts = px.imshow(
        ts_pivot,
        text_auto=".2%",
        color_continuous_scale="Reds",
        title="Topic Proportion Heatmap: Security Crisis Phases",
        labels=dict(x="Topic", y="Phase", color="Proportion"),
        aspect="auto",
    )
    fig_ts.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        coloraxis_colorbar_tickformat=".0%",
        margin=dict(t=50, b=100),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ── 2c: topic top words table ─────────────────────────────────────────────────
st.subheader("📋 Topic Keywords (Top 10 Terms per Topic)")

# Expand top_words pipe-delimited string into columns
tw_expanded = topic_words.copy()
tw_expanded["top_words"] = tw_expanded["top_words"].str.replace(" | ", " · ", regex=False)

# Color topic name cells to match chart colours
def color_topic(val):
    color = TOPIC_COLORS.get(val, "#EEEEEE")
    return f"background-color: {color}20; font-weight: bold;"

st.dataframe(
    tw_expanded[["topic_id", "topic_name", "top_words"]]
    .rename(columns={"topic_id": "ID", "topic_name": "Topic", "top_words": "Top Terms"}),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "**Interpretation:** Authoritarian discourse is dominated by centralised "
    "governance (Topic 1, 50 %) and geopolitical security (Topic 4, 45 %). "
    "Democratic discourse is split between economic–labor framing (Topic 2, 49 %) "
    "and personalised political rhetoric (Topic 3, 39 %). "
    "This confirms that topic structure is regime-conditioned, not merely crisis-driven."
)
