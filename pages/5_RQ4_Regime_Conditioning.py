"""
pages/5_RQ4_Regime_Conditioning.py
-------------------------------------
RQ4: Are these shifts universal or regime-conditioned?
Compares salience, topics, POS, entity density, and sentiment across regimes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RQ4 – Regime Conditioning",
    page_icon="⚖️",
    layout="wide",
)

DATA = Path(__file__).parent.parent / "dashboard_data"

# ── colour / style constants ──────────────────────────────────────────────────
REGIME_COLORS = {"Democratic": "#2196F3", "Authoritarian": "#E53935"}
DOMAIN_COLORS = {
    "Health":        "#43A047",
    "Security":      "#E53935",
    "Energy":        "#FB8C00",
    "Economic Risk": "#8E24AA",
}
TOPIC_COLORS = {
    "Global Development and Multilateral Cooperation":              "#1565C0",
    "Centralized Governance and Domestic State Authority":          "#C62828",
    "National Economic Identity and Labor Framing":                 "#2E7D32",
    "Personalized Political Rhetoric and Executive Communication":  "#F57F17",
    "Geopolitical Security and Strategic Relations":                "#6A1B9A",
}
SENTIMENT_COLORS = {
    "positive": "#2E7D32",
    "neutral":  "#9E9E9E",
    "negative": "#C62828",
}
POS_COLORS = {
    "Noun":    "#1565C0",
    "Verb":    "#2E7D32",
    "Adj":     "#F57F17",
    "Adverb":  "#6A1B9A",
    "Modal":   "#00838F",
}
ENTITY_COLORS = {
    "Person":  "#1565C0",
    "Org":     "#FB8C00",
    "GPE":     "#E53935",
    "NORP":    "#6A1B9A",
}

HEALTH_ORDER   = ["pre_health",   "during_health",   "post_health"]
SECURITY_ORDER = ["pre_security", "during_security"]

PHASE_LABELS = {
    "pre_health":       "Pre-Health",
    "during_health":    "During Health",
    "post_health":      "Post-Health",
    "pre_security":     "Pre-Security",
    "during_security":  "During Security",
}

SALIENCE_COLS   = ["health_salience", "security_salience",
                   "energy_salience", "economic_risk_salience"]
SALIENCE_LABELS = {
    "health_salience":        "Health",
    "security_salience":      "Security",
    "energy_salience":        "Energy",
    "economic_risk_salience": "Economic Risk",
}

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load(fname):
    df = pd.read_csv(DATA / fname)
    print(f"[RQ4] {fname}: {df.shape}")
    return df

sal_rh      = load("salience_by_regime_health_phase.csv")
sal_rs      = load("salience_by_regime_security_phase.csv")
topic_reg   = load("topic_share_by_regime.csv")
pos_reg     = load("pos_by_regime.csv")
ent_density = load("entity_density_by_regime.csv")
sent_reg    = load("sentiment_by_regime.csv")

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("RQ4 Filters")
sel_regimes = st.sidebar.multiselect(
    "Regime",
    ["Democratic", "Authoritarian"],
    default=["Democratic", "Authoritarian"],
)
sel_domains = st.sidebar.multiselect(
    "Salience Domains",
    list(SALIENCE_LABELS.values()),
    default=list(SALIENCE_LABELS.values()),
)
selected_sal_cols = [c for c, l in SALIENCE_LABELS.items() if l in sel_domains]

# ── page header ───────────────────────────────────────────────────────────────
st.title("⚖️ RQ4: Are These Shifts Universal or Regime-Conditioned?")
st.markdown(
    """
    Layers 1–3 established that crises restructure executive attention, narrative
    framing, and affective tone.  **RQ4 asks whether these restructurings affect
    all governments equally** — or whether institutional context (democratic vs
    authoritarian regime type) amplifies, dampens, or redirects them.

    The comparative analysis covers all four analytical layers:
    issue salience, topic structure, syntactic and actor features, and sentiment
    distribution. Regime type is treated as a **moderating variable** that
    conditions how strongly and in which direction crisis effects manifest.

    > *"While crisis-driven attention shifts are broadly systemic, affective
    > intensification and structural linguistic differences are more strongly
    > conditioned by regime type."*
    """
)

# ── key insight banner ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

auth_health = sal_rh[
    (sal_rh["regime"] == "Authoritarian") &
    (sal_rh["health_crisis_phase"] == "during_health")
]["health_salience"].values[0]
dem_health = sal_rh[
    (sal_rh["regime"] == "Democratic") &
    (sal_rh["health_crisis_phase"] == "during_health")
]["health_salience"].values[0]

auth_noun = pos_reg[pos_reg["regime"] == "Authoritarian"]["noun_share"].values[0]
dem_verb  = pos_reg[pos_reg["regime"] == "Democratic"]["verb_share"].values[0]

auth_ent  = ent_density[ent_density["regime"] == "Authoritarian"]["entity_total_rate"].values[0]
dem_ent   = ent_density[ent_density["regime"] == "Democratic"]["entity_total_rate"].values[0]

c1.metric("Auth. health salience (during)",  f"{auth_health:.5f}", help="Authoritarian mean health salience during COVID-19")
c2.metric("Dem. health salience (during)",   f"{dem_health:.5f}",  help="Democratic mean health salience during COVID-19 — ~50% higher spike")
c3.metric("Auth. noun share",  f"{auth_noun:.1%}", help="Authoritarian mean noun token share")
c4.metric("Dem. verb share",   f"{dem_verb:.1%}",  help="Democratic mean verb token share — contrasting structural style")

st.info(
    "**Key insight:** Regime type is not merely a baseline difference — it "
    "systematically moderates how crises restructure discourse.  Democratic "
    "regimes show stronger proportional health salience spikes and sharper "
    "affective amplification, while authoritarian regimes maintain higher "
    "baseline security salience and structurally denser, noun-dominated rhetoric "
    "that persists regardless of crisis phase."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — SALIENCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.header("A — Issue Salience: Regime × Phase Interaction")
st.caption(
    "Mean domain salience per speech, broken down by regime and crisis phase. "
    "Lines coloured by regime reveal whether crisis-driven attention shifts are "
    "amplified or dampened by institutional context."
)

tab_sal_h, tab_sal_s = st.tabs(["Health Crisis", "Security Crisis"])

def regime_salience_chart(df, phase_col, phase_order, domain_col,
                          domain_label, title, x_labels):
    sub = df[df["regime"].isin(sel_regimes)].copy()
    sub[phase_col] = pd.Categorical(sub[phase_col], categories=phase_order, ordered=True)
    sub = sub.sort_values(phase_col)
    sub["phase_label"] = sub[phase_col].astype(str).map(PHASE_LABELS).fillna(sub[phase_col].astype(str))

    fig = px.line(
        sub,
        x="phase_label",
        y=domain_col,
        color="regime",
        markers=True,
        color_discrete_map=REGIME_COLORS,
        category_orders={"phase_label": [PHASE_LABELS[p] for p in phase_order]},
        title=title,
        labels={"phase_label": "Phase", domain_col: f"Mean {domain_label} Salience", "regime": "Regime"},
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=10))
    fig.update_layout(
        legend_title_text="Regime",
        margin=dict(t=55, b=10),
    )
    return fig

def faceted_salience_chart(df, phase_col, phase_order, title):
    """Four-panel faceted line chart, one panel per salience domain."""
    sub = df[df["regime"].isin(sel_regimes)].copy()
    sub[phase_col] = pd.Categorical(sub[phase_col], categories=phase_order, ordered=True)
    sub = sub.sort_values(phase_col)
    sub["phase_label"] = sub[phase_col].astype(str).map(PHASE_LABELS).fillna(sub[phase_col].astype(str))

    long = sub[["regime", "phase_label"] + selected_sal_cols].melt(
        id_vars=["regime", "phase_label"],
        var_name="domain_col",
        value_name="salience",
    )
    long["domain"] = long["domain_col"].map(SALIENCE_LABELS)

    fig = px.line(
        long,
        x="phase_label",
        y="salience",
        color="regime",
        facet_col="domain",
        facet_col_wrap=2,
        markers=True,
        color_discrete_map=REGIME_COLORS,
        category_orders={"phase_label": [PHASE_LABELS[p] for p in phase_order]},
        title=title,
        labels={"phase_label": "Phase", "salience": "Mean Salience", "regime": "Regime"},
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(
        legend_title_text="Regime",
        margin=dict(t=65, b=10),
        height=500,
    )
    return fig

with tab_sal_h:
    col_h1, col_h2 = st.columns(2, gap="large")

    with col_h1:
        st.subheader("Health Salience by Regime")
        st.plotly_chart(
            regime_salience_chart(
                sal_rh, "health_crisis_phase", HEALTH_ORDER,
                "health_salience", "Health",
                "Health Salience: Regime × Health Phase",
                ["Pre-Health", "During Health", "Post-Health"],
            ),
            width="stretch",
        )

    with col_h2:
        st.subheader("Security Salience by Regime (Health Phases)")
        st.plotly_chart(
            regime_salience_chart(
                sal_rh, "health_crisis_phase", HEALTH_ORDER,
                "security_salience", "Security",
                "Security Salience: Regime × Health Phase",
                ["Pre-Health", "During Health", "Post-Health"],
            ),
            width="stretch",
        )

    if selected_sal_cols:
        st.subheader("All Domains — Faceted View")
        st.plotly_chart(
            faceted_salience_chart(
                sal_rh, "health_crisis_phase", HEALTH_ORDER,
                "Regime × Health Phase: All Salience Domains",
            ),
            width="stretch",
        )

    # Summary delta table
    st.subheader("Regime Comparison at Peak (During Health)")
    peak_h = sal_rh[sal_rh["health_crisis_phase"] == "during_health"][
        ["regime"] + SALIENCE_COLS
    ].set_index("regime").rename(columns=SALIENCE_LABELS)
    st.dataframe(
        peak_h.style.format("{:.6f}").highlight_max(axis=0, color="#E3F2FD"),
        use_container_width=True,
    )

    st.info(
        "**Regime interaction — health:** Democratic regimes exhibit a stronger "
        "proportional health salience spike (~6× pre-health vs ~4× for authoritarian). "
        "This aligns with the paper's significant Crisis × Democratic interaction "
        "term (p < .05), indicating regime-conditioned responsiveness to public "
        "health shocks. Security salience rises more steeply in post-health "
        "authoritarian discourse, reflecting geopolitical priority persistence."
    )

with tab_sal_s:
    col_s1, col_s2 = st.columns(2, gap="large")

    with col_s1:
        st.subheader("Security Salience by Regime")
        st.plotly_chart(
            regime_salience_chart(
                sal_rs, "security_crisis_phase", SECURITY_ORDER,
                "security_salience", "Security",
                "Security Salience: Regime × Security Phase",
                ["Pre-Security", "During Security"],
            ),
            width="stretch",
        )

    with col_s2:
        st.subheader("Health Salience by Regime (Security Phases)")
        st.plotly_chart(
            regime_salience_chart(
                sal_rs, "security_crisis_phase", SECURITY_ORDER,
                "health_salience", "Health",
                "Health Salience: Regime × Security Phase",
                ["Pre-Security", "During Security"],
            ),
            width="stretch",
        )

    if selected_sal_cols:
        st.subheader("All Domains — Faceted View")
        st.plotly_chart(
            faceted_salience_chart(
                sal_rs, "security_crisis_phase", SECURITY_ORDER,
                "Regime × Security Phase: All Salience Domains",
            ),
            width="stretch",
        )

    st.subheader("Regime Comparison at Peak (During Security)")
    peak_s = sal_rs[sal_rs["security_crisis_phase"] == "during_security"][
        ["regime"] + SALIENCE_COLS
    ].set_index("regime").rename(columns=SALIENCE_LABELS)
    st.dataframe(
        peak_s.style.format("{:.6f}").highlight_max(axis=0, color="#FFEBEE"),
        use_container_width=True,
    )

    st.info(
        "**Regime interaction — security:** Both regimes increase security "
        "salience during the Ukraine crisis, but the interaction term is **not** "
        "statistically significant — security reallocation operates as a "
        "systemic structural mechanism rather than a regime-contingent amplification. "
        "Authoritarian regimes maintain consistently higher baseline security "
        "salience (~0.009 vs ~0.006), amplifying from a higher starting point."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — TOPIC DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.header("B — Topic Structure: Regime-Conditioned Thematic Worlds")
st.caption(
    "Proportional topic share by regime, derived from the 5-topic LDA model. "
    "Dramatic divergence between regimes confirms that topic structure is "
    "institutionally anchored, not merely crisis-driven."
)

tr_filt = topic_reg[topic_reg["regime"].isin(sel_regimes)].copy()

col_t1, col_t2 = st.columns([3, 2], gap="large")

with col_t1:
    fig_topic = px.bar(
        tr_filt,
        x="regime",
        y="proportion",
        color="topic_name",
        text=tr_filt["proportion"].apply(lambda v: f"{v:.0%}" if v > 0.02 else ""),
        color_discrete_map=TOPIC_COLORS,
        title="Topic Share by Regime",
        labels={"proportion": "Proportion", "regime": "Regime", "topic_name": "Topic"},
    )
    fig_topic.update_traces(textposition="inside", textfont_size=10)
    fig_topic.update_layout(
        barmode="stack",
        yaxis_tickformat=".0%",
        legend_title_text="Topic",
        legend=dict(font=dict(size=10)),
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_topic, width="stretch")

with col_t2:
    st.markdown("**Dominant topics per regime**")
    for regime in [r for r in ["Authoritarian", "Democratic"] if r in sel_regimes]:
        top = (
            tr_filt[tr_filt["regime"] == regime]
            .sort_values("proportion", ascending=False)
            .head(2)
        )
        color = REGIME_COLORS[regime]
        st.markdown(f"<span style='color:{color}; font-weight:bold'>{regime}</span>", unsafe_allow_html=True)
        for _, row in top.iterrows():
            st.markdown(f"- **{row['topic_name'][:40]}…** ({row['proportion']:.0%})")
        st.markdown("")

st.info(
    "**Topic divergence:** Authoritarian discourse is almost entirely split between "
    "*Centralised Governance and Domestic State Authority* (50 %) and "
    "*Geopolitical Security and Strategic Relations* (45 %), reflecting state-centric "
    "and sovereignty-oriented framing.  Democratic discourse organises around "
    "*National Economic Identity and Labor Framing* (49 %) and "
    "*Personalised Political Rhetoric* (39 %), reflecting pluralist, economic, "
    "and identity-driven communication. This divergence is **structural**, "
    "not crisis-reactive."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — STRUCTURAL DIFFERENCES: POS + NER
# ══════════════════════════════════════════════════════════════════════════════
st.header("C — Structural Linguistic Differences")
st.caption(
    "Part-of-speech shares and named entity density reveal deep structural "
    "differences in how regimes construct discourse, independent of topic content."
)

col_c1, col_c2 = st.columns(2, gap="large")

# ── C1: POS comparison ────────────────────────────────────────────────────────
with col_c1:
    st.subheader("Syntactic Structure (POS)")

    pos_filt = pos_reg[pos_reg["regime"].isin(sel_regimes)].copy()
    pos_long = pos_filt.melt(
        id_vars="regime",
        value_vars=["noun_share", "verb_share", "adj_share", "adv_share", "modal_share"],
        var_name="pos",
        value_name="share",
    )
    pos_rename = {
        "noun_share":  "Noun",
        "verb_share":  "Verb",
        "adj_share":   "Adj",
        "adv_share":   "Adverb",
        "modal_share": "Modal",
    }
    pos_long["pos_label"] = pos_long["pos"].map(pos_rename)

    fig_pos = px.bar(
        pos_long,
        x="pos_label",
        y="share",
        color="regime",
        barmode="group",
        color_discrete_map=REGIME_COLORS,
        text=pos_long["share"].apply(lambda v: f"{v:.1%}"),
        title="POS Share by Regime",
        labels={"share": "Mean Token Share", "pos_label": "POS", "regime": "Regime"},
        category_orders={"pos_label": ["Noun", "Verb", "Adj", "Adverb", "Modal"]},
    )
    fig_pos.update_traces(textposition="outside", textfont_size=10)
    fig_pos.update_layout(
        yaxis_tickformat=".0%",
        legend_title_text="Regime",
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_pos, width="stretch")

    if len(pos_filt) >= 2:
        auth_noun_v = pos_filt[pos_filt["regime"] == "Authoritarian"]["noun_share"].values[0]
        dem_noun_v  = pos_filt[pos_filt["regime"] == "Democratic"]["noun_share"].values[0]
        auth_verb_v = pos_filt[pos_filt["regime"] == "Authoritarian"]["verb_share"].values[0]
        dem_verb_v  = pos_filt[pos_filt["regime"] == "Democratic"]["verb_share"].values[0]

        mc1, mc2 = st.columns(2)
        mc1.metric("Noun gap (Auth − Dem)",
                   f"{(auth_noun_v - dem_noun_v):+.1%}",
                   help="Authoritarian speeches use proportionally more nouns")
        mc2.metric("Verb gap (Dem − Auth)",
                   f"{(dem_verb_v - auth_verb_v):+.1%}",
                   help="Democratic speeches use proportionally more verbs")

    st.info(
        "**Syntactic asymmetry:** Authoritarian discourse is noun-dominated "
        "(37 % vs 24 % for democratic), consistent with declarative, "
        "state-centred rhetoric that anchors meaning in institutional entities. "
        "Democratic discourse is more verb-oriented (21 % vs 13 %), signalling "
        "procedural and action-driven framing."
    )

# ── C2: entity density comparison ────────────────────────────────────────────
with col_c2:
    st.subheader("Actor Density (NER)")

    ent_filt = ent_density[ent_density["regime"].isin(sel_regimes)].copy()
    ent_long = ent_filt.melt(
        id_vars="regime",
        value_vars=["person_rate", "org_rate", "gpe_rate", "norp_rate"],
        var_name="entity_type",
        value_name="rate",
    )
    ent_rename = {
        "person_rate": "Person",
        "org_rate":    "Org",
        "gpe_rate":    "GPE",
        "norp_rate":   "NORP",
    }
    ent_long["entity_label"] = ent_long["entity_type"].map(ent_rename)

    fig_ent = px.bar(
        ent_long,
        x="entity_label",
        y="rate",
        color="regime",
        barmode="group",
        color_discrete_map=REGIME_COLORS,
        text=ent_long["rate"].apply(lambda v: f"{v:.4f}"),
        title="Entity Density by Regime",
        labels={"rate": "Entity Rate", "entity_label": "Entity Type", "regime": "Regime"},
        category_orders={"entity_label": ["Person", "Org", "GPE", "NORP"]},
    )
    fig_ent.update_traces(textposition="outside", textfont_size=10)
    fig_ent.update_layout(
        legend_title_text="Regime",
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_ent, width="stretch")

    if len(ent_filt) >= 2:
        auth_tot = ent_filt[ent_filt["regime"] == "Authoritarian"]["entity_total_rate"].values[0]
        dem_tot  = ent_filt[ent_filt["regime"] == "Democratic"]["entity_total_rate"].values[0]
        ec1, ec2 = st.columns(2)
        ec1.metric("Auth. total entity rate", f"{auth_tot:.4f}")
        ec2.metric("Dem. total entity rate",  f"{dem_tot:.4f}",
                   delta=f"{(dem_tot - auth_tot):+.4f} vs authoritarian")

    st.info(
        "**Actor density:** Authoritarian discourse is consistently more "
        "entity-dense (0.077 vs 0.061 total), anchoring rhetoric in specific "
        "states, persons, and organisations — a geopolitical referential style. "
        "Democratic discourse is less entity-heavy but shows greater diversity "
        "of actor types, including partisan and ideological references."
    )

# ── C3: radar / spider comparison ────────────────────────────────────────────
st.subheader("Structural Profile Comparison — Radar View")
st.caption(
    "Normalised radar chart comparing key structural features across regimes. "
    "Each axis is scaled 0–1 relative to the maximum value across regimes."
)

if len(pos_reg[pos_reg["regime"].isin(sel_regimes)]) >= 1:
    radar_cols = {
        "Noun share":       ("pos", "noun_share"),
        "Verb share":       ("pos", "verb_share"),
        "Modal share":      ("pos", "modal_share"),
        "Person rate":      ("ent", "person_rate"),
        "GPE rate":         ("ent", "gpe_rate"),
        "Entity density":   ("ent", "entity_total_rate"),
    }

    raw = {}
    for regime in [r for r in ["Authoritarian", "Democratic"] if r in sel_regimes]:
        row = {}
        for label, (src, col) in radar_cols.items():
            src_df = pos_reg if src == "pos" else ent_density
            vals = src_df[src_df["regime"] == regime][col]
            row[label] = float(vals.values[0]) if len(vals) else 0.0
        raw[regime] = row

    # Normalise per axis
    categories = list(radar_cols.keys())
    maxima = {cat: max(raw[r].get(cat, 0) for r in raw) for cat in categories}
    norm = {
        regime: [raw[regime][cat] / maxima[cat] if maxima[cat] > 0 else 0
                 for cat in categories]
        for regime in raw
    }

    # Map regime → valid rgba fill (Plotly rejects 8-digit hex alpha notation)
    RADAR_FILLS = {
        "Democratic":    "rgba(33,150,243,0.18)",   # #2196F3 at 18% opacity
        "Authoritarian": "rgba(229,57,53,0.18)",    # #E53935 at 18% opacity
    }

    fig_radar = go.Figure()
    for regime, vals in norm.items():
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=RADAR_FILLS.get(regime, "rgba(128,128,128,0.18)"),
            line=dict(color=REGIME_COLORS[regime], width=2),
            name=regime,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        legend_title_text="Regime",
        title="Structural Feature Profile (normalised)",
        margin=dict(t=60, b=20),
        height=420,
    )
    st.plotly_chart(fig_radar, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — AFFECTIVE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.header("D — Affective Register: Sentiment by Regime")
st.caption(
    "Distribution of positive, neutral, and negative sentiment labels across "
    "regimes.  Differences reflect both underlying communicative styles and "
    "corpus composition (speech length, formality, translation effects)."
)

sent_filt = sent_reg[sent_reg["regime"].isin(sel_regimes)].copy()
sent_filt["sentiment"] = pd.Categorical(
    sent_filt["sentiment"],
    categories=["negative", "neutral", "positive"],
    ordered=True,
)
sent_filt = sent_filt.sort_values(["regime", "sentiment"])

col_d1, col_d2 = st.columns([3, 2], gap="large")

with col_d1:
    fig_sent = px.bar(
        sent_filt,
        x="regime",
        y="pct",
        color="sentiment",
        text=sent_filt["pct"].apply(lambda v: f"{v:.1f}%"),
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
        title="Sentiment Distribution by Regime",
        labels={"pct": "% of Speeches", "regime": "Regime", "sentiment": "Sentiment"},
    )
    fig_sent.update_traces(textposition="inside", textfont_size=11)
    fig_sent.update_layout(
        barmode="stack",
        yaxis_title="% of Speeches",
        legend_title_text="Sentiment",
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_sent, width="stretch")

with col_d2:
    st.markdown("**Sentiment breakdown**")
    pivot = (
        sent_filt.pivot_table(
            index="regime", columns="sentiment", values="pct", aggfunc="sum"
        )
        .reindex(columns=["positive", "neutral", "negative"])
        .round(1)
    )
    st.dataframe(
        pivot.style.format("{:.1f}%")
             .background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True,
    )

    gap_pos = None
    if "Democratic" in sel_regimes and "Authoritarian" in sel_regimes:
        dem_pos  = float(sent_filt[(sent_filt["regime"] == "Democratic")  & (sent_filt["sentiment"] == "positive")]["pct"].values[0])
        auth_pos = float(sent_filt[(sent_filt["regime"] == "Authoritarian") & (sent_filt["sentiment"] == "positive")]["pct"].values[0])
        gap_pos  = dem_pos - auth_pos
        st.metric("Positive gap (Dem − Auth)", f"{gap_pos:+.1f} pp",
                  help="Percentage point difference in positive sentiment")

st.info(
    "**Affective gap:** Democratic speeches are more positively labelled "
    "(87 % vs 67 % authoritarian), with the 20 percentage-point gap reflecting "
    "a structurally more optimistic, mobilisation-oriented communication style. "
    "Authoritarian discourse shows a substantially higher neutral share (32 % "
    "vs 13 %), consistent with declarative state-communication norms. "
    "Negative sentiment is rare in both regimes (<1 %), confirming that "
    "surface-level sentiment is a structurally constrained feature of "
    "executive rhetoric regardless of institutional type."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-LAYER SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════
st.header("Cross-Layer Synthesis")

c_l1, c_l2 = st.columns(2, gap="large")

with c_l1:
    st.markdown("#### 🔵 Democratic Regime Profile")
    st.markdown(
        """
        | Layer | Finding |
        |---|---|
        | **Salience** | Larger proportional health spike (~6×); weaker baseline security |
        | **Topics** | Economic identity (49 %) + political rhetoric (39 %) |
        | **Syntax** | Verb-dominated (21 %); less noun-heavy |
        | **NER** | Lower entity density (0.061); more partisan actor references |
        | **Sentiment** | More positive (87 %); lower neutral baseline |

        Democratic discourse is **action-oriented, economically framed, and
        identity-driven**, responding to health crises with stronger proportional
        amplification.  The pluralist communicative environment generates more
        diverse actor references and higher positivity but also stronger
        Crisis × Regime interaction effects.
        """
    )

with c_l2:
    st.markdown("#### 🔴 Authoritarian Regime Profile")
    st.markdown(
        """
        | Layer | Finding |
        |---|---|
        | **Salience** | Higher baseline security; moderate health spike (~4×) |
        | **Topics** | Governance authority (50 %) + geopolitical security (45 %) |
        | **Syntax** | Noun-dominated (37 %); state-centric declarative style |
        | **NER** | Higher entity density (0.077); state and leader anchoring |
        | **Sentiment** | Less positive (67 %); high neutral share (32 %) |

        Authoritarian discourse is **declarative, geopolitically anchored, and
        entity-dense**, maintaining elevated baseline security salience that
        crises amplify rather than create.  The state-centric communicative
        logic generates consistent institutional framing across all crisis types,
        with security reallocation showing no significant regime moderation.
        """
    )

st.divider()

st.subheader("⚖️ RQ4 Conclusion")
st.success(
    """
    **Main finding:** Crisis-driven discursive restructuring is **partially universal
    and partially regime-conditioned**, with the conditioning operating differently
    across analytical layers.

    - **Health salience shifts** are regime-conditioned: democratic regimes show
      significantly stronger proportional amplification (Crisis × Democratic
      interaction, p < .05), consistent with electoral accountability incentives
      and responsive governance norms.
    - **Security salience shifts** are systemic: both regimes increase security
      discourse during the Ukraine crisis at comparable rates, with no significant
      interaction term — geopolitical threat activates security framing universally.
    - **Topic structure** is fundamentally regime-conditioned and stable across
      phases: authoritarian discourse occupies a state-security thematic world;
      democratic discourse occupies an economic-identity world.
    - **Syntactic and actor structure** (POS, NER) are regime-level constants
      rather than crisis-reactive features, confirming that linguistic style is
      institutionally anchored.
    - **Sentiment** shows a structural 20 pp positive gap favouring democratic
      regimes, but both converge on near-zero negative rates — the crisis does
      not break the positivity norm in either institutional context.

    Together, these findings support a **dual-mechanism model**: crisis creates
    universal systemic pressures on executive discourse, but institutional context
    determines the magnitude, direction, and linguistic form of the response.
    """
)
