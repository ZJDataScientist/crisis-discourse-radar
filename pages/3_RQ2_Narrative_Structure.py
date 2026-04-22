"""
pages/3_RQ2_Narrative_Structure.py
------------------------------------
RQ2: How are topics narratively structured?
Covers bigram framing, semantic drift, POS syntax, and NER actor structure.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RQ2 – Narrative Structure",
    page_icon="🔍",
    layout="wide",
)

DATA = Path(__file__).parent.parent / "dashboard_data"

# ── colour constants ───────────────────────────────────────────────────────────
REGIME_COLORS  = {"Democratic": "#2196F3", "Authoritarian": "#E53935"}
DOMAIN_COLORS  = {"security": "#E53935",  "health": "#43A047"}
DRIFT_COLORS   = {"closer": "#2E7D32",    "farther": "#C62828"}
POS_COLORS = {
    "noun_share":  "#1565C0",
    "verb_share":  "#2E7D32",
    "adj_share":   "#F57F17",
    "adv_share":   "#6A1B9A",
    "modal_share": "#00838F",
}
POS_LABELS = {
    "noun_share":  "Noun",
    "verb_share":  "Verb",
    "adj_share":   "Adjective",
    "adv_share":   "Adverb",
    "modal_share": "Modal",
}
ENTITY_COLORS = {
    "PERSON": "#1565C0",
    "GPE":    "#E53935",
    "ORG":    "#FB8C00",
    "NORP":   "#6A1B9A",
}

HEALTH_ORDER   = ["pre_health",   "during_health",   "post_health"]
SECURITY_ORDER = ["pre_security", "during_security"]

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load(fname):
    df = pd.read_csv(DATA / fname)
    print(f"[RQ2] {fname}: {df.shape}")
    return df

bigrams_overall  = load("top_bigrams_overall.csv")
bigrams_health   = load("top_bigrams_by_health_phase.csv")
bigrams_security = load("top_bigrams_by_security_phase.csv")
semantic_drift   = load("semantic_drift_summary.csv")
pos_regime       = load("pos_by_regime.csv")
pos_phase        = load("pos_by_phase.csv")
ent_density      = load("entity_density_by_regime.csv")
ent_country      = load("top_entities_by_country.csv")
ent_regime       = load("top_entities_by_regime.csv")

# ── page header ───────────────────────────────────────────────────────────────
st.title("🔍 RQ2: How Are Topics Narratively Structured?")
st.markdown(
    """
    Beyond measuring *what* receives attention, this layer examines *how* salient
    topics are linguistically and rhetorically constructed.  The analysis moves
    from surface phrase combinations (bigrams) to deep semantic reorganisation
    (Word2Vec drift), and then to structural features — syntactic composition (POS)
    and actor-level density (NER).

    **Four analytical lenses:**

    | Layer | Method | Question |
    |---|---|---|
    | **Phrase framing** | TF-IDF bigrams by phase | What multi-word constructions dominate each period? |
    | **Semantic drift** | Co-occurrence shift across phases | How do anchor-word meanings reorganise? |
    | **Syntactic structure** | POS share by regime / phase | Do regimes differ in noun- vs verb-oriented rhetoric? |
    | **Actor structure** | NER entity density | Who and what is referenced, and how densely? |
    """
)

st.info(
    "**Key insight:** Crises do not merely increase word frequency — they "
    "reorganise the relational architecture of discourse through distinctive "
    "phrase constructions, semantic realignment toward legitimacy and governance "
    "concepts, and structural shifts toward noun-heavy, entity-dense framing "
    "in authoritarian contexts."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — PHRASE-LEVEL FRAMING (BIGRAMS)
# ══════════════════════════════════════════════════════════════════════════════
st.header("A — Phrase-Level Framing: Distinctive Bigrams by Phase")
st.caption(
    "Top bigrams extracted using TF-IDF on phase-level super-documents. "
    "Higher TF-IDF score = more distinctive to that phase relative to other phases. "
    "Select a phase from the dropdown to explore its characteristic phrase constructions."
)

tab_overall, tab_health_bg, tab_security_bg = st.tabs(
    ["Overall Corpus", "Health Crisis Phases", "Security Crisis Phases"]
)

with tab_overall:
    n_overall = st.slider("Number of bigrams to show", 10, 60, 25, key="n_overall")
    top_n = bigrams_overall.head(n_overall).sort_values("frequency")
    fig_ov = px.bar(
        top_n,
        x="frequency",
        y="bigram",
        orientation="h",
        text="frequency",
        title=f"Top {n_overall} Bigrams — Full Corpus (by raw frequency)",
        labels={"frequency": "Frequency", "bigram": ""},
        color_discrete_sequence=["#1565C0"],
    )
    fig_ov.update_traces(textposition="outside")
    fig_ov.update_layout(
        margin=dict(l=10, r=30, t=50, b=10),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(400, n_overall * 22),
    )
    st.plotly_chart(fig_ov, use_container_width=True)

with tab_health_bg:
    phases_h = HEALTH_ORDER
    phase_labels_h = {
        "pre_health":    "Pre-Health (baseline governance)",
        "during_health": "During Health Crisis (COVID-19)",
        "post_health":   "Post-Health (securitisation shift)",
    }
    sel_ph = st.selectbox(
        "Select health phase",
        options=phases_h,
        format_func=lambda p: phase_labels_h.get(p, p),
        key="sel_phase_health",
    )
    n_health = st.slider("Number of bigrams", 5, 25, 15, key="n_health")

    phase_data_h = (
        bigrams_health[bigrams_health["health_crisis_phase"] == sel_ph]
        .head(n_health)
        .sort_values("tfidf_score")
    )
    fig_bh = px.bar(
        phase_data_h,
        x="tfidf_score",
        y="bigram",
        orientation="h",
        text=phase_data_h["tfidf_score"].apply(lambda v: f"{v:.4f}"),
        title=f"Top Bigrams — {phase_labels_h[sel_ph]}",
        labels={"tfidf_score": "TF-IDF Score", "bigram": ""},
        color_discrete_sequence=[DOMAIN_COLORS["health"]],
    )
    fig_bh.update_traces(textposition="outside")
    fig_bh.update_layout(
        margin=dict(l=10, r=60, t=50, b=10),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(350, n_health * 26),
    )
    st.plotly_chart(fig_bh, use_container_width=True)

    interpretation_h = {
        "pre_health":    "Pre-health discourse centres on governance baselines: law enforcement, North Korea, and tax cuts frame a security-and-economy-first executive agenda.",
        "during_health": "During COVID-19, `covid 19` and `ambassador birx` emerge alongside ongoing security vocabulary, showing that pandemic response co-exists with geopolitical framing rather than replacing it.",
        "post_health":   "Post-health discourse is marked by `military operation` and `special military` — a clear securitisation shift as the Ukraine war reorganises executive rhetoric globally.",
    }
    st.info(f"**Phase interpretation:** {interpretation_h[sel_ph]}")

with tab_security_bg:
    phases_s = SECURITY_ORDER
    phase_labels_s = {
        "pre_security":    "Pre-Security (baseline period)",
        "during_security": "During Security Crisis (Ukraine war)",
    }
    sel_ps = st.selectbox(
        "Select security phase",
        options=phases_s,
        format_func=lambda p: phase_labels_s.get(p, p),
        key="sel_phase_security",
    )
    n_security = st.slider("Number of bigrams", 5, 25, 15, key="n_security")

    phase_data_s = (
        bigrams_security[bigrams_security["security_crisis_phase"] == sel_ps]
        .head(n_security)
        .sort_values("tfidf_score")
    )
    fig_bs = px.bar(
        phase_data_s,
        x="tfidf_score",
        y="bigram",
        orientation="h",
        text=phase_data_s["tfidf_score"].apply(lambda v: f"{v:.4f}"),
        title=f"Top Bigrams — {phase_labels_s[sel_ps]}",
        labels={"tfidf_score": "TF-IDF Score", "bigram": ""},
        color_discrete_sequence=[DOMAIN_COLORS["security"]],
    )
    fig_bs.update_traces(textposition="outside")
    fig_bs.update_layout(
        margin=dict(l=10, r=60, t=50, b=10),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(350, n_security * 26),
    )
    st.plotly_chart(fig_bs, use_container_width=True)

    interpretation_s = {
        "pre_security":    "Pre-security rhetoric is dominated by Trump-era governance phrases (`president trump`, `north korea`) alongside law enforcement framing — a pre-conflict security imaginary.",
        "during_security": "`military operation` and `special military` rise sharply during the Ukraine crisis. `president biden` and `xi jinping` surface as key actor references, signalling geopolitical coordination framing.",
    }
    st.info(f"**Phase interpretation:** {interpretation_s[sel_ps]}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — SEMANTIC DRIFT
# ══════════════════════════════════════════════════════════════════════════════
st.header("B — Semantic Drift: How Anchor Word Meanings Shift Across Phases")
st.caption(
    "Co-occurrence drift: change in conditional probability P(context | anchor) "
    "between pre- and during-crisis phases.  Positive = anchor became semantically "
    "*closer* to the context word during the crisis. "
    "Paper-validated Word2Vec cosine similarity values are shown where available."
)

# ── B1: paper-validated pairs (4 per anchor) ─────────────────────────────────
validated = semantic_drift[semantic_drift["paper_cosine_pre"].notna()].copy()

col_b1, col_b2 = st.columns(2, gap="large")

def make_cosine_bump(df_anchor, anchor_label, color):
    """Paired bar chart showing paper cosine similarity pre vs during."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Pre-crisis",
        x=df_anchor["context_word"],
        y=df_anchor["paper_cosine_pre"],
        marker_color="#90A4AE",
        text=df_anchor["paper_cosine_pre"].round(3),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="During crisis",
        x=df_anchor["context_word"],
        y=df_anchor["paper_cosine_during"],
        marker_color=color,
        text=df_anchor["paper_cosine_during"].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title=f'Semantic Proximity of "{anchor_label}" — Word2Vec Cosine Similarity',
        xaxis_title="Context Word",
        yaxis_title="Cosine Similarity",
        yaxis_range=[0, 0.65],
        legend_title_text="Phase",
        margin=dict(t=55, b=10),
    )
    return fig

with col_b1:
    st.subheader("Security Anchor: *\"security\"*")
    sec_val = validated[validated["domain"] == "security"].copy()
    st.plotly_chart(
        make_cosine_bump(sec_val, "security", DOMAIN_COLORS["security"]),
        use_container_width=True,
    )

with col_b2:
    st.subheader("Health Anchor: *\"health\"*")
    hlt_val = validated[validated["domain"] == "health"].copy()
    st.plotly_chart(
        make_cosine_bump(hlt_val, "health", DOMAIN_COLORS["health"]),
        use_container_width=True,
    )

# ── B2: diverging drift chart (all context words) ────────────────────────────
st.subheader("Drift Direction Across All Context Words")
st.caption(
    "Co-occurrence drift values for all anchor–context pairs. "
    "Green bars = anchor moved *closer* to context word during crisis. "
    "Red bars = anchor moved *farther*. "
    "Faceted by domain (security / health)."
)

# Filter out extreme outliers (care/public dominate health cooc)
drift_plot = semantic_drift[
    semantic_drift["cooc_drift"].abs() < 0.15
].copy()
drift_plot["label"] = drift_plot["context_word"] + " (" + drift_plot["domain"] + ")"
drift_plot["color"] = drift_plot["direction"].map(DRIFT_COLORS)
drift_plot = drift_plot.sort_values(["domain", "cooc_drift"])

fig_drift = px.bar(
    drift_plot,
    x="cooc_drift",
    y="context_word",
    facet_col="domain",
    color="direction",
    color_discrete_map=DRIFT_COLORS,
    orientation="h",
    text=drift_plot["cooc_drift"].apply(lambda v: f"{v:+.4f}"),
    title="Co-occurrence Drift: Pre → During Crisis  (anchor ↔ context)",
    labels={"cooc_drift": "Drift (Δ P(context | anchor))", "context_word": "", "direction": "Direction"},
)
fig_drift.update_traces(textposition="outside")
fig_drift.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title()))
fig_drift.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
fig_drift.update_layout(
    margin=dict(t=55, r=80, b=10),
    legend_title_text="Direction",
    height=520,
)
st.plotly_chart(fig_drift, use_container_width=True)

col_di1, col_di2 = st.columns(2)
with col_di1:
    st.info(
        "**Security drift:** During the Ukraine crisis, *security* moves closer to "
        "`sovereignty`, `territorial`, `integrity`, and `alliance` — signalling "
        "legitimacy-centred reframing rather than intensified militarisation. "
        "Association with `stability` and `military` slightly decreases."
    )
with col_di2:
    st.info(
        "**Health drift:** During COVID-19, *health* becomes more associated with "
        "`economy`, `crisis`, `governance`, `pandemic`, and `safety` — indicating "
        "expansion from biomedical containment toward systemic governance framing. "
        "Health drift is larger in magnitude (avg 0.041) than security (avg 0.005)."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — POS SYNTACTIC STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
st.header("C — Syntactic Structure: Part-of-Speech Shares")
st.caption(
    "Mean proportion of each POS category in speech tokens, by regime and crisis phase. "
    "Noun-heavy discourse is associated with declarative, entity-anchored rhetoric; "
    "verb-heavy discourse with procedural, action-oriented framing."
)

pos_cols_show = ["noun_share", "verb_share", "adj_share", "adv_share", "modal_share"]

tab_pos_r, tab_pos_p = st.tabs(["By Regime", "By Phase"])

with tab_pos_r:
    pos_r_long = (
        pos_regime[["regime"] + pos_cols_show]
        .melt(id_vars="regime", var_name="pos", value_name="share")
    )
    pos_r_long["pos_label"] = pos_r_long["pos"].map(POS_LABELS)

    fig_pos_r = px.bar(
        pos_r_long,
        x="pos_label",
        y="share",
        color="regime",
        barmode="group",
        color_discrete_map=REGIME_COLORS,
        text=pos_r_long["share"].apply(lambda v: f"{v:.1%}"),
        title="POS Share by Regime",
        labels={"share": "Mean Token Share", "pos_label": "POS Category", "regime": "Regime"},
    )
    fig_pos_r.update_traces(textposition="outside")
    fig_pos_r.update_layout(
        yaxis_tickformat=".0%",
        legend_title_text="Regime",
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_pos_r, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Authoritarian noun share", f"{pos_regime.loc[pos_regime.regime=='Authoritarian','noun_share'].values[0]:.1%}")
    c1.metric("Democratic noun share",    f"{pos_regime.loc[pos_regime.regime=='Democratic',  'noun_share'].values[0]:.1%}")
    c2.metric("Authoritarian verb share", f"{pos_regime.loc[pos_regime.regime=='Authoritarian','verb_share'].values[0]:.1%}")
    c2.metric("Democratic verb share",    f"{pos_regime.loc[pos_regime.regime=='Democratic',  'verb_share'].values[0]:.1%}")

    st.info(
        "**Structural asymmetry:** Authoritarian discourse is noun-dominated "
        "(37 % vs 24 %), reflecting declarative, state-centred rhetoric anchored "
        "in institutional and geopolitical entities. Democratic discourse is more "
        "verb-oriented (21 % vs 13 %), emphasising procedural responsiveness "
        "and policy action."
    )

with tab_pos_p:
    # Combine health + security phases, label them clearly
    pos_p = pos_phase.copy()
    phase_display = {
        "during_health":    "During Health",
        "post_health":      "Post-Health",
        "pre_health":       "Pre-Health",
        "during_security":  "During Security",
        "pre_security":     "Pre-Security",
    }
    pos_p["phase_label"] = pos_p["phase"].map(phase_display).fillna(pos_p["phase"])
    pos_p["phase_type_label"] = pos_p["phase_type"].str.title()

    pos_p_long = (
        pos_p[["phase_label", "phase_type_label"] + pos_cols_show]
        .melt(id_vars=["phase_label", "phase_type_label"], var_name="pos", value_name="share")
    )
    pos_p_long["pos_label"] = pos_p_long["pos"].map(POS_LABELS)

    fig_pos_p = px.bar(
        pos_p_long,
        x="phase_label",
        y="share",
        color="pos_label",
        barmode="group",
        color_discrete_map={v: POS_COLORS[k] for k, v in POS_LABELS.items()},
        facet_col="phase_type_label",
        text=pos_p_long["share"].apply(lambda v: f"{v:.1%}"),
        title="POS Share by Crisis Phase",
        labels={"share": "Mean Token Share", "phase_label": "Phase", "pos_label": "POS"},
    )
    fig_pos_p.update_traces(textposition="outside", textfont_size=9)
    fig_pos_p.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_pos_p.update_layout(
        yaxis_tickformat=".0%",
        legend_title_text="POS",
        margin=dict(t=60, b=10),
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_pos_p, use_container_width=True)

    st.info(
        "**Phase stability:** POS distributions are relatively stable across "
        "phases (noun ~33–35 %, verb ~15 %), suggesting that syntactic style is "
        "a regime-level structural property rather than a crisis-reactive feature."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — NER / ACTOR STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
st.header("D — Actor Structure: Named Entity Recognition (NER)")
st.caption(
    "Entity density = proportion of speech tokens that are named entities of each type. "
    "Higher density indicates more entity-anchored, referential rhetoric."
)

# ── D1: entity density by regime ─────────────────────────────────────────────
st.subheader("Entity Density by Regime")

ent_rate_cols = ["person_rate", "org_rate", "gpe_rate", "norp_rate"]
ent_rate_labels = {
    "person_rate": "Person",
    "org_rate":    "Organisation",
    "gpe_rate":    "Geopolitical Entity",
    "norp_rate":   "Nationality / Group",
}

ent_d_long = (
    ent_density[["regime"] + ent_rate_cols]
    .melt(id_vars="regime", var_name="entity_type", value_name="rate")
)
ent_d_long["entity_label"] = ent_d_long["entity_type"].map(ent_rate_labels)

col_ent1, col_ent2 = st.columns([3, 2], gap="large")

with col_ent1:
    fig_ent_d = px.bar(
        ent_d_long,
        x="entity_label",
        y="rate",
        color="regime",
        barmode="group",
        color_discrete_map=REGIME_COLORS,
        text=ent_d_long["rate"].apply(lambda v: f"{v:.4f}"),
        title="Mean Entity Rate by Regime (tokens per speech)",
        labels={"rate": "Entity Rate", "entity_label": "Entity Type", "regime": "Regime"},
    )
    fig_ent_d.update_traces(textposition="outside")
    fig_ent_d.update_layout(
        legend_title_text="Regime",
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_ent_d, use_container_width=True)

with col_ent2:
    st.markdown("**Total entity density**")
    for _, row in ent_density.iterrows():
        st.metric(
            label=row["regime"],
            value=f"{row['entity_total_rate']:.4f}",
            delta=f"{'higher' if row['entity_total_rate'] == ent_density['entity_total_rate'].max() else 'lower'} density",
        )
    st.info(
        "Authoritarian discourse is consistently more entity-dense (~0.077 vs ~0.061), "
        "indicating more referential, actor-anchored framing across all entity types."
    )

# ── D2: top entities by country / regime ─────────────────────────────────────
st.subheader("Top Named Entities")

tab_ent_c, tab_ent_r = st.tabs(["By Country", "By Regime"])

def plot_top_entities(df, group_col, group_val, label_type, n_top=15):
    sub = (
        df[(df[group_col] == group_val) & (df["label"] == label_type)]
        .head(n_top)
        .sort_values("count")
    )
    color = ENTITY_COLORS.get(label_type, "#455A64")
    fig = px.bar(
        sub,
        x="count",
        y="entity",
        orientation="h",
        text="count",
        title=f"Top {label_type} Entities — {group_val}",
        labels={"count": "Frequency", "entity": ""},
        color_discrete_sequence=[color],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=10, r=50, t=50, b=10),
        height=max(350, n_top * 26),
    )
    return fig

with tab_ent_c:
    sel_country = st.selectbox(
        "Country",
        sorted(ent_country["country"].unique()),
        key="ent_country_sel",
    )
    sel_label_c = st.selectbox(
        "Entity Type",
        ["PERSON", "GPE", "ORG", "NORP"],
        key="ent_label_c",
    )
    n_ent_c = st.slider("Top N", 5, 20, 15, key="n_ent_c")

    st.plotly_chart(
        plot_top_entities(ent_country, "country", sel_country, sel_label_c, n_ent_c),
        use_container_width=True,
    )

with tab_ent_r:
    sel_regime_e = st.selectbox(
        "Regime",
        sorted(ent_regime["regime"].unique()),
        key="ent_regime_sel",
    )
    sel_label_r = st.selectbox(
        "Entity Type",
        ["PERSON", "GPE", "ORG", "NORP"],
        key="ent_label_r",
    )
    n_ent_r = st.slider("Top N", 5, 20, 15, key="n_ent_r")

    st.plotly_chart(
        plot_top_entities(ent_regime, "regime", sel_regime_e, sel_label_r, n_ent_r),
        use_container_width=True,
    )

    entity_insights = {
        ("Authoritarian", "GPE"):    "Russia and China dominate GPE references in authoritarian speeches, reflecting strong territorial and state-centric anchoring.",
        ("Authoritarian", "PERSON"): "Vladimir Putin dominates PERSON references (22 K), confirming personalised executive authority framing in Russian discourse.",
        ("Democratic",    "GPE"):    "America, United States, and China lead GPE mentions — consistent with both domestic identity framing and foreign policy emphasis.",
        ("Democratic",    "PERSON"): "Trump, Biden, and Obama cycle through as dominant PERSON references — democratic discourse is pluralised across leaders.",
        ("Authoritarian", "ORG"):    "State institutions (Government, State Duma, Presidential Executive Office) anchor authoritarian organisational framing.",
        ("Democratic",    "ORG"):    "Congress, Senate, NATO, and White House reflect institutional pluralism and alliance-oriented framing.",
        ("Authoritarian", "NORP"):   "Russian and Chinese nationalities dominate, reinforcing ethno-national state identity.",
        ("Democratic",    "NORP"):   "American, Democrats, Republicans — partisan identity is a distinctive feature of democratic political framing.",
    }
    insight_key = (sel_regime_e, sel_label_r)
    if insight_key in entity_insights:
        st.info(f"**Insight:** {entity_insights[insight_key]}")

st.divider()

# ── synthesis box ─────────────────────────────────────────────────────────────
st.subheader("🔬 RQ2 Synthesis")
st.success(
    """
    **Main finding:** Crises reorganise executive discourse not only at the
    surface level (bigram frequency) but also at the structural and semantic
    levels, with the direction and magnitude of change varying systematically
    by crisis type and regime.

    - **Health crises** produce stronger semantic drift (avg Δ = 0.041 vs 0.005
      for security), embedding "health" within governance and economic stability
      rather than purely biomedical frames.
    - **Security crises** activate sovereignty- and legitimacy-centred phrase
      constructions (`military operation`, `special military`) without triggering
      proportional fear escalation — consistent with normative stabilisation logic.
    - **Regime structure** conditions both syntactic (noun vs verb dominance) and
      actor density (authoritarian: 0.077 vs democratic: 0.061), confirming that
      narrative architecture is institutionally anchored, not merely crisis-reactive.

    Together, bigram distinctiveness, semantic anchor drift, and POS/NER patterns
    converge on a consistent picture: *crises restructure how topics are narrated,
    not just which topics appear*.
    """
)
