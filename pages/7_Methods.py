"""
pages/7_Methods.py
-------------------
Methodology overview: six-layer analytical architecture,
RQ-to-layer mapping, corpus summary, and validation snapshot.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Methods",
    page_icon="📐",
    layout="wide",
)

DATA = Path(__file__).parent.parent / "dashboard_data"

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_assignments():
    return pd.read_csv(DATA / "topic_document_assignments_trimmed.csv")

@st.cache_data
def load_model_comp():
    return pd.read_csv(DATA / "model_comparison.csv")

@st.cache_data
def load_topic_res():
    return pd.read_csv(DATA / "topic_model_results.csv")

assignments = load_assignments()
model_comp  = load_model_comp()
topic_res   = load_topic_res()

# Derived values
n_speeches  = len(assignments)
n_countries = assignments["country"].nunique()
n_regimes   = assignments["regime"].nunique()
dates       = pd.to_datetime(assignments["date"], errors="coerce").dropna()
year_range  = f"{dates.min().year}–{dates.max().year}" if len(dates) else "2017–2024"

model_acc = (
    model_comp[
        (model_comp["metric"] == "accuracy") &
        (model_comp["class"]  == "overall") &
        (model_comp["epoch"].isna())
    ]
    .set_index("model")["value"]
)
topic_acc = float(topic_res[topic_res["label"] == "accuracy"]["accuracy"].values[0])

# ── page header ───────────────────────────────────────────────────────────────
st.title("📐 Methods — Analytical Architecture")
st.markdown(
    """
    This project implements a **six-layer CRISP-DM structured pipeline** to measure
    crisis-induced discursive restructuring in executive political communication.
    The corpus covers speeches from the United States, United Kingdom, Russia, and
    China between 2016 and 2024, spanning two systemic shocks — the COVID-19
    pandemic and the Ukraine security crisis.

    Each analytical layer builds on the previous, moving from descriptive
    measurement to inferential testing, comparative heterogeneity analysis,
    and predictive modeling using both classical machine learning and deep
    learning architectures.
    """
)

st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.subheader("Corpus at a Glance")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Speeches",           f"{n_speeches:,}")
k2.metric("Countries",          n_countries)
k3.metric("Regimes",            n_regimes)
k4.metric("Period",             year_range)
k5.metric("Analytical Layers",  6)
k6.metric("Research Questions", 5)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL FRAMEWORK  (figure + integrated layer expanders)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Analytical Framework")

st.markdown(
    "This section presents an interactive version of the analytical framework. "
    "The pipeline is operationalized to show how each analytical layer contributes "
    "to measuring discursive restructuring across attention, narrative structure, "
    "affective tone, institutional conditioning, and predictive modeling."
)

st.markdown("")

# ── pipeline CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .fw-block  { border-radius: 6px; overflow: hidden; max-width: 620px;
                 margin: 0 auto; }
    .fw-head   { padding: 10px 18px; color: white; font-weight: 700;
                 font-size: 0.87rem; letter-spacing: 0.07em; }
    .fw-body   { padding: 9px 18px 12px 18px; color: rgba(255,255,255,0.90);
                 font-size: 0.82rem; line-height: 1.80; }
    .fw-lbl    { opacity: 0.62; font-weight: 600; min-width: 90px;
                 display: inline-block; }
    .fw-arrow  { text-align: center; font-size: 1.35rem; color: #78909C;
                 margin: 4px 0; line-height: 1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: centred arrow
def _arrow():
    st.markdown('<div class="fw-arrow">↓</div>', unsafe_allow_html=True)

# ── render the pipeline in a centred column ───────────────────────────────
_lpad, _pipe, _rpad = st.columns([1, 4, 1])

with _pipe:

    # ── CRISIS ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="fw-block">'
        '  <div class="fw-head" style="background:#7B1818; text-align:center;'
        '       font-size:1.0rem; letter-spacing:0.10em; padding:13px 18px">'
        '    ⚡&nbsp; CRISIS'
        '  </div>'
        '  <div class="fw-body" style="background:#9B2020; text-align:center">'
        '    Exogenous Shock &nbsp;·&nbsp; COVID-19 Pandemic &nbsp;·&nbsp; Ukraine War'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _arrow()

    # ── RESEARCH QUESTIONS ───────────────────────────────────────────────
    st.markdown(
        '<div class="fw-block">'
        '  <div class="fw-head" style="background:#1A3A6C; text-align:center">'
        '    RESEARCH QUESTIONS &nbsp;·&nbsp; Analytical Structure'
        '  </div>'
        '  <div class="fw-body" style="background:#1E4480">'
        '    <span class="fw-lbl">RQ1 →</span> What topics receive attention?<br>'
        '    <span class="fw-lbl">RQ2 →</span> How are topics narratively structured?<br>'
        '    <span class="fw-lbl">RQ3 →</span> With what affective intensity?<br>'
        '    <span class="fw-lbl">RQ4 →</span> Are these shifts regime-conditioned?<br>'
        '    <span class="fw-lbl">RQ5 →</span> Can discourse structure be predicted?'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _arrow()

    # ── LAYER DATA ───────────────────────────────────────────────────────
    _LAYERS_VIS = [
        {
            "hc": "#2B5F8A", "bc": "#1E4A6E",
            "label": "LAYER 1 — ISSUE SALIENCE", "rq": "RQ1",
            "rows": [
                ("Methods",  "Dictionaries + TF-IDF"),
                ("Output",   "Topic Attention (Normalized)"),
            ],
        },
        {
            "hc": "#236B5C", "bc": "#1A5045",
            "label": "LAYER 2 — NARRATIVE STRUCTURE", "rq": "RQ2",
            "rows": [
                ("Methods",  "Bigrams + Word2Vec + POS + NER"),
                ("Output",   "Framing + Semantic Shifts + Actors"),
            ],
        },
        {
            "hc": "#B85A18", "bc": "#904510",
            "label": "LAYER 3 — AFFECTIVE TONE", "rq": "RQ3",
            "rows": [
                ("Methods",  "Logistic · SVM · LSTM Sentiment Classification"),
                ("Output",   "Emotional Intensity by Regime · Phase · Country"),
            ],
        },
        {
            "hc": "#852020", "bc": "#681818",
            "label": "LAYER 4 — MACHINE LEARNING + REGIME EFFECTS", "rq": "RQ4",
            "rows": [
                ("Unsupervised", "LDA (Topic Discovery)"),
                ("Supervised",   "SVM (Topic Classification ~95%) · Logistic (Baseline)"),
                ("Inference",    "Crisis × Regime Interaction Models"),
                ("Output",       "Topic Structure + Regime Conditioning"),
            ],
        },
        {
            "hc": "#4A2E80", "bc": "#382060",
            "label": "LAYER 5 — DEEP LEARNING (LSTM)", "rq": "RQ5",
            "rows": [
                ("Methods",  "Tokenization + Embeddings + Sequences"),
                ("Output",   "Sequence-based Sentiment Prediction (~70%)"),
            ],
        },
        {
            "hc": "#1A5C5C", "bc": "#124040",
            "label": "LAYER 6 — MODEL COMPARISON", "rq": "RQ5",
            "rows": [
                ("Models",   "Logistic  |  SVM  |  LSTM"),
                ("Metrics",  "Accuracy · Precision · Recall · F1"),
                ("Output",   "Performance + Interpretability Trade-offs"),
            ],
        },
    ]

    _LAYER_DETAILS = [
        {
            "icon": "📡",
            "objective": (
                "Measure how executive attention is redistributed across policy domains "
                "(health, security, energy, economic risk) before, during, and after "
                "crisis phases."
            ),
            "methods": [
                "Curated domain dictionaries (4 domains, 25–30 keywords each)",
                "Normalized term frequency: keyword count / speech length",
                "Mean salience aggregated by crisis phase",
                "OLS regression: salience ~ crisis_indicator + regime + crisis × regime",
                "Country fixed-effects robustness models",
            ],
            "measures": (
                "Normalised domain keyword density per speech.  "
                "Cohen's d quantifies crisis effect magnitude."
            ),
            "outputs": [
                "salience_by_health_phase.csv",
                "salience_by_security_phase.csv",
                "salience_by_regime_health_phase.csv",
                "salience_by_regime_security_phase.csv",
            ],
            "findings": (
                "Health salience spikes ×5 during COVID-19 (Cohen's d ≈ 0.80). "
                "Security salience rises ~27 % during Ukraine crisis (d ≈ 0.18). "
                "Health amplification is regime-conditioned; security reallocation is systemic."
            ),
        },
        {
            "icon": "🔍",
            "objective": (
                "Examine how salient topics are narratively framed through phrase-level "
                "constructions, semantic reorganisation, syntactic composition, and "
                "actor-level references."
            ),
            "methods": [
                "TF-IDF bigram extraction on phase-level super-documents",
                "PMI-informed noise filtering of structural / transcript artifacts",
                "Word2Vec phase-specific embeddings (vector size 200, window 8, skip-gram)",
                "Cosine similarity shift: anchor ↔ context across pre- and during-crisis phases",
                "POS tagging: noun / verb / adjective / adverb / modal shares",
                "Named Entity Recognition: PERSON, GPE, ORG, NORP density",
            ],
            "measures": (
                "TF-IDF score per phase-distinctive bigram. "
                "Cosine similarity Δ for anchor–context pairs. "
                "Mean POS token share. Mean entity density per speech."
            ),
            "outputs": [
                "top_bigrams_overall.csv",
                "top_bigrams_by_health_phase.csv",
                "top_bigrams_by_security_phase.csv",
                "semantic_drift_summary.csv",
                "pos_by_regime.csv / pos_by_phase.csv",
                "entity_density_by_regime.csv",
                "top_entities_by_country.csv / top_entities_by_regime.csv",
            ],
            "findings": (
                "Health drift avg Δ = 0.041 (health → stability/economy/governance). "
                "Security drift avg Δ = 0.016 (security → sovereignty/territorial integrity). "
                "Authoritarian discourse is noun-dominated (37 %) and entity-dense (0.077); "
                "democratic discourse is more verb-oriented (21 %) and actor-diverse."
            ),
        },
        {
            "icon": "🎭",
            "objective": (
                "Measure the emotional register in which executive discourse is delivered "
                "and whether crises alter affective intensity."
            ),
            "methods": [
                "Supervised sentiment classification: Logistic Regression, Linear SVM, LSTM",
                "Three-class labels: positive / neutral / negative",
                "TF-IDF features (max 5 000, unigrams + bigrams) for classical models",
                "Padded token sequences (max length 300) for LSTM",
                "Train/test split: 80 / 20, stratified, random_state = 42",
                "Evaluation: accuracy, precision, recall, F1 per class, weighted F1",
            ],
            "measures": (
                "% speeches per sentiment class by regime / country / phase. "
                "Class-level precision, recall, F1. "
                "LSTM training and validation accuracy / loss per epoch."
            ),
            "outputs": [
                "sentiment_by_regime.csv",
                "sentiment_by_country.csv",
                "sentiment_by_phase.csv",
                "lstm_history.csv",
                "lstm_metrics.csv",
                "model_comparison.csv",
            ],
            "findings": (
                "Discourse is structurally positive (67–87 % across groups). "
                "Negative sentiment < 1 % in all subgroups. "
                "Democratic speeches more positive (87 % vs 67 %). "
                "SVM outperforms LSTM due to class imbalance; LSTM achieves 0 % negative recall."
            ),
        },
        {
            "icon": "⚖️",
            "objective": (
                "Test whether crisis-driven discursive shifts are universal or conditioned "
                "by regime type, using inferential interaction models and supervised topic "
                "classification."
            ),
            "methods": [
                "Crisis × Regime interaction terms in OLS models",
                "Country fixed-effects for structural baseline control",
                "LDA topic modeling: 5 topics, CountVectorizer, random_state = 42",
                "TF-IDF + Linear SVM topic classifier on LDA-derived labels",
                "Iterative stopword refinement (4 rounds) for LDA coherence",
            ],
            "measures": (
                "Interaction coefficient β₃ (Crisis × Democratic). "
                "p-values and Cohen's d for regime-stratified comparisons. "
                "Topic classification accuracy and per-class F1."
            ),
            "outputs": [
                "topic_document_assignments.csv",
                "topic_top_words.csv",
                "topic_share_by_regime.csv",
                "topic_share_by_country.csv",
                "topic_share_by_health_phase.csv",
                "topic_share_by_security_phase.csv",
                "topic_share_by_regime_health_phase.csv",
                "topic_share_by_regime_security_phase.csv",
                "topic_model_results.csv",
            ],
            "findings": (
                "Health salience amplification is regime-conditioned (β₃ significant, p < .05). "
                "Security reallocation is systemic (β₃ not significant). "
                "Topic structure is fundamentally regime-anchored: authoritarian discourse "
                "occupies a state-security thematic world; democratic discourse an economic-identity world. "
                f"Topic SVM classification accuracy: {topic_acc:.1%}."
            ),
        },
        {
            "icon": "🧠",
            "objective": (
                "Evaluate whether sequential deep learning (LSTM) can capture "
                "discourse structure beyond bag-of-words representations."
            ),
            "methods": [
                "LSTM architecture: Embedding(10 000, 100) → LSTM(64) → Dense(3, softmax)",
                "Vocabulary size: 10 000 tokens",
                "Max sequence length: 300 tokens (padded / truncated)",
                "Training: 5 epochs, batch size 32, categorical cross-entropy loss",
                "No class-weight rebalancing (baseline configuration)",
            ],
            "measures": (
                "Per-epoch training and validation accuracy / loss. "
                "Class-level precision, recall, F1 on held-out test set. "
                "Overfitting diagnostic: train vs val loss divergence."
            ),
            "outputs": [
                "lstm_history.csv  (5 epochs × 4 metrics)",
                "lstm_metrics.csv  (per-class and aggregate)",
                "lstm_predictions.csv  (true vs predicted labels)",
                "lstm_confusion_matrix.png",
            ],
            "findings": (
                f"LSTM overall accuracy: {model_acc.get('LSTM', 0.701):.1%}. "
                "Validation loss rises from epoch 3 (overfitting). "
                "Negative class recall = 0 % — model collapses to majority class. "
                "Weighted F1 = 0.613, well below SVM (0.869). "
                "Class imbalance, short training, and no reweighting explain underperformance."
            ),
        },
        {
            "icon": "📊",
            "objective": (
                "Systematically compare classical (Logistic Regression, SVM) and deep "
                "learning (LSTM) models across both sentiment and topic classification tasks."
            ),
            "methods": [
                "Head-to-head accuracy comparison on the same held-out test set",
                "Per-class precision, recall, F1 for LSTM",
                "Confusion matrix analysis for all three sentiment models",
                "Training curve analysis for LSTM (overfitting diagnosis)",
                "Task-level comparison: sentiment vs topic prediction difficulty",
            ],
            "measures": (
                "Overall accuracy. Weighted F1. Per-class F1 for LSTM. "
                "Train vs validation loss gap at each epoch."
            ),
            "outputs": [
                "model_comparison.csv  (all models, metrics, epochs)",
                "logistic_confusion_matrix.png",
                "svm_confusion_matrix.png",
                "lstm_confusion_matrix.png",
            ],
            "findings": (
                f"Best sentiment model: Linear SVM ({model_acc.get('Linear SVM', 0.869):.1%}). "
                f"Best topic model: SVM + TF-IDF ({topic_acc:.1%}). "
                "SVM advantage over LSTM is structural: TF-IDF captures the lexical contrast "
                "between positive and neutral speeches efficiently without sequential context. "
                "LSTM requires class rebalancing and extended training to be competitive."
            ),
        },
    ]

    # ── render each layer: visual block + expander ────────────────────────
    for i, (vis, det) in enumerate(zip(_LAYERS_VIS, _LAYER_DETAILS)):
        # Build the body rows HTML
        rows_html = "".join(
            f'<span class="fw-lbl">{lbl}:</span> {val}<br>'
            for lbl, val in vis["rows"]
        )
        st.markdown(
            f'<div class="fw-block">'
            f'  <div class="fw-head" style="background:{vis["hc"]}">'
            f'    {vis["label"]}'
            f'    <span style="float:right; opacity:0.65; font-weight:400;'
            f'         font-size:0.78rem; letter-spacing:0">{vis["rq"]}</span>'
            f'  </div>'
            f'  <div class="fw-body" style="background:{vis["bc"]}">'
            f'    {rows_html}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        exp_label = (
            f"{det['icon']}  {vis['label'].split('—')[0].strip()} — "
            f"{vis['label'].split('—')[1].strip()}  ·  Full specification"
        )
        with st.expander(exp_label, expanded=False):
            _el, _er = st.columns([3, 2], gap="large")
            with _el:
                st.markdown("**Objective**")
                st.markdown(det["objective"])
                st.markdown("**Methods**")
                for m in det["methods"]:
                    st.markdown(f"- {m}")
                st.markdown("**What it measures**")
                st.markdown(det["measures"])
            with _er:
                st.markdown("**Outputs**")
                for o in det["outputs"]:
                    st.markdown(
                        f'<span style="font-family:monospace; font-size:0.81rem;'
                        f' background:#F5F5F5; padding:1px 6px; border-radius:3px">'
                        f'{o}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                st.markdown("**Key finding**")
                st.info(det["findings"])

        if i < len(_LAYERS_VIS) - 1:
            _arrow()

    _arrow()

    # ── OUTCOMES ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="fw-block">'
        '  <div class="fw-head" style="background:#37474F; text-align:center">'
        '    DISCURSIVE OUTCOMES'
        '  </div>'
        '  <div class="fw-body" style="background:#263238; text-align:center">'
        '    Cross-layer convergence &nbsp;·&nbsp; Regime conditioning &nbsp;·&nbsp;'
        '    Model benchmarks'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RQ-TO-LAYER MAPPING TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Research Question — Layer Mapping")
st.caption(
    "Each research question is addressed by one or more analytical layers.  "
    "The mapping shows the method used and the primary output at each stage."
)

rq_table = pd.DataFrame([
    {
        "Research Question": "RQ1 — What topics receive attention?",
        "Layer":             "Layer 1",
        "Methods":           "Dictionary salience · LDA topic modeling",
        "Primary Output":    "salience_by_*_phase.csv · topic_share_by_*.csv",
    },
    {
        "Research Question": "RQ2 — How are topics narratively structured?",
        "Layer":             "Layer 2",
        "Methods":           "TF-IDF bigrams · Word2Vec drift · POS · NER",
        "Primary Output":    "top_bigrams_*.csv · semantic_drift_summary.csv · pos_by_*.csv",
    },
    {
        "Research Question": "RQ3 — With what affective intensity?",
        "Layer":             "Layer 3",
        "Methods":           "Logistic · SVM · LSTM sentiment classification",
        "Primary Output":    "sentiment_by_*.csv · lstm_history.csv · lstm_metrics.csv",
    },
    {
        "Research Question": "RQ4 — Universal or regime-conditioned?",
        "Layer":             "Layers 1–3 + 4",
        "Methods":           "Crisis × Regime OLS interaction · topic classification",
        "Primary Output":    "*_by_regime_*.csv · topic_model_results.csv",
    },
    {
        "Research Question": "RQ5 — Can models predict discourse structure and tone?",
        "Layer":             "Layers 4–6",
        "Methods":           "SVM · Logistic · LSTM · confusion matrices · LDA + SVM topic",
        "Primary Output":    "model_comparison.csv · topic_model_results.csv",
    },
])

st.dataframe(rq_table, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL VALIDATION SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Model Validation Snapshot")
st.caption(
    "All models evaluated on the same held-out test set (20 %, n = 2 923).  "
    "Metrics reported: overall accuracy and, for LSTM, weighted F1."
)

v1, v2, v3, v4 = st.columns(4)

v1.metric(
    "Topic SVM (5-class)",
    f"{topic_acc:.1%}",
    help="TF-IDF + Linear SVM on LDA-derived topic labels",
)
v2.metric(
    "Sentiment SVM",
    f"{model_acc.get('Linear SVM', 0.869):.1%}",
    help="Linear SVM on TF-IDF features — best sentiment model",
)
v3.metric(
    "Sentiment Logistic",
    f"{model_acc.get('Logistic Regression', 0.859):.1%}",
    help="Logistic Regression on TF-IDF features",
)
v4.metric(
    "Sentiment LSTM",
    f"{model_acc.get('LSTM', 0.701):.1%}",
    delta="Weighted F1 = 0.613",
    delta_color="off",
    help="5-epoch LSTM — limited by class imbalance",
)

st.markdown("")
col_v1, col_v2 = st.columns(2, gap="large")

with col_v1:
    st.markdown("#### ✅ What the models can predict")
    st.markdown(
        """
        - **Topic structure** is highly predictable (95.3 %) because the five
          LDA topics correspond to genuinely separable vocabulary distributions
          (Russian state-security lexicon vs US economic-identity lexicon).
        - **Positive vs neutral sentiment** is distinguishable by lexical markers
          at ~87 % accuracy using SVM — no sequential context is required.
        - **Regime-level differences** in salience, POS, and NER are large and
          consistent enough to be learned by simple classifiers.
        """
    )

with col_v2:
    st.markdown("#### ⚠️ Model limitations")
    st.markdown(
        """
        - **Negative sentiment** is effectively undetectable under current
          class imbalance (19 test samples, 0 % LSTM recall).
          Remediation: SMOTE oversampling, class-weight adjustment.
        - **LSTM** requires longer training, rebalancing, and potentially a
          larger embedding dimension to match SVM performance.
        - **Salience dictionaries** are manually curated and corpus-specific;
          generalisation to other corpora requires re-validation.
        - **Sentiment labels** are machine-generated (not human-annotated),
          introducing systematic label noise in edge cases.
        """
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY NOTE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Reproducibility")

r1, r2 = st.columns(2, gap="large")

with r1:
    st.markdown("#### Scripts")
    scripts = {
        "build_dashboard_data.py":            "Generates sentiment, POS, NER, salience aggregates",
        "topic_pipeline_dashboard.py":        "Runs LDA + SVM topic pipeline",
        "build_dashboard_discourse_layers.py": "Generates bigram, drift, and salience outputs",
    }
    for script, desc in scripts.items():
        st.markdown(
            f'<code style="font-size:0.82rem">{script}</code><br>'
            f'<span style="color:#607D8B; font-size:0.84rem">{desc}</span><br>',
            unsafe_allow_html=True,
        )

with r2:
    st.markdown("#### Configuration")
    config_rows = [
        ("LDA random_state",          "42"),
        ("LDA n_topics",              "5"),
        ("SVM (sentiment)",           "LinearSVC, random_state=42"),
        ("SVM (topic)",               "LinearSVC, random_state=42"),
        ("TF-IDF max_features",       "5 000"),
        ("TF-IDF ngram_range",        "(1, 2)"),
        ("LSTM vocab_size",           "10 000"),
        ("LSTM max_seq_length",       "300"),
        ("LSTM embedding_dim",        "100"),
        ("LSTM epochs",               "5"),
        ("Train / test split",        "80 / 20, stratified"),
        ("random_state (all splits)", "42"),
    ]
    config_df = pd.DataFrame(config_rows, columns=["Parameter", "Value"])
    st.dataframe(config_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Dashboard built with Streamlit · Plotly · scikit-learn · pandas.  "
    "All outputs loaded exclusively from `dashboard_data/`.  "
    "No models are retrained when running this dashboard.  "
    "Corpus: executive_speech_master_corpus_FINAL.csv — 14 615 speeches."
)
