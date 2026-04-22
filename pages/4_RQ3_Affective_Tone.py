"""
pages/4_RQ3_Affective_Tone.py
-------------------------------
RQ3: With what affective intensity are topics communicated?
Covers sentiment distribution, model performance, and LSTM training dynamics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RQ3 – Affective Tone",
    page_icon="🎭",
    layout="wide",
)

DATA = Path(__file__).parent.parent / "dashboard_data"

# ── colour constants ───────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#2E7D32",
    "neutral":  "#9E9E9E",
    "negative": "#C62828",
}
REGIME_COLORS = {"Democratic": "#2196F3", "Authoritarian": "#E53935"}

HEALTH_ORDER   = ["pre_health",   "during_health",   "post_health"]
SECURITY_ORDER = ["pre_security", "during_security"]

PHASE_LABELS = {
    "pre_health":       "Pre-Health",
    "during_health":    "During Health (COVID-19)",
    "post_health":      "Post-Health",
    "pre_security":     "Pre-Security",
    "during_security":  "During Security (Ukraine)",
}

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load(fname):
    df = pd.read_csv(DATA / fname)
    print(f"[RQ3] {fname}: {df.shape}")
    return df

sent_regime   = load("sentiment_by_regime.csv")
sent_country  = load("sentiment_by_country.csv")
sent_phase    = load("sentiment_by_phase.csv")
lstm_hist     = load("lstm_history.csv")
lstm_met      = load("lstm_metrics.csv")
model_comp    = load("model_comparison.csv")

# ── tidy lstm_metrics ─────────────────────────────────────────────────────────
lstm_met = lstm_met.rename(columns={"Unnamed: 0": "label"})

# Add epoch index to lstm_hist for charting
lstm_hist = lstm_hist.copy()
lstm_hist.insert(0, "epoch", range(1, len(lstm_hist) + 1))

# ── page header ───────────────────────────────────────────────────────────────
st.title("🎭 RQ3: With What Affective Intensity Are Topics Communicated?")
st.markdown(
    """
    This layer measures whether crises alter not only *what* is said and *how*,
    but also the **emotional register** in which executive discourse is delivered.
    A supervised sentiment classifier (trained on the labelled corpus) assigns
    each speech a positive, neutral, or negative label.  The distribution of these
    labels — across regimes, countries, and crisis phases — reveals whether crises
    intensify or dampen affective tone.

    Three models are evaluated: Logistic Regression, Linear SVM, and a 5-epoch
    LSTM.  All are trained on the same corpus and evaluated on the same held-out
    test set (20 %, n = 2 923).
    """
)

# ── key insight banner ────────────────────────────────────────────────────────
col_i1, col_i2, col_i3 = st.columns(3)
col_i1.metric(
    "Positive sentiment",
    f"{sent_regime['pct'][sent_regime['sentiment'] == 'positive'].mean():.1f}%",
    help="Average % of speeches labelled positive across regimes",
)
col_i2.metric(
    "Neutral sentiment",
    f"{sent_regime['pct'][sent_regime['sentiment'] == 'neutral'].mean():.1f}%",
    help="Average % of speeches labelled neutral across regimes",
)
col_i3.metric(
    "Negative sentiment",
    f"{sent_regime['pct'][sent_regime['sentiment'] == 'negative'].mean():.1f}%",
    help="Average % of speeches labelled negative across regimes",
)

st.info(
    "**Key insight:** Executive political discourse is overwhelmingly positive "
    "(67–87 % depending on regime), with negative sentiment representing less "
    "than 1 % of all speeches across every group.  This structural positivity "
    "reflects the inherent legitimacy-maintenance function of executive rhetoric, "
    "and creates a severe class-imbalance challenge for supervised sentiment models."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — SENTIMENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
st.header("A — Sentiment Distribution")

# ── A1: by regime and country ─────────────────────────────────────────────────
col_a1, col_a2 = st.columns(2, gap="large")

with col_a1:
    st.subheader("By Regime")

    # Sort sentiments for consistent stacking
    regime_order   = ["Democratic", "Authoritarian"]
    sent_r = sent_regime.copy()
    sent_r["sentiment"] = pd.Categorical(
        sent_r["sentiment"], categories=["negative", "neutral", "positive"], ordered=True
    )
    sent_r = sent_r.sort_values(["regime", "sentiment"])

    fig_regime = px.bar(
        sent_r,
        x="regime",
        y="pct",
        color="sentiment",
        text=sent_r["pct"].apply(lambda v: f"{v:.1f}%"),
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"regime": regime_order,
                         "sentiment": ["positive", "neutral", "negative"]},
        title="Sentiment Distribution by Regime",
        labels={"pct": "% of Speeches", "regime": "Regime", "sentiment": "Sentiment"},
    )
    fig_regime.update_traces(textposition="inside", textfont_size=11)
    fig_regime.update_layout(
        barmode="stack",
        yaxis_title="% of Speeches",
        legend_title_text="Sentiment",
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_regime, width="stretch")

    st.info(
        "**Regime gap:** Democratic speeches are more positively toned (87 %) "
        "than authoritarian ones (67 %), with authoritarian discourse showing a "
        "much higher neutral share (32 % vs 13 %).  This aligns with authoritarian "
        "regimes' more declarative, bureaucratic communication style."
    )

with col_a2:
    st.subheader("By Country")

    # Normalise to 100 % for clean stacking (data already has pct)
    cty_order = ["United States", "United Kingdom", "Russia", "China"]
    sent_c = sent_country.copy()
    sent_c["sentiment"] = pd.Categorical(
        sent_c["sentiment"], categories=["negative", "neutral", "positive"], ordered=True
    )
    sent_c = sent_c.sort_values(["country", "sentiment"])

    fig_country = px.bar(
        sent_c,
        x="country",
        y="pct",
        color="sentiment",
        text=sent_c["pct"].apply(lambda v: f"{v:.1f}%"),
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"country": cty_order,
                         "sentiment": ["positive", "neutral", "negative"]},
        title="Sentiment Distribution by Country",
        labels={"pct": "% of Speeches", "country": "Country", "sentiment": "Sentiment"},
    )
    fig_country.update_traces(textposition="inside", textfont_size=11)
    fig_country.update_layout(
        barmode="stack",
        yaxis_title="% of Speeches",
        legend_title_text="Sentiment",
        margin=dict(t=50, b=10),
        xaxis_tickangle=-15,
    )
    st.plotly_chart(fig_country, width="stretch")

    st.info(
        "**Country comparison:** The US (88 %) and UK (75 %) lead in positive "
        "sentiment.  Russia shows the highest neutral share (33 %) while China "
        "produces zero negative-labelled speeches in this corpus, suggesting "
        "structurally constrained affective range in Chinese executive output."
    )

st.divider()

# ── A2: by crisis phase ────────────────────────────────────────────────────────
st.subheader("Sentiment Across Crisis Phases")
st.caption(
    "Stacked bars show how the balance of positive / neutral / negative sentiment "
    "shifts across crisis periods.  Phase labels are ordered chronologically."
)

tab_ph, tab_ps = st.tabs(["Health Crisis Phases", "Security Crisis Phases"])

def phase_sent_chart(phase_type_val, phase_order, title, color_scale):
    sub = sent_phase[sent_phase["phase_type"] == phase_type_val].copy()
    sub["phase"] = pd.Categorical(sub["phase"], categories=phase_order, ordered=True)
    sub["sentiment"] = pd.Categorical(
        sub["sentiment"], categories=["negative", "neutral", "positive"], ordered=True
    )
    sub = sub.sort_values(["phase", "sentiment"])
    phase_str = sub["phase"].astype(str)
    sub["phase_label"] = phase_str.map(PHASE_LABELS).fillna(phase_str)

    fig = px.bar(
        sub,
        x="phase_label",
        y="pct",
        color="sentiment",
        text=sub["pct"].apply(lambda v: f"{v:.1f}%"),
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
        title=title,
        labels={"pct": "% of Speeches", "phase_label": "Phase", "sentiment": "Sentiment"},
    )
    fig.update_traces(textposition="inside", textfont_size=11)
    fig.update_layout(
        barmode="stack",
        yaxis_title="% of Speeches",
        legend_title_text="Sentiment",
        margin=dict(t=50, b=10),
    )
    return fig

with tab_ph:
    st.plotly_chart(
        phase_sent_chart("health", HEALTH_ORDER,
                         "Sentiment by Health Crisis Phase", "Greens"),
        width="stretch",
    )
    st.info(
        "Positive sentiment is highest in the post-health period (73 %) and "
        "lowest in pre-health (71 %).  The differences are modest — overall "
        "positivity is structurally stable across health phases, suggesting "
        "that crisis does not dramatically shift surface-level sentiment tone "
        "even as salience and framing reorganise substantially."
    )

with tab_ps:
    st.plotly_chart(
        phase_sent_chart("security", SECURITY_ORDER,
                         "Sentiment by Security Crisis Phase", "Reds"),
        width="stretch",
    )
    st.info(
        "Security crisis phases also show stable positive dominance (~71–74 %). "
        "Negative sentiment remains below 1 % in both phases.  "
        "This surface stability contrasts with the deeper affective restructuring "
        "visible in the paper's salience-weighted emotion indices (Layer 3), "
        "where moralization and fear show statistically significant crisis effects."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.header("B — Model Performance: Sentiment Classification")
st.caption(
    "Three models trained on the same corpus (train 80 % / test 20 %, n_test = 2 923). "
    "Accuracy reflects the proportion of correctly classified speeches on the held-out set."
)

# ── B1: overall accuracy ──────────────────────────────────────────────────────
model_acc = (
    model_comp[
        (model_comp["metric"] == "accuracy") &
        (model_comp["class"] == "overall") &
        (model_comp["epoch"].isna())
    ]
    .sort_values("value", ascending=False)
    .copy()
)

col_b1, col_b2, col_b3 = st.columns(3)
model_icons = {"Logistic Regression": "📈", "Linear SVM": "⚡", "LSTM": "🧠"}
model_colors_list = ["#1565C0", "#2E7D32", "#E65100"]

for col, (_, row) in zip([col_b1, col_b2, col_b3], model_acc.iterrows()):
    icon = model_icons.get(row["model"], "📊")
    col.metric(f"{icon} {row['model']}", f"{row['value']:.1%}")

fig_acc = px.bar(
    model_acc,
    x="model",
    y="value",
    color="model",
    text=model_acc["value"].apply(lambda v: f"{v:.1%}"),
    color_discrete_sequence=model_colors_list,
    title="Overall Classification Accuracy (Test Set)",
    labels={"value": "Accuracy", "model": "Model"},
    range_y=[0.0, 1.0],
)
fig_acc.update_traces(textposition="outside")
fig_acc.update_layout(
    showlegend=False,
    yaxis_tickformat=".0%",
    margin=dict(t=50, b=10),
)
st.plotly_chart(fig_acc, width="stretch")

# ── B2: LSTM class-level performance ─────────────────────────────────────────
st.subheader("Class-Level Performance: LSTM")
st.caption(
    "Precision, recall, and F1-score broken down by sentiment class. "
    "Class imbalance (positive ≫ neutral ≫ negative) heavily distorts aggregate metrics."
)

# Filter to per-class rows only (exclude summary rows)
per_class = lstm_met[
    lstm_met["label"].isin(["negative", "neutral", "positive"])
].copy()

metrics_long = per_class.melt(
    id_vars="label",
    value_vars=["precision", "recall", "f1-score"],
    var_name="metric",
    value_name="score",
)

fig_class = px.bar(
    metrics_long,
    x="label",
    y="score",
    color="metric",
    barmode="group",
    text=metrics_long["score"].apply(lambda v: f"{v:.2f}"),
    color_discrete_sequence=["#1565C0", "#2E7D32", "#E65100"],
    category_orders={"label": ["negative", "neutral", "positive"]},
    title="LSTM: Precision / Recall / F1 by Sentiment Class",
    labels={"score": "Score", "label": "Sentiment Class", "metric": "Metric"},
    range_y=[0.0, 1.05],
)
fig_class.update_traces(textposition="outside", textfont_size=10)
fig_class.update_layout(
    legend_title_text="Metric",
    margin=dict(t=50, b=10),
)
st.plotly_chart(fig_class, width="stretch")

# Support count annotation
col_sup1, col_sup2, col_sup3 = st.columns(3)
support_data = per_class.set_index("label")["support"]
for col, cls, color in zip(
    [col_sup1, col_sup2, col_sup3],
    ["negative", "neutral", "positive"],
    ["#C62828", "#616161", "#2E7D32"],
):
    n = int(support_data.get(cls, 0))
    col.metric(
        f"Test samples — {cls}",
        f"{n:,}",
        help=f"Number of {cls} speeches in the held-out test set",
    )

st.warning(
    "**Class imbalance:** The test set contains 2 098 positive, 806 neutral, "
    "and only 19 negative speeches.  The LSTM achieves 0 % recall on the "
    "negative class — it learns to predict positive for almost every input.  "
    "Overall accuracy (70 %) is therefore misleadingly high; the weighted F1 "
    "(0.61) is a more informative measure of true performance."
)

# ── B3: all-model comparison with SVM/Logistic support note ──────────────────
st.subheader("Model Comparison: LSTM vs Classical")
st.caption(
    "LSTM F1-score per class vs overall accuracy for Logistic Regression and SVM. "
    "Classical models produce better overall accuracy despite simpler architecture."
)

compare_rows = []
# Classical — accuracy only
for _, row in model_acc[model_acc["model"] != "LSTM"].iterrows():
    compare_rows.append({"model": row["model"], "class": "overall (acc)", "score": row["value"]})
# LSTM — per class F1 + overall accuracy
for _, row in per_class.iterrows():
    compare_rows.append({"model": "LSTM", "class": f"{row['label']} (F1)", "score": row["f1-score"]})
lstm_overall = model_acc[model_acc["model"] == "LSTM"]["value"].values[0]
compare_rows.append({"model": "LSTM", "class": "overall (acc)", "score": lstm_overall})

compare_df = pd.DataFrame(compare_rows)

fig_cmp = px.bar(
    compare_df,
    x="class",
    y="score",
    color="model",
    barmode="group",
    text=compare_df["score"].apply(lambda v: f"{v:.2f}"),
    color_discrete_map={
        "Logistic Regression": "#1565C0",
        "Linear SVM":          "#2E7D32",
        "LSTM":                "#E65100",
    },
    title="Model Comparison: Accuracy and Class-Level F1 (LSTM)",
    labels={"score": "Score", "class": "Metric / Class", "model": "Model"},
    range_y=[0.0, 1.05],
)
fig_cmp.update_traces(textposition="outside", textfont_size=10)
fig_cmp.update_layout(
    legend_title_text="Model",
    margin=dict(t=55, b=10),
    xaxis_tickangle=-20,
)
st.plotly_chart(fig_cmp, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — LSTM TRAINING DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
st.header("C — LSTM Training Dynamics")
st.caption(
    "Training (solid) vs validation (dashed) curves across 5 epochs. "
    "Divergence between training and validation indicates overfitting."
)

tab_acc_curve, tab_loss_curve = st.tabs(["Accuracy Curves", "Loss Curves"])

def epoch_chart(metric_train, metric_val, y_label, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lstm_hist["epoch"],
        y=lstm_hist[metric_train],
        mode="lines+markers",
        name=f"Training {y_label}",
        line=dict(color="#1565C0", width=2.5),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=lstm_hist["epoch"],
        y=lstm_hist[metric_val],
        mode="lines+markers",
        name=f"Validation {y_label}",
        line=dict(color="#E65100", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="square"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        xaxis=dict(tickmode="linear", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=30),
        hovermode="x unified",
    )
    return fig

with tab_acc_curve:
    st.plotly_chart(
        epoch_chart("accuracy", "val_accuracy", "Accuracy",
                    "LSTM Accuracy: Training vs Validation"),
        width="stretch",
    )

    col_c1, col_c2 = st.columns(2)
    col_c1.metric(
        "Final training accuracy",
        f"{lstm_hist['accuracy'].iloc[-1]:.1%}",
        delta=f"+{lstm_hist['accuracy'].diff().dropna().mean():.1%} avg/epoch",
    )
    col_c2.metric(
        "Final validation accuracy",
        f"{lstm_hist['val_accuracy'].iloc[-1]:.1%}",
        delta=f"{lstm_hist['val_accuracy'].diff().dropna().mean():+.1%} avg/epoch",
    )
    st.info(
        "**Training trajectory:** Training accuracy improves steadily "
        f"({lstm_hist['accuracy'].iloc[0]:.1%} → {lstm_hist['accuracy'].iloc[-1]:.1%}), "
        "while validation accuracy peaks at epoch 1 "
        f"({lstm_hist['val_accuracy'].max():.1%}) and then plateaus or slightly declines. "
        "The gap signals the model has started to overfit to the dominant positive class."
    )

with tab_loss_curve:
    st.plotly_chart(
        epoch_chart("loss", "val_loss", "Loss",
                    "LSTM Loss: Training vs Validation"),
        width="stretch",
    )

    col_l1, col_l2 = st.columns(2)
    col_l1.metric(
        "Final training loss",
        f"{lstm_hist['loss'].iloc[-1]:.4f}",
        delta=f"{lstm_hist['loss'].diff().dropna().mean():.4f} avg/epoch",
    )
    col_l2.metric(
        "Final validation loss",
        f"{lstm_hist['val_loss'].iloc[-1]:.4f}",
        delta=f"{lstm_hist['val_loss'].diff().dropna().mean():+.4f} avg/epoch",
        delta_color="inverse",
    )
    st.info(
        "**Loss divergence:** Training loss falls continuously "
        f"({lstm_hist['loss'].iloc[0]:.3f} → {lstm_hist['loss'].iloc[-1]:.3f}), "
        "but validation loss begins rising from epoch 3 "
        f"({lstm_hist['val_loss'].iloc[2]:.3f} → {lstm_hist['val_loss'].iloc[-1]:.3f}). "
        "This divergence is a textbook overfitting signature and explains why "
        "the LSTM underperforms classical models despite higher model capacity."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — MODEL LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.header("D — Model Limitations and Diagnostic Notes")

col_d1, col_d2 = st.columns(2, gap="large")

with col_d1:
    st.markdown("#### ⚠️ Class Imbalance")
    st.markdown(
        """
        The corpus is structurally skewed: **72 % of speeches are labelled
        positive**, 28 % neutral, and under 1 % negative.  This reflects the
        inherent legitimacy-maintenance function of executive rhetoric — leaders
        systematically project confidence, progress, and optimism.

        **Consequence for models:**
        - A trivial classifier that always predicts *positive* achieves ~72 % accuracy.
        - The LSTM converges on exactly this strategy by epoch 2.
        - True negative-class recall = **0 %** for LSTM, meaning it fails
          completely on the rare but analytically most interesting category.
        """
    )

with col_d2:
    st.markdown("#### 📐 Accuracy vs True Performance")
    st.markdown(
        """
        | Metric | Logistic | SVM | LSTM |
        |---|---|---|---|
        | **Accuracy** | 85.9 % | 86.9 % | 70.1 % |
        | **Weighted F1** | — | — | 0.613 |
        | **Negative recall** | — | — | 0 % |
        | **Neutral F1** | — | — | 0.084 |

        Classical models (Logistic, SVM) achieve higher accuracy with simpler
        architectures because their bag-of-words TF-IDF representation captures
        the lexical patterns that distinguish positive from neutral speeches
        more robustly than sequential LSTM encodings under this level of
        class imbalance.

        **Recommended remediation:** SMOTE oversampling, class-weight adjustment,
        or threshold calibration for the negative class.
        """
    )

st.divider()

# ── synthesis box ─────────────────────────────────────────────────────────────
st.subheader("🎭 RQ3 Synthesis")
st.success(
    """
    **Main finding:** Executive political discourse is structurally positive across
    all countries, regimes, and crisis phases — crises do not overturn this
    baseline positivity at the surface level of sentiment labels.

    - **Democratic speeches** are more positively labelled (87 %) than
      **authoritarian ones** (67 %), with the latter showing a higher neutral
      share consistent with a more declarative, bureaucratic communication style.
    - **Negative sentiment** is near-absent (<1 %), making it analytically
      informative precisely because it marks genuine affective escalation when
      it appears.
    - **Crisis phases** produce only modest shifts in aggregate sentiment
      distribution, confirming that deeper affective restructuring — visible in
      the paper's moralization and fear indices (Layer 3) — is not captured by
      coarse three-class sentiment labels alone.
    - **The LSTM** is limited by class imbalance and overfits within 5 epochs;
      **SVM** remains the most reliable model for this corpus structure.

    Together, these results position affective tone as a structurally constrained
    feature of institutional rhetoric rather than a crisis-reactive variable —
    a finding that strengthens, rather than contradicts, the paper's deeper
    emotion-weighted analysis in Layer 3.
    """
)
