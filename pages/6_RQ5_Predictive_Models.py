"""
pages/6_RQ5_Predictive_Models.py
----------------------------------
RQ5: Can ML and deep learning models predict discourse structure and tone?
Covers model accuracy, LSTM training dynamics, class performance,
confusion matrices, and topic classification.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RQ5 – Predictive Models",
    page_icon="🤖",
    layout="wide",
)

DATA = Path(__file__).parent.parent / "dashboard_data"

# ── colour constants ───────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#1565C0",
    "Linear SVM":          "#2E7D32",
    "LSTM":                "#E65100",
    "SVM (Topic)":         "#6A1B9A",
}
SENTIMENT_COLORS = {
    "positive": "#2E7D32",
    "neutral":  "#9E9E9E",
    "negative": "#C62828",
}
TOPIC_COLORS = {
    "Global Development and Multilateral Cooperation":              "#1565C0",
    "Centralized Governance and Domestic State Authority":          "#C62828",
    "National Economic Identity and Labor Framing":                 "#2E7D32",
    "Personalized Political Rhetoric and Executive Communication":  "#F57F17",
    "Geopolitical Security and Strategic Relations":                "#6A1B9A",
}
METRIC_COLORS = {
    "precision": "#1565C0",
    "recall":    "#2E7D32",
    "f1-score":  "#E65100",
}

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load(fname):
    df = pd.read_csv(DATA / fname)
    print(f"[RQ5] {fname}: {df.shape}")
    return df

@st.cache_data
def load_image(fname):
    path = DATA / fname
    return path if path.exists() else None

model_comp   = load("model_comparison.csv")
lstm_hist    = load("lstm_history.csv")
lstm_met     = load("lstm_metrics.csv")
topic_res    = load("topic_model_results.csv")

lstm_hist    = lstm_hist.copy()
lstm_hist.insert(0, "epoch", range(1, len(lstm_hist) + 1))

lstm_met     = lstm_met.rename(columns={"Unnamed: 0": "label"})
per_class    = lstm_met[lstm_met["label"].isin(["negative", "neutral", "positive"])].copy()
topic_class  = topic_res[topic_res["label"].str.match(r"^\d+$")].copy()
topic_class["topic_name"] = topic_class["topic_name"].str.strip()

# Pull overall accuracy rows (no epoch rows)
model_acc = (
    model_comp[
        (model_comp["metric"] == "accuracy") &
        (model_comp["class"] == "overall") &
        (model_comp["epoch"].isna())
    ]
    .sort_values("value", ascending=False)
    .copy()
)
topic_accuracy = float(topic_res[topic_res["label"] == "accuracy"]["accuracy"].values[0])

# ── page header ───────────────────────────────────────────────────────────────
st.title("🤖 RQ5: Can Models Predict Discourse Structure and Emotional Tone?")
st.markdown(
    """
    This layer evaluates whether the discursive patterns identified in Layers 1–4
    are not only descriptively observable but also **computationally predictable**.
    Two prediction tasks are benchmarked:

    - **Sentiment classification** — predicting the affective label (positive /
      neutral / negative) of each speech using Logistic Regression, Linear SVM,
      and a 5-epoch LSTM.
    - **Topic classification** — predicting the dominant LDA topic of each speech
      using a TF-IDF + SVM pipeline trained on LDA-assigned labels.

    Model performance is evaluated on a held-out test set (20 %, n = 2 923)
    using accuracy, precision, recall, and F1-score per class.
    """
)

# ── KPI banner ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best sentiment model",    "Linear SVM",  help="Highest overall accuracy on sentiment task")
k2.metric("SVM sentiment accuracy",  f"{model_acc[model_acc['model']=='Linear SVM']['value'].values[0]:.1%}")
k3.metric("LSTM sentiment accuracy", f"{model_acc[model_acc['model']=='LSTM']['value'].values[0]:.1%}",
          delta=f"{model_acc[model_acc['model']=='LSTM']['value'].values[0] - model_acc[model_acc['model']=='Linear SVM']['value'].values[0]:+.1%} vs SVM")
k4.metric("Topic classification accuracy", f"{topic_accuracy:.1%}", help="TF-IDF + SVM on LDA topic labels")

st.info(
    "**Key insight:** Classical models (SVM, Logistic Regression) outperform the "
    "LSTM on sentiment classification despite lower model complexity.  The LSTM "
    "achieves 0 % recall on the negative class, collapsing predictions to the "
    "dominant positive class under severe class imbalance.  "
    "Topic classification with SVM + TF-IDF reaches **95.3 % accuracy**, "
    "demonstrating that discourse structure is highly predictable once LDA-derived "
    "labels provide a balanced, policy-relevant target variable."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — MODEL ACCURACY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.header("A — Model Accuracy: Sentiment Classification")
st.caption(
    "Overall classification accuracy on the held-out test set (n = 2 923). "
    "All models trained on the same TF-IDF features (Logistic / SVM) or "
    "tokenised sequence embeddings (LSTM)."
)

col_a1, col_a2 = st.columns([3, 2], gap="large")

with col_a1:
    fig_acc = px.bar(
        model_acc,
        x="model",
        y="value",
        color="model",
        text=model_acc["value"].apply(lambda v: f"{v:.1%}"),
        color_discrete_map=MODEL_COLORS,
        title="Overall Accuracy by Model — Sentiment Task",
        labels={"value": "Accuracy", "model": "Model"},
        range_y=[0.0, 1.0],
        category_orders={"model": ["Linear SVM", "Logistic Regression", "LSTM"]},
    )
    fig_acc.update_traces(textposition="outside", textfont_size=13)
    fig_acc.update_layout(
        showlegend=False,
        yaxis_tickformat=".0%",
        margin=dict(t=55, b=10),
    )
    st.plotly_chart(fig_acc, width="stretch")

with col_a2:
    st.markdown("**Model summary**")
    for _, row in model_acc.iterrows():
        color = MODEL_COLORS.get(row["model"], "#455A64")
        st.markdown(
            f"<span style='color:{color}; font-weight:bold'>■</span> "
            f"**{row['model']}** — {row['value']:.1%}",
            unsafe_allow_html=True,
        )
    st.markdown("")
    st.markdown(
        """
        **Why does SVM beat LSTM?**

        - SVM operates on TF-IDF vectors that capture the sharp lexical
          contrast between positive and neutral speeches efficiently.
        - The LSTM receives padded sequences of 300 tokens and learns
          sequential dependencies, but the 72 % positive class dominance
          causes it to converge on a trivial majority-class strategy.
        - With only 5 epochs and no class-weight rebalancing, the LSTM
          never learns to distinguish minority classes.
        """
    )

st.info(
    "**Interpretation:** The SVM advantage is not a failure of deep learning — "
    "it reflects the mismatch between LSTM capacity and data structure.  "
    "A bag-of-words representation is sufficient when sentiment is carried by "
    "lexical markers (positive: *growth*, *proud*, *achieve*; "
    "neutral: *said*, *stated*, *noted*) rather than sequential dependencies."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — LSTM TRAINING DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
st.header("B — LSTM Training Dynamics")
st.caption(
    "Training vs validation curves across 5 epochs.  "
    "Solid lines = training set.  Dashed lines = validation set."
)

tab_acc_c, tab_loss_c = st.tabs(["Accuracy Curves", "Loss Curves"])

def training_curve(y_train, y_val, y_label, title, y_fmt=".2f"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lstm_hist["epoch"], y=lstm_hist[y_train],
        mode="lines+markers", name=f"Train {y_label}",
        line=dict(color="#1565C0", width=2.5),
        marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=lstm_hist["epoch"], y=lstm_hist[y_val],
        mode="lines+markers", name=f"Validation {y_label}",
        line=dict(color="#E65100", width=2.5, dash="dash"),
        marker=dict(size=9, symbol="square"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        xaxis=dict(tickmode="linear", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=65, b=30),
    )
    return fig

with tab_acc_c:
    st.plotly_chart(
        training_curve("accuracy", "val_accuracy", "Accuracy",
                       "LSTM Accuracy — Training vs Validation"),
        width="stretch",
    )
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Train accuracy (epoch 1)", f"{lstm_hist['accuracy'].iloc[0]:.1%}")
    col_b2.metric("Train accuracy (epoch 5)", f"{lstm_hist['accuracy'].iloc[-1]:.1%}",
                  delta=f"{lstm_hist['accuracy'].iloc[-1] - lstm_hist['accuracy'].iloc[0]:+.1%}")
    col_b3.metric("Best val accuracy",
                  f"{lstm_hist['val_accuracy'].max():.1%}",
                  help=f"Epoch {lstm_hist['val_accuracy'].idxmax() + 1}")
    st.info(
        f"Training accuracy climbs steadily "
        f"({lstm_hist['accuracy'].iloc[0]:.1%} → {lstm_hist['accuracy'].iloc[-1]:.1%}), "
        f"while validation accuracy peaks at epoch "
        f"{lstm_hist['val_accuracy'].idxmax() + 1} "
        f"({lstm_hist['val_accuracy'].max():.1%}) and then plateaus.  "
        "The gap widens from epoch 3 onward, indicating the model is "
        "memorising training distribution rather than generalising."
    )

with tab_loss_c:
    st.plotly_chart(
        training_curve("loss", "val_loss", "Loss",
                       "LSTM Loss — Training vs Validation"),
        width="stretch",
    )
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.metric("Train loss (epoch 1)", f"{lstm_hist['loss'].iloc[0]:.4f}")
    col_l2.metric("Train loss (epoch 5)", f"{lstm_hist['loss'].iloc[-1]:.4f}",
                  delta=f"{lstm_hist['loss'].iloc[-1] - lstm_hist['loss'].iloc[0]:+.4f}")
    col_l3.metric("Final val loss",       f"{lstm_hist['val_loss'].iloc[-1]:.4f}",
                  delta=f"{lstm_hist['val_loss'].iloc[-1] - lstm_hist['val_loss'].iloc[0]:+.4f}",
                  delta_color="inverse")
    st.info(
        "Training loss decreases monotonically "
        f"({lstm_hist['loss'].iloc[0]:.3f} → {lstm_hist['loss'].iloc[-1]:.3f}), "
        "but validation loss begins rising from epoch 3 "
        f"({lstm_hist['val_loss'].iloc[2]:.3f} → {lstm_hist['val_loss'].iloc[-1]:.3f}).  "
        "This divergence is a textbook overfitting signature: the model is "
        "reducing training error by assigning high confidence to the majority "
        "class, rather than improving discrimination across all three classes."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — CLASS-LEVEL PERFORMANCE + CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
st.header("C — Class-Level Performance: Sentiment Models")

# ── C1: LSTM precision / recall / F1 per class ───────────────────────────────
st.subheader("LSTM — Per-Class Metrics")
st.caption(
    "Precision, recall, and F1-score for each sentiment class on the test set.  "
    "Class imbalance (positive: 2 098 · neutral: 806 · negative: 19) "
    "makes the negative class practically unlearnable without rebalancing."
)

metrics_long = per_class.melt(
    id_vars="label",
    value_vars=["precision", "recall", "f1-score"],
    var_name="metric",
    value_name="score",
)

fig_cls = px.bar(
    metrics_long,
    x="label",
    y="score",
    color="metric",
    barmode="group",
    text=metrics_long["score"].apply(lambda v: f"{v:.2f}"),
    color_discrete_map=METRIC_COLORS,
    category_orders={"label": ["negative", "neutral", "positive"]},
    title="LSTM: Precision · Recall · F1 per Sentiment Class",
    labels={"score": "Score", "label": "Sentiment Class", "metric": "Metric"},
    range_y=[0.0, 1.1],
)
fig_cls.update_traces(textposition="outside", textfont_size=10)
fig_cls.update_layout(
    legend_title_text="Metric",
    margin=dict(t=55, b=10),
)
st.plotly_chart(fig_cls, width="stretch")

# Summary metrics row
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("LSTM overall accuracy",   f"{float(lstm_met[lstm_met['label']=='accuracy']['accuracy'].values[0]):.1%}")
s2.metric("Positive F1",             f"{float(per_class[per_class['label']=='positive']['f1-score'].values[0]):.3f}")
s3.metric("Neutral F1",              f"{float(per_class[per_class['label']=='neutral' ]['f1-score'].values[0]):.3f}")
s4.metric("Negative F1",             f"{float(per_class[per_class['label']=='negative']['f1-score'].values[0]):.3f}")
s5.metric("Weighted F1",             f"{float(lstm_met[lstm_met['label']=='weighted avg']['f1-score'].values[0]):.3f}")

st.warning(
    "**Class imbalance alert:** The LSTM achieves F1 = 0.000 on the negative "
    "class and F1 = 0.084 on neutral — both near-zero.  The high overall accuracy "
    "(70.1 %) is misleading: a naive majority-class classifier achieves ~72 % "
    "without learning anything.  Weighted F1 (0.613) is the appropriate "
    "performance summary for this imbalanced distribution."
)

st.divider()

# ── C2: confusion matrices ────────────────────────────────────────────────────
st.subheader("Confusion Matrices")
st.caption(
    "Each cell shows the number of test speeches classified into each "
    "predicted class.  Diagonal = correct predictions."
)

cm_tab_lstm, cm_tab_svm, cm_tab_lr = st.tabs(
    ["LSTM", "Linear SVM", "Logistic Regression"]
)

def show_confusion_matrix(fname, caption_text):
    img_path = load_image(fname)
    if img_path:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(img_path), caption=caption_text, width=500)
    else:
        st.warning(f"Image not found: {fname}")

with cm_tab_lstm:
    show_confusion_matrix(
        "lstm_confusion_matrix.png",
        "LSTM Confusion Matrix — test set (n = 2 923)",
    )
    st.info(
        "The LSTM matrix reveals the collapse to majority-class prediction: "
        "virtually all speeches are classified as *positive*, with almost no "
        "true positives for *negative* or *neutral* classes.  "
        "This pattern is consistent with a model that has not learned "
        "meaningful class boundaries under imbalance."
    )

with cm_tab_svm:
    show_confusion_matrix(
        "svm_confusion_matrix.png",
        "Linear SVM Confusion Matrix — test set (n = 2 923)",
    )
    st.info(
        "SVM achieves substantially better class separation, particularly "
        "for the neutral class, while maintaining high positive-class recall.  "
        "Some negative speeches are misclassified as neutral or positive — "
        "a consequence of the extremely small negative class (19 samples)."
    )

with cm_tab_lr:
    show_confusion_matrix(
        "logistic_confusion_matrix.png",
        "Logistic Regression Confusion Matrix — test set (n = 2 923)",
    )
    st.info(
        "Logistic Regression shows a performance profile similar to SVM.  "
        "The slight accuracy gap (85.9 % vs 86.9 %) suggests SVM's maximum-margin "
        "hyperplane marginally better separates the positive / neutral boundary "
        "in TF-IDF feature space."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — TOPIC PREDICTABILITY
# ══════════════════════════════════════════════════════════════════════════════
st.header("D — Topic Classification: TF-IDF + SVM")
st.caption(
    "5-class topic classifier trained on LDA-assigned dominant topic labels.  "
    "TF-IDF bigram features (n-gram 1–2, max 5 000 features), Linear SVM."
)

col_d1, col_d2 = st.columns([2, 3], gap="large")

with col_d1:
    st.metric(
        "Topic Classification Accuracy",
        f"{topic_accuracy:.1%}",
        help="TF-IDF + Linear SVM on 5-class LDA topic labels, test set n = 2 923",
    )
    st.markdown("")

    # Macro and weighted averages
    macro_row    = topic_res[topic_res["label"] == "macro avg"].iloc[0]
    weighted_row = topic_res[topic_res["label"] == "weighted avg"].iloc[0]

    st.metric("Macro F1",    f"{float(macro_row['f1-score']):.3f}")
    st.metric("Weighted F1", f"{float(weighted_row['f1-score']):.3f}")

    st.success(
        "**Why is topic accuracy so high?**\n\n"
        "Topic labels are derived from the same TF-IDF feature space used by "
        "the SVM classifier — the classification task is essentially learning "
        "to reproduce the LDA-derived clustering using a supervised linear "
        "boundary.  With 5 well-separated topics and strong vocabulary "
        "differentiation (Russian vs US lexicon, economic vs security terms), "
        "SVM achieves near-ceiling performance."
    )

with col_d2:
    st.subheader("Per-Topic F1 Score")

    topic_f1_df = topic_class.sort_values("f1-score", ascending=True).copy()
    topic_f1_df["f1_label"] = topic_f1_df["f1-score"].apply(lambda v: f"{v:.3f}")

    fig_topic_f1 = px.bar(
        topic_f1_df,
        x="f1-score",
        y="topic_name",
        orientation="h",
        text="f1_label",
        color="topic_name",
        color_discrete_map=TOPIC_COLORS,
        title="TF-IDF + SVM: F1-Score per Topic Class",
        labels={"f1-score": "F1-Score", "topic_name": ""},
        range_x=[0.0, 1.05],
    )
    fig_topic_f1.update_traces(textposition="outside", showlegend=False)
    fig_topic_f1.update_layout(
        showlegend=False,
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=10, r=60, t=55, b=10),
        height=320,
    )
    st.plotly_chart(fig_topic_f1, width="stretch")

    st.subheader("Per-Topic Precision · Recall · F1")
    topic_prec_df = topic_class.melt(
        id_vars="topic_name",
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric", value_name="score",
    ).copy()
    topic_prec_df["score_label"] = topic_prec_df["score"].apply(lambda v: f"{v:.2f}")

    fig_topic_prec = px.bar(
        topic_prec_df,
        x="topic_name",
        y="score",
        color="metric",
        barmode="group",
        text="score_label",
        color_discrete_map=METRIC_COLORS,
        title="Per-Topic: Precision · Recall · F1",
        labels={"score": "Score", "topic_name": "Topic", "metric": "Metric"},
        range_y=[0.0, 1.1],
    )
    fig_topic_prec.update_traces(textposition="outside", textfont_size=9)
    fig_topic_prec.update_layout(
        legend_title_text="Metric",
        xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
        margin=dict(t=55, b=60),
    )
    st.plotly_chart(fig_topic_prec, width="stretch")

st.info(
    "**Weakest class:** *Global Development and Multilateral Cooperation* "
    f"(F1 = {float(topic_class[topic_class['topic_name'].str.contains('Global')]['f1-score'].values[0]):.3f}) "
    "is the smallest class (149 test samples) and shows the most overlap "
    "with the Geopolitical Security topic — both share international and "
    "cooperation vocabulary.  All other topics exceed F1 = 0.94."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON INSIGHT BLOCK
# ══════════════════════════════════════════════════════════════════════════════
st.header("Model Comparison Insight")

col_e1, col_e2, col_e3 = st.columns(3, gap="large")

with col_e1:
    st.markdown(
        f"""
        #### 📈 Logistic Regression
        **Accuracy: {model_acc[model_acc['model']=='Logistic Regression']['value'].values[0]:.1%}**

        - Simple linear classifier on TF-IDF vectors
        - Strong baseline; interpretable coefficients
        - Slightly outperformed by SVM on this corpus
        - Better calibrated probability estimates than SVM
        - **Best use:** Interpretability + probability outputs
        """
    )

with col_e2:
    st.markdown(
        f"""
        #### ⚡ Linear SVM
        **Accuracy: {model_acc[model_acc['model']=='Linear SVM']['value'].values[0]:.1%}**

        - Maximum-margin linear classifier on TF-IDF
        - Best overall accuracy on sentiment task
        - Excellent performance on topic task (95.3 %)
        - Robust to high-dimensional sparse features
        - **Best use:** Production sentiment/topic classifier for this corpus
        """
    )

with col_e3:
    st.markdown(
        f"""
        #### 🧠 LSTM
        **Accuracy: {model_acc[model_acc['model']=='LSTM']['value'].values[0]:.1%}**

        - Sequential deep learning on padded token embeddings
        - High model capacity; learns temporal dependencies
        - Underperforms due to class imbalance + 5-epoch training
        - Weighted F1 = 0.613 vs accuracy 70.1 % — misleading metric
        - **Best use:** Longer training + class reweighting + more data
        """
    )

# Full comparison table
st.subheader("Full Performance Table")

comparison_rows = [
    {"Model": "Logistic Regression", "Task": "Sentiment",
     "Accuracy": f"{model_acc[model_acc['model']=='Logistic Regression']['value'].values[0]:.1%}",
     "Weighted F1": "—", "Neg. Recall": "—", "Notes": "Strong TF-IDF baseline"},
    {"Model": "Linear SVM",          "Task": "Sentiment",
     "Accuracy": f"{model_acc[model_acc['model']=='Linear SVM']['value'].values[0]:.1%}",
     "Weighted F1": "—", "Neg. Recall": "—", "Notes": "Best sentiment model"},
    {"Model": "LSTM",                 "Task": "Sentiment",
     "Accuracy": f"{model_acc[model_acc['model']=='LSTM']['value'].values[0]:.1%}",
     "Weighted F1": f"{float(lstm_met[lstm_met['label']=='weighted avg']['f1-score'].values[0]):.3f}",
     "Neg. Recall": "0 %", "Notes": "Collapses to majority class"},
    {"Model": "SVM (Topic)",          "Task": "Topic (5-class)",
     "Accuracy": f"{topic_accuracy:.1%}",
     "Weighted F1": f"{float(weighted_row['f1-score']):.3f}",
     "Neg. Recall": "N/A", "Notes": "Near-ceiling topic prediction"},
]
st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

st.divider()

# ── conclusion ────────────────────────────────────────────────────────────────
st.subheader("🤖 RQ5 Conclusion")
st.success(
    """
    **Main finding:** Executive discourse structure and emotional tone are
    **computationally predictable**, but the degree of predictability is
    task- and model-dependent.

    - **Topic structure is highly predictable** (SVM accuracy 95.3 %, weighted
      F1 = 0.953), confirming that the five LDA-derived topics correspond to
      genuinely separable vocabulary distributions that a linear classifier
      can reliably recover.
    - **Sentiment is moderately predictable** (SVM 86.9 %, Logistic 85.9 %),
      with performance limited by the structural near-absence of negative
      speeches rather than model inadequacy.
    - **The LSTM underperforms** classical models on this corpus due to three
      compounding factors: severe class imbalance (72 % positive), limited
      training (5 epochs), and the absence of class-weight rebalancing.
      These are tractable engineering problems, not fundamental limitations
      of deep learning for political text.
    - **SVM is the recommended production model** for both tasks on this
      corpus: it combines high accuracy, interpretable decision boundaries,
      and robustness to the sparse, high-dimensional TF-IDF feature space.

    Together, these results confirm that crisis-driven discursive restructuring
    is not only observable and theoretically coherent but also **learnable** —
    supporting the integration of predictive NLP into comparative political
    communication research.
    """
)
