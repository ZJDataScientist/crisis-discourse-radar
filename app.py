"""
app.py  –  Crisis Discourse Dashboard
Standard Streamlit multi-page entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Discourse Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Discourse Radar")
st.markdown("*Executive Political Communication Across Regimes and Crisis Phases*")
st.markdown(
    """
    A computational analysis of executive political speech across the United States,
    United Kingdom, Russia, and China (2016–2024), spanning two systemic crises —
    the COVID-19 pandemic and the Ukraine war. A six-layer analytical pipeline
    measures how crisis restructures attention, framing, emotional tone, and
    predictability, and whether those shifts are universal or regime-conditioned.
    """
)

st.divider()

st.subheader("Conceptual Framework")
st.markdown(
    """
    Crisis is treated as an exogenous shock that disrupts communicative equilibrium
    and forces leaders to recalibrate how they allocate attention, frame issues, and
    express emotional tone. The framework analyzes discourse across multiple layers:
    issue salience captures shifts in policy attention; framing and semantic modeling
    measure how meaning is reorganized; affective tone captures emotional intensity;
    and Named Entity Recognition (NER) identifies changes in political agency and actor emphasis.
    
    Building on this structure, supervised and deep learning models test whether these
    discursive transformations are not only observable, but also predictable. Regime type
    functions as a moderating condition, allowing the analysis to distinguish universal
    crisis adaptation from institutionally conditioned communication strategies.
    """
)

st.divider()
st.subheader("Analytical Sections")
st.markdown(
    """
    | Section | Research Question |
    |---|---|
    | **Overview** | Corpus summary · country map · model performance snapshot |
    | **RQ1 — Attention** | What topics receive priority under crisis? |
    | **RQ2 — Narrative Structure** | How are topics linguistically framed? |
    | **RQ3 — Affective Tone** | With what emotional intensity are topics communicated? |
    | **RQ4 — Regime Conditioning** | Are discursive shifts universal or regime-specific? |
    | **RQ5 — Predictive Models** | Can ML and DL predict discourse structure and tone? |
    | **Methods** | Six-layer analytical pipeline specification |
    """
)

st.divider()
st.info("👈 Select a section from the sidebar to begin.")
