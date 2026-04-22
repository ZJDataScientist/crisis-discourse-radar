"""
pages/0_Landing.py
------------------
Discourse Radar — landing page.

Visual concept: "Analytical Observatory"
Light off-white background with an abstract SVG composition:
  • Sparse coordinate grid — reference frame / analytical scaffolding
  • Concentric radar arcs — scanning / listening field
  • Left + right network clusters — semantic co-occurrence graphs
  • Vector arrows — directional embedding shift
  • Sinusoidal waveform — discourse signal trace

All CSS is injected via st.markdown and therefore lives inside
Streamlit's React component tree. It is automatically cleared when
the user navigates to any other page — no global leakage.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Discourse Radar",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DATA = Path(__file__).parent.parent / "dashboard_data"

@st.cache_data
def _corpus():
    try:
        df = pd.read_csv(DATA / "topic_document_assignments_trimmed.csv")
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        yr = f"{dates.min().year}–{dates.max().year}"
        return f"{len(df):,}", df["country"].nunique(), df["regime"].nunique(), yr
    except Exception:
        return "14,615", 4, 2, "2016–2024"

n_sp, n_co, n_re, yr = _corpus()

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
#MainMenu, header, footer { visibility: hidden; }

/* page base colour lives on body so it shows through transparent layers */
html, body { background-color: #EEF2F6 !important; }

/* every Streamlit container is transparent — lets the fixed SVG show */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMainBlockContainer"],
section.main,
.main > div,
.appview-container {
    background: transparent !important;
    background-image: none !important;
}

/* centred content column — above illustration */
.block-container {
    max-width: 640px;
    padding-top: 10vh  !important;
    padding-bottom: 6vh !important;
    background: transparent !important;
    position: relative;
    z-index: 10;
}

/* top-right brand */
.lab-badge {
    position: fixed;
    top: 20px; right: 26px;
    font-family: "Courier New", monospace;
    font-size: 0.62rem;
    color: #96B2C0;
    letter-spacing: 0.07em;
    z-index: 999;
}

/* footer */
.dr-footer {
    position: fixed;
    bottom: 16px; left: 0; right: 0;
    text-align: center;
    font-family: "Courier New", monospace;
    font-size: 0.57rem;
    color: #B0C8D4;
    letter-spacing: 0.12em;
    z-index: 999;
}

/* overline tag */
.dr-over {
    font-family: "Courier New", monospace;
    font-size: 0.60rem;
    color: #4A8098;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 1.0rem;
}

/* hero title */
.dr-title {
    font-size: 4.5rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #0A1828;
    line-height: 1.00;
    margin: 0 0 1.1rem 0;
}

/* subtitle */
.dr-subtitle {
    font-size: 0.98rem;
    font-style: italic;
    color: #4A6E82;
    letter-spacing: 0.01em;
    line-height: 1.55;
    margin-bottom: 2.2rem;
}

/* primary pill button (Maple-style) */
div[data-testid="stButton"] > button {
    width: 100% !important;
    padding: 0.74rem 2.4rem !important;
    font-size: 0.93rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 50px !important;
    background: #0A1828 !important;
    color: #EEF2F6 !important;
    border: none !important;
    box-shadow: 0 2px 12px rgba(10,24,40,0.14) !important;
    transition: background 0.16s ease, box-shadow 0.16s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1A3A5C !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(10,24,40,0.22) !important;
}

/* corpus metadata */
.dr-meta {
    font-family: "Courier New", monospace;
    font-size: 0.59rem;
    color: #88AAB8;
    letter-spacing: 0.05em;
    margin-top: 1.5rem;
    line-height: 2.0;
}
.dr-meta-val {
    color: #2A7898;
    font-weight: 700;
}
</style>
"""

# ── ILLUSTRATED BACKGROUND ────────────────────────────────────────────────────
#
# Injected via st.markdown — scoped to this page only.
# When Streamlit navigates to any other page it rebuilds the React component
# tree from scratch, discarding every st.markdown element from the old page.
# Nothing here can leak to Overview, RQ pages, or Methods.
#
# Composition (viewBox 0 0 1440 600):
#   • Sparse coordinate grid — analytical reference frame
#   • Concentric radar arcs centred below the viewport — scanning field
#   • Left + right network clusters — semantic graph motif
#   • Vector arrows — directional embedding shift
#   • Sinusoidal waveform — discourse signal trace
#   • Gradient overlay — fades the top 28-44 % to keep hero text readable
#
ILLUSTRATION = """
<div style="position:fixed; top:0; left:0; width:100vw; height:100vh;
            z-index:1; pointer-events:none; overflow:hidden;">
  <svg width="100%" height="100%" viewBox="0 0 1440 600"
       preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="dr-fade" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%"  stop-color="#EEF2F6" stop-opacity="1"/>
        <stop offset="28%" stop-color="#EEF2F6" stop-opacity="1"/>
        <stop offset="44%" stop-color="#EEF2F6" stop-opacity="0"/>
      </linearGradient>
      <marker id="dr-arr" markerWidth="6" markerHeight="6"
              refX="5" refY="3" orient="auto">
        <path d="M0,0.5 L5,3 L0,5.5 Z" fill="#1B4E78" fill-opacity="0.30"/>
      </marker>
    </defs>

    <!-- COORDINATE GRID — vertical guides -->
    <line x1="148"  y1="0" x2="148"  y2="600" stroke="#1B4E78" stroke-opacity="0.055" stroke-width="0.8"/>
    <line x1="360"  y1="0" x2="360"  y2="600" stroke="#1B4E78" stroke-opacity="0.055" stroke-width="0.8"/>
    <line x1="1080" y1="0" x2="1080" y2="600" stroke="#1B4E78" stroke-opacity="0.055" stroke-width="0.8"/>
    <line x1="1292" y1="0" x2="1292" y2="600" stroke="#1B4E78" stroke-opacity="0.055" stroke-width="0.8"/>

    <!-- COORDINATE GRID — horizontal guides -->
    <line x1="0" y1="160" x2="1440" y2="160" stroke="#1B4E78" stroke-opacity="0.045" stroke-width="0.8"/>
    <line x1="0" y1="320" x2="1440" y2="320" stroke="#1B4E78" stroke-opacity="0.045" stroke-width="0.8"/>
    <line x1="0" y1="480" x2="1440" y2="480" stroke="#1B4E78" stroke-opacity="0.045" stroke-width="0.8"/>

    <!-- RADAR ARCS — centre (720, 690), partial arcs emerge from below -->
    <circle cx="720" cy="690" r="190" fill="none" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <circle cx="720" cy="690" r="340" fill="none" stroke="#1B4E78" stroke-opacity="0.11" stroke-width="1"/>
    <circle cx="720" cy="690" r="490" fill="none" stroke="#1B4E78" stroke-opacity="0.08" stroke-width="1"/>
    <circle cx="720" cy="690" r="640" fill="none" stroke="#1B4E78" stroke-opacity="0.06" stroke-width="1"/>
    <!-- radar crosshairs -->
    <line x1="720" y1="120" x2="720" y2="600"
          stroke="#1B4E78" stroke-opacity="0.055" stroke-width="0.8"
          stroke-dasharray="5,10"/>
    <line x1="150" y1="445" x2="1290" y2="445"
          stroke="#1B4E78" stroke-opacity="0.055" stroke-width="0.8"
          stroke-dasharray="5,10"/>

    <!-- LEFT NETWORK — nodes + edges -->
    <line x1="72"  y1="315" x2="148" y2="362" stroke="#1B4E78" stroke-opacity="0.18" stroke-width="1"/>
    <line x1="148" y1="362" x2="112" y2="428" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <line x1="72"  y1="315" x2="46"  y2="392" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <line x1="148" y1="362" x2="192" y2="300" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <line x1="72"  y1="315" x2="192" y2="300" stroke="#1B4E78" stroke-opacity="0.10" stroke-width="0.9"/>
    <line x1="112" y1="428" x2="46"  y2="392" stroke="#1B4E78" stroke-opacity="0.10" stroke-width="0.9"/>
    <line x1="148" y1="362" x2="218" y2="410" stroke="#1B4E78" stroke-opacity="0.11" stroke-width="0.9"/>
    <line x1="218" y1="410" x2="112" y2="428" stroke="#1B4E78" stroke-opacity="0.10" stroke-width="0.9"/>
    <line x1="218" y1="410" x2="262" y2="460" stroke="#1B4E78" stroke-opacity="0.08" stroke-width="0.8"/>
    <circle cx="72"  cy="315" r="4.5" fill="#1B4E78" fill-opacity="0.28"/>
    <circle cx="148" cy="362" r="6"   fill="#1B4E78" fill-opacity="0.34"/>
    <circle cx="112" cy="428" r="3.5" fill="#1B4E78" fill-opacity="0.22"/>
    <circle cx="46"  cy="392" r="3"   fill="#1B4E78" fill-opacity="0.20"/>
    <circle cx="192" cy="300" r="4"   fill="#1B4E78" fill-opacity="0.25"/>
    <circle cx="218" cy="410" r="3"   fill="#1B4E78" fill-opacity="0.20"/>
    <circle cx="262" cy="460" r="2.5" fill="#1B4E78" fill-opacity="0.16"/>
    <circle cx="88"  cy="510" r="2.5" fill="#1B4E78" fill-opacity="0.16"/>
    <circle cx="152" cy="545" r="2"   fill="#1B4E78" fill-opacity="0.13"/>
    <circle cx="205" cy="518" r="2"   fill="#1B4E78" fill-opacity="0.13"/>
    <line x1="88"  y1="510" x2="152" y2="545" stroke="#1B4E78" stroke-opacity="0.09" stroke-width="0.8"/>
    <line x1="152" y1="545" x2="205" y2="518" stroke="#1B4E78" stroke-opacity="0.09" stroke-width="0.8"/>

    <!-- RIGHT NETWORK — mirrored -->
    <line x1="1368" y1="315" x2="1292" y2="362" stroke="#1B4E78" stroke-opacity="0.18" stroke-width="1"/>
    <line x1="1292" y1="362" x2="1328" y2="428" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <line x1="1368" y1="315" x2="1394" y2="392" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <line x1="1292" y1="362" x2="1248" y2="300" stroke="#1B4E78" stroke-opacity="0.14" stroke-width="1"/>
    <line x1="1368" y1="315" x2="1248" y2="300" stroke="#1B4E78" stroke-opacity="0.10" stroke-width="0.9"/>
    <line x1="1328" y1="428" x2="1394" y2="392" stroke="#1B4E78" stroke-opacity="0.10" stroke-width="0.9"/>
    <line x1="1292" y1="362" x2="1222" y2="410" stroke="#1B4E78" stroke-opacity="0.11" stroke-width="0.9"/>
    <line x1="1222" y1="410" x2="1328" y2="428" stroke="#1B4E78" stroke-opacity="0.10" stroke-width="0.9"/>
    <line x1="1222" y1="410" x2="1178" y2="460" stroke="#1B4E78" stroke-opacity="0.08" stroke-width="0.8"/>
    <circle cx="1368" cy="315" r="4.5" fill="#1B4E78" fill-opacity="0.28"/>
    <circle cx="1292" cy="362" r="6"   fill="#1B4E78" fill-opacity="0.34"/>
    <circle cx="1328" cy="428" r="3.5" fill="#1B4E78" fill-opacity="0.22"/>
    <circle cx="1394" cy="392" r="3"   fill="#1B4E78" fill-opacity="0.20"/>
    <circle cx="1248" cy="300" r="4"   fill="#1B4E78" fill-opacity="0.25"/>
    <circle cx="1222" cy="410" r="3"   fill="#1B4E78" fill-opacity="0.20"/>
    <circle cx="1178" cy="460" r="2.5" fill="#1B4E78" fill-opacity="0.16"/>
    <circle cx="1352" cy="510" r="2.5" fill="#1B4E78" fill-opacity="0.16"/>
    <circle cx="1288" cy="545" r="2"   fill="#1B4E78" fill-opacity="0.13"/>
    <circle cx="1235" cy="518" r="2"   fill="#1B4E78" fill-opacity="0.13"/>
    <line x1="1352" y1="510" x2="1288" y2="545" stroke="#1B4E78" stroke-opacity="0.09" stroke-width="0.8"/>
    <line x1="1288" y1="545" x2="1235" y2="518" stroke="#1B4E78" stroke-opacity="0.09" stroke-width="0.8"/>

    <!-- VECTOR ARROWS — directional indicators -->
    <line x1="305"  y1="496" x2="390"  y2="474"
          stroke="#1B4E78" stroke-opacity="0.22" stroke-width="1.4"
          marker-end="url(#dr-arr)"/>
    <line x1="430"  y1="534" x2="532"  y2="516"
          stroke="#1B4E78" stroke-opacity="0.18" stroke-width="1.2"
          marker-end="url(#dr-arr)"/>
    <line x1="1135" y1="496" x2="1050" y2="474"
          stroke="#1B4E78" stroke-opacity="0.22" stroke-width="1.4"
          marker-end="url(#dr-arr)"/>
    <line x1="1010" y1="534" x2="908"  y2="516"
          stroke="#1B4E78" stroke-opacity="0.18" stroke-width="1.2"
          marker-end="url(#dr-arr)"/>

    <!-- SIGNAL WAVEFORM — bottom trace -->
    <path d="M 0 554 Q 90 534 180 554 Q 270 574 360 554
                     Q 450 534 540 554 Q 630 574 720 554
                     Q 810 534 900 554 Q 990 574 1080 554
                     Q 1170 534 1260 554 Q 1350 534 1440 554"
          fill="none" stroke="#1B4E78" stroke-opacity="0.16" stroke-width="1.5"/>
    <path d="M 0 574 Q 90 562 180 574 Q 270 586 360 574
                     Q 450 562 540 574 Q 630 586 720 574
                     Q 810 562 900 574 Q 990 586 1080 574
                     Q 1170 562 1260 574 Q 1350 562 1440 574"
          fill="none" stroke="#1B4E78" stroke-opacity="0.09" stroke-width="1"/>

    <!-- FADE OVERLAY — protects hero text area -->
    <rect x="0" y="0" width="1440" height="600" fill="url(#dr-fade)"/>
  </svg>
</div>
"""

st.markdown(CSS + ILLUSTRATION, unsafe_allow_html=True)

# ── fixed chrome ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-badge">[ ZJ Analytics Labs ]</div>
<div class="dr-footer">— Powered by ZJ Analytics —</div>
""", unsafe_allow_html=True)

# ── overline ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dr-over">◈ &nbsp; Discourse Intelligence Platform</div>',
    unsafe_allow_html=True,
)

# ── hero title ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dr-title">Discourse Radar.</div>',
    unsafe_allow_html=True,
)

# ── subtitle ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dr-subtitle">'
    'Mapping how executive language restructures under crisis'
    '</div>',
    unsafe_allow_html=True,
)

# ── enter button ──────────────────────────────────────────────────────────────
if st.button("Enter the Analysis →", type="primary"):
    st.switch_page("pages/1_Overview.py")

# ── corpus metadata ───────────────────────────────────────────────────────────
st.markdown(
    f'<div class="dr-meta">'
    f'<span class="dr-meta-val">{n_sp}</span> speeches'
    f'&nbsp;&nbsp;·&nbsp;&nbsp;<span class="dr-meta-val">{n_co}</span> countries'
    f'&nbsp;&nbsp;·&nbsp;&nbsp;<span class="dr-meta-val">{n_re}</span> regimes'
    f'&nbsp;&nbsp;·&nbsp;&nbsp;<span class="dr-meta-val">{yr}</span>'
    f'&nbsp;&nbsp;·&nbsp;&nbsp;<span class="dr-meta-val">6</span> analytical layers'
    f'</div>',
    unsafe_allow_html=True,
)
