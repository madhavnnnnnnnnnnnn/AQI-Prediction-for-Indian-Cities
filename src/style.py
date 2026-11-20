"""Visual styling for the Streamlit app.

Plain CSS injected with st.markdown: a tinted smog background photo,
drifting smog clouds, a frosted-glass form card, fade-in blocks,
a glowing Predict button and a pulsing AQI number.
"""

import base64

import streamlit as st

from src.config import ROOT_DIR

BACKGROUND_PATH = ROOT_DIR / "background.jpg"

STYLE_CSS = """
<style>
/* Background: smog photo with a dark tint so text stays readable */
.stApp {
    background-color: #0f1720;
    background-image:
        linear-gradient(rgba(10, 15, 25, 0.72), rgba(10, 15, 25, 0.82)),
        url("data:image/jpeg;base64,__BG__");
    background-size: cover;
    background-position: center bottom;
    background-attachment: fixed;
}

/* Drifting smog clouds behind everything */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    width: 560px; height: 560px;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.30;
    z-index: 0;
    pointer-events: none;
}
.stApp::before {
    background: #cfd8dc;
    top: -150px; left: -150px;
    animation: drift1 24s ease-in-out infinite alternate;
}
.stApp::after {
    background: #90a4ae;
    bottom: -180px; right: -180px;
    animation: drift2 28s ease-in-out infinite alternate;
}
@keyframes drift1 { to { transform: translate(60vw, 40vh) scale(1.3); } }
@keyframes drift2 { to { transform: translate(-55vw, -45vh) scale(1.2); } }

[data-testid="stAppViewContainer"],
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"] { position: relative; z-index: 1; }

/* Frosted-glass cards: the form and bordered containers */
[data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
}

/* Fade-up entry for each block */
[data-testid="stVerticalBlock"] > div {
    animation: fadeUp 0.6s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Glowing Predict button */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: #0b1320;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 18px rgba(79, 172, 254, 0.8);
    color: #0b1320;
}

/* Prediction result */
.aqi-result { display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; }
.aqi-value {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    animation: pulse 2.2s ease-in-out infinite;
}
.aqi-range { font-size: 1.05rem; opacity: 0.85; }
.aqi-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 700;
    color: #0b1320;
}
@keyframes pulse {
    0%, 100% { text-shadow: 0 0 0 rgba(255, 255, 255, 0); }
    50%      { text-shadow: 0 0 22px var(--glow, rgba(255, 99, 99, 0.9)); }
}

/* Respect users who turn off motion */
@media (prefers-reduced-motion: reduce) {
    *, .stApp, .stApp::before, .stApp::after { animation: none !important; }
}
</style>
"""


@st.cache_data
def _background_b64() -> str:
    """Read background.jpg as base64 text (empty string if the file is missing)."""
    if BACKGROUND_PATH.exists():
        return base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()
    return ""


def apply_style():
    """Inject the app's CSS. Call once, right after st.set_page_config."""
    st.markdown(STYLE_CSS.replace("__BG__", _background_b64()), unsafe_allow_html=True)
