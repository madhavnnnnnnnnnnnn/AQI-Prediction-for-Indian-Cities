"""
AQI Prediction App
------------------
A simple Streamlit front-end for a trained scikit-learn regressor that
predicts tomorrow's Air Quality Index (AQI) from today's pollutant levels.

Run with:
    streamlit run app.py
"""

import base64
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths to the saved artefacts (relative to this file, so the app works no
# matter which directory you launch it from).
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
FILL_VALUES_PATH = MODELS_DIR / "fill_values.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"
BACKGROUND_PATH = BASE_DIR / "background.jpg"

# The 7 pollutant inputs, in the order they appear in the form.
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]


# ---------------------------------------------------------------------------
# 1. Load the model, median fill values and feature column list once.
#    st.cache_resource keeps them in memory across reruns of the script.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    fill_values = joblib.load(FILL_VALUES_PATH)  # pandas Series of medians
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)  # exact column order the model expects
    return model, fill_values, feature_cols


# ---------------------------------------------------------------------------
# Styling: background image, drifting smog, glass card, fade-ins,
# glowing button and pulsing AQI number
# ---------------------------------------------------------------------------
@st.cache_data
def load_background_b64() -> str:
    """Read background.jpg as base64 text (empty string if the file is missing)."""
    if BACKGROUND_PATH.exists():
        return base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()
    return ""


STYLE_CSS = """
<style>
/* Background: smog photo with a dark tint so text stays readable */
.stApp {
    background-color: #0f1720;
    background-image:
        linear-gradient(rgba(10, 15, 25, 0.70), rgba(10, 15, 25, 0.80)),
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

/* Frosted-glass form card */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.40);
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

/* Pulsing AQI number */
[data-testid="stMetricValue"] {
    font-size: 3rem !important;
    font-weight: 700;
    animation: pulse 2.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { text-shadow: 0 0 0 rgba(255, 99, 99, 0); }
    50%      { text-shadow: 0 0 22px rgba(255, 99, 99, 0.9); }
}

/* Respect users who turn off motion */
@media (prefers-reduced-motion: reduce) {
    *, .stApp, .stApp::before, .stApp::after { animation: none !important; }
}
</style>
"""


def add_motion():
    st.markdown(STYLE_CSS.replace("__BG__", load_background_b64()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_season(month: int) -> str:
    """Map a month number (1-12) to an Indian season name."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"  # 10, 11


def aqi_category(aqi: int) -> tuple[str, str]:
    """Return (category, short health message) for an AQI value."""
    if aqi <= 50:
        return "Good", "Minimal impact. Enjoy the fresh air!"
    if aqi <= 100:
        return "Satisfactory", "Minor breathing discomfort for sensitive people."
    if aqi <= 200:
        return "Moderate", "Breathing discomfort for people with lung or heart disease, children and older adults."
    if aqi <= 300:
        return "Poor", "Breathing discomfort for most people on prolonged exposure."
    if aqi <= 400:
        return "Very Poor", "Respiratory illness likely on prolonged exposure. Limit outdoor activity."
    return "Severe", "Serious health impact, even for healthy people. Stay indoors if possible."


def build_features(pollutant_values: dict, chosen_date: date, feature_cols: list) -> pd.DataFrame:
    """Turn the form inputs into a single-row DataFrame matching the model's features."""
    row = dict(pollutant_values)

    # Date-based features
    row["month"] = chosen_date.month
    row["day_of_week"] = chosen_date.weekday()  # Monday = 0 ... Sunday = 6

    # One-hot encode the season (exactly one of these will be 1)
    season = get_season(chosen_date.month)
    for s in ["monsoon", "post_monsoon", "summer", "winter"]:
        row[f"season_{s}"] = 1 if season == s else 0

    # Reindex so the columns are in the exact order the model was trained on;
    # any column we didn't create is filled with 0.
    df = pd.DataFrame([row])
    return df.reindex(columns=feature_cols, fill_value=0)


# ---------------------------------------------------------------------------
# 2. Page title and description
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️")
add_motion()  # must come after set_page_config
st.title("🌫️ Tomorrow's AQI Predictor")
st.write(
    "Enter today's pollutant readings and the date, and the model will predict "
    "**tomorrow's Air Quality Index (AQI)**. Pollutant fields are pre-filled "
    "with typical (median) values from the training data."
)

model, fill_values, feature_cols = load_artifacts()

# ---------------------------------------------------------------------------
# 3. Input form
# ---------------------------------------------------------------------------
with st.form("aqi_form"):
    city = st.text_input("City", value="Delhi")  # display only, not a feature
    chosen_date = st.date_input("Date", value=date.today())

    st.subheader("Pollutant levels")
    pollutant_values = {}
    cols = st.columns(2)  # lay the 7 inputs out in two columns
    for i, p in enumerate(POLLUTANTS):
        with cols[i % 2]:
            pollutant_values[p] = st.number_input(
                p,
                min_value=0.0,
                value=float(fill_values[p]),  # pre-fill with the median
                step=0.1,
                format="%.2f",
            )

    submitted = st.form_submit_button("Predict AQI")

# ---------------------------------------------------------------------------
# 4 & 5. Build the feature row, predict and show the result
# ---------------------------------------------------------------------------
if submitted:
    X = build_features(pollutant_values, chosen_date, feature_cols)

    prediction = model.predict(X)[0]
    aqi = max(0, int(round(prediction)))  # AQI can't be negative
    category, message = aqi_category(aqi)

    place = f" in {city}" if city.strip() else ""
    st.subheader(f"Predicted AQI for tomorrow{place}")
    st.metric(label="AQI", value=aqi)
    st.markdown(f"**Category: {category}**")

    # Colour the message box by severity
    if category in ("Good", "Satisfactory"):
        st.success(message)
    elif category == "Moderate":
        st.warning(message)
    else:
        st.error(message)

    # Optional: let the user inspect exactly what was sent to the model
    with st.expander("Show model input"):
        st.dataframe(X)
