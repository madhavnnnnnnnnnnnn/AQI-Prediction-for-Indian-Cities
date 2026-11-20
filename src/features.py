"""Feature engineering shared by the modeling notebook and the Streamlit app.

Keeping these functions in one place guarantees that the app prepares a
new input exactly the same way the training data was prepared.
"""

from datetime import date

import pandas as pd

from src.config import FEATURE_COLS, SEASONS

MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
}


def get_season(month: int) -> str:
    """Map a month number (1-12) to an Indian season name."""
    return MONTH_TO_SEASON[month]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, day of week and one-hot season columns from the Date column.

    The season columns are created from a fixed list (not pd.get_dummies),
    so train, test and app inputs always have the same columns.
    """
    df = df.copy()
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek  # Monday = 0 ... Sunday = 6
    season = df["month"].map(MONTH_TO_SEASON)
    for s in SEASONS:
        df[f"season_{s}"] = (season == s).astype(int)
    return df


def build_feature_row(pollutant_values: dict, chosen_date: date) -> pd.DataFrame:
    """Turn one set of app inputs into a single-row DataFrame for the model."""
    row = pd.DataFrame([{**pollutant_values, "Date": pd.Timestamp(chosen_date)}])
    row = add_date_features(row)
    return row[FEATURE_COLS]


def aqi_category(aqi: float) -> tuple[str, str]:
    """Return (category, short health message) for an AQI value (CPCB scale)."""
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


AQI_BINS = [0, 50, 100, 200, 300, 400, 500]
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
