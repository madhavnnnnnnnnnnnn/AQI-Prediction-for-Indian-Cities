"""Project-wide settings: file paths, feature lists and constants.

Paths are built from this file's location, so the notebooks, scripts and
the Streamlit app all work no matter which folder they are launched from.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "city_day.csv"
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
FILL_VALUES_PATH = MODELS_DIR / "fill_values.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# The 7 pollutant readings used as inputs.
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]

# Season names, in a fixed order so the one-hot columns never change.
SEASONS = ["monsoon", "post_monsoon", "summer", "winter"]
SEASON_COLS = [f"season_{s}" for s in SEASONS]

# Final model inputs, in the exact order the model is trained on.
FEATURE_COLS = POLLUTANTS + ["month", "day_of_week"] + SEASON_COLS

TARGET = "AQI_tomorrow"

# Train on 2015-2018, test on 2019 onwards (time-based split).
SPLIT_DATE = "2019-01-01"

# Official AQI scale maximum; higher values are capped.
AQI_CAP = 500

RANDOM_STATE = 42
