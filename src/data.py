"""Loading and cleaning the city_day.csv dataset.

Each step mirrors a decision explained in notebooks/01_eda.ipynb and
notebooks/02_modeling.ipynb.
"""

import numpy as np
import pandas as pd

from src.config import AQI_CAP, DATA_PATH, SPLIT_DATE, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, sorted by city and date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create AQI_tomorrow: the same city's AQI on the next calendar day.

    If the next row for a city is not exactly one day later (a gap in the
    data), the target is set to NaN instead of borrowing a later day's AQI.
    """
    df = df.copy()
    df[TARGET] = df.groupby("City")["AQI"].shift(-1)
    next_date = df.groupby("City")["Date"].shift(-1)
    df.loc[(next_date - df["Date"]).dt.days != 1, TARGET] = np.nan
    return df


def remove_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Ahmedabad rows whose CO column is a copy of NO, and cap AQI at 500."""
    copied_co = (df["City"] == "Ahmedabad") & (df["NO"] == df["CO"])
    df = df[~copied_co].copy()
    df["AQI"] = df["AQI"].clip(upper=AQI_CAP)
    df[TARGET] = df[TARGET].clip(upper=AQI_CAP)
    return df


def time_split(df: pd.DataFrame, split_date: str = SPLIT_DATE):
    """Split by date: everything before split_date is train, the rest is test."""
    df = df.dropna(subset=[TARGET])
    cutoff = pd.Timestamp(split_date)
    train = df[df["Date"] < cutoff].sort_values(["Date", "City"]).copy()
    test = df[df["Date"] >= cutoff].sort_values(["Date", "City"]).copy()
    return train, test
