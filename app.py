"""
AQI Prediction App
------------------
A Streamlit app that forecasts tomorrow's Air Quality Index (AQI) for an
Indian city from today's pollutant readings, explains the forecast, and
shows how the model was built and how well it performs.

Run with:
    streamlit run app.py
"""

import json
from datetime import date

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (DATA_PATH, FEATURE_COLS, FEATURE_COLS_PATH, FILL_VALUES_PATH,
                        METRICS_PATH, MODEL_PATH, POLLUTANTS, SPLIT_DATE)
from src.data import add_target, load_raw, remove_bad_rows
from src.features import aqi_category, build_feature_row
from src.style import apply_style

# Official CPCB colours for each AQI category.
CATEGORY_COLOURS = {
    "Good": "#55a84f",
    "Satisfactory": "#a3c853",
    "Moderate": "#fff833",
    "Poor": "#f29c33",
    "Very Poor": "#e93f33",
    "Severe": "#af2d24",
}

# Unit and a sensible upper limit for each input (well above anything in the data).
POLLUTANT_INFO = {
    "PM2.5": ("µg/m³", 1000.0, "Fine particles from exhaust, burning and industry"),
    "PM10": ("µg/m³", 1000.0, "Coarser dust from roads and construction"),
    "NO2": ("µg/m³", 500.0, "Nitrogen dioxide, mostly from traffic"),
    "SO2": ("µg/m³", 500.0, "Sulphur dioxide, from coal and industrial fuel"),
    "CO": ("mg/m³", 50.0, "Carbon monoxide, from incomplete burning (note: mg, not µg)"),
    "O3": ("µg/m³", 500.0, "Ground-level ozone, worse on sunny days"),
    "NH3": ("µg/m³", 500.0, "Ammonia, from agriculture and waste"),
}


# ---------------------------------------------------------------------------
# Loading (cached, with friendly errors instead of crashes)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the saved Pipeline, training medians and feature order."""
    model = joblib.load(MODEL_PATH)
    fill_values = joblib.load(FILL_VALUES_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    return model, fill_values, feature_cols


@st.cache_data
def load_metrics():
    """Load results saved by notebooks/02_modeling.ipynb (None if missing)."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_clean_data():
    """Cleaned dataset for the Data Insights tab (None if the CSV is missing)."""
    if not DATA_PATH.exists():
        return None
    return remove_bad_rows(add_target(load_raw()))


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict(model, pollutant_values: dict, chosen_date: date) -> float:
    """Predict tomorrow's AQI, kept inside the official 0-500 scale."""
    X = build_feature_row(pollutant_values, chosen_date)
    return float(np.clip(model.predict(X)[0], 0, 500))


def explain(model, pollutant_values: dict, chosen_date: date, fill_values) -> pd.Series:
    """How much each reading moved the forecast compared with a typical day.

    For each pollutant we swap the user's value for the training median and
    re-predict. The difference is that reading's effect on this forecast.
    """
    base = predict(model, pollutant_values, chosen_date)
    effects = {}
    for p in POLLUTANTS:
        typical = dict(pollutant_values)
        typical[p] = float(fill_values[p])
        effects[p] = base - predict(model, typical, chosen_date)
    return pd.Series(effects)


def validate_inputs(values: dict) -> tuple[list, list]:
    """Return (errors, warnings) for the form inputs."""
    errors, warnings = [], []
    if all(v == 0 for v in values.values()):
        errors.append("All readings are 0. Please enter today's actual pollutant levels.")
    if values["PM2.5"] > values["PM10"] > 0:
        warnings.append("PM2.5 is higher than PM10. PM2.5 is part of PM10, so this is unusual; "
                        "please double-check the readings.")
    if values["CO"] > 20:
        warnings.append("CO is above 20 mg/m³, which is very rare. Check that CO is in mg/m³, not µg/m³.")
    return errors, warnings


def interval_for(aqi: float, category: str, metrics) -> float:
    """80% error range for this kind of forecast, from the test-set errors."""
    if not metrics:
        return 0.0
    return metrics["interval_80_by_category"].get(category, metrics["interval_80"])


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")
apply_style()

st.title("🌫️ Tomorrow's AQI Predictor")
st.write(
    "Forecast **tomorrow's Air Quality Index (AQI)** for an Indian city from today's "
    "pollutant readings, so you can plan outdoor time, exercise or a mask in advance."
)

try:
    model, fill_values, feature_cols = load_artifacts()
except Exception:
    st.error(
        "The trained model could not be loaded. Make sure the `models/` folder contains "
        "`best_model.pkl`, `fill_values.pkl` and `feature_cols.json`. You can recreate them by "
        "running `notebooks/02_modeling.ipynb`."
    )
    st.stop()

if feature_cols != FEATURE_COLS:
    st.error("The saved model expects different features than this app builds. "
             "Re-run `notebooks/02_modeling.ipynb` to retrain it.")
    st.stop()

metrics = load_metrics()
data = load_clean_data()

tab_predict, tab_insights, tab_perf, tab_about = st.tabs(
    ["🔮 Predict", "📊 Data Insights", "📈 Model Performance", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    with st.expander("How to use this app", expanded=False):
        st.markdown(
            "1. Pick your city and today's date.\n"
            "2. Enter today's pollutant readings from a monitoring station or an app such as "
            "[CPCB](https://airquality.cpcb.gov.in/AQI_India/). Fields start at typical (median) values.\n"
            "3. Press **Predict AQI** to see tomorrow's forecast, its likely range and what drove it."
        )

    with st.form("aqi_form"):
        cities = sorted(data["City"].unique()) if data is not None else ["Delhi"]
        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0,
                                help="Shown in the result. The model uses the readings, not the city name.")
        with c2:
            chosen_date = st.date_input("Today's date", value=date.today(),
                                        help="Used for month, day of week and season.")

        st.subheader("Today's pollutant levels")
        pollutant_values = {}
        cols = st.columns(2)
        for i, p in enumerate(POLLUTANTS):
            unit, max_value, description = POLLUTANT_INFO[p]
            with cols[i % 2]:
                pollutant_values[p] = st.number_input(
                    f"{p} ({unit})",
                    min_value=0.0,
                    max_value=max_value,
                    value=float(fill_values[p]),
                    step=0.1,
                    format="%.2f",
                    help=f"{description}. Typical value: {fill_values[p]:.2f} {unit}.",
                )

        submitted = st.form_submit_button("Predict AQI")

    if submitted:
        errors, warnings = validate_inputs(pollutant_values)
        for message in errors:
            st.error(message)
        for message in warnings:
            st.warning(message)

        if not errors:
            try:
                aqi = predict(model, pollutant_values, chosen_date)
                effects = explain(model, pollutant_values, chosen_date, fill_values)
            except Exception:
                st.error("Something went wrong while making the prediction. Please check the inputs and try again.")
                st.stop()

            category, message = aqi_category(aqi)
            colour = CATEGORY_COLOURS[category]
            margin = interval_for(aqi, category, metrics)
            low, high = max(0, aqi - margin), min(500, aqi + margin)
            text_colour = "#ffffff" if category in ("Very Poor", "Severe") else "#0b1320"

            with st.container(border=True):
                st.subheader(f"Predicted AQI for tomorrow in {city}")
                range_text = (f"likely between <b>{low:.0f}</b> and <b>{high:.0f}</b>"
                              if margin else "")
                st.markdown(
                    f"""
                    <div class="aqi-result" style="--glow: {colour}">
                        <span class="aqi-value" style="color: {colour}">{aqi:.0f}</span>
                        <span class="aqi-badge" style="background: {colour}; color: {text_colour}">{category}</span>
                        <span class="aqi-range">{range_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if margin:
                    st.caption(f"The range covers 80% of the model's errors on {category.lower()} "
                               "forecasts during the 2019-2020 test period.")
                if category in ("Good", "Satisfactory"):
                    st.success(message)
                elif category == "Moderate":
                    st.warning(message)
                else:
                    st.error(message)

            st.subheader("Why this prediction?")
            st.write("Each bar shows how much a reading pushed tomorrow's forecast **up** (red) or "
                     "**down** (green) compared with a typical day, where that pollutant is at its median.")
            effect_df = effects.rename("effect").reset_index().rename(columns={"index": "pollutant"})
            chart = (
                alt.Chart(effect_df)
                .mark_bar()
                .encode(
                    x=alt.X("effect:Q", title="Change in forecast (AQI points)"),
                    y=alt.Y("pollutant:N", sort="-x", title=None),
                    color=alt.condition("datum.effect > 0", alt.value("#e93f33"), alt.value("#55a84f")),
                    tooltip=["pollutant", alt.Tooltip("effect:Q", format="+.1f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")

            top = effects.abs().idxmax()
            direction = "raised" if effects[top] > 0 else "lowered"
            st.info(f"**{top}** had the biggest effect: it {direction} the forecast by "
                    f"{abs(effects[top]):.0f} AQI points compared with a typical day.")
            st.caption("Effects are measured one reading at a time, so when several readings are high "
                       "together they do not add up exactly to the full forecast.")

            with st.expander("Show model input"):
                st.dataframe(build_feature_row(pollutant_values, chosen_date), hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: Data Insights
# ---------------------------------------------------------------------------
with tab_insights:
    if data is None:
        st.info("The dataset (`data/city_day.csv`) was not found, so insights are unavailable.")
    else:
        st.subheader("The dataset")
        m1, m2, m3 = st.columns(3)
        m1.metric("Daily records", f"{len(data):,}")
        m2.metric("Cities", data["City"].nunique())
        m3.metric("Period", f"{data['Date'].dt.year.min()}–{data['Date'].dt.year.max()}")
        st.caption("Source: Kaggle, *Air Quality Data in India (2015-2020)*, `city_day.csv`, "
                   "after removing Ahmedabad's faulty CO rows and capping AQI at 500.")

        st.subheader("Which cities are most polluted?")
        city_aqi = data.groupby("City")["AQI"].median().dropna().sort_values(ascending=False).reset_index()
        st.altair_chart(
            alt.Chart(city_aqi).mark_bar().encode(
                x=alt.X("AQI:Q", title="Median AQI"),
                y=alt.Y("City:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                color=alt.Color("AQI:Q", scale=alt.Scale(scheme="orangered"), legend=None),
                tooltip=["City", alt.Tooltip("AQI:Q", format=".0f")],
            ).properties(height=26 * 24),
            width="stretch",
        )
        st.caption("Northern cities on the Indo-Gangetic plain (Delhi, Patna, Gurugram, Lucknow) are the most "
                   "polluted; north-eastern and southern cities such as Aizawl and Shillong are the cleanest.")

        st.subheader("Air quality through the year")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        by_month = data.groupby(data["Date"].dt.month)["AQI"].mean().reset_index()
        by_month["Month"] = by_month["Date"].map(lambda m: months[m - 1])
        st.altair_chart(
            alt.Chart(by_month).mark_bar().encode(
                x=alt.X("Month:N", sort=months, title=None),
                y=alt.Y("AQI:Q", title="Average AQI"),
                color=alt.Color("AQI:Q", scale=alt.Scale(scheme="orangered"), legend=None),
                tooltip=["Month", alt.Tooltip("AQI:Q", format=".0f")],
            ).properties(height=280),
            width="stretch",
        )
        st.caption("Winter (November to January) is the worst season, with still air and burning; "
                   "monsoon rain (July to September) cleans the air.")

        st.subheader("Which pollutants move with AQI?")
        corr = data[POLLUTANTS].corrwith(data["AQI"]).sort_values(ascending=False).rename("correlation").reset_index()
        st.altair_chart(
            alt.Chart(corr).mark_bar(color="#4facfe").encode(
                x=alt.X("correlation:Q", title="Correlation with AQI", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("index:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("index:N", title="Pollutant"), alt.Tooltip("correlation:Q", format=".2f")],
            ).properties(height=260),
            width="stretch",
        )
        st.caption("PM10 and PM2.5 are by far the most related to AQI; the gases matter much less.")

        st.subheader("AQI over time for one city")
        city_choice = st.selectbox("Choose a city", sorted(data["City"].unique()),
                                   index=sorted(data["City"].unique()).index("Delhi"), key="trend_city")
        trend = (data[data["City"] == city_choice].set_index("Date")["AQI"]
                 .resample("ME").mean().dropna().reset_index())
        st.altair_chart(
            alt.Chart(trend).mark_line(point=True, color="#f29c33").encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("AQI:Q", title="Monthly average AQI"),
                tooltip=[alt.Tooltip("Date:T", format="%b %Y"), alt.Tooltip("AQI:Q", format=".0f")],
            ).properties(height=280),
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Tab 3: Model Performance
# ---------------------------------------------------------------------------
with tab_perf:
    if metrics is None:
        st.info("`models/metrics.json` was not found. Run `notebooks/02_modeling.ipynb` to create it.")
    else:
        best = next(r for r in metrics["test_results"] if r["Model"] == metrics["best_model"])
        improvement = (1 - best["MAE"] / metrics["baseline_test_mae"]) * 100

        st.subheader(f"Chosen model: {metrics['best_model']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE", f"{best['MAE']:.1f}", help="Average error in AQI points")
        k2.metric("RMSE", f"{best['RMSE']:.1f}", help="Like MAE, but punishes big misses more")
        k3.metric("R²", f"{best['R2']:.3f}", help="Share of the variation in AQI the model explains")
        k4.metric("Better than baseline", f"{improvement:.0f}%",
                  help=f"Baseline always predicts the average (MAE {metrics['baseline_test_mae']:.1f})")
        st.caption(f"Trained on {metrics['train_period'][0]} to {metrics['train_period'][1]} "
                   f"({metrics['n_train']:,} rows), tested on {metrics['test_period'][0]} to "
                   f"{metrics['test_period'][1]} ({metrics['n_test']:,} rows), which the model never saw.")

        st.subheader("Model comparison on the test set")
        st.dataframe(pd.DataFrame(metrics["test_results"]).round(2), hide_index=True, width="stretch")

        with st.expander("Cross-validation results (5 time-based folds on 2015-2018)"):
            st.dataframe(pd.DataFrame(metrics["cv_results"]).round(2), hide_index=True, width="stretch")
            st.caption("Models were chosen by cross-validated MAE on the training years only; "
                       "the test years were used once, at the end.")
            for name, params in metrics["best_params"].items():
                st.markdown(f"**{name}** best settings: `{params}`")

        st.subheader("What drives the predictions")
        imp = pd.Series(metrics["permutation_importance"]).rename("importance").reset_index()
        st.altair_chart(
            alt.Chart(imp).mark_bar(color="#4facfe").encode(
                x=alt.X("importance:Q", title="MAE increase when the feature is shuffled"),
                y=alt.Y("index:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("index:N", title="Feature"), alt.Tooltip("importance:Q", format=".2f")],
            ).properties(height=440),
            width="stretch",
        )

        st.subheader("Where the model makes mistakes")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Error by actual AQI category**")
            cat = pd.DataFrame(metrics["error_by_category"]).T.reset_index().rename(columns={"index": "Category"})
            st.dataframe(cat.round(1), hide_index=True, width="stretch")
            st.caption("Bias below 0 means the model predicts too low.")
        with e2:
            st.markdown("**Error by season**")
            season = pd.Series(metrics["error_by_season"]).rename("MAE").reset_index().rename(columns={"index": "Season"})
            st.dataframe(season.round(1), hide_index=True, width="stretch")
            st.markdown("**Worst cities**")
            worst = pd.Series(metrics["error_by_city"]).head(5).rename("MAE").reset_index().rename(columns={"index": "City"})
            st.dataframe(worst.round(1), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: About
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("The problem")
    st.write(
        "Commuters, parents, outdoor workers, the elderly and people with asthma or heart and lung "
        "conditions need to know **tomorrow's** air quality to plan ahead: limit time outside, move "
        "exercise indoors, wear a mask, or keep children inside. This app forecasts it from today's readings."
    )
    st.subheader("How it works")
    st.markdown(
        "- **Data:** daily pollution readings for 26 Indian cities, 2015 to 2020 (Kaggle).\n"
        "- **Cleaning:** removed Ahmedabad rows where the CO column was a copy of NO, capped AQI at 500.\n"
        f"- **Split by date:** trained on data before {SPLIT_DATE[:4]}, tested on {SPLIT_DATE[:4]}-2020, "
        "so the model is judged on future days it never saw.\n"
        "- **Features:** 7 pollutants plus month, day of week and season.\n"
        "- **Models:** baseline, Linear Regression, Decision Tree, Random Forest and Gradient Boosting, "
        "compared with time-based cross-validation and tuned with Grid/RandomizedSearchCV.\n"
        "- **Pipeline:** missing values are filled with training medians inside a scikit-learn Pipeline."
    )
    st.subheader("Limitations")
    st.markdown(
        "- Trained on historical data (2015-2020), not live sensor feeds.\n"
        "- No weather inputs (wind, rain, temperature), which strongly affect pollution.\n"
        "- Less accurate on rare Severe days, which it tends to underpredict.\n"
        "- The 2020 COVID lockdown made that year unusually clean.\n"
        "- **This is a student project, not an official forecast.** Check CPCB for official AQI."
    )
    st.subheader("Project")
    st.markdown("Source code and notebooks: "
                "[GitHub](https://github.com/madhavnnnnnnnnnnnn/AQI-Prediction-for-Indian-Cities)")
