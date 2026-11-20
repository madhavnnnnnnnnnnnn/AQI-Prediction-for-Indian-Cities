# 🌫️ AQI Prediction for Indian Cities

> A machine learning app that predicts daily Air Quality Index (AQI) from pollutant readings, helping authorities estimate air quality when sensor data is missing and issue timely health advisories.

---

## 🎬 Demo

| Input form | Prediction result |
|---|---|
| ![App input form](images/app_screenshot.png) | ![App prediction result](images/app_prediction.png) |

- **Live app:** [add Streamlit Community Cloud link here, if deployed]

---

## ✨ App Features

The Streamlit app (`app.py`) predicts **tomorrow's AQI** from today's pollutant readings.

**How it works**
- Enter a city and a date, plus today's levels of PM2.5, PM10, NO2, SO2, CO, O3 and NH3.
- Pollutant fields are pre-filled with median values from the training data.
- The date is turned into model features: month, day of week and season (winter, summer, monsoon, post-monsoon).
- The result shows the predicted AQI, its category (Good, Satisfactory, Moderate, Poor, Very Poor or Severe) and a colour-coded health message.
- A **Show model input** panel shows exactly what was sent to the model.

**Design**
- Smog photo background (`background.jpg`) with a dark tint so text stays readable.
- Slowly drifting smog clouds animated with CSS.
- Frosted-glass card around the input form.
- Blocks fade in when the page loads.
- Gradient Predict button with a hover glow.
- Large AQI number with a pulsing glow.
- Animations turn off automatically for users who choose reduced motion in their system settings.

All styling is plain CSS injected with `st.markdown`, so no extra packages are needed.

---

## 🎯 Problem Statement and Impact

Air quality sensors across Indian cities often have gaps, delays, or failures. This project predicts daily AQI from pollutant readings [and previous days' values] so that:

- **Pollution control boards** can estimate AQI when a sensor is missing or faulty.
- **Health authorities** can issue advisories when air quality turns Poor or Severe.

**Task type:** Regression
**Target variable:** `AQI`
**Success metric:** MAE and RMSE on a held-out test set, compared against a baseline model.

---

## 📊 Dataset

- **Source:** [Air Quality Data in India (2015 - 2020)](PASTE-KAGGLE-LINK-HERE) on Kaggle (file used: `city_day.csv`)
- **Size:** [rows] rows, [columns] columns, [number] cities
- **Period:** 2015 to 2020
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, City, Date [edit to match what you actually use]
- **Target:** AQI

---

## 🛠️ Approach

1. **Problem definition:** user, target, and success metric defined before touching data.
2. **EDA:** missing values, duplicates, outliers, trends over time, city-wise comparison, correlations.
3. **Preprocessing:** [missing value method and why], [outlier treatment and why], date-based features, encoding of City, scaling.
4. **Data split:** time-based split ([train years] for training, [test years] for testing) to avoid leakage. All transformers fit on training data only.
5. **Modeling:** baseline Linear Regression, then Decision Tree, Random Forest, and [Gradient Boosting / other].
6. **Validation and tuning:** time-aware cross-validation and [GridSearchCV / RandomizedSearchCV], with fixed random seeds.
7. **Interpretation:** [feature importance / permutation importance / SHAP].
8. **Error analysis:** where and why the model fails.
9. **Deployment:** saved scikit-learn Pipeline (joblib) served through a Streamlit app.

---

## 📈 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | [ ] | [ ] | [ ] |
| Decision Tree | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] |
| [Gradient Boosting] | [ ] | [ ] | [ ] |

**Best model:** [name], with [x]% lower MAE than the baseline.

---

## 💡 Key Insights and Findings

- [Insight 1, e.g. which pollutant drives AQI the most]
- [Insight 2, e.g. seasonal pattern: winter months show the highest AQI]
- [Insight 3, e.g. which cities are the most polluted]
- **Error analysis:** [e.g. the model underpredicts extreme AQI values because they are rare in the training data]

---

## 🧗 Challenges Faced

| Challenge | How I solved it |
|---|---|
| [e.g. High share of missing values in some columns] | [your solution] |
| [e.g. Risk of data leakage in time-series data] | [your solution] |
| [Add your own] | [your solution] |

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/madhavnnnnnnnnnnnn/AQI-Prediction-for-Indian-Cities.git
cd AQI-Prediction-for-Indian-Cities

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
AQI-Prediction-for-Indian-Cities/
|-- README.md
|-- requirements.txt
|-- app.py                   # Streamlit app (prediction + styling)
|-- background.jpg           # app background image
|-- 01_eda.ipynb
|-- 02_preprocessing.ipynb
|-- Data/                    # sample data or link to source
|-- Notebooks/
|-- src/                     # preprocessing, training, utility code
|-- models/
|   |-- best_model.pkl       # trained model
|   |-- fill_values.pkl      # median values used to pre-fill the form
|   |-- feature_cols.json    # feature order the model expects
|-- images/
|   |-- app_screenshot.png   # input form screenshot
|   |-- app_prediction.png   # prediction result screenshot
|-- .gitignore
```

---

## ⚠️ Limitations and Future Improvements

**Limitations**
- Trained on historical data (2015 to 2020), not live sensor feeds.
- 2020 lockdowns make that year's readings unusual.
- Performance varies across cities and is weaker for extreme AQI values.

**Future improvements**
- Add weather data (temperature, humidity, wind) as features.
- Connect live readings from a public API (e.g. OpenAQ) to the saved model.
- Next-day AQI forecasting with lag features.
- Deploy online with Streamlit Community Cloud or FastAPI.

---

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib / Seaborn · Scikit-learn · [SHAP] · Joblib · Streamlit · CSS animations · Git / GitHub

---

## 📬 Contact

**Madhav N Aju**
- LinkedIn: [your LinkedIn URL]
- Email: [your email]
- GitHub: [madhavnnnnnnnnnnnn](https://github.com/madhavnnnnnnnnnnnn)
