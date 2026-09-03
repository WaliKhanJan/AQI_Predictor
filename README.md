# 🌫️ AQI Predictor — Serverless Air Quality Forecasting for Islamabad

A fully serverless, end-to-end machine learning system that forecasts Islamabad's Air Quality Index (AQI) 24, 48, and 72 hours ahead, using historical and live weather + pollutant data.

**Live dashboard:** [isb-aqi-predictor.streamlit.app](https://isb-aqi-predictor.streamlit.app)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Automation (CI/CD)](#automation-cicd)
- [Dashboard](#dashboard)
- [Known Issues / Gotchas](#known-issues--gotchas)
- [Project Report](#project-report)

---

## Architecture Overview

```
                     ┌─────────────────────┐
                     │   Open-Meteo APIs    │
                     │ (Weather + Air Qual.)│
                     └──────────┬───────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                    │
     ┌────────▼─────────┐                ┌─────────▼──────────┐
     │   backfill.py     │                │ feature_pipeline.py│
     │ (one-time, 2.5yrs │                │  (hourly, live —   │
     │  historical data) │                │  GitHub Actions)   │
     └────────┬──────────┘                └─────────┬──────────┘
              │                                      │
              └──────────────────┬───────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │   Hopsworks Feature Store  │
                     │      (aqi_features)        │
                     └────────────┬────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │   training_pipeline.py     │
                     │ (daily — GitHub Actions)   │
                     │ trains XGBoost + Ridge,    │
                     │ registers to Model Registry│
                     └────────────┬────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Hopsworks Model Registry   │
                     │ aqi_model_24h / 48h / 72h  │
                     └────────────┬────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │        app.py               │
                     │  Streamlit Dashboard        │
                     │ (deployed — Streamlit Cloud)│
                     └──────────────────────────────┘
```

**Nothing in this system runs on a dedicated, always-on server managed by the project.** GitHub Actions runners are ephemeral (spun up, run, torn down), and Hopsworks + Streamlit Community Cloud are both managed platforms.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Data source | [Open-Meteo](https://open-meteo.com) | Free, no API key, historical + live weather & air quality from one provider |
| Feature Store / Model Registry | [Hopsworks](https://www.hopsworks.ai) (serverless free tier) | Centralized, versioned storage for features and models |
| Modelling | scikit-learn (Ridge, Random Forest), XGBoost | Compared and selected per forecast horizon |
| Explainability | SHAP | Feature importance per model |
| Dashboard | Streamlit + Altair | Interactive, color-coded forecast UI |
| Automation | GitHub Actions | Hourly feature pipeline, daily training pipeline |
| Deployment | Streamlit Community Cloud | Free, serverless dashboard hosting |

---

## Project Structure

```
AQI_Predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     # hourly live data collection
│       └── training_pipeline.yml    # daily model retraining
├── notebooks/
│   ├── eda.ipynb                    # exploratory data analysis
│   ├── shap_analysis.ipynb          # SHAP explainability
│   └── shap_summary_*.png           # saved SHAP plots (used by dashboard)
├── src/
│   ├── backfill.py                  # one-time historical data pull
│   ├── feature_engineering.py       # builds engineered_features.csv
│   ├── push_to_hopsworks.py         # pushes engineered features to Feature Store
│   ├── feature_pipeline.py          # hourly live feature pipeline
│   ├── training_pipeline.py         # daily training + Model Registry push
│   ├── model_comparison.py          # Ridge vs Random Forest vs XGBoost comparison
│   ├── aqi_utils.py                 # AQI category / alert helper functions
│   └── app.py                       # Streamlit dashboard
├── data/                            # generated CSVs (gitignored)
├── models/                          # locally saved model files (gitignored)
├── requirements.txt
├── runtime.txt                      # pins Python 3.11 for Streamlit Cloud
├── .env                             # local secrets (gitignored, never committed)
└── README.md
```

---

## Prerequisites

- Python 3.11 (matches both local development and the pinned deployment runtime)
- A free [Hopsworks](https://www.hopsworks.ai) account and API key
- Git and a GitHub account (for automation)
- **Windows only:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload — required to install the `hopsworks` package (see [Known Issues](#known-issues--gotchas))

No API key is needed for Open-Meteo.

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/WaliKhanJan/AQI_Predictor.git
cd AQI_Predictor
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Hopsworks credentials**

Create a `.env` file at the project root:
```
HOPSWORKS_API_KEY=your_api_key_here
```

**5. (Windows only) Create a `/tmp` directory**

The Hopsworks client assumes a Unix-style `/tmp` path, which doesn't exist on Windows by default:
```bash
mkdir D:\tmp
```
(Scripts also create this programmatically at runtime, so this step is a safety net rather than strictly required.)

---

## Running the Pipeline

Run these once, in order, to build the system from scratch:

```bash
# 1. Pull ~2.5 years of historical weather + air quality data
python src/backfill.py

# 2. Compute time-based features, lags, change rate, and future targets
python src/feature_engineering.py

# 3. Push the engineered dataset to the Hopsworks Feature Store
python src/push_to_hopsworks.py

# 4. Compare candidate models (optional — for reference/reproducibility)
python src/model_comparison.py

# 5. Train the final selected models and register them to the Model Registry
python src/training_pipeline.py
```

After this, the live pipeline can be run manually or left to GitHub Actions:
```bash
python src/feature_pipeline.py
```

---

## Automation (CI/CD)

Two GitHub Actions workflows, both requiring a `HOPSWORKS_API_KEY` repository secret (**Settings → Secrets and variables → Actions**):

| Workflow | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Hourly (`cron: "5 * * * *"`) | Fetches live weather + air quality, computes features, inserts a new row into the Feature Store |
| `training_pipeline.yml` | Daily (`cron: "30 2 * * *"`) | Retrains all three models on current Feature Store data, registers new versions to the Model Registry |

Both can also be triggered manually from the **Actions** tab (`workflow_dispatch`).

> **Note:** GitHub Actions' free tier does not guarantee scheduled workflows run at exact hourly boundaries — actual run times can be irregular. `feature_pipeline.py` accounts for this by matching lag features to the *nearest available* timestamp within a 2-hour tolerance, rather than requiring an exact match.

---

## Dashboard

Run locally:
```bash
streamlit run src/app.py
```

The deployed version on Streamlit Community Cloud connects to the same Hopsworks Feature Store and Model Registry — no local data required. To deploy your own copy:

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file path `src/app.py`
4. Under **Advanced settings**, add the `HOPSWORKS_API_KEY` secret
5. Deploy

The dashboard caches Feature Store reads for 1 hour (`@st.cache_data(ttl=3600)`) to match the pipeline's own update frequency — data may lag by up to an hour after a fresh pipeline run, which is expected.

---

## Known Issues / Gotchas

These were real issues hit during development — documented here so they don't need to be re-diagnosed:

- **`hopsworks` fails to install on Windows** with a `twofish`/C++ compiler error → install Visual C++ Build Tools first.
- **`hopsworks.login()` fails with a missing `/tmp` path on Windows** → create the directory, or rely on the `os.makedirs("/tmp", exist_ok=True)` already present in the scripts.
- **Hopsworks' Arrow Flight Query Service intermittently fails to read data** (`FlightUnavailableError`) → all read paths fall back to `read_options={"use_hive": True}` on failure.
- **Streamlit Cloud deploy fails with `ModuleNotFoundError: No module named 'imp'`** → caused by Streamlit Cloud defaulting to Python 3.14, which removed the `imp` module; fixed via `runtime.txt` pinning Python 3.11.
- **Lag features come back `None` on live rows** → caused by GitHub Actions' irregular scheduling breaking an exact-timestamp lag match; resolved with a nearest-match-within-tolerance approach in `feature_pipeline.py`.

---

## Project Report

A detailed report covering data source selection reasoning, EDA findings, model comparison and selection, SHAP explainability results, and the engineering problems encountered (and how each was resolved) is available separately: **AQI_Predictor_Final_Report.docx**.

---

**Built by Wali Muhammad Nasir** — Internship Project, 10Pearls Shine Program - Cohort 9
