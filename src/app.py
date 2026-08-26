import streamlit as st
import os
import joblib
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from aqi_utils import get_aqi_category, check_forecast_alerts

os.makedirs("/tmp", exist_ok=True)
load_dotenv()

st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide")
st.title("🌫️ Islamabad AQI Predictor")

FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide",
    "hour", "month", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate_1h", "aqi_change_rate_24h"
]


def safe_read(aqi_fg):
    """Try the default read; fall back to use_hive if the Query Service is flaky."""
    try:
        return aqi_fg.read()
    except Exception:
        return aqi_fg.read(read_options={"use_hive": True})


@st.cache_resource
def load_models():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()

    model_24h_dir = mr.get_model("aqi_model_24h", version=2).download()
    model_48h_dir = mr.get_model("aqi_model_48h", version=2).download()
    model_72h_dir = mr.get_model("aqi_model_72h", version=2).download()

    model_24h = joblib.load(os.path.join(model_24h_dir, "model_24h.pkl"))
    model_48h = joblib.load(os.path.join(model_48h_dir, "model_48h.pkl"))
    model_72h = joblib.load(os.path.join(model_72h_dir, "model_72h.pkl"))

    return model_24h, model_48h, model_72h


@st.cache_data(ttl=3600)
def get_feature_store_data():
    """Single read of the full feature table — reused for both the latest row and the trend chart."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df = safe_read(aqi_fg)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
with st.spinner("Loading models..."):
    model_24h, model_48h, model_72h = load_models()
st.success("Models loaded successfully")

# ---------------------------------------------------------------------------
# Load data (one read, used everywhere below)
# ---------------------------------------------------------------------------
with st.spinner("Fetching latest data..."):
    df = get_feature_store_data()

latest_row = df.iloc[-1]

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
X_latest = pd.DataFrame([latest_row[FEATURE_COLS]])

pred_24h = model_24h.predict(X_latest)[0]
pred_48h = model_48h.predict(X_latest)[0]
pred_72h = model_72h.predict(X_latest)[0]

st.subheader("3-Day AQI Forecast")
col1, col2, col3 = st.columns(3)

for col, horizon, pred in zip([col1, col2, col3], ["24h", "48h", "72h"], [pred_24h, pred_48h, pred_72h]):
    info = get_aqi_category(pred)
    with col:
        st.metric(label=f"In {horizon}", value=f"{pred:.0f}")
        st.caption(info["category"])
        if info["alert"]:
            st.warning(f"⚠️ {info['category']}")

# ---------------------------------------------------------------------------
# Recent trend chart
# ---------------------------------------------------------------------------
st.subheader("Recent AQI Trend")
trend_df = df.tail(72)[["time", "us_aqi"]]
st.caption(f"Showing {len(trend_df)} most recent stored hours "
           f"({trend_df['time'].min()} to {trend_df['time'].max()})")
st.line_chart(trend_df.set_index("time"))

# ---------------------------------------------------------------------------
# Forecast trajectory chart
# ---------------------------------------------------------------------------
st.subheader("Forecast Trajectory")
forecast_data = pd.DataFrame({
    "Time": ["Now", "+24h", "+48h", "+72h"],
    "AQI": [latest_row["us_aqi"], pred_24h, pred_48h, pred_72h]
})
st.bar_chart(forecast_data.set_index("Time"))

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
predictions_dict = {"24h": pred_24h, "48h": pred_48h, "72h": pred_72h}
alerts = check_forecast_alerts(predictions_dict)

if alerts:
    st.subheader("⚠️ Air Quality Alerts")
    for alert in alerts:
        st.error(f"**{alert['horizon']} forecast**: AQI {alert['aqi']:.0f} — {alert['category']}")
else:
    st.subheader("✅ No Hazardous Air Quality Alerts")
    st.info("Forecasted AQI levels remain within acceptable ranges for the next 3 days.")

st.caption(f"Latest data: {latest_row['time']} (UTC) | City: {latest_row['city']}")
