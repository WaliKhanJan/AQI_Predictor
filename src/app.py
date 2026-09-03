import streamlit as st
import os
import joblib
import hopsworks
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from aqi_utils import get_aqi_category, check_forecast_alerts

os.makedirs("/tmp", exist_ok=True)
load_dotenv()

st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide", page_icon="🌫️")

FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide",
    "hour", "month", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate_1h", "aqi_change_rate_24h"
]

AQI_COLORS = {
    "Good": "#00e676",
    "Moderate": "#ffeb3b",
    "Unhealthy for Sensitive Groups": "#ff9800",
    "Unhealthy": "#f44336",
    "Very Unhealthy": "#9c27b0",
    "Hazardous": "#7e0023",
}

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups (children, elderly, respiratory conditions) should reduce prolonged outdoor exertion.",
    "Unhealthy": "Everyone may experience health effects. Limit prolonged outdoor exertion.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Avoid outdoor exertion.",
    "Hazardous": "Health emergency: entire population is likely affected. Avoid all outdoor activity.",
}


def safe_read(aqi_fg):
    try:
        return aqi_fg.read()
    except Exception:
        return aqi_fg.read(read_options={"use_hive": True})

def get_latest_model_dir(mr, name):
    models = mr.get_models(name)
    latest = max(models, key=lambda m: m.version)
    return latest.download()

@st.cache_resource(ttl=86400)
def load_models():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()

    model_24h_dir = get_latest_model_dir(mr, "aqi_model_24h")
    model_48h_dir = get_latest_model_dir(mr, "aqi_model_48h")
    model_72h_dir = get_latest_model_dir(mr, "aqi_model_72h")

    model_24h = joblib.load(os.path.join(model_24h_dir, "model_24h.pkl"))
    model_48h = joblib.load(os.path.join(model_48h_dir, "model_48h.pkl"))
    model_72h = joblib.load(os.path.join(model_72h_dir, "model_72h.pkl"))

    return model_24h, model_48h, model_72h


@st.cache_data(ttl=3600)
def get_feature_store_data():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df = safe_read(aqi_fg)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def colored_line_chart(data, x_col, y_col, title=""):
    base = alt.Chart(data).mark_line(color="#4fc3f7", strokeWidth=2).encode(
        x=alt.X(f"{x_col}:T", title=""),
        y=alt.Y(f"{y_col}:Q", title="US AQI"),
        tooltip=[x_col, y_col],
    )
    return base.properties(height=300, title=title).interactive()


def colored_forecast_bar(forecast_df):
    forecast_df = forecast_df.copy()
    forecast_df["Category"] = forecast_df["AQI"].apply(lambda v: get_aqi_category(v)["category"])

    chart = alt.Chart(forecast_df).mark_bar().encode(
        x=alt.X("Time:N", sort=None, title=""),
        y=alt.Y("AQI:Q"),
        color=alt.Color("Category:N", scale=alt.Scale(domain=list(AQI_COLORS.keys()), range=list(AQI_COLORS.values())), legend=alt.Legend(title="AQI Category")),
        tooltip=["Time", "AQI", "Category"],
    ).properties(height=300)
    return chart


with st.sidebar:
    st.header("About")
    st.write("Serverless AQI prediction system for Islamabad, using weather + pollutant data.")
    st.write("**Models**: XGBoost (24h), Ridge Regression (48h, 72h)")
    st.write("**Data source**: Open-Meteo")
    st.write("**Update frequency**: Hourly (automated via GitHub Actions)")
    st.divider()
    st.caption("10Pearls Internship Project — Wali Muhammad Nasir")

st.title("🌫️ Islamabad AQI Predictor")

with st.spinner("Loading models..."):
    model_24h, model_48h, model_72h = load_models()

with st.spinner("Fetching latest data..."):
    df = get_feature_store_data()

complete_rows = df.dropna(subset=FEATURE_COLS)
latest_row = complete_rows.iloc[-1]

X_latest = pd.DataFrame([latest_row[FEATURE_COLS]])

pred_24h = model_24h.predict(X_latest)[0]
pred_48h = model_48h.predict(X_latest)[0]
pred_72h = model_72h.predict(X_latest)[0]

st.caption(f"Prediction based on: {latest_row['time']} (UTC) — most recent row with complete feature data")

current_aqi = latest_row["us_aqi"]
current_info = get_aqi_category(current_aqi)

st.metric(label="Current AQI", value=f"{current_aqi:.0f}")
st.caption(f"Category: {current_info['category']}")

st.subheader("Current Conditions")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Temperature", f"{latest_row['temperature_2m']:.1f}°C")
c2.metric("Humidity", f"{latest_row['relative_humidity_2m']:.0f}%")
c3.metric("Wind Speed", f"{latest_row['wind_speed_10m']:.1f} km/h")
c4.metric("Pressure", f"{latest_row['surface_pressure']:.0f} hPa")

st.write("")

tab1, tab2, tab3 = st.tabs(["📊 Forecast", "📈 Trends", "🔍 Model Insights"])

with tab1:
    st.subheader("3-Day AQI Forecast")
    col1, col2, col3 = st.columns(3)

    for col, horizon, pred in zip([col1, col2, col3], ["24h", "48h", "72h"], [pred_24h, pred_48h, pred_72h]):
        info = get_aqi_category(pred)
        color = AQI_COLORS[info["category"]]
        with col:
            st.markdown(
                f"""
                <div style="border-left: 6px solid {color}; padding: 10px 16px; border-radius: 6px; background-color: rgba(255,255,255,0.03);">
                    <div style="font-size: 14px; opacity: 0.7;">In {horizon}</div>
                    <div style="font-size: 36px; font-weight: 700;">{pred:.0f}</div>
                    <div style="font-size: 14px; color: {color}; font-weight: 600;">{info['category']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(HEALTH_ADVICE[info["category"]])

    st.write("")
    st.subheader("Forecast Trajectory")
    forecast_data = pd.DataFrame({
        "Time": ["Now", "+24h", "+48h", "+72h"],
        "AQI": [current_aqi, pred_24h, pred_48h, pred_72h]
    })
    st.altair_chart(colored_forecast_bar(forecast_data), use_container_width=True)

    predictions_dict = {"24h": pred_24h, "48h": pred_48h, "72h": pred_72h}
    alerts = check_forecast_alerts(predictions_dict)

    if alerts:
        st.subheader("⚠️ Air Quality Alerts")
        for alert in alerts:
            st.error(f"**{alert['horizon']} forecast**: AQI {alert['aqi']:.0f} — {alert['category']}")
    else:
        st.subheader("✅ No Hazardous Air Quality Alerts")
        st.info("Forecasted AQI levels remain within acceptable ranges for the next 3 days.")

with tab2:
    st.subheader("Recent AQI Trend")
    trend_df = df.tail(72)[["time", "us_aqi"]]
    st.altair_chart(colored_line_chart(trend_df, "time", "us_aqi"), use_container_width=True)
    st.caption(f"{len(trend_df)} most recent stored hours "
               f"({trend_df['time'].min()} to {trend_df['time'].max()})")

    st.subheader("Full Historical Trend (Daily Average)")
    daily_df = df.set_index("time")["us_aqi"].resample("D").mean().reset_index()
    st.altair_chart(colored_line_chart(daily_df, "time", "us_aqi"), use_container_width=True)
    st.caption(f"Daily average AQI, {daily_df['time'].min().date()} to {daily_df['time'].max().date()}")

    st.subheader("Current Pollutant Levels")
    pollutants = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
    pollutant_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
        "Value": [latest_row[p] for p in pollutants]
    })
    st.bar_chart(pollutant_df.set_index("Pollutant"))

with tab3:
    st.subheader("SHAP Feature Importance")
    st.write("What drives each model's predictions — computed via SHAP on the test set.")

    shap_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")

    for horizon, fname in [("24h (XGBoost)", "shap_summary_24h.png"),
                            ("48h (Ridge)", "shap_summary_48h.png"),
                            ("72h (Ridge)", "shap_summary_72h.png")]:
        img_path = os.path.join(shap_dir, fname)
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Feature importance — {horizon} model")
        else:
            st.caption(f"SHAP plot not found for {horizon}")

st.divider()
st.caption(f"Latest data: {latest_row['time']} (UTC) | City: {latest_row['city']}")
