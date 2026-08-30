# Fetch live data from Open-Meteo

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

LAT, LON = 33.6844, 73.0479
CITY = "Islamabad"

def fetch_current_weather():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m",
        "past_hours": 1,
        "forecast_hours": 1,
    }
    response = requests.get(WEATHER_URL, params=params)
    response.raise_for_status()
    return response.json()

def fetch_current_air_quality():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,us_aqi",
        "past_hours": 1,
        "forecast_hours": 1,
    }
    response = requests.get(AIR_QUALITY_URL, params=params)
    response.raise_for_status()
    return response.json()

# Build the new row, using Feature Store history for lags
import hopsworks
from dotenv import load_dotenv

os.makedirs("/tmp", exist_ok=True)
load_dotenv()

def get_recent_history(fs, hours_needed=24):
    """Pull the most recent rows from the Feature Store to compute lag features."""
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    try:
        df = aqi_fg.read()
    except Exception:
        df = aqi_fg.read(read_options={"use_hive": True})
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(hours_needed + 48)  # wider buffer so the tolerance match below has enough rows to search


def get_lag_value(history, target_time, tolerance_hours=2):
    if len(history) == 0:
        return None
    time_diffs = (history["time"] - target_time).abs()
    closest_idx = time_diffs.idxmin()
    if time_diffs[closest_idx] <= pd.Timedelta(hours=tolerance_hours):
        return history.loc[closest_idx, "us_aqi"]
    return None


def build_live_row(fs):
    weather_json = fetch_current_weather()
    air_json = fetch_current_air_quality()

    weather_df = pd.DataFrame(weather_json["hourly"])
    air_df = pd.DataFrame(air_json["hourly"])

    print("Weather rows:", len(weather_df))
    print(weather_df)
    print("\nAir quality rows:", len(air_df))
    print(air_df)

    merged = pd.merge(weather_df, air_df, on="time", how="inner")
    print("\nMerged rows:", len(merged))
    print(merged)

    merged["city"] = CITY
    merged["time"] = pd.to_datetime(merged["time"])

    # Take the most recent complete hour 
    new_row = merged.iloc[[-2]].copy().reset_index(drop=True)

    # Time-based features
    new_row["hour"] = new_row["time"].dt.hour
    new_row["day"] = new_row["time"].dt.day
    new_row["month"] = new_row["time"].dt.month
    new_row["day_of_week"] = new_row["time"].dt.dayofweek

    # Pull recent history to compute lags
    history = get_recent_history(fs)
    row_time = new_row["time"].iloc[0]
    if row_time.tzinfo is None:
        row_time = row_time.tz_localize("UTC")
        
    lag_1h = get_lag_value(history, row_time - timedelta(hours=1))
    lag_3h = get_lag_value(history, row_time - timedelta(hours=3))
    lag_6h = get_lag_value(history, row_time - timedelta(hours=6))
    lag_24h = get_lag_value(history, row_time - timedelta(hours=24))

    new_row["aqi_lag_1h"] = lag_1h
    new_row["aqi_lag_3h"] = lag_3h
    new_row["aqi_lag_6h"] = lag_6h
    new_row["aqi_lag_24h"] = lag_24h

    new_row["aqi_change_rate_1h"] = new_row["us_aqi"].iloc[0] - lag_1h if lag_1h is not None else None
    new_row["aqi_change_rate_24h"] = new_row["us_aqi"].iloc[0] - lag_24h if lag_24h is not None else None

    new_row["hour"] = new_row["hour"].astype("int64")
    new_row["day"] = new_row["day"].astype("int64")
    new_row["month"] = new_row["month"].astype("int64")
    new_row["day_of_week"] = new_row["day_of_week"].astype("int64")

    new_row["target_24h"] = None
    new_row["target_48h"] = None
    new_row["target_72h"] = None

    float_cols = ["aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h", "aqi_change_rate_1h", "aqi_change_rate_24h"]
    for col in float_cols:
        new_row[col] = new_row[col].astype("float64")

    return new_row

    return new_row

# Insert into Feature Store

def main():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    new_row = build_live_row(fs)
    print(new_row)

    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    aqi_fg.insert(new_row)
    print(f"Inserted new row for {new_row['time'].iloc[0]}")

if __name__ == "__main__":
    main()
