# Step 1: Fetch live data from Open-Meteo

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

# Step 2: Build the new row, using Feature Store history for lags
import hopsworks
from dotenv import load_dotenv

os.makedirs("/tmp", exist_ok=True)
load_dotenv()

def get_recent_history(fs, hours_needed=24):
    """Pull the most recent rows from the Feature Store to compute lag features."""
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df = aqi_fg.read()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    return df.tail(hours_needed + 5)  # small buffer

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

    # Take the most recent complete hour (not the forecast hour)
    new_row = merged.iloc[[-2]].copy().reset_index(drop=True)

    # Time-based features
    new_row["hour"] = new_row["time"].dt.hour
    new_row["day"] = new_row["time"].dt.day
    new_row["month"] = new_row["time"].dt.month
    new_row["day_of_week"] = new_row["time"].dt.dayofweek

    # Pull recent history to compute lags
    history = get_recent_history(fs)

    def get_lag_value(hours_back):
        target_time = new_row["time"].iloc[0] - timedelta(hours=hours_back)
        match = history[history["time"] == target_time]
        return match["us_aqi"].iloc[0] if len(match) > 0 else None

    new_row["aqi_lag_1h"] = get_lag_value(1)
    new_row["aqi_lag_3h"] = get_lag_value(3)
    new_row["aqi_lag_6h"] = get_lag_value(6)
    new_row["aqi_lag_24h"] = get_lag_value(24)

    new_row["aqi_change_rate_1h"] = new_row["us_aqi"].iloc[0] - new_row["aqi_lag_1h"].iloc[0] if new_row["aqi_lag_1h"].iloc[0] is not None else None
    new_row["aqi_change_rate_24h"] = new_row["us_aqi"].iloc[0] - new_row["aqi_lag_24h"].iloc[0] if new_row["aqi_lag_24h"].iloc[0] is not None else None


    new_row["hour"] = new_row["hour"].astype("int64")
    new_row["day"] = new_row["day"].astype("int64")
    new_row["month"] = new_row["month"].astype("int64")
    new_row["day_of_week"] = new_row["day_of_week"].astype("int64")

    new_row["target_24h"] = None
    new_row["target_48h"] = None
    new_row["target_72h"] = None

    return new_row


    return new_row

#Step 3: Insert into Feature Store

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