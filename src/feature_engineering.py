import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "data", "historical_features.csv"))
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)  # ensure chronological order — critical for shift() to work correctly

# 1. Time-based features
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["day_of_week"] = df["time"].dt.dayofweek  # 0 = Monday, 6 = Sunday

# 2. Lag features — AQI at past points in time
df["aqi_lag_1h"] = df["us_aqi"].shift(1)
df["aqi_lag_3h"] = df["us_aqi"].shift(3)
df["aqi_lag_6h"] = df["us_aqi"].shift(6)
df["aqi_lag_24h"] = df["us_aqi"].shift(24)

# 3. AQI change rate — momentum/trend
df["aqi_change_rate_1h"] = df["us_aqi"] - df["aqi_lag_1h"]
df["aqi_change_rate_24h"] = df["us_aqi"] - df["aqi_lag_24h"]

# 4. Future targets — what we're actually predicting
df["target_24h"] = df["us_aqi"].shift(-24)
df["target_48h"] = df["us_aqi"].shift(-48)
df["target_72h"] = df["us_aqi"].shift(-72)

# Check the result
print(f"Total rows before cleanup: {len(df)}")
print(f"Rows with any missing lag/target values: {df.isnull().any(axis=1).sum()}")


# Save the engineered dataset
output_path = os.path.join(BASE_DIR, "data", "engineered_features.csv")
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")