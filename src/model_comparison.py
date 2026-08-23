import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)

data_path = os.path.join(BASE_DIR, "data", "engineered_features.csv")
df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])

print(f"Loaded {len(df)} rows from local file")
print(df.head()) 

# Split chronologically — never shuffle time-series data
train = df[df["time"] < "2026-02-01"].copy()
test = df[df["time"] >= "2026-02-01"].copy()

print(f"Train: {len(train)} rows ({train['time'].min()} to {train['time'].max()})")
print(f"Test: {len(test)} rows ({test['time'].min()} to {test['time'].max()})")

# Drop rows with missing values (the 96 rows from lag/target shifting)
train = train.dropna()
test = test.dropna()

print(f"\nAfter dropping NaN rows:")
print(f"Train: {len(train)} rows")
print(f"Test: {len(test)} rows")

# Define feature columns (inputs) — excluding time, targets, raw us_aqi, city (non-numeric), and 'day'
feature_cols = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide",
    "hour", "month", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate_1h", "aqi_change_rate_24h"
]

target_cols = ["target_24h", "target_48h", "target_72h"]

X_train = train[feature_cols]
y_train = train[target_cols]

X_test = test[feature_cols]
y_test = test[target_cols]

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Ridge Regression baseline
ridge = MultiOutputRegressor(Ridge(alpha=1.0))
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

# Evaluate each horizon separately
for i, horizon in enumerate(["24h", "48h", "72h"]):
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"Ridge - {horizon}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")


# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1  # use all CPU cores to speed up training
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest results:")
for i, horizon in enumerate(["24h", "48h", "72h"]):
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_rf[:, i]))
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_rf[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_rf[:, i])
    print(f"RF - {horizon}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")


# XGBoost
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

xgb = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("XGBoost results:")
for i, horizon in enumerate(["24h", "48h", "72h"]):
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_xgb[:, i]))
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_xgb[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_xgb[:, i])
    print(f"XGB - {horizon}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")

