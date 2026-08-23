import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
import os
import hopsworks
from dotenv import load_dotenv

os.makedirs("/tmp", exist_ok=True)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
df = aqi_fg.read()
df["time"] = pd.to_datetime(df["time"])

print(f"Loaded {len(df)} rows from Hopsworks Feature Store")

# ---------------------------------------------------------------------------
# Chronological train/test split — never shuffle time-series data
# ---------------------------------------------------------------------------
SPLIT_DATE = "2026-02-01"
train = df[df["time"] < SPLIT_DATE].copy()
test = df[df["time"] >= SPLIT_DATE].copy()

train = train.dropna()
test = test.dropna()

print(f"Train: {len(train)} rows ({train['time'].min()} to {train['time'].max()})")
print(f"Test: {len(test)} rows ({test['time'].min()} to {test['time'].max()})")

# ---------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------
feature_cols = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide",
    "hour", "month", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_rate_1h", "aqi_change_rate_24h"
]

X_train = train[feature_cols]
X_test = test[feature_cols]

# ---------------------------------------------------------------------------
# Final model selection (chosen after comparing Ridge / Random Forest / XGBoost
# across all three horizons — see src/model_comparison.py for the full comparison
# and reasoning):
#   - 24h: XGBoost  (best performer at the shortest, strongest-signal horizon)
#   - 48h: Ridge    (matched/outperformed tree-based models as signal weakens)
#   - 72h: Ridge    (same reasoning as 48h)
# ---------------------------------------------------------------------------

def train_and_evaluate(model, horizon_name, target_col):
    y_train = train[target_col]
    y_test = test[target_col]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{horizon_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


model_24h, metrics_24h = train_and_evaluate(
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
    "24h (XGBoost)",
    "target_24h",
)

model_48h, metrics_48h = train_and_evaluate(
    Ridge(alpha=1.0),
    "48h (Ridge)",
    "target_48h",
)

model_72h, metrics_72h = train_and_evaluate(
    Ridge(alpha=1.0),
    "72h (Ridge)",
    "target_72h",
)

print("\nTraining complete. Models ready for saving to Model Registry.")



# SAVING MODELS TO HOPSWORK
import hopsworks
from dotenv import load_dotenv
import joblib

os.makedirs("/tmp", exist_ok=True)
load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key)
mr = project.get_model_registry()

# ---------------------------------------------------------------------------
# Save each model locally first (Hopsworks needs a file path to upload)
# ---------------------------------------------------------------------------
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model_24h, os.path.join(models_dir, "model_24h.pkl"))
joblib.dump(model_48h, os.path.join(models_dir, "model_48h.pkl"))
joblib.dump(model_72h, os.path.join(models_dir, "model_72h.pkl"))

# ---------------------------------------------------------------------------
# Register each model in the Model Registry, with its metrics attached
# ---------------------------------------------------------------------------
def register_model(model_name, model_path, metrics, description):
    model_meta = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=description,
    )
    model_meta.save(model_path)
    print(f"Registered: {model_name}")

register_model(
    "aqi_model_24h",
    os.path.join(models_dir, "model_24h.pkl"),
    metrics_24h,
    "XGBoost model predicting AQI 24 hours ahead"
)

register_model(
    "aqi_model_48h",
    os.path.join(models_dir, "model_48h.pkl"),
    metrics_48h,
    "Ridge Regression model predicting AQI 48 hours ahead"
)

register_model(
    "aqi_model_72h",
    os.path.join(models_dir, "model_72h.pkl"),
    metrics_72h,
    "Ridge Regression model predicting AQI 72 hours ahead"
)

print("\nAll models registered to Hopsworks Model Registry.")