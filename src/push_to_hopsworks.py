import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

os.makedirs("/tmp", exist_ok=True)  

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)

api_key = os.getenv("HOPSWORKS_API_KEY")

# Connect
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
print(f"Connected to project: {project.name}")

# Load engineered data
data_path = os.path.join(BASE_DIR, "data", "engineered_features.csv")
df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])
print(f"Loaded {len(df)} rows from {data_path}")

# Create (or get existing) Feature Group
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Engineered weather + air quality features for AQI prediction",
    primary_key=["time", "city"],
    event_time="time",
    time_travel_format="HUDI",
)

# Insert data
aqi_fg.insert(df)
print("Data inserted into Feature Store successfully.")