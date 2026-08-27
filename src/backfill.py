import requests
import pandas as pd

CITIES = {
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
}
START_DATE = "2024-01-01"
END_DATE = "2026-08-25"

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_air_quality(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,us_aqi"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def json_to_df(json_data):
    hourly_data = json_data["hourly"]
    df = pd.DataFrame(hourly_data)
    return df

def build_city_dataset(city_name, lat, lon):
    print(f"Fetching weather for {city_name}...")
    weather_json = fetch_weather(lat, lon, START_DATE, END_DATE)
    weather_df = json_to_df(weather_json)

    print(f"Fetching air quality for {city_name}...")
    air_json = fetch_air_quality(lat, lon, START_DATE, END_DATE)
    air_df = json_to_df(air_json)

    merged_df = pd.merge(weather_df, air_df, on="time", how="inner")
    merged_df["city"] = city_name

    return merged_df

def main():
    all_cities_data = []

    for city_name, coords in CITIES.items():
        city_df = build_city_dataset(city_name, coords["lat"], coords["lon"])
        all_cities_data.append(city_df)
        print(f"{city_name}: {len(city_df)} rows collected.")

    final_df = pd.concat(all_cities_data, ignore_index=True)

    output_path = "data/historical_features.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(final_df)} total rows to {output_path}")


if __name__ == "__main__":
    main()