def get_aqi_category(aqi_value):
    """
    Maps a US AQI value to its official category, alert level, and color.
    Based on standard US EPA AQI breakpoints.
    """
    if aqi_value <= 50:
        return {"category": "Good", "alert": False, "level": 0, "color": "green"}
    elif aqi_value <= 100:
        return {"category": "Moderate", "alert": False, "level": 1, "color": "yellow"}
    elif aqi_value <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "alert": False, "level": 2, "color": "orange"}
    elif aqi_value <= 200:
        return {"category": "Unhealthy", "alert": True, "level": 3, "color": "red"}
    elif aqi_value <= 300:
        return {"category": "Very Unhealthy", "alert": True, "level": 4, "color": "purple"}
    else:
        return {"category": "Hazardous", "alert": True, "level": 5, "color": "maroon"}


def check_forecast_alerts(predictions_dict):
    """
    Takes a dict like {"24h": 145, "48h": 210, "72h": 95}
    and returns which horizons trigger a hazardous alert.
    """
    alerts = []
    for horizon, aqi_value in predictions_dict.items():
        info = get_aqi_category(aqi_value)
        if info["alert"]:
            alerts.append({
                "horizon": horizon,
                "aqi": aqi_value,
                "category": info["category"],
            })
    return alerts

