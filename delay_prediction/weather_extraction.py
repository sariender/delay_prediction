# +
"""Open-Meteo weather extraction and join utilities for the SBB delay model.

Provides three things:
  1. fetch_forecast_weather  — D-1 / D-2 archived forecasts (Previous Runs API)
  2. fetch_observed_weather  — hourly observed weather (Archive API)
  3. enrich_with_weather     — join cached weather onto a features DataFrame
"""

import requests
import pandas as pd
from pyspark.sql import functions as F


LAUSANNE_LAT = 46.5168
LAUSANNE_LON = 6.6291

FORECAST_API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HOURLY_VARS = [
    # D-1 forecast
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "rain_previous_day1",
    "snowfall_previous_day1",
    "weather_code_previous_day1",
    "wind_speed_10m_previous_day1",
    "wind_gusts_10m_previous_day1",
    # D-2 forecast
    "temperature_2m_previous_day2",
    "relative_humidity_2m_previous_day2",
    "precipitation_previous_day2",
    "rain_previous_day2",
    "snowfall_previous_day2",
    "weather_code_previous_day2",
    "wind_speed_10m_previous_day2",
    "wind_gusts_10m_previous_day2",
]

OBS_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
]

REVISION_PAIRS = [
    ("temperature_2m_fcst_d1", "temperature_2m_fcst_d2", "temperature_2m_fcst_d1_minus_d2"),
    ("precipitation_fcst_d1",  "precipitation_fcst_d2",  "precipitation_fcst_d1_minus_d2"),
    ("rain_fcst_d1",           "rain_fcst_d2",           "rain_fcst_d1_minus_d2"),
    ("snowfall_fcst_d1",       "snowfall_fcst_d2",       "snowfall_fcst_d1_minus_d2"),
    ("wind_speed_10m_fcst_d1", "wind_speed_10m_fcst_d2", "wind_speed_10m_fcst_d1_minus_d2"),
]

WEATHER_FEATURE_COLS = [
    # D-1
    "temperature_2m_fcst_d1",
    "relative_humidity_2m_fcst_d1",
    "precipitation_fcst_d1",
    "rain_fcst_d1",
    "snowfall_fcst_d1",
    "weather_code_fcst_d1",
    "wind_speed_10m_fcst_d1",
    "wind_gusts_10m_fcst_d1",

    # D-2
    "temperature_2m_fcst_d2",
    "relative_humidity_2m_fcst_d2",
    "precipitation_fcst_d2",
    "rain_fcst_d2",
    "snowfall_fcst_d2",
    "weather_code_fcst_d2",
    "wind_speed_10m_fcst_d2",
    "wind_gusts_10m_fcst_d2",

    # Forecast revision
    "temperature_2m_fcst_d1_minus_d2",
    "precipitation_fcst_d1_minus_d2",
    "rain_fcst_d1_minus_d2",
    "snowfall_fcst_d1_minus_d2",
    "wind_speed_10m_fcst_d1_minus_d2",

    # Short-notice observed proxy
    "temperature_2m_obs",
    "relative_humidity_2m_obs",
    "precipitation_obs",
    "rain_obs",
    "snowfall_obs",
    "weather_code_obs",
    "wind_speed_10m_obs",
    "wind_gusts_10m_obs",
]


def fetch_forecast_weather(
    spark,
    cache_path,
    start_date,
    end_date,
    lat=LAUSANNE_LAT,
    lon=LAUSANNE_LON,
    timezone="Europe/Zurich",
):
    """Fetch D-1 and D-2 archived forecasts and write to Parquet at `cache_path`.

    Returns the cached Spark DataFrame.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": FORECAST_HOURLY_VARS,
    }

    response = requests.get(FORECAST_API_URL, params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo forecast request failed ({response.status_code}): "
            f"{response.text[:500]}"
        )

    hourly = response.json()["hourly"]

    forecast_pdf = pd.DataFrame({
        "weather_hour": pd.to_datetime(hourly["time"]),
        # D-1
        "temperature_2m_fcst_d1":       hourly.get("temperature_2m_previous_day1"),
        "relative_humidity_2m_fcst_d1": hourly.get("relative_humidity_2m_previous_day1"),
        "precipitation_fcst_d1":        hourly.get("precipitation_previous_day1"),
        "rain_fcst_d1":                 hourly.get("rain_previous_day1"),
        "snowfall_fcst_d1":             hourly.get("snowfall_previous_day1"),
        "weather_code_fcst_d1":         hourly.get("weather_code_previous_day1"),
        "wind_speed_10m_fcst_d1":       hourly.get("wind_speed_10m_previous_day1"),
        "wind_gusts_10m_fcst_d1":       hourly.get("wind_gusts_10m_previous_day1"),
        # D-2
        "temperature_2m_fcst_d2":       hourly.get("temperature_2m_previous_day2"),
        "relative_humidity_2m_fcst_d2": hourly.get("relative_humidity_2m_previous_day2"),
        "precipitation_fcst_d2":        hourly.get("precipitation_previous_day2"),
        "rain_fcst_d2":                 hourly.get("rain_previous_day2"),
        "snowfall_fcst_d2":             hourly.get("snowfall_previous_day2"),
        "weather_code_fcst_d2":         hourly.get("weather_code_previous_day2"),
        "wind_speed_10m_fcst_d2":       hourly.get("wind_speed_10m_previous_day2"),
        "wind_gusts_10m_fcst_d2":       hourly.get("wind_gusts_10m_previous_day2"),
    })

    forecast_df = spark.createDataFrame(forecast_pdf)
    for d1_col, d2_col, diff_col in REVISION_PAIRS:
        forecast_df = forecast_df.withColumn(diff_col, F.col(d1_col) - F.col(d2_col))

    forecast_df.write.mode("overwrite").parquet(cache_path)
    return spark.read.parquet(cache_path).cache()


def fetch_observed_weather(
    spark,
    cache_path,
    start_date,
    end_date,
    lat=LAUSANNE_LAT,
    lon=LAUSANNE_LON,
    timezone="Europe/Zurich",
):
    """Fetch hourly observed weather and write to Parquet at `cache_path`.

    Returns the cached Spark DataFrame.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": OBS_HOURLY_VARS,
    }

    response = requests.get(ARCHIVE_API_URL, params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo observed weather request failed ({response.status_code}): "
            f"{response.text[:500]}"
        )

    obs_hourly = response.json()["hourly"]

    observed_pdf = pd.DataFrame({
        "observed_weather_hour":    pd.to_datetime(obs_hourly["time"]),
        "temperature_2m_obs":       obs_hourly.get("temperature_2m"),
        "relative_humidity_2m_obs": obs_hourly.get("relative_humidity_2m"),
        "precipitation_obs":        obs_hourly.get("precipitation"),
        "rain_obs":                 obs_hourly.get("rain"),
        "snowfall_obs":             obs_hourly.get("snowfall"),
        "weather_code_obs":         obs_hourly.get("weather_code"),
        "wind_speed_10m_obs":       obs_hourly.get("wind_speed_10m"),
        "wind_gusts_10m_obs":       obs_hourly.get("wind_gusts_10m"),
    })

    observed_df = spark.createDataFrame(observed_pdf)
    observed_df.write.mode("overwrite").parquet(cache_path)
    return spark.read.parquet(cache_path).cache()


def load_cached_weather(spark, forecast_cache_path, obs_cache_path):
    """Load forecast + observed weather from Parquet caches."""
    forecast_df = spark.read.parquet(forecast_cache_path).cache()
    observed_df = spark.read.parquet(obs_cache_path).cache()
    return forecast_df, observed_df


def enrich_with_weather(features_df, forecast_df, observed_df, scheduled_ts_col="scheduled_arrival_ts"):
    """Join forecast on `weather_hour` and observed on `weather_hour - 1h`."""
    return (
        features_df
        .withColumn("weather_hour", F.date_trunc("hour", F.col(scheduled_ts_col)))
        .withColumn("observed_weather_hour", F.col("weather_hour") - F.expr("INTERVAL 1 HOUR"))
        .join(forecast_df, on="weather_hour", how="left")
        .join(observed_df, on="observed_weather_hour", how="left")
    )
