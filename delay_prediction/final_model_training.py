# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prod Model

# %% [markdown]
# ## Setup 

# %%
import os
import pwd
import numpy as np
import pandas as pd
import sys
from pyspark.sql import Window


from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F


username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = "H1"

print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")

# %%
spark = (SparkSession\
            .builder
            .appName(username + '-final-2')
            .config('spark.ui.port', randrange(4050, 4450, 5))
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path))
            .config('spark.jars',
                    f'{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar,'
                    f'{hadoopFS}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar,'
                    f'{hadoopFS}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar'
            )
            .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')
            .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')
            .config('spark.sql.catalog.iceberg.type', 'hadoop')
            .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/silver/')
            .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')
            .config('spark.sql.catalog.spark_catalog.type', 'hadoop')
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/assignment-3/warehouse')
            .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/assignment-3/spark/warehouse')
            .config("spark.executor.memory", "6g")
            .config("spark.executor.cores", "4")
            .config("spark.executor.instances", "4")
        ).master('yarn').getOrCreate()

# %%
spark.sparkContext

# %% [markdown]
# ----

# %%
base = f"{hadoopFS}/user/groups/com-490/H1/final/v1"

# %%
print(repr(hadoopFS)); print(repr(base))

# %%
# !hadoop fs -ls /user/groups/com-490/H1/final/v1

# %%
#feature_v3_df.write.mode("overwrite").parquet(f"{base}/full_data_w_o_hist_weather_specialdays.parquet")

# %% [markdown]
# ### Full Data Training

# %%
feature_v3_df = spark.read.parquet(f"{base}/full_data_w_o_hist_weather_specialdays.parquet")
print(feature_v3_df.count())

# %%
full_base_df = feature_v3_df

# %%
full_base_df.agg(
    F.count("*").alias("n_rows"),
    F.min("operating_day").alias("min_day"),
    F.max("operating_day").alias("max_day"),
    F.countDistinct("operating_day").alias("n_days")
).show(truncate=False)


# %%
def make_delay_stats(df, group_cols, prefix):
    return (
        df
        .groupBy(*group_cols)
        .agg(
            F.count("*").alias(f"{prefix}_n_obs"),
            F.round(F.avg("arrival_delay_seconds"), 3).alias(f"{prefix}_mean_delay"),
            F.expr("percentile_approx(arrival_delay_seconds, 0.5)").alias(f"{prefix}_median_delay"),
            F.expr("percentile_approx(arrival_delay_seconds, 0.9)").alias(f"{prefix}_p90_delay")
        )
    )


# %%
def make_all_hist_tables(history_df):
    hist_operator_hour_df = make_delay_stats(
        history_df,
        ["operator_id", "hour_of_day"],
        "hist_operator_hour"
    )

    hist_line_hour_df = make_delay_stats(
        history_df,
        ["line_id", "hour_of_day"],
        "hist_line_hour"
    )

    hist_stop_hour_df = make_delay_stats(
        history_df,
        ["bpuic", "hour_of_day"],
        "hist_stop_hour"
    )

    hist_transport_hour_df = make_delay_stats(
        history_df,
        ["transport_clean", "hour_of_day"],
        "hist_transport_hour"
    )

    return (
        hist_operator_hour_df,
        hist_line_hour_df,
        hist_stop_hour_df,
        hist_transport_hour_df
    )


def add_historical_features(df, hist_tables):
    (
        hist_operator_hour_df,
        hist_line_hour_df,
        hist_stop_hour_df,
        hist_transport_hour_df
    ) = hist_tables

    return (
        df
        .join(hist_operator_hour_df, on=["operator_id", "hour_of_day"], how="left")
        .join(hist_line_hour_df, on=["line_id", "hour_of_day"], how="left")
        .join(hist_stop_hour_df, on=["bpuic", "hour_of_day"], how="left")
        .join(hist_transport_hour_df, on=["transport_clean", "hour_of_day"], how="left")
    )


# %%
#columns
hist_feature_cols = [
    "hist_operator_hour_n_obs",
    "hist_operator_hour_mean_delay",
    "hist_operator_hour_median_delay",
    "hist_operator_hour_p90_delay",

    "hist_line_hour_n_obs",
    "hist_line_hour_mean_delay",
    "hist_line_hour_median_delay",
    "hist_line_hour_p90_delay",

    "hist_stop_hour_n_obs",
    "hist_stop_hour_mean_delay",
    "hist_stop_hour_median_delay",
    "hist_stop_hour_p90_delay",

    "hist_transport_hour_n_obs",
    "hist_transport_hour_mean_delay",
    "hist_transport_hour_median_delay",
    "hist_transport_hour_p90_delay",
]


# %%
def make_hist_fill_values(history_df):
    global_delay_stats = history_df.agg(
        F.round(F.avg("arrival_delay_seconds"), 3).alias("global_mean_delay"),
        F.expr("percentile_approx(arrival_delay_seconds, 0.5)").alias("global_median_delay"),
        F.expr("percentile_approx(arrival_delay_seconds, 0.9)").alias("global_p90_delay")
    ).collect()[0]

    GLOBAL_MEAN_DELAY = float(global_delay_stats["global_mean_delay"])
    GLOBAL_MEDIAN_DELAY = float(global_delay_stats["global_median_delay"])
    GLOBAL_P90_DELAY = float(global_delay_stats["global_p90_delay"])

    print(f"Global mean delay:   {GLOBAL_MEAN_DELAY}")
    print(f"Global median delay: {GLOBAL_MEDIAN_DELAY}")
    print(f"Global p90 delay:    {GLOBAL_P90_DELAY}")

    fill_values = {}

    for c in hist_feature_cols:
        if c.endswith("_n_obs"):
            fill_values[c] = 0
        elif c.endswith("_mean_delay"):
            fill_values[c] = GLOBAL_MEAN_DELAY
        elif c.endswith("_median_delay"):
            fill_values[c] = GLOBAL_MEDIAN_DELAY
        elif c.endswith("_p90_delay"):
            fill_values[c] = GLOBAL_P90_DELAY

    return fill_values


# %%
hist_tables_full = make_all_hist_tables(full_base_df)
fill_values_full = make_hist_fill_values(full_base_df)

full_features_df = (
    add_historical_features(full_base_df, hist_tables_full)
    .fillna(fill_values_full)
)

# %%
from weather_extraction import load_cached_weather, enrich_with_weather, WEATHER_FEATURE_COLS

FORECAST_CACHE_PATH =  f"{base}/forecast_weather_2024_07_2026_01.parquet"
OBS_CACHE_PATH      =  f"{base}/open_meteo_lausanne_observed_hourly_2024_07_2026_01.parquet"

forecast_df, observed_weather_df = load_cached_weather(spark, FORECAST_CACHE_PATH, OBS_CACHE_PATH)

print(f"Forecast rows: {forecast_df.count():,}")
print(f"Observed weather rows: {observed_weather_df.count():,}")

# %%
full_weather_enriched_df = enrich_with_weather(
    full_features_df,
    forecast_df,
    observed_weather_df
)

# %%
from special_days import build_special_day_tables, add_special_day_features

special_days_df, school_holiday_ranges_df = build_special_day_tables(spark)

full_final_features_df = add_special_day_features(
    full_weather_enriched_df,
    special_days_df,
    school_holiday_ranges_df
)

# %%
LABEL_COL = "arrival_delay_seconds"

ID_COLS = [
    "operating_day",
    "scheduled_arrival_ts",
    "weather_hour",
    "observed_weather_hour",
    "trip_id",
    "bpuic",
    "stop_name",
]

CATEGORICAL_COLS = [
    "operator_id",
    "transport_clean",
    "line_text",
    # weather condition categories
    "weather_code_fcst_d1",
    "weather_code_fcst_d2",
    # special-day categories
    "special_day_type",
    "scheduled_timestamp_pattern"
]

NUMERIC_COLS = [
    # scheduled time / calendar
    "month",
    "day_of_week",
    "is_weekend",
    "hour_of_day",
    "minute_of_hour",
    "scheduled_minute_of_day",
    "is_morning_peak",
    "is_evening_peak",
    "is_late_night",

    # observed trip-position proxy
    "stop_sequence_idx",
    "n_stops_in_trip_observed",
    "relative_stop_position_v2",
    "scheduled_minutes_since_first_observed_stop",
    "is_first_observed_stop",
    "is_last_observed_stop",

    # historical delay features
    "hist_operator_hour_n_obs",
    "hist_operator_hour_mean_delay",
    "hist_operator_hour_median_delay",
    "hist_operator_hour_p90_delay",
    "hist_line_hour_n_obs",
    "hist_line_hour_mean_delay",
    "hist_line_hour_median_delay",
    "hist_line_hour_p90_delay",
    "hist_stop_hour_n_obs",
    "hist_stop_hour_mean_delay",
    "hist_stop_hour_median_delay",
    "hist_stop_hour_p90_delay",
    "hist_transport_hour_n_obs",
    "hist_transport_hour_mean_delay",
    "hist_transport_hour_median_delay",
    "hist_transport_hour_p90_delay",

    # D-1 forecast weather
    "temperature_2m_fcst_d1",
    "relative_humidity_2m_fcst_d1",
    "precipitation_fcst_d1",
    "snowfall_fcst_d1",
    "wind_speed_10m_fcst_d1",
    "wind_gusts_10m_fcst_d1",

    # D-2 forecast weather
    "temperature_2m_fcst_d2",
    "relative_humidity_2m_fcst_d2",
    "precipitation_fcst_d2",

    "snowfall_fcst_d2",
    "wind_speed_10m_fcst_d2",
    "wind_gusts_10m_fcst_d2",

    # forecast revision features
    "temperature_2m_fcst_d1_minus_d2",
    "precipitation_fcst_d1_minus_d2",
    "snowfall_fcst_d1_minus_d2",
    "wind_speed_10m_fcst_d1_minus_d2",

    # special-day features
    "is_special_day",
    "special_day_intensity",
    "is_school_holiday",
]
EXCLUDE_COLS = [
    # direct target / leakage
    "arrival_delay_seconds_raw",
    "arrival_delay_minutes",
    "arrival_delay_was_capped",
    "actual_arrival_ts",
    "actual_departure_ts",

    # post-event status
    "arr_status",
    "dep_status",

    # already used for filtering / diagnostics
    "unplanned",
    "failed",
    "transit",

    # raw / redundant identifiers
    "operator_abrv",
    "operator_name",
    "product_id",
    "transport",
    "line_id",
    "circuit_id",
    "stop_name",
    "trip_id",

    # leaky or overly specific categorical fields
    "timestamp_pattern",
    "diagnostic_timestamp_pattern",
    "school_holiday_name",
    "special_day_name",

    # timestamps not directly used as ML features
    "operating_day",
    "scheduled_arrival_ts",
    "scheduled_departure_ts",
    "scheduled_event_ts",
    "first_scheduled_event_ts",
    "weather_hour",
    "observed_weather_hour",

    # duplicate / older version
    "relative_stop_position",

    # not available in real time / observed weather
    "temperature_2m_obs",
    "relative_humidity_2m_obs",
    "precipitation_obs",
    "rain_obs",
    "snowfall_obs",
    "wind_speed_10m_obs",
    "wind_gusts_10m_obs",
    "weather_code_obs",

    # excluded: many NaN values; precipitation + snowfall already capture wet-weather signal
    "rain_fcst_d1",
    "rain_fcst_d2",
    "rain_fcst_d1_minus_d2",
]

# %%
a.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col(LABEL_COL).isNull().cast("int")).alias("label_nulls"),
    F.min(LABEL_COL).alias("label_min"),
    F.expr(f"percentile_approx({LABEL_COL}, 0.5)").alias("label_median"),
    F.expr(f"percentile_approx({LABEL_COL}, 0.9)").alias("label_p90"),
    F.max(LABEL_COL).alias("label_max"),
).show(truncate=False)

# %%
model_input_cols = NUMERIC_COLS + CATEGORICAL_COLS

bad_model_cols = [c for c in EXCLUDE_COLS if c in model_input_cols]

print("Excluded columns used as model inputs:", bad_model_cols)

# %%
bad_obs_cols = [
    c for c in model_input_cols
    if c.endswith("_obs") and not c.endswith("_n_obs")
]

print("Suspicious observed columns:", bad_obs_cols)

# %%
full_final_features_df

# %%
FINAL_DATA_PATH = f"{base}/final_full_data.parquet"
full_final_features_df.write.mode("overwrite").parquet(FINAL_DATA_PATH)

# %% [markdown]
# ### Model

# %%
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor, DecisionTreeRegressor
from model_pipeline import build_pipeline, evaluate
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# %%
LABEL_COL = "arrival_delay_seconds"

MODEL_NUMERIC_COLS = NUMERIC_COLS
MODEL_CATEGORICAL_COLS = CATEGORICAL_COLS

model_input_cols = NUMERIC_COLS + CATEGORICAL_COLS

bad_model_cols = [c for c in EXCLUDE_COLS if c in model_input_cols]
print("Excluded columns used as model inputs:", bad_model_cols)

# %%
n_full = full_final_features_df.count()

null_summary_df = (
    full_final_features_df
    .select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in model_input_cols
    ])
)

# Convert wide one-row table into long format
stack_expr = "stack({0}, {1}) as (feature, n_nulls)".format(
    len(model_input_cols),
    ", ".join([f"'{c}', `{c}`" for c in model_input_cols])
)

null_summary_long_df = (
    null_summary_df
    .selectExpr(stack_expr)
    .withColumn(
        "null_pct",
        F.round(100 * F.col("n_nulls") / F.lit(n_full), 3)
    )
    .orderBy(F.desc("null_pct"), F.desc("n_nulls"), "feature")
)

null_summary_long_df.show(100, truncate=False)

# %% [markdown]
# FINAL_MODEL_PATH = f"{base}/prod_gbt_model_full_data"
#
# gbt = GBTRegressor(
#     featuresCol="features",
#     labelCol=LABEL_COL,
#     predictionCol="pred_delay",
#     maxDepth=8,
#     maxBins=256,
#     maxIter=100,
#     stepSize=0.05,
#     seed=42
# )
#
# pipe = build_pipeline(gbt,MODEL_NUMERIC_COLS,MODEL_CATEGORICAL_COLS)
#
# print("Training final GBT on full available labeled data...")
# #final_model = pipe.fit(full_final_features_df)
# print("Done.")
#
# #final_model.write().overwrite().save(FINAL_MODEL_PATH)
#
# print("Saved final model:")
# print(FINAL_MODEL_PATH)

# %%
FINAL_MODEL_PATH = f"{base}/trained_models/prod_gbt_model_full_data"

# %%
from pyspark.ml import PipelineModel

loaded_final_model = PipelineModel.load(FINAL_MODEL_PATH)
print("Final model loaded successfully.")

# %%
final_model = loaded_final_model

# %%
sample_pred_df = (
    loaded_final_model
    .transform(full_final_features_df.limit(10))
    .withColumn("error", F.col(LABEL_COL) - F.col("pred_delay"))
    .select(
        "operating_day",
        "trip_id",
        "bpuic",
        "stop_name",
        LABEL_COL,
        F.round("pred_delay", 2).alias("pred_delay"),
        F.round("error", 2).alias("error")
    )
)

sample_pred_df.show(10, truncate=False)

# %%
for i, stage in enumerate(final_model.stages):
    print(i, stage.__class__.__name__)
    print(stage)

# %%
model_to_check = final_model   # or loaded_final_model

for i, stage in enumerate(model_to_check.stages):
    class_name = stage.__class__.__name__

    if "StringIndexer" in class_name:
        print("\nStage:", i)
        print("Class:", class_name)
        print("Input col:", stage.getInputCol())
        print("Output col:", stage.getOutputCol())

        if stage.hasParam("handleInvalid"):
            print("handleInvalid:", stage.getOrDefault(stage.getParam("handleInvalid")))
        else:
            print("No handleInvalid parameter")

# %%
model_to_check = final_model   # or loaded_final_model

for i, stage in enumerate(model_to_check.stages):
    class_name = stage.__class__.__name__

    if "StringIndexer" in class_name:
        print("\nStage:", i)
        print("Class:", class_name)

        if hasattr(stage, "getInputCol"):
            print("Input col:", stage.getInputCol())

        if hasattr(stage, "getOutputCol"):
            print("Output col:", stage.getOutputCol())

        if stage.hasParam("handleInvalid"):
            print("handleInvalid:", stage.getOrDefault(stage.getParam("handleInvalid")))
        else:
            print("No handleInvalid parameter")

# %%
for i, stage in enumerate(model_to_check.stages):
    class_name = stage.__class__.__name__

    if "VectorAssembler" in class_name:
        print("\nStage:", i)
        print("Class:", class_name)

        if stage.hasParam("handleInvalid"):
            print("handleInvalid:", stage.getOrDefault(stage.getParam("handleInvalid")))

# %%
EVAL_CALIB_PRED_PATH = (
    f"{base}/calibration_model_data/evaluation_outputs/"
    "eval_calib_predictions_full.parquet"
)

calib_pred_df = spark.read.parquet(EVAL_CALIB_PRED_PATH)

print(calib_pred_df.count())

# %%
EVAL_TEST_PRED_PATH = (
    f"{base}/calibration_model_data/evaluation_outputs/"
    "eval_test_predictions_full.parquet"
)

EVAL_TEST_QUANTILE_PATH = (
    f"{base}/calibration_model_data/evaluation_outputs/"
    "eval_test_quantiles_backoff.parquet"
)

EVAL_COVERAGE_PATH = (
    f"{base}/calibration_model_data/evaluation_outputs/"
    "eval_coverage_backoff.parquet"
)

# %%
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

LABEL_COL = "arrival_delay_seconds"
PRED_COL = "pred_delay"

rmse_eval = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol=PRED_COL,
    metricName="rmse"
)

mae_eval = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol=PRED_COL,
    metricName="mae"
)

r2_eval = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol=PRED_COL,
    metricName="r2"
)

calib_metrics = {
    "rmse": rmse_eval.evaluate(calib_pred_df),
    "mae": mae_eval.evaluate(calib_pred_df),
    "r2": r2_eval.evaluate(calib_pred_df),
}

print(
    f"{'calibration':<20}  "
    f"RMSE={calib_metrics['rmse']:.2f}  "
    f"MAE={calib_metrics['mae']:.2f}  "
    f"R²={calib_metrics['r2']:.4f}"
)

# %%
from pyspark.sql import functions as F

LABEL_COL = "arrival_delay_seconds"
PRED_COL = "pred_delay"

calib_error_df = (
    calib_pred_df
    .withColumn("residual", F.col(LABEL_COL) - F.col(PRED_COL))
    .withColumn("abs_error", F.abs(F.col("residual")))
)

calib_error_df.select(
    F.count("*").alias("n_rows"),

    F.round(F.avg("residual"), 2).alias("mean_residual"),
    F.round(F.expr("percentile_approx(residual, 0.5)"), 2).alias("median_residual"),
    F.round(F.expr("percentile_approx(residual, 0.8)"), 2).alias("residual_q80"),
    F.round(F.expr("percentile_approx(residual, 0.9)"), 2).alias("residual_q90"),
    F.round(F.expr("percentile_approx(residual, 0.95)"), 2).alias("residual_q95"),
    F.round(F.expr("percentile_approx(residual, 0.99)"), 2).alias("residual_q99"),

    F.round(F.expr("percentile_approx(abs_error, 0.5)"), 2).alias("abs_error_p50"),
    F.round(F.expr("percentile_approx(abs_error, 0.8)"), 2).alias("abs_error_p80"),
    F.round(F.expr("percentile_approx(abs_error, 0.9)"), 2).alias("abs_error_p90"),
    F.round(F.expr("percentile_approx(abs_error, 0.95)"), 2).alias("abs_error_p95"),
    F.round(F.expr("percentile_approx(abs_error, 0.99)"), 2).alias("abs_error_p99"),
).show(truncate=False)

# %%
residual_band_df = (
    calib_error_df
    .withColumn(
        "residual_band",
        F.when(F.col("residual") < -300, F.lit("< -300 sec"))
         .when(F.col("residual") < -180, F.lit("-300 to -180 sec"))
         .when(F.col("residual") < -60, F.lit("-180 to -60 sec"))
         .when(F.col("residual") < 60, F.lit("-60 to 60 sec"))
         .when(F.col("residual") < 180, F.lit("60 to 180 sec"))
         .when(F.col("residual") < 300, F.lit("180 to 300 sec"))
         .when(F.col("residual") < 600, F.lit("300 to 600 sec"))
         .otherwise(F.lit("> 600 sec"))
    )
)

n_calib = calib_error_df.count()

residual_band_df.groupBy("residual_band") \
    .agg(F.count("*").alias("n_rows")) \
    .withColumn("pct", F.round(100 * F.col("n_rows") / F.lit(n_calib), 2)) \
    .orderBy("residual_band") \
    .show(truncate=False)

# %% [markdown]
# From our assignment 2:
#
# We added feature groups one at a time, training a GBT regressor at each step,
# to measure how much each group improves the model:
#
# | Step             | RMSE (s) | MAE (s) | R²    | Improvement |
# |------------------|----------|---------|-------|-------------|
# | Calendar only    | 178.36   | 99.47   | 0.212 | Baseline    |
# | + History        | 176.95   | 98.39   | 0.224 | −1.41 s     |
# | + Weather        | 174.87   | 97.94   | 0.242 | −2.08 s     |
# | + Special days   | 174.22   | 97.64   | 0.248 | −0.65 s     |
#
# Adding all engineered features improves RMSE by 4.14 seconds (2.3 %) and R²
# from 0.21 to 0.25 (a 17 % relative gain). Weather has the largest single
# effect; historical delay statistics and special-day flags are smaller but
# positive. All groups are kept in the final model.

# %% [markdown]
# We compared four MLlib regressors on the full Step 12 feature set:
#
# | Algorithm | RMSE (s) | MAE (s) | R²    |
# |-----------|----------|---------|-------|
# | GBT       | 174.22   | 97.64   | 0.248 |
# | Random Forest | 187.27 | 103.51 | 0.131 |
# | Decision Tree | 188.45 | 104.32 | 0.120 |
# | Linear Regression | 190.69 | 106.52 | 0.099 |
#
# Gradient-Boosted Trees outperform the alternatives by a substantial margin
# (13 s RMSE over Random Forest, 16 s over Linear Regression). The ranking
# follows the expected pattern for tabular data: sequential boosting > bagging >
# single tree > linear. We select GBT as the final model and use it for all evaluations.

# %% [markdown]
# **Confidence intervals.** Using a normal-approximation 95 % CI on the
# ~8.6M test predictions, we obtain:
#
# - RMSE = **174.22 s**  (95 % CI 173.64 – 174.79)
# - MAE  = **97.64 s**   (95 % CI 97.54 – 97.73)
#
# The very tight CIs (RMSE ±0.6 s, MAE ±0.1 s) reflect the large test sample.
# They confirm that the algorithm-comparison gaps (GBT vs. RF: 13 s; GBT vs.
# Linear: 16 s) and the feature-group deltas (history −1.4 s, weather
# −2.1 s, special days −0.7 s) are all statistically significant.
#

# %% [markdown]
# **By line:** The per-line residual pattern matches transit-mode hierarchy:
# the metro m1 is the easiest to predict (MAE 54 s), regional buses (701, 702)
# sit in the middle (~78 s), and the busy TL urban tram/bus lines (1, 2, 6, 8,
# 9, 18, 21, 25) are hardest (MAE 110-150 s). Dedicated infrastructure is easy;
# shared-road urban services under heavy traffic load are hard. Real-time
# inputs would mostly help the latter group.

# %%
spark.stop()

# %%
