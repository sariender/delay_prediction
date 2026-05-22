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
# # Useful Visualizations

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
            .appName(username + 'final')
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

# %% [markdown]
# - Plot 1: The point model uses stable operational signals.
# - Plot 2: Residuals provide data-driven safety margins.
# - Plot 3: Those calibrated quantiles are validated on held-out test data.
# - Plot 4: How context-specific are the February quantiles used by the router?

# %%
BASE = f"{hadoopFS}/user/groups/com-490/H1/final/v1"

# %%
print(repr(hadoopFS)); print(repr(BASE))

# %%
# !hadoop fs -ls /user/groups/com-490/H1/final/v1

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
LABEL_COL = "arrival_delay_seconds"

MODEL_NUMERIC_COLS = NUMERIC_COLS
MODEL_CATEGORICAL_COLS = CATEGORICAL_COLS

model_input_cols = NUMERIC_COLS + CATEGORICAL_COLS

# %%
from pyspark.sql import functions as F

CONFIG_PATH = (
    f"{base}/calibration_model_data/calibration_artifacts/"
    "calibration_config.parquet"
)

config_df = spark.read.parquet(CONFIG_PATH)
config_df.show(truncate=False)

GROUP_LEVELS = {
    row["level_name"]: row["group_cols_csv"].split(",")
    for row in config_df.collect()
}

ALPHAS = [
    float(x)
    for x in config_df.select("alphas_csv").first()["alphas_csv"].split(",")
]

MIN_GROUP_ROWS = int(config_df.select("min_group_rows").first()["min_group_rows"])

def alpha_to_name(alpha):
    return f"q{int(round(alpha * 100)):02d}"

LEVEL_ORDER_RICH_TO_SIMPLE = [
    "L6_line_specific",
    "L5_ultra_rich",
    "L4_rich",
    "L3_operational",
    "L2_context",
    "L1_core",
    "L0_simple",
]

print(GROUP_LEVELS)
print(ALPHAS)
print(MIN_GROUP_ROWS)

# %%
print("Loaded levels from config:")
for level_name, group_cols in GROUP_LEVELS.items():
    print(level_name, "->", group_cols)

print("ALPHAS:", ALPHAS)
print("MIN_GROUP_ROWS:", MIN_GROUP_ROWS)

# %%
CALIB_TABLES_PATH = (
    f"{base}/calibration_model_data/calibration_artifacts/"
    "calibration_residual_tables"
)

# %%
group_residual_tables = {}

for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
    path = f"{CALIB_TABLES_PATH}/{level_name}.parquet"
    df = spark.read.parquet(path)
    group_residual_tables[level_name] = df

    print(level_name)
    print("rows:", df.count())
    df.printSchema()

# %%
GLOBAL_QUANTILES_PATH = (
    f"{base}/calibration_model_data/calibration_artifacts/"
    "global_residual_quantiles.parquet"
)

# %%
global_quantiles_df = spark.read.parquet(GLOBAL_QUANTILES_PATH)
global_quantiles_df.show(truncate=False)

# %%
global_residual_quantiles = {
    row["quantile"]: float(row["residual_quantile"])
    for row in global_quantiles_df.collect()
}
global_residual_quantiles

# %%
expected_qnames = [alpha_to_name(a) for a in ALPHAS]

missing_global_q = [
    q for q in expected_qnames
    if q not in global_residual_quantiles
]

print("Missing global quantiles:", missing_global_q)


# %%
def residual_backoff_expr(qname):
    expr = None

    for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
        n_col = f"{level_name}_n_rows"
        residual_col = f"{level_name}_residual_{qname}"

        condition = (
            F.col(n_col).isNotNull()
            & (F.col(n_col) >= F.lit(MIN_GROUP_ROWS))
            & F.col(residual_col).isNotNull()
        )

        if expr is None:
            expr = F.when(condition, F.col(residual_col))
        else:
            expr = expr.when(condition, F.col(residual_col))

    return expr.otherwise(F.lit(global_residual_quantiles[qname]))


def chosen_level_expr(qname):
    expr = None

    for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
        n_col = f"{level_name}_n_rows"
        residual_col = f"{level_name}_residual_{qname}"

        condition = (
            F.col(n_col).isNotNull()
            & (F.col(n_col) >= F.lit(MIN_GROUP_ROWS))
            & F.col(residual_col).isNotNull()
        )

        if expr is None:
            expr = F.when(condition, F.lit(level_name))
        else:
            expr = expr.when(condition, F.lit(level_name))

    return expr.otherwise(F.lit("global"))


# %%
def add_uncertainty_buckets(df):
    return (
        df
        .withColumn(
            "pred_delay_bucket",
            F.when(F.col("pred_delay") < 60, F.lit("low_pred_delay"))
             .when(F.col("pred_delay") < 180, F.lit("medium_pred_delay"))
             .when(F.col("pred_delay") < 420, F.lit("high_pred_delay"))
             .otherwise(F.lit("very_high_pred_delay"))
        )
        .withColumn(
            "time_bucket",
            F.when(F.col("is_morning_peak") == 1, F.lit("morning_peak"))
             .when(F.col("is_evening_peak") == 1, F.lit("evening_peak"))
             .when(F.col("is_late_night") == 1, F.lit("late_night"))
             .otherwise(F.lit("offpeak"))
        )
        .withColumn(
            "day_bucket",
            F.when(F.col("is_weekend") == 1, F.lit("weekend"))
             .otherwise(F.lit("weekday"))
        )
        .withColumn(
            "weather_bucket",
            F.when(F.col("snowfall_fcst_d1") > 0, F.lit("snow"))
             .when(F.col("precipitation_fcst_d1") >= 2.0, F.lit("heavy_wet"))
             .when(F.col("precipitation_fcst_d1") > 0.1, F.lit("light_wet"))
             .when(F.col("wind_gusts_10m_fcst_d1") > 50, F.lit("windy"))
             .otherwise(F.lit("dry"))
        )
        .withColumn(
            "forecast_instability_bucket",
            F.when(
                (F.abs(F.col("precipitation_fcst_d1_minus_d2")) > 1.0) |
                (F.abs(F.col("wind_speed_10m_fcst_d1_minus_d2")) > 10.0) |
                (F.abs(F.col("temperature_2m_fcst_d1_minus_d2")) > 3.0),
                F.lit("unstable_forecast")
            ).otherwise(F.lit("stable_forecast"))
        )
        .withColumn(
            "line_risk_bucket",
            F.when(F.col("hist_line_hour_p90_delay") < 180, F.lit("low_line_risk"))
             .when(F.col("hist_line_hour_p90_delay") < 420, F.lit("medium_line_risk"))
             .otherwise(F.lit("high_line_risk"))
        )
        .withColumn(
            "trip_stage_bucket",
            F.when(F.col("is_first_observed_stop") == 1, F.lit("origin"))
             .when(F.col("is_last_observed_stop") == 1, F.lit("terminal"))
             .when(F.col("relative_stop_position_v2") < 0.33, F.lit("early"))
             .when(F.col("relative_stop_position_v2") < 0.66, F.lit("middle"))
             .otherwise(F.lit("late"))
        )
        .withColumn(
            "special_bucket",
            F.when(F.col("special_day_type") == "public_holiday", F.lit("public_holiday"))
             .when(F.col("is_school_holiday") == 1, F.lit("school_holiday"))
             .when(F.col("is_special_day") == 1, F.lit("special_day"))
             .otherwise(F.lit("normal_day"))
        )
    )


# %%
for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
    df = group_residual_tables[level_name]

    expected_cols = [
        f"{level_name}_group",
        f"{level_name}_n_rows",
    ] + [
        f"{level_name}_residual_{alpha_to_name(a)}"
        for a in ALPHAS
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]

    print(level_name, "missing cols:", missing_cols)

# %%
FEB_PRED_PATH = (
    f"{base}/route_demo_outputs/predictions/"
    "feb_predictions_full.parquet"
)

feb_pred_df = spark.read.parquet(FEB_PRED_PATH)

print(feb_pred_df.count())
print("feb_pred cols:", len(feb_pred_df.columns))

# %%
required_for_buckets = [
    "pred_delay",
    "transport_clean",

    # time/day
    "is_morning_peak",
    "is_evening_peak",
    "is_late_night",
    "is_weekend",

    # weather
    "snowfall_fcst_d1",
    "precipitation_fcst_d1",
    "wind_gusts_10m_fcst_d1",

    # forecast instability
    "precipitation_fcst_d1_minus_d2",
    "wind_speed_10m_fcst_d1_minus_d2",
    "temperature_2m_fcst_d1_minus_d2",

    # historical risk
    "hist_line_hour_p90_delay",

    # trip stage
    "relative_stop_position_v2",
    "is_first_observed_stop",
    "is_last_observed_stop",

    # special days
    "special_day_type",
    "is_special_day",
    "is_school_holiday",

    # grouping / joins
    "line_text",
    "bpuic",
]

missing = [c for c in required_for_buckets if c not in feb_pred_df.columns]
print("Missing required columns:", missing)

# %%
FULL_DATA_PATH = f"{base}/features_and_training_data/final_full_data.parquet"

full_base_df = spark.read.parquet(FULL_DATA_PATH)
print(full_base_df.count())

# %%
stop_hub_full_df = (
    full_base_df
    .groupBy("bpuic")
    .agg(
        F.countDistinct("line_text").alias("n_lines_at_stop"),
        F.countDistinct("transport_clean").alias("n_transport_types_at_stop"),
        F.count("*").alias("n_events_at_stop")
    )
    .withColumn(
        "stop_hub_bucket",
        F.when(F.col("n_lines_at_stop") >= 8, F.lit("major_hub"))
         .when(F.col("n_lines_at_stop") >= 3, F.lit("medium_hub"))
         .otherwise(F.lit("local_stop"))
    )
    .select("bpuic", "stop_hub_bucket")
)

# %%
feb_pred_grouped_df = (
    feb_pred_df
    .join(stop_hub_full_df, on="bpuic", how="left")
    .fillna({"stop_hub_bucket": "unknown_hub"})
)

feb_pred_grouped_df = add_uncertainty_buckets(feb_pred_grouped_df)

# %%
for level_name, group_cols in GROUP_LEVELS.items():
    feb_pred_grouped_df = feb_pred_grouped_df.withColumn(
        f"{level_name}_group",
        F.concat_ws("__", *[F.col(c).cast("string") for c in group_cols])
    )

# %%
group_key_cols = [f"{level_name}_group" for level_name in GROUP_LEVELS.keys()]
feb_pred_grouped_df.select(*group_key_cols).show(5, truncate=False, vertical=True)

# %%
feb_with_calib_df = feb_pred_grouped_df
for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
    group_col = f"{level_name}_group"

    feb_with_calib_df = feb_with_calib_df.join(
        group_residual_tables[level_name],
        on=group_col,
        how="left"
    )

# %%
n_before = feb_pred_grouped_df.count()
n_after = feb_with_calib_df.count()

print("before:", n_before)
print("after: ", n_after)

# %%
feb_quantile_df = feb_with_calib_df

for alpha in ALPHAS:
    qname = alpha_to_name(alpha)

    feb_quantile_df = feb_quantile_df.withColumn(
        qname,
        F.col("pred_delay") + residual_backoff_expr(qname)
    )

feb_quantile_df = feb_quantile_df.withColumn(
    "chosen_calibration_level_q90",
    chosen_level_expr("q90")
)

# %%
feb_quantile_df.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col("pred_delay").isNull().cast("int")).alias("pred_nulls"),
    F.sum(F.col("q90").isNull().cast("int")).alias("q90_nulls"),
    F.expr("percentile_approx(pred_delay, 0.5)").alias("median_pred"),
    F.expr("percentile_approx(q50, 0.5)").alias("median_q50"),
    F.expr("percentile_approx(q90, 0.5)").alias("median_q90"),
    F.expr("percentile_approx(q95, 0.5)").alias("median_q95"),
    F.expr("percentile_approx(q99, 0.5)").alias("median_q99")
).show(truncate=False)

# %%
n_feb = feb_quantile_df.count()

feb_quantile_df.groupBy("chosen_calibration_level_q90") \
    .agg(F.count("*").alias("n_rows")) \
    .withColumn(
        "pct",
        F.round(100 * F.col("n_rows") / F.lit(n_feb), 3)
    ) \
    .orderBy(F.desc("n_rows")) \
    .show(truncate=False)

# %%
for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
    n_col = f"{level_name}_n_rows"
    residual_col = f"{level_name}_residual_q90"

    print("\n", level_name)

    feb_with_calib_df.select(
        F.count("*").alias("n_rows"),
        F.sum(F.col(n_col).isNotNull().cast("int")).alias("matched_group_rows"),
        F.round(
            100 * F.avg(F.col(n_col).isNotNull().cast("int")),
            3
        ).alias("matched_group_pct"),
        F.sum(
            (
                F.col(n_col).isNotNull()
                & (F.col(n_col) >= F.lit(MIN_GROUP_ROWS))
                & F.col(residual_col).isNotNull()
            ).cast("int")
        ).alias("usable_group_rows"),
        F.round(
            100 * F.avg(
                (
                    F.col(n_col).isNotNull()
                    & (F.col(n_col) >= F.lit(MIN_GROUP_ROWS))
                    & F.col(residual_col).isNotNull()
                ).cast("int")
            ),
            3
        ).alias("usable_group_pct")
    ).show(truncate=False)

# %%
for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
    print(level_name)
    print(GROUP_LEVELS[level_name])
    print()

# %%
for level_name in LEVEL_ORDER_RICH_TO_SIMPLE:
    group_col = f"{level_name}_group"

    feb_groups_df = feb_pred_grouped_df.select(group_col).distinct()
    calib_groups_df = group_residual_tables[level_name].select(group_col).distinct()

    n_feb_groups = feb_groups_df.count()
    n_calib_groups = calib_groups_df.count()
    n_overlap = feb_groups_df.join(calib_groups_df, on=group_col, how="inner").count()

    print(
        level_name,
        "| feb groups:", n_feb_groups,
        "| calib groups:", n_calib_groups,
        "| overlap:", n_overlap
    )

# %%
FEB_QUANTILE_PATH = (
    f"{base}/route_demo_outputs/quantiles/"
    "feb_quantiles_backoff.parquet"
)

feb_quantile_df = spark.read.parquet(FEB_QUANTILE_PATH)

# %%
QUANTILE_COLS = [
    "q01", "q05", "q10", "q20", "q35", "q50",
    "q65", "q80", "q90", "q95", "q99"
]

#ROUTE_LOOKUP_PATH = f"{base}/route_demo_lookup_february.parquet"

demo_feb_lookup_df = feb_quantile_df.select(
    # timetable / lookup keys
    "operating_day",
    "trip_id",
    "bpuic",
    "stop_name",
    "scheduled_arrival_ts",

    # route info
    "line_text",
    "transport_clean",

    # point prediction
    "pred_delay",

    # uncertainty / calibration
    "chosen_calibration_level_q90",
    *QUANTILE_COLS,

    # diagnostics
    "line_seen_in_training",
    "line_id_source"
)


# %%
demo_feb_lookup_df.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col("pred_delay").isNull().cast("int")).alias("pred_nulls"),
    F.sum(F.col("q50").isNull().cast("int")).alias("q50_nulls"),
    F.sum(F.col("q90").isNull().cast("int")).alias("q90_nulls"),
    F.expr("percentile_approx(pred_delay, 0.5)").alias("median_pred"),
    F.expr("percentile_approx(q50, 0.5)").alias("median_q50"),
    F.expr("percentile_approx(q90, 0.5)").alias("median_q90"),
    F.expr("percentile_approx(q95, 0.5)").alias("median_q95"),
    F.expr("percentile_approx(q99, 0.5)").alias("median_q99")
).show(truncate=False)

# %%
#demo_feb_lookup_df
ROUTE_LOOKUP_PATH = (
    f"{base}/route_demo_outputs/lookup/"
    "route_demo_lookup_february.parquet"
)

# %%
key_cols = [
    "operating_day",
    "trip_id",
    "bpuic",
    "scheduled_arrival_ts",
]

dup_check_df = (
    demo_feb_lookup_df
    .groupBy(*key_cols)
    .agg(F.count("*").alias("n_rows"))
    .filter(F.col("n_rows") > 1)
)

print("duplicate keys:", dup_check_df.count())
dup_check_df.show(20, truncate=False)

# %%
n_lookup = demo_feb_lookup_df.count()

demo_feb_lookup_df.groupBy("chosen_calibration_level_q90") \
    .agg(F.count("*").alias("n_rows")) \
    .withColumn(
        "pct",
        F.round(100 * F.col("n_rows") / F.lit(n_lookup), 3)
    ) \
    .orderBy(F.desc("n_rows")) \
    .show(truncate=False)

# %% [markdown]
# ## 1- Robustness Feature Check w final model
# - Does the final model rely on broadly similar signals after retraining?
# - As a robustness check, I compare feature importance between the evaluation-stage model, which was trained only on the training split, and the final production model, which was retrained on all available historical labeled data before February prediction. I do not interpret feature importance causally. I use it only to check whether the main predictive signals remain broadly stable after retraining. Similar importance patterns would suggest that the February production model is relying on the same operational information validated in the evaluation pipeline.
# - Stable feature importance does not prove that predictions are well calibrated.
# 1. Feature-family importance stability: evaluation vs production model
# 2. Calibration residual distribution
# 3. Empirical coverage plot
# 4. February backoff-level distribution
# 5. Transfer probability curve
#

# %% [markdown]
# Evaluation model importance:
#     interpretable and linked to honest validation.
#
# Final production model importance:
#     shows what the February deployment model uses.
#
# If rankings are broadly similar:
#     the model's main signals are stable after retraining on all historical data.
#
# Evaluation-stage model:
# trained only on train split
# used for calibration/test validation
#
# Production model:
# trained on all available labeled historical data
# used for February lookup generation
# After retraining on more historical data, does the model still rely on broadly similar predictive signals?

# %%
from pyspark.ml import PipelineModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Evaluation-stage model: trained only on train split
EVAL_MODEL_PATH = (
    f"{BASE}/calibration_model_data/models/"
    "eval_calib_gbt_model"
)

# Production model: trained on full historical labeled data
PROD_MODEL_PATH = (
    f"{BASE}/trained_models/"
    "prod_gbt_model_full_data"
)

# Data used only to recover feature-vector metadata
TRAIN_FEATURES_PATH = (
    f"{BASE}/calibration_model_data/feature_splits/"
    "train_final_features_df.parquet"
)

FULL_DATA_PATH = (
    f"{BASE}/features_and_training_data/"
    "final_full_data.parquet"
)

eval_model = PipelineModel.load(EVAL_MODEL_PATH)
prod_model = PipelineModel.load(PROD_MODEL_PATH)

train_features_df = spark.read.parquet(TRAIN_FEATURES_PATH)
full_data_df = spark.read.parquet(FULL_DATA_PATH)

print("Loaded evaluation model and production model.")

# %%
print("Evaluation model stages:")
for i, stage in enumerate(eval_model.stages):
    print(i, type(stage).__name__, stage.uid)

print("\nProduction model stages:")
for i, stage in enumerate(prod_model.stages):
    print(i, type(stage).__name__, stage.uid)


# %%
def extract_feature_importance(pipeline_model, input_df, model_name):
    """
    Extract feature importances and feature names from a Spark PipelineModel.
    
    Parameters
    ----------
    pipeline_model : pyspark.ml.PipelineModel
        Trained Spark ML pipeline containing a tree-based regression model.
    input_df : pyspark.sql.DataFrame
        Dataframe compatible with the pipeline, used only to recover
        assembled-feature metadata.
    model_name : str
        Label used in the returned pandas table.
    
    Returns
    -------
    pandas.DataFrame
        One row per encoded feature with raw importance, percentage importance,
        and rank.
    """
    
    # Find the final tree-based model carrying featureImportances
    fitted_model = next(
        stage for stage in reversed(pipeline_model.stages)
        if hasattr(stage, "featureImportances")
    )
    
    features_col = fitted_model.getFeaturesCol()
    importances = fitted_model.featureImportances.toArray()
    
    # Transform one row so that Spark exposes the feature vector metadata
    transformed_sample = pipeline_model.transform(input_df.limit(1))
    metadata = transformed_sample.schema[features_col].metadata
    
    attrs = metadata.get("ml_attr", {}).get("attrs", {})
    
    feature_attrs = []
    for attr_type in ["numeric", "binary", "nominal"]:
        feature_attrs.extend(attrs.get(attr_type, []))
    
    feature_attrs = sorted(feature_attrs, key=lambda x: x["idx"])
    feature_names = [attr["name"] for attr in feature_attrs]
    
    print(f"{model_name}: {type(fitted_model).__name__}")
    print(f"{model_name}: {len(importances)} importance values")
    print(f"{model_name}: {len(feature_names)} metadata feature names")
    
    if len(feature_names) != len(importances):
        raise ValueError(
            f"{model_name}: feature-name mismatch. "
            f"Found {len(feature_names)} names but {len(importances)} importance values."
        )
    
    importance_pdf = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    
    importance_pdf["importance_pct"] = 100 * importance_pdf["importance"]
    importance_pdf["model"] = model_name
    
    importance_pdf = (
        importance_pdf
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    
    importance_pdf["rank"] = importance_pdf.index + 1
    
    return importance_pdf


# %%
eval_imp_pdf = extract_feature_importance(
    eval_model,
    train_features_df,
    model_name="Evaluation model"
)

prod_imp_pdf = extract_feature_importance(
    prod_model,
    full_data_df,
    model_name="Production model"
)

display(eval_imp_pdf.head(20))
display(prod_imp_pdf.head(20))

# %%
eval_features = set(eval_imp_pdf["feature"])
prod_features = set(prod_imp_pdf["feature"])

only_eval = sorted(eval_features - prod_features)
only_prod = sorted(prod_features - eval_features)

print("Features only in evaluation model:", only_eval)
print("Features only in production model:", only_prod)
print("Same feature space:", eval_features == prod_features)

# %%
importance_comparison_pdf = (
    eval_imp_pdf[["feature", "importance_pct", "rank"]]
    .rename(columns={
        "importance_pct": "eval_importance_pct",
        "rank": "eval_rank"
    })
    .merge(
        prod_imp_pdf[["feature", "importance_pct", "rank"]]
        .rename(columns={
            "importance_pct": "prod_importance_pct",
            "rank": "prod_rank"
        }),
        on="feature",
        how="outer"
    )
    .fillna({
        "eval_importance_pct": 0.0,
        "prod_importance_pct": 0.0
    })
)

importance_comparison_pdf["max_importance_pct"] = importance_comparison_pdf[
    ["eval_importance_pct", "prod_importance_pct"]
].max(axis=1)

importance_comparison_pdf["importance_difference_pct"] = (
    importance_comparison_pdf["prod_importance_pct"]
    - importance_comparison_pdf["eval_importance_pct"]
)

importance_comparison_pdf = importance_comparison_pdf.sort_values(
    "max_importance_pct",
    ascending=False
)

display(importance_comparison_pdf.head(25))

# %%
TOP_N = 15

top_eval_features = set(eval_imp_pdf.head(TOP_N)["feature"])
top_prod_features = set(prod_imp_pdf.head(TOP_N)["feature"])
top_union_features = top_eval_features | top_prod_features

top_comparison_pdf = (
    importance_comparison_pdf[
        importance_comparison_pdf["feature"].isin(top_union_features)
    ]
    .sort_values("max_importance_pct", ascending=True)
)

y_pos = np.arange(len(top_comparison_pdf))
bar_height = 0.38

plt.figure(figsize=(12, 8))

plt.barh(
    y_pos - bar_height / 2,
    top_comparison_pdf["eval_importance_pct"],
    height=bar_height,
    label="Evaluation model"
)

plt.barh(
    y_pos + bar_height / 2,
    top_comparison_pdf["prod_importance_pct"],
    height=bar_height,
    label="Production model"
)

plt.yticks(y_pos, top_comparison_pdf["feature"])
plt.xlabel("Feature importance (%)")
plt.title("Feature Importance Stability: Evaluation vs Production GBT Model")
plt.legend()
plt.tight_layout()
plt.show()
#If the same features remain important in both models, the main predictive signals
#are reasonably stable after retraining on all historical data.

# %%
TOP_K = 10

eval_top_k = set(eval_imp_pdf.head(TOP_K)["feature"])
prod_top_k = set(prod_imp_pdf.head(TOP_K)["feature"])

top_k_overlap = eval_top_k & prod_top_k

print(f"Top-{TOP_K} overlap:", len(top_k_overlap), "/", TOP_K)
print("Shared top features:")
for feature in sorted(top_k_overlap):
    print("-", feature)

# %%
from scipy.stats import spearmanr

common_comparison_pdf = importance_comparison_pdf.dropna(
    subset=["eval_rank", "prod_rank"]
).copy()

spearman_corr, spearman_pvalue = spearmanr(
    common_comparison_pdf["eval_rank"],
    common_comparison_pdf["prod_rank"]
)

print(f"Spearman rank correlation: {spearman_corr:.3f}")
print(f"p-value: {spearman_pvalue:.4g}")

# %%
pearson_corr = common_comparison_pdf[
    ["eval_importance_pct", "prod_importance_pct"]
].corr().iloc[0, 1]

print(f"Pearson correlation of importance percentages: {pearson_corr:.3f}")

# %%
plt.figure(figsize=(8, 7))

plt.scatter(
    importance_comparison_pdf["eval_importance_pct"],
    importance_comparison_pdf["prod_importance_pct"]
)

max_value = importance_comparison_pdf[
    ["eval_importance_pct", "prod_importance_pct"]
].to_numpy().max()

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Importance in evaluation model (%)")
plt.ylabel("Importance in production model (%)")
plt.title("Stability of Feature Importance After Final Retraining")
plt.tight_layout()
plt.show()
#Features close to the diagonal have similar importance in both models.
#Features far from the diagonal became more or less important after retraining.

# %%
def feature_family(feature_name):
    name = feature_name.lower()
    
    if "hist_operator_hour" in name:
        return "Historical operator-hour"
    if "hist_line_hour" in name:
        return "Historical line-hour"
    if "hist_stop_hour" in name:
        return "Historical stop-hour"
    if "hist_transport_hour" in name:
        return "Historical transport-hour"
    
    if "weather" in name or "temperature" in name or "precipitation" in name or "snowfall" in name or "wind" in name:
        return "Forecast weather"
    
    if "special_day" in name or "school_holiday" in name or "public_holiday" in name:
        return "Special days"
    
    if "relative_stop" in name or "stop_sequence" in name or "first_observed" in name or "last_observed" in name:
        return "Trip position"
    
    if "line_id" in name or "line_text" in name:
        return "Line identity"
    
    if "operator_id" in name:
        return "Operator identity"
    
    if "transport_clean" in name:
        return "Transport mode"
    
    if "bpuic" in name or "stop" in name:
        return "Stop identity/context"
    
    if "hour" in name or "day_of_week" in name or "weekend" in name or "month" in name:
        return "Scheduled time/calendar"
    
    return "Other"


# %%
eval_family_pdf = eval_imp_pdf.copy()
eval_family_pdf["feature_family"] = eval_family_pdf["feature"].apply(feature_family)

prod_family_pdf = prod_imp_pdf.copy()
prod_family_pdf["feature_family"] = prod_family_pdf["feature"].apply(feature_family)

eval_family_summary_pdf = (
    eval_family_pdf
    .groupby("feature_family", as_index=False)["importance_pct"]
    .sum()
    .rename(columns={"importance_pct": "eval_importance_pct"})
)

prod_family_summary_pdf = (
    prod_family_pdf
    .groupby("feature_family", as_index=False)["importance_pct"]
    .sum()
    .rename(columns={"importance_pct": "prod_importance_pct"})
)

family_comparison_pdf = (
    eval_family_summary_pdf
    .merge(prod_family_summary_pdf, on="feature_family", how="outer")
    .fillna(0.0)
)

family_comparison_pdf["max_importance_pct"] = family_comparison_pdf[
    ["eval_importance_pct", "prod_importance_pct"]
].max(axis=1)

family_comparison_pdf = family_comparison_pdf.sort_values(
    "max_importance_pct",
    ascending=True
)

display(family_comparison_pdf)

# %%
y_pos = np.arange(len(family_comparison_pdf))
bar_height = 0.38

plt.figure(figsize=(11, 7))

plt.barh(
    y_pos - bar_height / 2,
    family_comparison_pdf["eval_importance_pct"],
    height=bar_height,
    label="Evaluation model"
)

plt.barh(
    y_pos + bar_height / 2,
    family_comparison_pdf["prod_importance_pct"],
    height=bar_height,
    label="Production model"
)

plt.yticks(y_pos, family_comparison_pdf["feature_family"])
plt.xlabel("Total feature importance (%)")
plt.title("Feature-Family Importance Stability After Final Retraining")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 2 — Calibration Residual Distribution

# %% [raw]
# The point model makes prediction errors.
# Positive residuals mean the actual delay was larger than predicted.
# The upper residual quantiles become the safety margins used in calibration.
# residual = actual_delay - pred_delay
# Full-data Spark histogram for Plot 2

# %%
#Load calibration prediction and global quantile artifacts
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/user/groups/com-490/H1/final/v1"

EVAL_CALIB_PRED_PATH = (
    f"{BASE}/calibration_model_data/evaluation_outputs/"
    "eval_calib_predictions_full.parquet"
)

GLOBAL_QUANTILES_PATH = (
    f"{BASE}/calibration_model_data/calibration_artifacts/"
    "global_residual_quantiles.parquet"
)

calib_pred_df = spark.read.parquet(EVAL_CALIB_PRED_PATH)
global_quantiles_df = spark.read.parquet(GLOBAL_QUANTILES_PATH)

print("Calibration prediction rows:", calib_pred_df.count())
global_quantiles_df.show(truncate=False)

# %%
#Create residual column
LABEL_COL = "arrival_delay_seconds"
PRED_COL = "pred_delay"

calib_error_df = (
    calib_pred_df
    .withColumn(
        "residual",
        F.col(LABEL_COL) - F.col(PRED_COL)
    )
    .withColumn(
        "absolute_error",
        F.abs(F.col("residual"))
    )
)

calib_error_df.select(
    LABEL_COL,
    PRED_COL,
    "residual",
    "absolute_error"
).show(5, truncate=False)
#residual > 0  → actual delay was worse than predicted
#residual < 0  → actual delay was smaller than predicted

# %%
print(global_quantiles_df.columns)
global_quantiles_df.printSchema()
global_quantiles_df.show(1, truncate=False, vertical=True)

# %%
from pyspark.sql import functions as F

QUANTILE_ORDER = [
    "q01", "q05", "q10", "q20", "q35",
    "q50", "q65", "q80", "q90", "q95", "q99"
]

# Collecting is safe here because this artifact contains only a few quantile rows
all_residual_quantiles = {
    row["quantile"]: float(row["residual_quantile"])
    for row in global_quantiles_df.collect()
}

# Check that all expected quantiles exist
missing_quantiles = [
    q for q in QUANTILE_ORDER
    if q not in all_residual_quantiles
]

print("Missing quantiles:", missing_quantiles)

for q in QUANTILE_ORDER:
    print(f"{q}: {all_residual_quantiles[q]:.2f} sec")

# %%
#Prepare a sample only for visualization

# %%
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# global_quantiles_df is stored in long format:
# quantile | residual_quantile
# q01      | ...
# q05      | ...
# ...

QUANTILE_ORDER = [
    "q01", "q05", "q10", "q20", "q35",
    "q50", "q65", "q80", "q90", "q95", "q99"
]

# Safe to collect: this table has only one row per quantile
all_residual_quantiles = {
    row["quantile"]: float(row["residual_quantile"])
    for row in global_quantiles_df.collect()
}

# Check that the required quantiles exist
missing_quantiles = [
    q for q in QUANTILE_ORDER
    if q not in all_residual_quantiles
]

if missing_quantiles:
    raise ValueError(f"Missing residual quantiles: {missing_quantiles}")

# Quantile reference lines to display in the plot
QUANTILES_TO_SHOW = ["q50", "q80", "q90", "q95", "q99"]

residual_quantiles = {
    q: all_residual_quantiles[q]
    for q in QUANTILES_TO_SHOW
}

# Quantiles used to define a readable visible range
q01 = all_residual_quantiles["q01"]
q99 = all_residual_quantiles["q99"]

# Display approximately the central 98% of residuals, with small padding
PLOT_MIN = np.floor((q01 - 50) / 50) * 50
PLOT_MAX = np.ceil((q99 + 50) / 50) * 50

N_BINS = 70
BIN_WIDTH = (PLOT_MAX - PLOT_MIN) / N_BINS

print("All saved residual quantiles:")
for q in QUANTILE_ORDER:
    print(f"{q}: {all_residual_quantiles[q]:.2f} sec")

print("\nPlot range:", PLOT_MIN, "to", PLOT_MAX)
print("Bin width:", round(BIN_WIDTH, 2))
print("Quantile lines shown on plot:", residual_quantiles)

# %%
histogram_df = (
    calib_error_df
    .filter(
        (F.col("residual") >= F.lit(float(PLOT_MIN))) &
        (F.col("residual") <= F.lit(float(PLOT_MAX)))
    )
    .withColumn(
        "bin_idx",
        F.floor(
            (F.col("residual") - F.lit(float(PLOT_MIN))) /
            F.lit(float(BIN_WIDTH))
        ).cast("int")
    )
    # Put the exact upper endpoint in the final bin
    .withColumn(
        "bin_idx",
        F.when(F.col("bin_idx") >= N_BINS, F.lit(N_BINS - 1))
         .otherwise(F.col("bin_idx"))
    )
    .groupBy("bin_idx")
    .agg(F.count("*").alias("count"))
)

# Create all bins so missing bins appear as zero
all_bins_pdf = pd.DataFrame({"bin_idx": list(range(N_BINS))})

histogram_pdf = (
    histogram_df
    .toPandas()
    .merge(all_bins_pdf, on="bin_idx", how="right")
    .fillna({"count": 0})
    .sort_values("bin_idx")
)

histogram_pdf["bin_left"] = PLOT_MIN + histogram_pdf["bin_idx"] * BIN_WIDTH
histogram_pdf["bin_center"] = histogram_pdf["bin_left"] + BIN_WIDTH / 2

histogram_pdf.head()

# %%
display_range_check_df = (
    calib_error_df
    .select(
        F.count("*").alias("n_total"),
        F.sum((F.col("residual") < F.lit(float(PLOT_MIN))).cast("int")).alias("n_below_plot_range"),
        F.sum((F.col("residual") > F.lit(float(PLOT_MAX))).cast("int")).alias("n_above_plot_range"),
    )
    .withColumn(
        "pct_below_plot_range",
        F.round(100 * F.col("n_below_plot_range") / F.col("n_total"), 3)
    )
    .withColumn(
        "pct_above_plot_range",
        F.round(100 * F.col("n_above_plot_range") / F.col("n_total"), 3)
    )
)

display_range_check_df.show(truncate=False)

# %% [markdown]
# The plot displays the central range of residuals for readability; quantiles were computed on all calibration observations.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------

n_calib = calib_error_df.count()

plot_pdf = histogram_pdf.copy()
plot_pdf["pct_of_all_rows"] = 100 * plot_pdf["count"] / n_calib

q50 = residual_quantiles["q50"]
q80 = residual_quantiles["q80"]
q90 = residual_quantiles["q90"]
q95 = residual_quantiles["q95"]
q99 = residual_quantiles["q99"]

margin_names = ["q80", "q90", "q95", "q99"]
margin_values = [q80, q90, q95, q99]

# For the histogram, do not let q99 visually flatten the main distribution.
# Show the range where the central pattern and q90/q95 are readable.
HIST_X_MIN = PLOT_MIN
HIST_X_MAX = 350

hist_display_pdf = plot_pdf[
    plot_pdf["bin_center"].between(HIST_X_MIN, HIST_X_MAX)
].copy()

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, (ax_hist, ax_margin) = plt.subplots(
    1,
    2,
    figsize=(13, 5.6),
    gridspec_kw={"width_ratios": [2.5, 1]}
)

# ------------------------------------------------------------
# Left panel: residual distribution
# ------------------------------------------------------------

ax_hist.bar(
    hist_display_pdf["bin_center"],
    hist_display_pdf["pct_of_all_rows"],
    width=BIN_WIDTH,
    color="#B9C6D8",
    edgecolor="white",
    linewidth=0.4
)

# Positive residual area: actual delay was larger than predicted
ax_hist.axvspan(
    0,
    HIST_X_MAX,
    color="#F4A261",
    alpha=0.10
)

# No-error reference
ax_hist.axvline(
    0,
    color="#303030",
    linewidth=1.6
)

# Main safety-margin line only
ax_hist.axvline(
    q90,
    color="#C0392B",
    linestyle="--",
    linewidth=2.2
)

# Minimal labels
y_max = ax_hist.get_ylim()[1]

ax_hist.text(
    -8,
    y_max * 0.92,
    "0 sec",
    ha="right",
    va="top",
    fontsize=10,
    color="#303030"
)

ax_hist.text(
    q90 + 8,
    y_max * 0.72,
    f"q90 safety margin\n+{q90:.0f} sec",
    ha="left",
    va="top",
    fontsize=10,
    color="#C0392B",
    fontweight="bold"
)

ax_hist.text(
    HIST_X_MIN + 15,
    y_max * 0.98,
    "Model overpredicted delay",
    ha="left",
    va="top",
    fontsize=9.5,
    color="#5C6F82"
)

ax_hist.text(
    15,
    y_max * 0.98,
    "Actual delay exceeded prediction",
    ha="left",
    va="top",
    fontsize=9.5,
    color="#A65A16"
)

ax_hist.set_title(
    "Calibration residual distribution",
    fontsize=12,
    fontweight="bold"
)

ax_hist.set_xlabel(
    "Residual = actual delay − predicted delay (seconds)",
    fontsize=10
)

ax_hist.set_ylabel(
    "Share of calibration events (%)",
    fontsize=10
)

ax_hist.set_xlim(HIST_X_MIN, HIST_X_MAX)
ax_hist.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))
ax_hist.grid(axis="y", alpha=0.20)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

# ------------------------------------------------------------
# Right panel: upper-tail safety-margin summary
# ------------------------------------------------------------

y_positions = np.arange(len(margin_names))

ax_margin.barh(
    y_positions,
    margin_values,
    color=["#9DB7D5", "#C0392B", "#A93226", "#7B241C"],
    height=0.58
)

ax_margin.set_yticks(y_positions)
ax_margin.set_yticklabels(margin_names)
ax_margin.invert_yaxis()

for y, value in zip(y_positions, margin_values):
    ax_margin.text(
        value + 10,
        y,
        f"+{value:.0f} sec",
        va="center",
        fontsize=10
    )

ax_margin.set_title(
    "Upper-tail safety margins",
    fontsize=12,
    fontweight="bold"
)

ax_margin.set_xlabel(
    "Residual threshold (seconds)",
    fontsize=10
)

ax_margin.set_xlim(0, q99 + 110)
ax_margin.grid(axis="x", alpha=0.20)
ax_margin.spines["top"].set_visible(False)
ax_margin.spines["right"].set_visible(False)
ax_margin.spines["left"].set_visible(False)

# ------------------------------------------------------------
# Overall title and note
# ------------------------------------------------------------

fig.suptitle(
    "From Point-Prediction Errors to Calibrated Delay Buffers",
    fontsize=14,
    fontweight="bold",
    y=1.03
)

fig.text(
    0.5,
    -0.02,
    "Histogram uses all calibration events; the displayed range is restricted for readability.",
    ha="center",
    fontsize=9,
    color="#555555"
)

plt.tight_layout()
plt.show()

# %% [raw]
# The left panel shows the residual distribution from the calibration split. Positive residuals are the risky cases for routing because the actual delay exceeded the model prediction. The q90 residual is about 140 seconds, which becomes a conservative safety margin. The right panel shows that higher confidence levels require increasingly larger buffers, reaching about 666 seconds at q99.

# %%
from pathlib import Path

FIGURE_DIR = Path("../reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = FIGURE_DIR / "calibration_residual_distribution_full_data.png"

fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")

print("Saved:", FIGURE_PATH)

# %% [markdown]
# ## Plot 3 — Empirical Coverage of Calibrated Quantiles

# %% [raw]
# Does a predicted q90 actually contain the realized delay about 90% of the time on unseen test data?
# q50 coverage ≈ 50%
# q80 coverage ≈ 80%
# q90 coverage ≈ 90%
# q95 coverage ≈ 95%
# q99 coverage ≈ 99%

# %%
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = "/user/groups/com-490/H1/final/v1"

EVAL_TEST_QUANTILE_PATH = (
    f"{BASE}/calibration_model_data/evaluation_outputs/"
    "eval_test_quantiles_backoff.parquet"
)

test_quantile_df = spark.read.parquet(EVAL_TEST_QUANTILE_PATH).cache()

print("Test quantile rows:", test_quantile_df.count())
print("Columns:")
print(test_quantile_df.columns)

# %%
LABEL_COL = "arrival_delay_seconds"

QUANTILE_LEVELS = [
    (0.01, "q01"),
    (0.05, "q05"),
    (0.10, "q10"),
    (0.20, "q20"),
    (0.35, "q35"),
    (0.50, "q50"),
    (0.65, "q65"),
    (0.80, "q80"),
    (0.90, "q90"),
    (0.95, "q95"),
    (0.99, "q99"),
]

required_cols = [LABEL_COL] + [qcol for _, qcol in QUANTILE_LEVELS]
missing_cols = [c for c in required_cols if c not in test_quantile_df.columns]

print("Missing columns:", missing_cols)

# %%
null_check_df = test_quantile_df.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col(LABEL_COL).isNull().cast("int")).alias("label_nulls"),
    *[
        F.sum(F.col(qcol).isNull().cast("int")).alias(f"{qcol}_nulls")
        for _, qcol in QUANTILE_LEVELS
    ]
)

null_check_df.show(truncate=False)

# %%
coverage_exprs = []

for target_level, qcol in QUANTILE_LEVELS:
    coverage_exprs.append(
        F.avg(
            (F.col(LABEL_COL) <= F.col(qcol)).cast("double")
        ).alias(f"coverage_{qcol}")
    )

coverage_row = test_quantile_df.select(*coverage_exprs).first().asDict()

coverage_rows = []

for target_level, qcol in QUANTILE_LEVELS:
    empirical_coverage = float(coverage_row[f"coverage_{qcol}"])

    coverage_rows.append({
        "quantile": qcol,
        "target_coverage": target_level,
        "empirical_coverage": empirical_coverage,
        "coverage_error": empirical_coverage - target_level,
        "absolute_error": abs(empirical_coverage - target_level),
    })

coverage_pdf = pd.DataFrame(coverage_rows)

coverage_pdf["target_pct"] = 100 * coverage_pdf["target_coverage"]
coverage_pdf["empirical_pct"] = 100 * coverage_pdf["empirical_coverage"]
coverage_pdf["error_pp"] = 100 * coverage_pdf["coverage_error"]
coverage_pdf["absolute_error_pp"] = 100 * coverage_pdf["absolute_error"]

coverage_pdf

# %%
display_table_pdf = coverage_pdf[
    [
        "quantile",
        "target_pct",
        "empirical_pct",
        "error_pp",
        "absolute_error_pp"
    ]
].copy()

display_table_pdf = display_table_pdf.rename(columns={
    "quantile": "Quantile",
    "target_pct": "Target coverage (%)",
    "empirical_pct": "Empirical coverage (%)",
    "error_pp": "Error (percentage points)",
    "absolute_error_pp": "Absolute error (pp)",
})

display_table_pdf = display_table_pdf.round(2)

display(display_table_pdf)
#Error > 0  → quantile is conservative; it covers more rows than targeted.
#Error < 0  → quantile is under-covering; it covers fewer rows than targeted.

# %%
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(
    coverage_pdf["target_coverage"],
    coverage_pdf["empirical_coverage"],
    marker="o",
    linewidth=2,
    label="Calibrated quantiles on test set"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Perfect calibration"
)

for _, row in coverage_pdf.iterrows():
    ax.annotate(
        row["quantile"],
        (
            row["target_coverage"],
            row["empirical_coverage"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9
    )

ax.set_title(
    "Validation of Calibrated Delay Quantiles on Held-Out Test Data",
    fontsize=13
)
ax.set_xlabel("Target coverage", fontsize=11)
ax.set_ylabel("Empirical coverage on test set", fontsize=11)

ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

# %%
ROUTING_QUANTILES = ["q80", "q90", "q95", "q99"]

upper_coverage_pdf = (
    coverage_pdf[
        coverage_pdf["quantile"].isin(ROUTING_QUANTILES)
    ]
    .copy()
)

fig_upper, ax_upper = plt.subplots(figsize=(8, 5))

x_positions = range(len(upper_coverage_pdf))

ax_upper.bar(
    x_positions,
    upper_coverage_pdf["empirical_pct"],
    label="Empirical coverage"
)

ax_upper.scatter(
    x_positions,
    upper_coverage_pdf["target_pct"],
    marker="_",
    s=350,
    linewidths=3,
    label="Target coverage"
)

ax_upper.set_xticks(list(x_positions))
ax_upper.set_xticklabels(upper_coverage_pdf["quantile"])

ax_upper.set_title(
    "Routing-Relevant Quantile Coverage on Held-Out Test Data",
    fontsize=13
)
ax_upper.set_xlabel("Calibrated delay quantile")
ax_upper.set_ylabel("Coverage (%)")
ax_upper.set_ylim(70, 101)

for i, (_, row) in enumerate(upper_coverage_pdf.iterrows()):
    ax_upper.text(
        i,
        row["empirical_pct"] + 0.5,
        f"{row['empirical_pct']:.1f}%",
        ha="center",
        fontsize=10
    )

ax_upper.legend()
ax_upper.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# %%
FIGURE_DIR = Path("../reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FULL_COVERAGE_FIGURE_PATH = (
    FIGURE_DIR / "test_empirical_coverage_all_quantiles.png"
)

UPPER_COVERAGE_FIGURE_PATH = (
    FIGURE_DIR / "test_empirical_coverage_routing_quantiles.png"
)

fig.savefig(
    FULL_COVERAGE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

fig_upper.savefig(
    UPPER_COVERAGE_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", FULL_COVERAGE_FIGURE_PATH)
print("Saved:", UPPER_COVERAGE_FIGURE_PATH)

# %%
#Check whether quantiles are monotonic
QUANTILE_COLS = [qcol for _, qcol in QUANTILE_LEVELS]

violation_condition = None

for q_low, q_high in zip(QUANTILE_COLS[:-1], QUANTILE_COLS[1:]):
    current_violation = F.col(q_low) > F.col(q_high)

    if violation_condition is None:
        violation_condition = current_violation
    else:
        violation_condition = violation_condition | current_violation

monotonicity_check_df = test_quantile_df.select(
    F.count("*").alias("n_rows"),
    F.sum(violation_condition.cast("int")).alias("n_rows_with_quantile_violation")
).withColumn(
    "violation_pct",
    F.round(
        100 * F.col("n_rows_with_quantile_violation") / F.col("n_rows"),
        4
    )
)

monotonicity_check_df.show(truncate=False)

# %%
#Calculate a one-number calibration summary
mean_absolute_coverage_error_pp = coverage_pdf[
    "absolute_error_pp"
].mean()

max_absolute_coverage_error_pp = coverage_pdf[
    "absolute_error_pp"
].max()

worst_quantile = coverage_pdf.loc[
    coverage_pdf["absolute_error_pp"].idxmax(),
    "quantile"
]

print(
    f"Mean absolute coverage error: "
    f"{mean_absolute_coverage_error_pp:.2f} percentage points"
)

print(
    f"Maximum absolute coverage error: "
    f"{max_absolute_coverage_error_pp:.2f} percentage points "
    f"at {worst_quantile}"
)

# %% [markdown]
# The point model is not evaluated only through RMSE or MAE, because the routing algorithm needs probabilities. After learning residual-based quantiles from the calibration split, I evaluate them on an untouched test split. For each predicted quantile qα, I measure the proportion of actual test delays that fall below it. A well-calibrated q90 should cover around 90% of realized delays. Therefore, closeness to the diagonal indicates that the uncertainty estimates can be meaningfully used as routing probabilities.
# If coverage is slightly above the diagonal
# The quantiles are slightly conservative: they cover somewhat more cases than their nominal target. For robust routing, moderate conservatism is acceptable because missed connections are costly.
# If coverage is below the diagonal
# The quantiles under-cover at this level, meaning the confidence estimate may be too optimistic. This is a weakness that should be considered when using that quantile for routing decisions.

# %% [markdown]
# ## Plot 4 — February Calibration Backoff-Level Distribution

# %% [raw]
# When producing February delay quantiles, how specific was the calibration source used for q90?
# Most February predictions use rich context-specific residual calibration, while the global residual distribution remains available as a fallback for sparse cases.

# %%
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = "/user/groups/com-490/H1/final/v1"

FEB_LOOKUP_PATH = (
    f"{BASE}/route_demo_outputs/lookup/"
    "route_demo_lookup_february.parquet"
)

feb_lookup_df = spark.read.parquet(FEB_LOOKUP_PATH).cache()

print("February lookup rows:", feb_lookup_df.count())
print("Has chosen calibration level column:",
      "chosen_calibration_level_q90" in feb_lookup_df.columns)

# %%
feb_lookup_df.select(
    F.count("*").alias("n_rows"),
    F.sum(
        F.col("chosen_calibration_level_q90").isNull().cast("int")
    ).alias("chosen_level_q90_nulls")
).show(truncate=False)

# %%
LEVEL_ORDER = [
    "L6_line_specific",
    "L5_ultra_rich",
    "L4_rich",
    "L3_operational",
    "L2_context",
    "L1_core",
    "L0_simple",
    "global",
]

n_feb = feb_lookup_df.count()

level_summary_df = (
    feb_lookup_df
    .groupBy("chosen_calibration_level_q90")
    .agg(
        F.count("*").alias("n_rows")
    )
    .withColumn(
        "pct",
        100 * F.col("n_rows") / F.lit(n_feb)
    )
)

level_summary_pdf = level_summary_df.toPandas()

# Ensure the plot follows the intended rich-to-simple backoff order
level_summary_pdf["chosen_calibration_level_q90"] = pd.Categorical(
    level_summary_pdf["chosen_calibration_level_q90"],
    categories=LEVEL_ORDER,
    ordered=True
)

level_summary_pdf = (
    level_summary_pdf
    .sort_values("chosen_calibration_level_q90")
    .reset_index(drop=True)
)

level_summary_pdf["pct"] = level_summary_pdf["pct"].round(3)

level_summary_pdf

# %%
rich_levels = ["L6_line_specific", "L5_ultra_rich", "L4_rich"]
context_specific_levels = [
    "L6_line_specific",
    "L5_ultra_rich",
    "L4_rich",
    "L3_operational",
    "L2_context",
    "L1_core",
    "L0_simple",
]

rich_pct = level_summary_pdf.loc[
    level_summary_pdf["chosen_calibration_level_q90"].isin(rich_levels),
    "pct"
].sum()

context_specific_pct = level_summary_pdf.loc[
    level_summary_pdf["chosen_calibration_level_q90"].isin(context_specific_levels),
    "pct"
].sum()

global_pct = level_summary_pdf.loc[
    level_summary_pdf["chosen_calibration_level_q90"] == "global",
    "pct"
].sum()

print(f"Rich calibration levels L4-L6: {rich_pct:.2f}%")
print(f"Any context-specific level L0-L6: {context_specific_pct:.2f}%")
print(f"Global fallback: {global_pct:.2f}%")

# %%
LEVEL_LABELS = {
    "L6_line_specific": "L6: Line-specific",
    "L5_ultra_rich": "L5: Ultra-rich context",
    "L4_rich": "L4: Rich context",
    "L3_operational": "L3: Operational context",
    "L2_context": "L2: Basic context",
    "L1_core": "L1: Core context",
    "L0_simple": "L0: Simple context",
    "global": "Global fallback",
}

plot_pdf = level_summary_pdf.copy()

plot_pdf["label"] = plot_pdf["chosen_calibration_level_q90"].map(LEVEL_LABELS)

plot_pdf[["label", "n_rows", "pct"]]

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    plot_pdf["label"],
    plot_pdf["pct"]
)

ax.invert_yaxis()

for i, row in plot_pdf.iterrows():
    ax.text(
        row["pct"] + 0.4,
        i,
        f"{row['pct']:.1f}%",
        va="center",
        fontsize=10
    )

ax.set_title(
    "Calibration Specificity Used in the February Routing Lookup",
    fontsize=13
)

ax.set_xlabel(
    "Share of February trip-stop predictions (%)",
    fontsize=11
)

ax.set_ylabel("")

ax.set_xlim(0, max(plot_pdf["pct"]) + 7)

ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.show()

# %%
FIGURE_DIR = Path("../reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BACKOFF_FIGURE_PATH = (
    FIGURE_DIR / "february_calibration_backoff_level_distribution.png"
)

fig.savefig(
    BACKOFF_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", BACKOFF_FIGURE_PATH)

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    plot_pdf["label"],
    plot_pdf["pct"]
)

ax.invert_yaxis()

for i, row in plot_pdf.iterrows():
    ax.text(
        row["pct"] + 0.4,
        i,
        f"{row['pct']:.1f}%",
        va="center",
        fontsize=10
    )

ax.set_title(
    "February Calibration Backoff Distribution\n"
    f"Rich levels L4–L6: {rich_pct:.1f}%  |  Global fallback: {global_pct:.1f}%",
    fontsize=13
)

ax.set_xlabel("Share of February trip-stop predictions (%)")
ax.set_ylabel("")
ax.set_xlim(0, max(plot_pdf["pct"]) + 7)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.show()

# %% [raw]
# For each February trip-stop prediction, the calibration layer first tries to use the most specific residual distribution supported by enough calibration observations. If a highly specific group is too sparse, it backs off to broader contexts and eventually to the global residual distribution.
#
# This plot shows the calibration level selected for q90 in the final February lookup. Around 81% of predictions use rich calibration levels L4 to L6, including about 35% that use line-specific calibration. Only about 7% fall back to the global residual distribution. This suggests that the final lookup mainly uses context-aware safety margins, while still retaining a reliable fallback for sparse situations.

# %% [raw]
# This plot shows calibration specificity, not calibration accuracy. Accuracy is evaluated separately through held-out test-set empirical coverage.

# %% [markdown]
# ## Plot 5 — Transfer Probability Curve

# %%

# %% [raw]
# How the February delay lookup converts usable transfer slack X into a probability of making the transfer.

# %%
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = "/user/groups/com-490/H1/final/v1"

FEB_LOOKUP_PATH = (
    f"{BASE}/route_demo_outputs/lookup/"
    "route_demo_lookup_february.parquet"
)

feb_lookup_df = spark.read.parquet(FEB_LOOKUP_PATH)

# %%
INCOMING_TRIP_ID = "1334.TA.91-m1-j26-1.3.H"
TRANSFER_BPUIC = 8591818
OPERATING_DAY = "2026-02-04"
SCHEDULED_ARRIVAL_TS = "2026-02-04 12:44:00"

transfer_row = (
    feb_lookup_df
    .filter(
        (F.col("trip_id") == INCOMING_TRIP_ID) &
        (F.col("bpuic") == TRANSFER_BPUIC) &
        (F.col("operating_day") == F.to_date(F.lit(OPERATING_DAY))) &
        (
            F.col("scheduled_arrival_ts")
            == F.to_timestamp(F.lit(SCHEDULED_ARRIVAL_TS))
        )
    )
    .select(
        "operating_day",
        "trip_id",
        "bpuic",
        "stop_name",
        "scheduled_arrival_ts",
        "line_text",
        "transport_clean",
        "pred_delay",
        "chosen_calibration_level_q90",
        "q01", "q05", "q10", "q20", "q35", "q50",
        "q65", "q80", "q90", "q95", "q99"
    )
    .first()
)

if transfer_row is None:
    raise ValueError("Transfer row not found in February lookup.")

transfer_dict = transfer_row.asDict()

transfer_dict

# %%
QUANTILE_POINTS = [
    (0.01, "q01"),
    (0.05, "q05"),
    (0.10, "q10"),
    (0.20, "q20"),
    (0.35, "q35"),
    (0.50, "q50"),
    (0.65, "q65"),
    (0.80, "q80"),
    (0.90, "q90"),
    (0.95, "q95"),
    (0.99, "q99"),
]

quantile_curve_pdf = pd.DataFrame(
    {
        "probability": [p for p, _ in QUANTILE_POINTS],
        "delay_seconds": [
            float(transfer_dict[qcol])
            for _, qcol in QUANTILE_POINTS
        ],
        "quantile": [qcol for _, qcol in QUANTILE_POINTS],
    }
)

quantile_curve_pdf

# %%
RAW_TRANSFER_GAP_SEC = 180
MIN_TRANSFER_SEC = 120
EXTRA_TRANSFER_SEC = 0

X_SECONDS = (
    RAW_TRANSFER_GAP_SEC
    - MIN_TRANSFER_SEC
    - EXTRA_TRANSFER_SEC
)

print("Usable transfer slack X:", X_SECONDS, "seconds")


# %%
def prob_from_quantiles_for_plot(row_dict, X):
    """
    Approximate P(delay <= X) from calibrated delay quantiles.

    Note:
    This version lets the quantile curve determine P(delay <= X),
    including at X = 0. If your current production function forces
    X <= 0 to probability 0, keep that as a separate documented
    routing-rule decision.
    """
    points = sorted(
        [
            (float(row_dict[qcol]), float(p))
            for p, qcol in QUANTILE_POINTS
            if row_dict.get(qcol) is not None
        ],
        key=lambda item: item[0]
    )

    if not points:
        return np.nan

    if X <= points[0][0]:
        return points[0][1]

    if X >= points[-1][0]:
        return points[-1][1]

    for (delay_low, p_low), (delay_high, p_high) in zip(points[:-1], points[1:]):
        if delay_low <= X <= delay_high:
            if delay_high == delay_low:
                return p_high

            weight = (X - delay_low) / (delay_high - delay_low)

            return p_low + weight * (p_high - p_low)

    return np.nan


# %%
transfer_probability = prob_from_quantiles_for_plot(
    transfer_dict,
    X_SECONDS
)

print(
    f"P(delay <= {X_SECONDS} sec) = "
    f"{transfer_probability:.3f} "
    f"({100 * transfer_probability:.1f}%)"
)

# %%
min_x = min(-60, int(np.floor(quantile_curve_pdf["delay_seconds"].min() / 30) * 30))
max_x = int(np.ceil(quantile_curve_pdf["delay_seconds"].max() / 30) * 30) + 30

x_grid = np.arange(min_x, max_x + 1, 1)

curve_pdf = pd.DataFrame({
    "X_seconds": x_grid
})

curve_pdf["probability"] = curve_pdf["X_seconds"].apply(
    lambda x: prob_from_quantiles_for_plot(transfer_dict, x)
)

curve_pdf.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))

# Approximate CDF curve derived from quantiles
ax.plot(
    curve_pdf["X_seconds"],
    100 * curve_pdf["probability"],
    linewidth=2.5,
    label="Estimated transfer success probability"
)

# Quantile anchor points stored in the lookup
ax.scatter(
    quantile_curve_pdf["delay_seconds"],
    100 * quantile_curve_pdf["probability"],
    s=55,
    zorder=3,
    label="Calibrated quantile anchors"
)

# Label the most interpretable upper quantiles
LABEL_QUANTILES = ["q50", "q80", "q90", "q95", "q99"]

for _, row in quantile_curve_pdf[
    quantile_curve_pdf["quantile"].isin(LABEL_QUANTILES)
].iterrows():
    ax.annotate(
        f"{row['quantile']}\n{row['delay_seconds']:.0f}s",
        (row["delay_seconds"], 100 * row["probability"]),
        xytext=(5, -28),
        textcoords="offset points",
        fontsize=9
    )

# Highlight the actual transfer slack
ax.axvline(
    X_SECONDS,
    linestyle="--",
    linewidth=1.8,
    label=f"Actual usable slack X = {X_SECONDS} sec"
)

ax.scatter(
    [X_SECONDS],
    [100 * transfer_probability],
    s=85,
    zorder=4
)

ax.annotate(
    f"Transfer probability ≈ {100 * transfer_probability:.1f}%",
    (X_SECONDS, 100 * transfer_probability),
    xytext=(20, 15),
    textcoords="offset points",
    fontsize=10,
    arrowprops={"arrowstyle": "->", "lw": 1}
)

ax.set_title(
    "From Transfer Slack to Success Probability: Example February Connection",
    fontsize=13
)

ax.set_xlabel(
    "Usable transfer slack X (seconds)",
    fontsize=11
)

ax.set_ylabel(
    "P(delay ≤ X) (%)",
    fontsize=11
)

ax.set_ylim(0, 102)
ax.set_xlim(min_x, max_x)
ax.grid(alpha=0.25)
ax.legend(loc="lower right")

plt.tight_layout()
plt.show()

# %%
print("Example transfer used in Plot 5")
print("--------------------------------")
print("Incoming trip:", transfer_dict["trip_id"])
print("Transfer stop:", transfer_dict["stop_name"])
print("Scheduled incoming arrival:", transfer_dict["scheduled_arrival_ts"])
print("Transport:", transfer_dict["transport_clean"])
print("Line:", transfer_dict["line_text"])
print("Point prediction:", round(float(transfer_dict["pred_delay"]), 2), "sec")
print("q50:", round(float(transfer_dict["q50"]), 2), "sec")
print("q80:", round(float(transfer_dict["q80"]), 2), "sec")
print("q90:", round(float(transfer_dict["q90"]), 2), "sec")
print("q95:", round(float(transfer_dict["q95"]), 2), "sec")
print("q99:", round(float(transfer_dict["q99"]), 2), "sec")
print("Calibration source for q90:", transfer_dict["chosen_calibration_level_q90"])
print("Usable slack X:", X_SECONDS, "sec")
print("P(delay <= X):", round(100 * transfer_probability, 2), "%")

# %%
FIGURE_DIR = Path("../reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRANSFER_CURVE_PATH = (
    FIGURE_DIR / "transfer_probability_curve_example_february.png"
)

fig.savefig(
    TRANSFER_CURVE_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", TRANSFER_CURVE_PATH)

# %% [raw]
# This figure shows how the calibrated delay lookup is used by the routing algorithm. For one actual February connection, the scheduled transfer gap is three minutes. After subtracting the two-minute minimum transfer requirement, the passenger has 60 seconds of usable delay tolerance. The lookup provides calibrated delay quantiles for the incoming trip-stop event, and I linearly interpolate between these quantiles to estimate P(delay <= X). In this example, the estimated probability of making the transfer is about 62%.
#
# This is why the final output of my model is not only one predicted delay. The router needs a probability for any possible transfer slack X.
# A remaining routing-rule decision is whether X = 0 should be treated as automatically unsafe or evaluated as P(delay <= 0), i.e. the probability that the incoming service arrives on time or early.

# %%

# %%

# %%

# %%

# %%
spark.stop()

# %%

# %%

# %%
