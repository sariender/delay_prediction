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
# # February Lookup Table Prep 

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

# %%
base = f"{hadoopFS}/user/groups/com-490/H1/final/v1"

# %%
print(repr(hadoopFS)); print(repr(base))

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

# %% [markdown]
# ## Historical data

# %%
from pyspark.sql import functions as F
from pyspark.sql.window import Window

FULL_DATA_PATH = f"{base}/features_and_training_data/final_full_data.parquet"

full_base_df = spark.read.parquet(FULL_DATA_PATH)

print("full_base_df rows:", full_base_df.count())
print("full_base_df cols:", len(full_base_df.columns))

full_base_df.agg(
    F.min("operating_day").alias("min_day"),
    F.max("operating_day").alias("max_day"),
    F.countDistinct("operating_day").alias("n_days")
).show(truncate=False)

# %%
required_history_cols = [
    "arrival_delay_seconds",
    "operator_id",
    "line_id",
    "line_text",
    "bpuic",
    "transport_clean",
    "hour_of_day",
]

missing_history_cols = [
    c for c in required_history_cols
    if c not in full_base_df.columns
]

print("Missing history columns:", missing_history_cols)


# %% [markdown]
# ## Historical features

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


def make_hist_fill_values(history_df):
    global_delay_stats = history_df.agg(
        F.round(F.avg("arrival_delay_seconds"), 3).alias("global_mean_delay"),
        F.expr("percentile_approx(arrival_delay_seconds, 0.5)").alias("global_median_delay"),
        F.expr("percentile_approx(arrival_delay_seconds, 0.9)").alias("global_p90_delay")
    ).collect()[0]

    GLOBAL_MEAN_DELAY = float(global_delay_stats["global_mean_delay"])
    GLOBAL_MEDIAN_DELAY = float(global_delay_stats["global_median_delay"])
    GLOBAL_P90_DELAY = float(global_delay_stats["global_p90_delay"])

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


# %% [markdown]
# ## Historical tables

# %%
hist_tables_full = make_all_hist_tables(full_base_df)
fill_values_full = make_hist_fill_values(full_base_df)

hist_operator_hour_df, hist_line_hour_df, hist_stop_hour_df, hist_transport_hour_df = hist_tables_full

for name, df in [
    ("operator_hour", hist_operator_hour_df),
    ("line_hour", hist_line_hour_df),
    ("stop_hour", hist_stop_hour_df),
    ("transport_hour", hist_transport_hour_df),
]:
    print(name, df.count())
    df.show(3, truncate=False)

# %% [markdown]
# ## Timetable Prep. (February for the demo)

# %%
## you should change here if you want to use another timetable.

# %%
FEB_TIMETABLE_PATH = f"{base}/input_from_data_side/timetable_february.parquet"
feb_raw_df = spark.read.parquet(FEB_TIMETABLE_PATH)

print("feb_raw rows:", feb_raw_df.count())
feb_raw_df.printSchema()

# %% [markdown]
# ## from february base data to lookup ready data

# %%
feb_base_df = (
    feb_raw_df
    .withColumnRenamed("arrival_timestamp", "scheduled_arrival_ts")
    .withColumnRenamed("departure_timestamp", "scheduled_departure_ts")
    .withColumn(
        "operating_day",
        F.to_date(F.coalesce("scheduled_arrival_ts", "scheduled_departure_ts"))
    )
    .withColumn(
        "bpuic",
        F.split(F.col("stop_id"), ":").getItem(0).cast("int")
    )
    .withColumn("line_text", F.col("route_short_name").cast("string"))
    .withColumn("transport_clean_raw", F.col("transport_clean"))
    .filter(F.col("scheduled_arrival_ts").isNotNull())
)

# %%
feb_base_df = (
    feb_base_df
    .withColumn(
        "transport_clean",
        F.when(F.upper(F.col("transport_clean_raw")).isin("B", "BN", "BUA", "BUS"), F.lit("Bus"))
         .when(F.upper(F.col("transport_clean_raw")).isin("M", "METRO"), F.lit("Metro"))
         .when(
             F.upper(F.col("transport_clean_raw")).isin(
                 "IC", "ICE", "IR", "IRE", "RE", "R", "S", "SN",
                 "TGV", "EC", "RJX", "TER", "PE", "EXT", "NJ", "ZUG"
             ),
             F.lit("Zug")
         )
         .when(F.upper(F.col("transport_clean_raw")).isin("T", "TN", "TRAM"), F.lit("Tram"))
         .when(F.upper(F.col("transport_clean_raw")).isin("BAT", "SCHIFF"), F.lit("Schiff"))
         .when(F.upper(F.col("transport_clean_raw")).isin("CC", "ZAHNRADBAHN"), F.lit("Zahnradbahn"))
         .when(F.upper(F.col("transport_clean_raw")).isin("TAXI"), F.lit("Taxi"))
         .otherwise(F.lit("Other"))
    )
)

feb_base_df.groupBy("transport_clean_raw", "transport_clean") \
    .agg(F.count("*").alias("n_rows")) \
    .orderBy(F.desc("n_rows")) \
    .show(50, truncate=False)

# %% [markdown]
# ## Add line support flag for diagnostics

# %%
training_line_text_df = (
    full_base_df
    .select("operator_id", "line_text")
    .distinct()
    .withColumn("line_seen_in_training", F.lit(1))
)

feb_base_df = (
    feb_base_df
    .join(
        training_line_text_df,
        on=["operator_id", "line_text"],
        how="left"
    )
    .withColumn(
        "line_seen_in_training",
        F.coalesce(F.col("line_seen_in_training"), F.lit(0))
    )
)

feb_base_df.groupBy("line_seen_in_training") \
    .agg(F.count("*").alias("n_rows")) \
    .orderBy("line_seen_in_training") \
    .show(truncate=False)

# %% [markdown]
# ##  Create line_id lookup from historical full data

# %%
line_lookup_counts_df = (
    full_base_df
    .withColumn("line_text_norm", F.upper(F.trim(F.col("line_text"))))
    .groupBy(
        "operator_id",
        "line_text_norm",
        "line_id",
        "line_text"
    )
    .agg(F.count("*").alias("n_rows"))
)

line_lookup_window = Window.partitionBy(
    "operator_id",
    "line_text_norm"
).orderBy(F.desc("n_rows"))

line_lookup_df = (
    line_lookup_counts_df
    .withColumn("rn", F.row_number().over(line_lookup_window))
    .filter(F.col("rn") == 1)
    .select(
        "operator_id",
        "line_text_norm",
        F.col("line_id").alias("hist_line_id"),
        F.col("line_text").alias("hist_line_text")
    )
)

# %%
feb_base_df = (
    feb_base_df
    .withColumn("line_text_norm", F.upper(F.trim(F.col("line_text"))))
    .join(
        line_lookup_df,
        on=["operator_id", "line_text_norm"],
        how="left"
    )
    .withColumn("line_id", F.col("hist_line_id"))
    .withColumn(
        "line_text",
        F.coalesce(F.col("hist_line_text"), F.col("line_text"))
    )
    .drop("line_text_norm", "hist_line_id", "hist_line_text")
)

# %%
feb_base_df.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col("line_id").isNull().cast("int")).alias("missing_line_id"),
    F.round(
        100 * F.avg(F.col("line_id").isNull().cast("int")),
        3
    ).alias("missing_line_id_pct")
).show(truncate=False)

feb_base_df.filter(F.col("line_id").isNull()) \
    .groupBy("operator_id", "line_text", "transport_clean") \
    .agg(F.count("*").alias("n_rows")) \
    .orderBy(F.desc("n_rows")) \
    .show(30, truncate=False)

# %%
feb_base_df = (
    feb_base_df
    .withColumn(
        "line_id_source",
        F.when(F.col("line_id").isNotNull(), F.lit("historical_lookup"))
         .otherwise(F.lit("fallback_unseen_line"))
    )
    .withColumn(
        "line_id",
        F.when(
            F.col("line_id").isNotNull(),
            F.col("line_id")
        ).otherwise(
            F.concat_ws(
                ":",
                F.coalesce(F.col("operator_id"), F.lit("unknown_operator")),
                F.coalesce(F.col("line_text"), F.lit("unknown_line"))
            )
        )
    )
)

# %%
feb_base_df.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col("line_id").isNull().cast("int")).alias("missing_line_id"),
    F.round(
        100 * F.avg(F.col("line_id").isNull().cast("int")),
        3
    ).alias("missing_line_id_pct")
).show(truncate=False)

# %%
training_transport_values = [
    r["transport_clean"]
    for r in full_base_df
        .select("transport_clean")
        .distinct()
        .collect()
]

print("Training transport values:", training_transport_values)

feb_base_df = (
    feb_base_df
    .withColumn(
        "transport_clean_model",
        F.when(
            F.col("transport_clean").isin(training_transport_values),
            F.col("transport_clean")
        ).otherwise(F.lit("Other"))
    )
    .withColumnRenamed("transport_clean", "transport_clean_mapped")
    .withColumnRenamed("transport_clean_model", "transport_clean")
)

# %%
feb_base_df.groupBy(
    "transport_clean_raw",
    "transport_clean_mapped",
    "transport_clean"
).agg(
    F.count("*").alias("n_rows")
).orderBy(
    F.desc("n_rows")
).show(50, truncate=False)

# %% [markdown]
# ## Calendar/time features

# %%
feb_feature_v1_df = (
    feb_base_df

    .withColumn("year", F.year("operating_day"))
    .withColumn("month", F.month("operating_day"))
    .withColumn("day_of_week", F.dayofweek("operating_day"))
    .withColumn(
        "is_weekend",
        F.when(F.dayofweek("operating_day").isin(1, 7), 1).otherwise(0)
    )

    .withColumn("hour_of_day", F.hour("scheduled_arrival_ts"))
    .withColumn("minute_of_hour", F.minute("scheduled_arrival_ts"))
    .withColumn(
        "scheduled_minute_of_day",
        F.hour("scheduled_arrival_ts") * 60 + F.minute("scheduled_arrival_ts")
    )

    .withColumn(
        "is_morning_peak",
        F.when(F.col("hour_of_day").between(7, 9), 1).otherwise(0)
    )
    .withColumn(
        "is_evening_peak",
        F.when(F.col("hour_of_day").between(16, 18), 1).otherwise(0)
    )
    .withColumn(
        "is_late_night",
        F.when((F.col("hour_of_day") <= 5) | (F.col("hour_of_day") >= 23), 1).otherwise(0)
    )
)

# %% [markdown]
# ## Scheduled timestamp pattern

# %%
feb_feature_v1_df = (
    feb_feature_v1_df
    .withColumn(
        "scheduled_timestamp_pattern",
        F.when(
            F.col("scheduled_arrival_ts").isNotNull()
            & F.col("scheduled_departure_ts").isNotNull(),
            F.lit("scheduled_arrival_and_departure")
        )
        .when(
            F.col("scheduled_arrival_ts").isNotNull()
            & F.col("scheduled_departure_ts").isNull(),
            F.lit("scheduled_arrival_only_likely_terminal")
        )
        .when(
            F.col("scheduled_arrival_ts").isNull()
            & F.col("scheduled_departure_ts").isNotNull(),
            F.lit("scheduled_departure_only_likely_origin")
        )
        .otherwise(F.lit("other_scheduled_timestamp_pattern"))
    )
)

# %% [markdown]
# ## Trip-position features

# %%
TRIP_GROUP_COLS = [
    "operating_day",
    "operator_id",
    "trip_id",
    "line_id",
]

feb_feature_v3_base_df = feb_feature_v1_df.withColumn(
    "scheduled_event_ts",
    F.coalesce(
        F.col("scheduled_arrival_ts"),
        F.col("scheduled_departure_ts")
    )
)

scheduled_stop_events_df = (
    feb_feature_v3_base_df
    .select(
        *TRIP_GROUP_COLS,
        "scheduled_event_ts",
        "bpuic"
    )
    .dropDuplicates()
)

trip_sequence_window = (
    Window
    .partitionBy(*TRIP_GROUP_COLS)
    .orderBy(
        F.col("scheduled_event_ts").asc_nulls_last(),
        F.col("bpuic").asc_nulls_last()
    )
)

trip_summary_window = Window.partitionBy(*TRIP_GROUP_COLS)

scheduled_stop_features_df = (
    scheduled_stop_events_df
    .withColumn(
        "stop_sequence_idx",
        F.row_number().over(trip_sequence_window)
    )
    .withColumn(
        "n_stops_in_trip_observed",
        F.count("*").over(trip_summary_window)
    )
    .withColumn(
        "relative_stop_position",
        F.col("stop_sequence_idx") / F.col("n_stops_in_trip_observed")
    )
    .withColumn(
        "is_first_observed_stop",
        F.when(F.col("stop_sequence_idx") == 1, 1).otherwise(0)
    )
    .withColumn(
        "is_last_observed_stop",
        F.when(
            F.col("stop_sequence_idx") == F.col("n_stops_in_trip_observed"),
            1
        ).otherwise(0)
    )
)

feb_feature_v3_df = (
    feb_feature_v3_base_df
    .join(
        scheduled_stop_features_df,
        on=TRIP_GROUP_COLS + ["scheduled_event_ts", "bpuic"],
        how="left"
    )
    .withColumn(
        "first_scheduled_event_ts",
        F.min("scheduled_event_ts").over(trip_summary_window)
    )
    .withColumn(
        "scheduled_minutes_since_first_observed_stop",
        (
            F.unix_timestamp("scheduled_event_ts")
            - F.unix_timestamp("first_scheduled_event_ts")
        ) / 60.0
    )
    .withColumn(
        "relative_stop_position_v2",
        F.when(
            F.col("n_stops_in_trip_observed") > 1,
            (F.col("stop_sequence_idx") - 1) / (F.col("n_stops_in_trip_observed") - 1)
        ).otherwise(F.lit(0.0))
    )
)

# %%
feb_feature_v3_df.select(
    "operating_day",
    "trip_id",
    "line_text",
    "line_id",
    "transport_clean",
    "bpuic",
    "stop_name",
    "scheduled_event_ts",
    "stop_sequence_idx",
    "n_stops_in_trip_observed",
    "relative_stop_position_v2",
    "scheduled_minutes_since_first_observed_stop",
    "line_seen_in_training"
).show(5, truncate=False, vertical=True)

# %% [markdown]
# ## Add historical features to February

# %%
feb_hist_debug_df = add_historical_features(
    feb_feature_v3_df,
    hist_tables_full
)

feb_hist_debug_df.select(
    F.count("*").alias("n_rows"),
    F.round(
        100 * F.avg(F.col("hist_operator_hour_mean_delay").isNull().cast("int")),
        3
    ).alias("operator_hour_missing_pct"),
    F.round(
        100 * F.avg(F.col("hist_line_hour_mean_delay").isNull().cast("int")),
        3
    ).alias("line_hour_missing_pct"),
    F.round(
        100 * F.avg(F.col("hist_stop_hour_mean_delay").isNull().cast("int")),
        3
    ).alias("stop_hour_missing_pct"),
    F.round(
        100 * F.avg(F.col("hist_transport_hour_mean_delay").isNull().cast("int")),
        3
    ).alias("transport_hour_missing_pct"),
).show(truncate=False)

# %%
feb_features_df = feb_hist_debug_df.fillna(fill_values_full)

# %%
feb_features_df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in hist_feature_cols
]).show(truncate=False)

# %% [markdown]
# ## Weather

# %%
FEB_FORECAST_CACHE_PATH = f"{base}/weather_data/forecast/forecast_weather_2026_02.parquet"
forecast_feb_df = spark.read.parquet(FEB_FORECAST_CACHE_PATH)
print(f"February forecast rows: {forecast_feb_df.count():,}")

# %%
forecast_feb_df.agg(
    F.min("weather_hour").alias("min_weather_hour"),
    F.max("weather_hour").alias("max_weather_hour")
).show(truncate=False)

feb_features_df.agg(
    F.min("scheduled_arrival_ts").alias("min_scheduled_arrival"),
    F.max("scheduled_arrival_ts").alias("max_scheduled_arrival")
).show(truncate=False)


# %%
def enrich_with_forecast_weather_only(features_df, forecast_df, scheduled_ts_col="scheduled_arrival_ts"):
    return (
        features_df
        .withColumn("weather_hour", F.date_trunc("hour", F.col(scheduled_ts_col)))
        .join(forecast_df, on="weather_hour", how="left")
    )


# %%
feb_weather_enriched_df = enrich_with_forecast_weather_only(
    feb_features_df,
    forecast_feb_df
)

# %%
weather_cols = [
    "temperature_2m_fcst_d1",
    "relative_humidity_2m_fcst_d1",
    "precipitation_fcst_d1",
    "snowfall_fcst_d1",
    "wind_speed_10m_fcst_d1",
    "wind_gusts_10m_fcst_d1",
    "temperature_2m_fcst_d2",
    "relative_humidity_2m_fcst_d2",
    "precipitation_fcst_d2",
    "snowfall_fcst_d2",
    "wind_speed_10m_fcst_d2",
    "wind_gusts_10m_fcst_d2",
    "temperature_2m_fcst_d1_minus_d2",
    "precipitation_fcst_d1_minus_d2",
    "snowfall_fcst_d1_minus_d2",
    "wind_speed_10m_fcst_d1_minus_d2",
    "weather_code_fcst_d1",
    "weather_code_fcst_d2",
]

n_feb = feb_weather_enriched_df.count()

feb_weather_enriched_df.select([
    F.round(
        100 * F.sum(F.col(c).isNull().cast("int")) / F.lit(n_feb),
        3
    ).alias(c)
    for c in weather_cols
    if c in feb_weather_enriched_df.columns
]).show(truncate=False)

# %% [markdown]
# ## Special days

# %%
from special_days import build_special_day_tables, add_special_day_features

special_days_df, school_holiday_ranges_df = build_special_day_tables(spark)

feb_final_features_df = add_special_day_features(
    feb_weather_enriched_df,
    special_days_df,
    school_holiday_ranges_df
)

# %% [markdown]
# ## Null_check

# %%
numeric_null_check_df = feb_final_features_df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in MODEL_NUMERIC_COLS
])

stack_expr = "stack({0}, {1}) as (feature, n_nulls)".format(
    len(MODEL_NUMERIC_COLS),
    ", ".join([f"'{c}', `{c}`" for c in MODEL_NUMERIC_COLS])
)

numeric_null_long_df = (
    numeric_null_check_df
    .selectExpr(stack_expr)
    .filter(F.col("n_nulls") > 0)
    .orderBy(F.desc("n_nulls"))
)

numeric_null_long_df.show(100, truncate=False)

# %%
categorical_null_check_df = feb_final_features_df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in MODEL_CATEGORICAL_COLS
])

stack_expr = "stack({0}, {1}) as (feature, n_nulls)".format(
    len(MODEL_CATEGORICAL_COLS),
    ", ".join([f"'{c}', `{c}`" for c in MODEL_CATEGORICAL_COLS])
)

categorical_null_long_df = (
    categorical_null_check_df
    .selectExpr(stack_expr)
    .filter(F.col("n_nulls") > 0)
    .orderBy(F.desc("n_nulls"))
)

categorical_null_long_df.show(100, truncate=False)

# %%
cat_fill_values = {c: "unknown" for c in MODEL_CATEGORICAL_COLS}
feb_final_features_df = feb_final_features_df.fillna(cat_fill_values)

# %%
numeric_fill_values = {}

for c in MODEL_NUMERIC_COLS:
    if c.startswith("hist_") and c.endswith("_n_obs"):
        numeric_fill_values[c] = 0
    elif c.startswith("hist_") and c.endswith("_mean_delay"):
        numeric_fill_values[c] = fill_values_full.get(c, 0.0)
    elif c.startswith("hist_") and c.endswith("_median_delay"):
        numeric_fill_values[c] = fill_values_full.get(c, 0.0)
    elif c.startswith("hist_") and c.endswith("_p90_delay"):
        numeric_fill_values[c] = fill_values_full.get(c, 0.0)
    else:
        numeric_fill_values[c] = 0.0

feb_final_features_df = feb_final_features_df.fillna(numeric_fill_values)

# %% [markdown]
# ## Check every feature distribution

# %%
model_input_cols = MODEL_NUMERIC_COLS + MODEL_CATEGORICAL_COLS

for name, df in [
    ("historical", full_base_df),
    ("february", feb_final_features_df),
]:
    missing = [c for c in model_input_cols if c not in df.columns]
    print(name, "missing model columns:", missing)


# %%
def null_summary(df, cols, dataset_name):
    n = df.count()

    wide_df = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in cols
    ])

    stack_expr = "stack({0}, {1}) as (feature, n_nulls)".format(
        len(cols),
        ", ".join([f"'{c}', `{c}`" for c in cols])
    )

    return (
        wide_df
        .selectExpr(stack_expr)
        .withColumn("dataset", F.lit(dataset_name))
        .withColumn("null_pct", F.round(100 * F.col("n_nulls") / F.lit(n), 3))
        .select("dataset", "feature", "n_nulls", "null_pct")
    )


# %%
hist_nulls_df = null_summary(full_base_df, model_input_cols, "historical")
feb_nulls_df = null_summary(feb_final_features_df, model_input_cols, "february")

null_compare_df = (
    hist_nulls_df
    .withColumnRenamed("n_nulls", "hist_n_nulls")
    .withColumnRenamed("null_pct", "hist_null_pct")
    .drop("dataset")
    .join(
        feb_nulls_df
        .withColumnRenamed("n_nulls", "feb_n_nulls")
        .withColumnRenamed("null_pct", "feb_null_pct")
        .drop("dataset"),
        on="feature",
        how="outer"
    )
    .orderBy(F.desc("feb_null_pct"), F.desc("hist_null_pct"))
)

null_compare_df.show(100, truncate=False)

# %%
full_types = {
    field.name: str(field.dataType)
    for field in full_base_df.schema.fields
}

feb_types = {
    field.name: str(field.dataType)
    for field in feb_final_features_df.schema.fields
}

type_mismatch = []

for c in model_input_cols:
    if c in full_types and c in feb_types:
        if full_types[c] != feb_types[c]:
            type_mismatch.append((c, full_types[c], feb_types[c]))

type_mismatch


# %%
def categorical_distribution(df, col, dataset_name, top_n=30):
    n = df.count()

    return (
        df
        .groupBy(col)
        .agg(F.count("*").alias("n_rows"))
        .withColumn("pct", F.round(100 * F.col("n_rows") / F.lit(n), 3))
        .withColumn("dataset", F.lit(dataset_name))
        .withColumnRenamed(col, "value")
        .select("dataset", F.lit(col).alias("feature"), "value", "n_rows", "pct")
        .orderBy(F.desc("n_rows"))
        .limit(top_n)
    )


# %%
for c in MODEL_CATEGORICAL_COLS:
    print("\n====================")
    print(c)
    print("====================")

    categorical_distribution(full_base_df, c, "historical").show(30, truncate=False)
    categorical_distribution(feb_final_features_df, c, "february").show(30, truncate=False)


# %%
def numeric_summary(df, cols, dataset_name):
    rows = []

    for c in cols:
        row = (
            df
            .select(
                F.lit(dataset_name).alias("dataset"),
                F.lit(c).alias("feature"),
                F.count("*").alias("n_rows"),
                F.sum(F.col(c).isNull().cast("int")).alias("n_nulls"),
                F.min(c).alias("min"),
                F.expr(f"percentile_approx({c}, 0.1)").alias("p10"),
                F.expr(f"percentile_approx({c}, 0.5)").alias("median"),
                F.expr(f"percentile_approx({c}, 0.9)").alias("p90"),
                F.max(c).alias("max"),
                F.round(F.avg(c), 3).alias("mean"),
            )
        )
        rows.append(row)

    out = rows[0]
    for r in rows[1:]:
        out = out.unionByName(r)

    return out


# %%
hist_num_summary_df = numeric_summary(full_base_df, MODEL_NUMERIC_COLS, "historical")
feb_num_summary_df = numeric_summary(feb_final_features_df, MODEL_NUMERIC_COLS, "february")

num_compare_df = (
    hist_num_summary_df
    .select(
        "feature",
        F.col("min").alias("hist_min"),
        F.col("median").alias("hist_median"),
        F.col("p90").alias("hist_p90"),
        F.col("max").alias("hist_max"),
        F.col("mean").alias("hist_mean"),
    )
    .join(
        feb_num_summary_df
        .select(
            "feature",
            F.col("min").alias("feb_min"),
            F.col("median").alias("feb_median"),
            F.col("p90").alias("feb_p90"),
            F.col("max").alias("feb_max"),
            F.col("mean").alias("feb_mean"),
        ),
        on="feature",
        how="outer"
    )
    .withColumn(
        "median_diff",
        F.round(F.col("feb_median") - F.col("hist_median"), 3)
    )
    .withColumn(
        "mean_diff",
        F.round(F.col("feb_mean") - F.col("hist_mean"), 3)
    )
)

num_compare_df.orderBy(F.desc(F.abs(F.col("mean_diff")))).show(100, truncate=False)

# %%
feb_raw_df.filter(
    F.lower(F.col("route_short_name")) == "m2"
).groupBy(
    "operator_id",
    "operator_name",
    "route_short_name",
    "transport_clean"
).agg(
    F.count("*").alias("n_rows"),
    F.min("arrival_timestamp").alias("min_arrival_ts"),
    F.max("arrival_timestamp").alias("max_arrival_ts"),
    F.countDistinct("trip_id").alias("n_trips"),
    F.countDistinct("stop_id").alias("n_stops")
).orderBy(
    F.desc("n_rows")
).show(50, truncate=False)

# %%
for df_name, df in [("historical", full_base_df), ("february", feb_final_features_df)]:
    print(df_name)
    df.groupBy("transport_clean") \
        .agg(F.count("*").alias("n_rows")) \
        .orderBy(F.desc("n_rows")) \
        .show(truncate=False)

# %%
feb_final_features_df.groupBy("line_seen_in_training") \
    .agg(F.count("*").alias("n_rows")) \
    .withColumn(
        "pct",
        F.round(100 * F.col("n_rows") / F.lit(feb_final_features_df.count()), 3)
    ) \
    .orderBy("line_seen_in_training") \
    .show(truncate=False)

# %%
feb_final_features_df.filter(F.col("line_seen_in_training") == 0) \
    .groupBy("operator_id", "line_text", "transport_clean") \
    .agg(F.count("*").alias("n_rows")) \
    .orderBy(F.desc("n_rows")) \
    .show(50, truncate=False)

# %%
feb_final_features_df.select(
    F.count("*").alias("n_rows"),
    F.round(100 * F.avg((F.col("hist_operator_hour_n_obs") == 0).cast("int")), 3).alias("operator_hist_zero_pct"),
    F.round(100 * F.avg((F.col("hist_line_hour_n_obs") == 0).cast("int")), 3).alias("line_hist_zero_pct"),
    F.round(100 * F.avg((F.col("hist_stop_hour_n_obs") == 0).cast("int")), 3).alias("stop_hist_zero_pct"),
    F.round(100 * F.avg((F.col("hist_transport_hour_n_obs") == 0).cast("int")), 3).alias("transport_hist_zero_pct"),
).show(truncate=False)

# %% [markdown]
# ## Diagnostics

# %%
trip_span_check_df = (
    feb_feature_v3_df
    .groupBy(
        "operating_day",
        "operator_id",
        "trip_id",
        "line_id",
        "line_text",
        "transport_clean"
    )
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("bpuic").alias("n_stops"),
        F.min("scheduled_event_ts").alias("first_ts"),
        F.max("scheduled_event_ts").alias("last_ts"),
        F.round(
            (
                F.unix_timestamp(F.max("scheduled_event_ts"))
                - F.unix_timestamp(F.min("scheduled_event_ts"))
            ) / 60.0,
            2
        ).alias("trip_span_minutes"),
        F.max("scheduled_minutes_since_first_observed_stop").alias("max_minutes_since_first")
    )
    .orderBy(F.desc("trip_span_minutes"))
)

trip_span_check_df.show(10, truncate=False)

# %%
bad_trip = (
    trip_span_check_df
    .orderBy(F.desc("trip_span_minutes"))
    .limit(5)
    .collect()[0]
)

bad_operating_day = bad_trip["operating_day"]
bad_operator_id = bad_trip["operator_id"]
bad_trip_id = bad_trip["trip_id"]
bad_line_id = bad_trip["line_id"]

print(bad_operating_day, bad_operator_id, bad_trip_id, bad_line_id)

# %% [markdown]
# ## Get prediction

# %%
from pyspark.ml import PipelineModel
FINAL_MODEL_PATH = f"{base}/trained_models/prod_gbt_model_full_data"
loaded_final_model = PipelineModel.load(FINAL_MODEL_PATH)
print("Final model loaded successfully.")

# %%
feb_pred_df = loaded_final_model.transform(feb_final_features_df)

feb_pred_df.select(
    F.count("*").alias("n_rows"),
    F.sum(F.col("pred_delay").isNull().cast("int")).alias("pred_nulls"),
    F.min("pred_delay").alias("min_pred_delay"),
    F.expr("percentile_approx(pred_delay, 0.5)").alias("median_pred_delay"),
    F.expr("percentile_approx(pred_delay, 0.9)").alias("p90_pred_delay"),
    F.max("pred_delay").alias("max_pred_delay")
).show(truncate=False)

# %%
#FEB_PRED_PATH = f"{base}/route_demo_outputs/predictions/feb_predictions_full.parquet"

# %% [markdown]
# ## Load Saved Evaluation Calibration

# %% [markdown]
# ### Load Config

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

# %% [markdown]
# ### Residual quantiles

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

# %% [markdown]
# ### Load global residual quantiles

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


# %% [markdown]
# ### Define backoff funcitons

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


# %% [markdown]
# ### Add_uncertainty_buckets

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

# %%
spark.stop()

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

BASE = "/user/groups/com-490/H1/final/v1"

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

# %%

# %%
