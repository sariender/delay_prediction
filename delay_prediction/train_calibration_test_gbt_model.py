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
# # Evaluation Calibration GBT Model

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
            .appName(username + '-assignment-2')
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
base

# %%
#You can go to directly Step 3.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Step 1: Data Prep

# %%
from pyspark.sql import functions as F

spark.table("iceberg.sbb.istdaten").agg(
    F.min("operating_day").alias("min_day"),
    F.max("operating_day").alias("max_day"),
).show()


# %%
tse = pd.read_parquet(f"{base}/input_from_data_side/trip_stop_events_norm.parquet")

# %%
tse.head(1).T

# %%
tme  = pd.read_parquet(f"{base}/input_from_data_side/timetable_february.parquet")

# %%
# Continuous 12-month window selected in Step 1: 2024-07-01 through 2025-06-30
START_DATE = "2024-07-01"
END_DATE   = "2026-01-31"
tse = spark.read.parquet(f"{base}/trip_stop_events_norm.parquet")
tse.select("operator_id").distinct().createOrReplaceTempView("operators")
tse.select("bpuic").distinct().createOrReplaceTempView("stops_served")

# %%
# # %%time
base_candidate_df = spark.sql(f"""
    SELECT
        -- Service / trip identifiers
        i.operating_day,
        i.trip_id,

        -- Operator information
        i.operator_id,
        i.operator_abrv,
        i.operator_name,

        -- Line / route information
        i.product_id,
        i.transport,
        i.line_id,
        i.line_text,
        i.circuit_id,

        -- Stop information
        i.bpuic,
        i.stop_name,

        -- Data-quality / operational flags
        i.unplanned,
        i.failed,
        i.transit,

        -- Arrival information
        i.arr_time   AS scheduled_arrival_ts,
        i.arr_actual AS actual_arrival_ts,
        i.arr_status,

        -- Departure information
        -- We kept for diagnostics (when arrival fields are missing)
        i.dep_time   AS scheduled_departure_ts,
        i.dep_actual AS actual_departure_ts,
        i.dep_status

    FROM iceberg.sbb.istdaten i

    -- Restrict to operators identified in Part IIa
    INNER JOIN operators o
        ON i.operator_id = o.operator_id

    -- Restrict to stops served by TL and MBC, identified in Part IIb
    INNER JOIN stops_served ss
        ON i.bpuic = ss.bpuic

    WHERE i.operating_day >= DATE('{START_DATE}')
      AND i.operating_day <  DATE('{END_DATE}')
      AND i.unplanned = FALSE
""")

# %%
base_candidate_df = base_candidate_df.cache()

n_candidate = base_candidate_df.count()
print(f"Candidate dataset size: {n_candidate:,} rows")

base_candidate_df.printSchema()

# %%
# Base data description
base_candidate_df.select(
    F.count("*").alias("n_rows"),
    F.countDistinct("trip_id").alias("n_trips"),
    F.countDistinct("bpuic").alias("n_stops"),
    F.countDistinct("operator_id").alias("n_operators"),
    F.min("operating_day").alias("min_day"),
    F.max("operating_day").alias("max_day")
).show(truncate=False)

# %%
# # %%time
# Missingness summary for every column.
# Helps separate rows where the arrival/departure delay target can be computed
# from rows useful only for diagnostics.
missing_counts_wide_df = base_candidate_df.agg(*[
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in base_candidate_df.columns
])
missing_summary_df = (
    missing_counts_wide_df
    .select(
        F.explode(
            F.array(*[
                F.struct(
                    F.lit(c).alias("column"),
                    F.col(c).alias("missing_count"),
                    F.round(
                        100 * F.col(c) / F.lit(n_candidate),
                        3
                    ).alias("missing_percentage")
                )
                for c in base_candidate_df.columns
            ])
        ).alias("x")
    )
    .select("x.*")
    .orderBy(F.desc("missing_percentage"))
)
missing_summary_df.show(100, truncate=False)

# %%
timestamp_missing_pattern_df = (
    base_candidate_df
    .withColumn("scheduled_arrival_missing", F.col("scheduled_arrival_ts").isNull().cast("int"))
    .withColumn("actual_arrival_missing", F.col("actual_arrival_ts").isNull().cast("int"))
    .withColumn("scheduled_departure_missing", F.col("scheduled_departure_ts").isNull().cast("int"))
    .withColumn("actual_departure_missing", F.col("actual_departure_ts").isNull().cast("int"))
    .groupBy(
        "scheduled_arrival_missing",
        "actual_arrival_missing",
        "scheduled_departure_missing",
        "actual_departure_missing"
    )
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage",
        F.round(100 * F.col("count") / F.lit(n_candidate), 3)
    )
    .orderBy(F.desc("count"))
)

timestamp_missing_pattern_df.show(50, truncate=False)

# %%
base_candidate_check_df = (
    base_candidate_df
    .withColumn(
        "diagnostic_timestamp_pattern",
        F.when(
            F.col("scheduled_arrival_ts").isNotNull()
            & F.col("actual_arrival_ts").isNotNull()
            & F.col("scheduled_departure_ts").isNotNull()
            & F.col("actual_departure_ts").isNotNull(),
            F.lit("arrival_and_departure_available")
        )
        .when(
            F.col("scheduled_arrival_ts").isNotNull()
            & F.col("actual_arrival_ts").isNotNull()
            & F.col("scheduled_departure_ts").isNull()
            & F.col("actual_departure_ts").isNull(),
            F.lit("arrival_only_likely_terminal")
        )
        .when(
            F.col("scheduled_arrival_ts").isNull()
            & F.col("actual_arrival_ts").isNull()
            & F.col("scheduled_departure_ts").isNotNull()
            & F.col("actual_departure_ts").isNotNull(),
            F.lit("departure_only_likely_origin")
        )
        .otherwise(F.lit("other_timestamp_pattern"))
    )
)

timestamp_pattern_summary_df = (
    base_candidate_check_df
    .groupBy("diagnostic_timestamp_pattern")
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage",
        F.round(100 * F.col("count") / F.lit(n_candidate), 3)
    )
    .orderBy(F.desc("count"))
)

timestamp_pattern_summary_df.show(truncate=False)

# %%
base_candidate_check_df = (
    base_candidate_df
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

# %%
scheduled_timestamp_pattern_summary_df = (
    base_candidate_check_df
    .groupBy("scheduled_timestamp_pattern")
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage",
        F.round(100 * F.col("count") / F.lit(n_candidate), 3)
    )
    .orderBy(F.desc("count"))
)

scheduled_timestamp_pattern_summary_df.show(truncate=False)

# %%
structural_patterns_df = base_candidate_check_df.filter(
    F.col("scheduled_timestamp_pattern").isin(
        "scheduled_arrival_only_likely_terminal",
        "scheduled_departure_only_likely_origin"
    )
)

n_structural = structural_patterns_df.count()

structural_status_check_df = (
    structural_patterns_df
    .groupBy(
        "scheduled_timestamp_pattern",
        "arr_status",
        "dep_status",
        "failed",
        "transit"
    )
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage_within_structural_patterns",
        F.round(100 * F.col("count") / F.lit(n_structural), 3)
    )
    .orderBy(
        "scheduled_timestamp_pattern",
        F.desc("count")
    )
)

structural_status_check_df.show(100, truncate=False)

# %%
arrival_delay_by_status_df = (
    arrival_available_df
    .withColumn(
        "arrival_delay_seconds_raw",
        F.unix_timestamp("actual_arrival_ts") - F.unix_timestamp("scheduled_arrival_ts")
    )
)

arrival_delay_by_status_df.groupBy("arr_status") \
    .agg(
        F.count("*").alias("n_rows"),
        F.round(F.avg("arrival_delay_seconds_raw"), 2).alias("mean_delay"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.5)").alias("median_delay"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.9)").alias("p90_delay"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.95)").alias("p95_delay"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.99)").alias("p99_delay"),
        F.min("arrival_delay_seconds_raw").alias("min_delay"),
        F.max("arrival_delay_seconds_raw").alias("max_delay"),
    ) \
    .orderBy(F.desc("n_rows")) \
    .show(truncate=False)

# %%
from pyspark.sql import functions as F

arrival_available_df = (
    base_candidate_check_df
    .filter(F.col("scheduled_arrival_ts").isNotNull())
    .filter(F.col("actual_arrival_ts").isNotNull())
)

arrival_available_df.groupBy("arr_status") \
    .agg(F.count("*").alias("n_rows")) \
    .withColumn(
        "percentage",
        F.round(100 * F.col("n_rows") / arrival_available_df.count(), 3)
    ) \
    .orderBy(F.desc("n_rows")) \
    .show(truncate=False)

# %%
base_df_arrival = (
    base_candidate_check_df
    .filter(F.col("scheduled_arrival_ts").isNotNull())
    .filter(F.col("actual_arrival_ts").isNotNull())
    .filter(F.col("arr_status").isin("REAL", "PROGNOSE"))
)

base_df_arrival = base_df_arrival.withColumn(
    "arrival_delay_seconds_raw",
    F.unix_timestamp("actual_arrival_ts") - F.unix_timestamp("scheduled_arrival_ts")
)

base_df_arrival.select(
    F.count("*").alias("n_arrival_model_rows"),
    F.min("arrival_delay_seconds_raw").alias("min_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.5)").alias("median_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.9)").alias("p90_delay_sec"),
    F.max("arrival_delay_seconds_raw").alias("max_delay_sec")
).show(truncate=False)

# %%
# # %%time
operational_flag_summary_df = (
    base_candidate_check_df
    .groupBy("unplanned", "failed", "transit")
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage",
        F.round(100 * F.col("count") / F.lit(n_candidate), 3)
    )
    .orderBy(F.desc("count"))
)

operational_flag_summary_df.show(50, truncate=False)

# %%
failed_timestamp_pattern_df = (
    base_candidate_check_df
    .groupBy("failed", "scheduled_timestamp_pattern")
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage_total",
        F.round(100 * F.col("count") / F.lit(n_candidate), 3)
    )
    .orderBy("failed", F.desc("count"))
)

failed_timestamp_pattern_df.show(50, truncate=False)

# %%
failed_df = base_candidate_check_df.filter(F.col("failed") == True)
n_failed = failed_df.count()

failed_by_operator_df = (
    failed_df
    .groupBy("operator_abrv", "operator_id", "operator_name")
    .agg(F.count("*").alias("failed_count"))
    .withColumn(
        "failed_percentage",
        F.round(100 * F.col("failed_count") / F.lit(n_failed), 3)
    )
    .orderBy(F.desc("failed_count"))
)

failed_by_operator_df.show(50, truncate=False)
# Among all failed rows, what percentage comes from each operator?

# %%
# Within-operator failed rate (TL has highest volume, so contribution alone is misleading)
failed_rate_by_operator_df = (
    base_candidate_check_df
    .groupBy("operator_abrv", "operator_id", "operator_name")
    .agg(
        F.count("*").alias("total_count"),
        F.sum(F.when(F.col("failed") == True, 1).otherwise(0)).alias("failed_count")
    )
    .withColumn(
        "failed_rate_pct",
        F.round(100 * F.col("failed_count") / F.col("total_count"), 3)
    )
    .orderBy(F.desc("failed_rate_pct"))
)

failed_rate_by_operator_df.show(50, truncate=False)

# %%
failed_rate_by_line_df = (
    base_candidate_check_df
    .filter(F.col("operator_abrv") == "TL")
    .groupBy("operator_abrv", "line_text")
    .agg(
        F.count("*").alias("total_count"),
        F.sum(F.when(F.col("failed") == True, 1).otherwise(0)).alias("failed_count")
    )
    .withColumn(
        "failed_rate_pct",
        F.round(100 * F.col("failed_count") / F.col("total_count"), 3)
    )
    .orderBy(F.desc("failed_rate_pct"))
)

failed_rate_by_line_df.show(50, truncate=False)

# %%
# Apply failed-filtering for the clean delay-model dataset.
base_candidate_clean_df = (
    base_candidate_check_df
    .filter(F.col("failed") == False)
)

n_candidate_clean = base_candidate_clean_df.count()

print(f"Rows before failed-filtering: {n_candidate:,}")
print(f"Rows after failed-filtering:  {n_candidate_clean:,}")
print(f"Rows removed:                {n_candidate - n_candidate_clean:,}")
print(f"Removed percentage:          {100 * (n_candidate - n_candidate_clean) / n_candidate:.3f}%")

# %%
clean_timestamp_pattern_df = (
    base_candidate_clean_df
    .groupBy("scheduled_timestamp_pattern")
    .agg(F.count("*").alias("count"))
    .withColumn(
        "percentage",
        F.round(100 * F.col("count") / F.lit(n_candidate_clean), 3)
    )
    .orderBy(F.desc("count"))
)

clean_timestamp_pattern_df.show(truncate=False)

# %%
# Create arrival-delay modeling dataset.
# This keeps rows where arrival delay can be computed.

base_df_arrival = (
    base_candidate_clean_df
    .filter(F.col("scheduled_arrival_ts").isNotNull())
    .filter(F.col("actual_arrival_ts").isNotNull())
    .filter(F.col("arr_status").isin("REAL", "PROGNOSE"))
)

n_arrival = base_df_arrival.count()

print(f"Clean candidate rows: {n_candidate_clean:,}")
print(f"Arrival-delay rows:   {n_arrival:,}")
print(f"Kept percentage:      {100 * n_arrival / n_candidate_clean:.3f}%")

# %%
base_df_arrival.groupBy("scheduled_timestamp_pattern") \
    .agg(F.count("*").alias("count")) \
    .withColumn(
        "percentage",
        F.round(100 * F.col("count") / F.lit(n_arrival), 3)
    ) \
    .orderBy(F.desc("count")) \
    .show(truncate=False)

# %%
# Raw arrival-delay target in seconds
base_df_arrival = base_df_arrival.withColumn(
    "arrival_delay_seconds_raw",
    F.unix_timestamp("actual_arrival_ts") - F.unix_timestamp("scheduled_arrival_ts")
)

# Approximate quantiles
arrival_delay_summary_df = base_df_arrival.select(
    F.count("*").alias("n_rows"),
    F.min("arrival_delay_seconds_raw").alias("min_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.001)").alias("p001"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.01)").alias("p01"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.05)").alias("p05"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.5)").alias("median"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.9)").alias("p90"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.95)").alias("p95"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.99)").alias("p99"),
    F.expr("percentile_approx(arrival_delay_seconds_raw, 0.999)").alias("p999"),
    F.max("arrival_delay_seconds_raw").alias("max_delay_sec")
)

arrival_delay_summary_df.show(truncate=False)

# %%
arrival_delay_thresholds_df = base_df_arrival.agg(
    F.count("*").alias("n_rows"),

    # ±10 minutes
    F.sum(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 600, 1).otherwise(0)).alias("n_abs_gt_10min"),
    F.round(
        100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 600, 1).otherwise(0)),
        3
    ).alias("pct_abs_gt_10min"),

    # ±30 minutes
    F.sum(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 1800, 1).otherwise(0)).alias("n_abs_gt_30min"),
    F.round(
        100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 1800, 1).otherwise(0)),
        3
    ).alias("pct_abs_gt_30min"),

    # ±1 hour
    F.sum(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 3600, 1).otherwise(0)).alias("n_abs_gt_1h"),
    F.round(
        100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 3600, 1).otherwise(0)),
        3
    ).alias("pct_abs_gt_1h"),

    # ±2 hours
    F.sum(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 7200, 1).otherwise(0)).alias("n_abs_gt_2h"),
    F.round(
        100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 7200, 1).otherwise(0)),
        3
    ).alias("pct_abs_gt_2h")
)

arrival_delay_thresholds_df.show(truncate=False)

# %% [markdown]
# ±10 min affects ~2.15% (too aggressive — many real delays); ±30 min affects 0.154%; ±1 h affects 0.037%. Use **±1 h as a conservative cap**.

# %%
arrival_delay_extremes_direction_df = base_df_arrival.agg(
    F.sum(F.when(F.col("arrival_delay_seconds_raw") < -600, 1).otherwise(0)).alias("n_early_gt_10min"),
    F.sum(F.when(F.col("arrival_delay_seconds_raw") > 600, 1).otherwise(0)).alias("n_late_gt_10min"),

    F.sum(F.when(F.col("arrival_delay_seconds_raw") < -1800, 1).otherwise(0)).alias("n_early_gt_30min"),
    F.sum(F.when(F.col("arrival_delay_seconds_raw") > 1800, 1).otherwise(0)).alias("n_late_gt_30min"),

    F.sum(F.when(F.col("arrival_delay_seconds_raw") < -3600, 1).otherwise(0)).alias("n_early_gt_1h"),
    F.sum(F.when(F.col("arrival_delay_seconds_raw") > 3600, 1).otherwise(0)).alias("n_late_gt_1h"),

    F.sum(F.when(F.col("arrival_delay_seconds_raw") < -7200, 1).otherwise(0)).alias("n_early_gt_2h"),
    F.sum(F.when(F.col("arrival_delay_seconds_raw") > 7200, 1).otherwise(0)).alias("n_late_gt_2h")
)

arrival_delay_extremes_direction_df.show(truncate=False)

# %%
# Add basic calendar features for diagnostics
base_df_arrival_diag = (
    base_df_arrival
    .withColumn("year", F.year("operating_day"))
    .withColumn("month", F.month("operating_day"))
    .withColumn("day_of_week", F.dayofweek("operating_day"))
    .withColumn("hour_of_day", F.hour("scheduled_arrival_ts"))
)

# Delay by month
delay_by_month_df = (
    base_df_arrival_diag
    .groupBy("year", "month")
    .agg(
        F.count("*").alias("n_rows"),
        F.round(F.avg("arrival_delay_seconds_raw"), 2).alias("mean_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.5)").alias("median_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.9)").alias("p90_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.95)").alias("p95_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.99)").alias("p99_delay_sec"),
        F.round(
            100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 600, 1).otherwise(0)),
            3
        ).alias("abs_delay_gt_10min_pct"),
        F.round(
            100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 3600, 1).otherwise(0)),
            3
        ).alias("abs_delay_gt_1h_pct")
    )
    .orderBy("year", "month")
)

delay_by_month_df.show(truncate=False)

# %%
delay_by_hour_df = (
    base_df_arrival_diag
    .groupBy("hour_of_day")
    .agg(
        F.count("*").alias("n_rows"),
        F.round(F.avg("arrival_delay_seconds_raw"), 2).alias("mean_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.5)").alias("median_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.9)").alias("p90_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.95)").alias("p95_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.99)").alias("p99_delay_sec"),
        F.round(
            100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 600, 1).otherwise(0)),
            3
        ).alias("abs_delay_gt_10min_pct"),
        F.round(
            100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 3600, 1).otherwise(0)),
            3
        ).alias("abs_delay_gt_1h_pct")
    )
    .orderBy("hour_of_day")
)

delay_by_hour_df.show(30, truncate=False)

# %%
delay_by_day_of_week_df = (
    base_df_arrival_diag
    .groupBy("day_of_week")
    .agg(
        F.count("*").alias("n_rows"),
        F.round(F.avg("arrival_delay_seconds_raw"), 2).alias("mean_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.5)").alias("median_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.9)").alias("p90_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.95)").alias("p95_delay_sec"),
        F.expr("percentile_approx(arrival_delay_seconds_raw, 0.99)").alias("p99_delay_sec"),
        F.round(
            100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 600, 1).otherwise(0)),
            3
        ).alias("abs_delay_gt_10min_pct"),
        F.round(
            100 * F.avg(F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 3600, 1).otherwise(0)),
            3
        ).alias("abs_delay_gt_1h_pct")
    )
    .orderBy("day_of_week")
)

delay_by_day_of_week_df.show(20, truncate=False)

# %% [markdown]
# Sun/Mon have lower delays; Tue–Fri higher; Saturday has the highest >10-min share. Supports keeping `day_of_week` and `is_weekend`.

# %%
# Final clean arrival-delay dataset (consolidates all earlier filters)
base_df_arrival = (
    base_candidate_check_df
    .filter(F.col("failed") == False)
    .filter(F.col("scheduled_arrival_ts").isNotNull())
    .filter(F.col("actual_arrival_ts").isNotNull())
    .filter(F.col("arr_status").isin("REAL", "PROGNOSE"))
    .filter(
        F.col("diagnostic_timestamp_pattern").isin(
            "arrival_and_departure_available",
            "arrival_only_likely_terminal"
        )
    )
)

# Raw arrival-delay target
base_df_arrival = base_df_arrival.withColumn(
    "arrival_delay_seconds_raw",
    F.unix_timestamp("actual_arrival_ts") - F.unix_timestamp("scheduled_arrival_ts")
)

# Capped target (±1 h) for the first regression model
base_df_arrival = (
    base_df_arrival
    .withColumn(
        "arrival_delay_seconds",
        F.least(
            F.greatest(F.col("arrival_delay_seconds_raw"), F.lit(-3600)),
            F.lit(3600)
        )
    )
    .withColumn(
        "arrival_delay_minutes",
        F.col("arrival_delay_seconds") / 60.0
    )
    .withColumn(
        "arrival_delay_was_capped",
        F.when(F.abs(F.col("arrival_delay_seconds_raw")) > 3600, 1).otherwise(0)
    )
)

base_df_arrival = base_df_arrival.cache()

n_arrival = base_df_arrival.count()

print(f"Final arrival-delay dataset size: {n_arrival:,} rows")
print(f"Share of clean candidate rows kept: {100 * n_arrival / n_candidate_clean:.3f}%")

base_df_arrival.printSchema()

# %%
base_df_arrival.select(
    F.count("*").alias("n_rows"),
    F.min("arrival_delay_seconds").alias("min_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds, 0.5)").alias("median_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds, 0.9)").alias("p90_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds, 0.95)").alias("p95_delay_sec"),
    F.expr("percentile_approx(arrival_delay_seconds, 0.99)").alias("p99_delay_sec"),
    F.max("arrival_delay_seconds").alias("max_delay_sec"),
    F.sum("arrival_delay_was_capped").alias("n_capped"),
    F.round(100 * F.avg("arrival_delay_was_capped"), 3).alias("capped_percentage")
).show(truncate=False)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Step 2: Feature engineering

# %% [markdown]
# #### 2.1 Baseline raw input (`baseline_raw_input_df`)
#
# Reference snapshot using only raw/static fields. Categorical encoding will happen later in the ML pipeline; here we just snapshot the candidate columns.

# %%
# # %%time

baseline_raw_input_df = base_df_arrival.select(
    # Kept for time-based splitting and diagnostics, not necessarily as model features
    "operating_day",
    "scheduled_arrival_ts",

    # Raw/static candidate predictors
    "operator_id",
    "operator_abrv",
    "operator_name",
    "product_id",
    "transport",
    "line_id",
    "line_text",
    "circuit_id",
    "bpuic",
    "diagnostic_timestamp_pattern",
    "scheduled_timestamp_pattern",

    # Target
    "arrival_delay_seconds"
)

baseline_raw_input_df = baseline_raw_input_df.cache()

n_baseline_raw = baseline_raw_input_df.count()
print(f"Raw baseline input dataset size: {n_baseline_raw:,} rows")

baseline_raw_input_df.printSchema()

# %%
baseline_raw_missing_df = baseline_raw_input_df.agg(*[
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in baseline_raw_input_df.columns
])

baseline_raw_missing_summary_df = (
    baseline_raw_missing_df
    .select(
        F.explode(
            F.array(*[
                F.struct(
                    F.lit(c).alias("column"),
                    F.col(c).alias("missing_count"),
                    F.round(100 * F.col(c) / F.lit(n_baseline_raw), 3).alias("missing_percentage")
                )
                for c in baseline_raw_input_df.columns
            ])
        ).alias("x")
    )
    .select("x.*")
    .orderBy(F.desc("missing_percentage"))
)

baseline_raw_missing_summary_df.show(100, truncate=False)

# %% [markdown]
# #### 2.2 Calendar and scheduled-time features (`feature_v1_df`)
#
# Calendar features derived from `operating_day` and `scheduled_arrival_ts` — pre-event, no leakage.
# Includes month, day of week, weekend flag, hour, minute, minute-of-day, and peak-period indicators.

# %%
# # %%time

feature_v1_df = (
    base_df_arrival

    # Calendar features from service day
    .withColumn("year", F.year("operating_day"))
    .withColumn("month", F.month("operating_day"))
    .withColumn("day_of_week", F.dayofweek("operating_day"))  # Spark: 1=Sunday, 7=Saturday
    .withColumn(
        "is_weekend",
        F.when(F.dayofweek("operating_day").isin(1, 7), 1).otherwise(0)
    )

    # Scheduled-arrival time features
    .withColumn("hour_of_day", F.hour("scheduled_arrival_ts"))
    .withColumn("minute_of_hour", F.minute("scheduled_arrival_ts"))
    .withColumn(
        "scheduled_minute_of_day",
        F.hour("scheduled_arrival_ts") * 60 + F.minute("scheduled_arrival_ts")
    )

    # Peak-period indicators from scheduled arrival time
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

feature_v1_df = feature_v1_df.cache()

n_feature_v1 = feature_v1_df.count()
print(f"Feature v1 dataset size: {n_feature_v1:,} rows")

feature_v1_df.select(
    "operating_day",
    "scheduled_arrival_ts",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_of_day",
    "minute_of_hour",
    "scheduled_minute_of_day",
    "is_morning_peak",
    "is_evening_peak",
    "is_late_night",
    "arrival_delay_seconds"
).show(5, truncate=False)

# %%
feature_v1_cols = [
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_of_day",
    "minute_of_hour",
    "scheduled_minute_of_day",
    "is_morning_peak",
    "is_evening_peak",
    "is_late_night"
]

feature_v1_df.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in feature_v1_cols
]).show(truncate=False)

# %% [markdown]
# #### 2.3 Cleaned transport type (`feature_v2_df`)
#
# `product_id` and `transport` describe service type at different granularities. Combine into a single standardized `transport_clean` so detailed labels (IC, IR, RE, S, …) aren't treated as unrelated categories.

# %%
feature_v2_df = (
    feature_v1_df
    .withColumn(
        "transport_clean",
        F.when(
            F.upper(F.col("product_id")).like("%BUS%") | (F.upper(F.col("product_id")) == "BUA"),
            F.lit("Bus")
        )
        .when(F.col("product_id") == "Zug", F.lit("Zug"))
        .when(F.col("product_id") == "Tram", F.lit("Tram"))
        .when(F.col("product_id") == "Metro", F.lit("Metro"))
        .when(F.col("product_id") == "Schiff", F.lit("Schiff"))
        .when(F.col("product_id") == "Zahnradbahn", F.lit("Zahnradbahn"))
        .when(F.col("product_id") == "Taxi", F.lit("Taxi"))
        .when(
            F.upper(F.col("transport")).isin(
                "IC", "ICE", "IR", "IRE", "RE", "R", "S", "SN",
                "TGV", "EC", "RJX", "TER", "PE", "EXT", "NJ"
            ),
            F.lit("Zug")
        )
        .when(F.upper(F.col("transport")) == "BAT", F.lit("Schiff"))
        .when(F.upper(F.col("transport")) == "CC", F.lit("Zahnradbahn"))
        .when(F.upper(F.col("transport")) == "M", F.lit("Metro"))
        .when(F.upper(F.col("transport")).isin("T", "TN"), F.lit("Tram"))
        .otherwise(F.lit("Other"))
    )
)
feature_v2_df.groupBy("transport_clean") \
    .agg(F.count("*").alias("count")) \
    .orderBy(F.desc("count")) \
    .show(truncate=False)

# %%
feature_v2_df.filter(F.col("transport_clean") == "Other") \
    .groupBy("product_id", "transport") \
    .agg(F.count("*").alias("count")) \
    .orderBy(F.desc("count")) \
    .show(50, truncate=False)

# %% [markdown]
# Restricted sample contains only `Bus` and `Metro`; no rows fall into `Other`. Kept as a model feature because bus and metro may have different delay behavior.

# %% [markdown]
# #### 2.4 Observed stop-sequence features (`feature_v3_df`)
#
# The official timetable has `stop_sequence`, but the timetable `trip_id` does not directly match `istdaten.trip_id`. The Part III matching covers only ~5.1% of trip IDs in the modeling sample (verified below), so we build an observed stop-sequence proxy from `istdaten`: order stops within each `(trip_id, operating_day)` by scheduled event time. Useful as a proxy for delay accumulation along a trip.

# %%
TRIP_GROUP_COLS = [
    "operating_day",
    "operator_id",
    "trip_id",
    "line_id",
]

feature_v3_base_df = feature_v2_df.withColumn(
    "scheduled_event_ts",
    F.coalesce(
        F.col("scheduled_arrival_ts"),
        F.col("scheduled_departure_ts")
    )
)

# One row per scheduled stop event
scheduled_stop_events_df = (
    feature_v3_base_df
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

feature_v3_df = (
    feature_v3_base_df
    .join(
        scheduled_stop_features_df,
        on=TRIP_GROUP_COLS + ["scheduled_event_ts", "bpuic"],
        how="left"
    )
    .cache()
)

n_feature_v3 = feature_v3_df.count()
print(f"Feature v3 dataset size: {n_feature_v3:,} rows")

# %%
feature_v3_df.select(
    "trip_id",
    "operating_day",
    "bpuic",
    "stop_name",
    "scheduled_event_ts",
    "stop_sequence_idx",
    "n_stops_in_trip_observed",
    "relative_stop_position",
    "is_first_observed_stop",
    "is_last_observed_stop"
).orderBy(
    "trip_id",
    "operating_day",
    "stop_sequence_idx"
).show(30, truncate=False)

# %%
feature_v3_df.select(
    F.sum(F.when(F.col("scheduled_event_ts").isNull(), 1).otherwise(0)).alias("missing_scheduled_event_ts"),
    F.sum(F.when(F.col("stop_sequence_idx").isNull(), 1).otherwise(0)).alias("missing_stop_sequence_idx"),
    F.sum(F.when(F.col("relative_stop_position").isNull(), 1).otherwise(0)).alias("missing_relative_stop_position")
).show(truncate=False)

# %%
TRIP_GROUP_COLS = [
    "operating_day",
    "operator_id",
    "trip_id",
    "line_id",
]

trip_summary_window = Window.partitionBy(*TRIP_GROUP_COLS)

feature_v3_df = (
    feature_v3_df
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
feature_v3_df.select(
    "operating_day",
    "operator_id",
    "trip_id",
    "line_id",
    "bpuic",
    "stop_name",
    "scheduled_event_ts",
    "first_scheduled_event_ts",
    "stop_sequence_idx",
    "n_stops_in_trip_observed",
    "relative_stop_position_v2",
    "scheduled_minutes_since_first_observed_stop",
    "is_first_observed_stop",
    "is_last_observed_stop"
).orderBy(
    "operating_day",
    "operator_id",
    "trip_id",
    "line_id",
    "stop_sequence_idx"
).show(30, truncate=False)

# %%
#feature_v3_df.write.mode("overwrite").parquet(f"{base}/full_data_w_o_hist_weather_specialdays.parquet")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Step 3 — Train/test split

# %%
FEATURE_V3_PATH = f"{base}/features_and_training_data/full_data_w_o_hist_weather_specialdays.parquet"
feature_v3_df = spark.read.parquet(FEATURE_V3_PATH)
print(feature_v3_df.count())

# %%
from pyspark.sql import functions as F

TEST_DAYS = 28
CALIB_MONTHS = 3

max_date = (
    feature_v3_df
    .agg(F.max("operating_day").alias("max_date"))
    .collect()[0]["max_date"]
)

test_start = (
    spark.range(1)
    .select(F.date_sub(F.lit(max_date), TEST_DAYS).alias("test_start"))
    .collect()[0]["test_start"]
)

calib_start = (
    spark.range(1)
    .select(
        F.add_months(
            F.trunc(F.lit(test_start), "MM"),
            -CALIB_MONTHS
        ).alias("calib_start")
    )
    .collect()[0]["calib_start"]
)

print("max_date:", max_date)
print("calib_start:", calib_start)
print("test_start:", test_start)

train_base_df = feature_v3_df.filter(
    F.col("operating_day") < F.lit(calib_start)
)

calib_base_df = feature_v3_df.filter(
    (F.col("operating_day") >= F.lit(calib_start)) &
    (F.col("operating_day") < F.lit(test_start))
)

test_base_df = feature_v3_df.filter(
    F.col("operating_day") >= F.lit(test_start)
)

for name, df in [
    ("train", train_base_df),
    ("calib", calib_base_df),
    ("test", test_base_df),
]:
    print(name)
    df.agg(
        F.count("*").alias("n_rows"),
        F.min("operating_day").alias("min_day"),
        F.max("operating_day").alias("max_day"),
        F.countDistinct("operating_day").alias("n_days")
    ).show(truncate=False)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Step 4: Historical delay features
#
# Group-level delay summaries computed from the training set only and joined to both train and test. Missing groups in the test set are filled with global training stats.

# %% [markdown]
# #### Stats Function

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


# %% [markdown]
# #### Historical Tables Function

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
hist_tables_train = make_all_hist_tables(train_base_df)
fill_values_train = make_hist_fill_values(train_base_df)

train_features_df = (
    add_historical_features(train_base_df, hist_tables_train)
    .fillna(fill_values_train)
)

calib_features_df = (
    add_historical_features(calib_base_df, hist_tables_train)
    .fillna(fill_values_train)
)

print("train features:", train_features_df.count())
print("calib features:", calib_features_df.count())

# %%
train_calib_base_df = train_base_df.unionByName(calib_base_df)

hist_tables_train_calib = make_all_hist_tables(train_calib_base_df)
fill_values_train_calib = make_hist_fill_values(train_calib_base_df)

test_features_df = (
    add_historical_features(test_base_df, hist_tables_train_calib)
    .fillna(fill_values_train_calib)
)

print("test features:", test_features_df.count())

# %% [markdown]
# ### Step 5: Weather features (Open-Meteo)
#
# Weather is sourced from [Open-Meteo](https://open-meteo.com/) and joined onto each scheduled stop event. To avoid leakage we never use realized future weather. We consider only forecasts available *before* the event:
#
# - **D−1 forecast** (main advance-planning feature, one day ahead) — Previous Runs API
# - **D−2 forecast** (earlier forecast, used to derive a *forecast revision* signal `d1 − d2`) — Previous Runs API
# - **Hour−1 observed** (short-notice proxy: what the realized weather looked like one hour before the event) — Archive API
#
# **Implementation.** All API calls and the join logic live in [`weather_extraction.py`](weather_extraction.py).
#
# We cached parquets live in the shared group folder
# `/user/groups/com-490/{groupName}/assignment-2/`, so any group member can re-run the notebook end-to-end without hitting the API. We used the following code:
# ```python
# from weather_extraction import fetch_forecast_weather, fetch_observed_weather
#
# FORECAST_CACHE_PATH = f"/user/groups/com-490/{groupName}/assignment-2/open_meteo_lausanne_previous_day1_day2_2024_07_2025_06.parquet"
# OBS_CACHE_PATH      = f"/user/groups/com-490/{groupName}/assignment-2/open_meteo_lausanne_observed_hourly_2024_07_2025_06.parquet"
#
# # Fetch forecast weather (D-1 and D-2 archived forecasts)
# forecast_df = fetch_forecast_weather(
#     spark=spark,
#     cache_path=FORECAST_CACHE_PATH,
#     start_date="2024-07-01",
#     end_date="2025-06-30",
# )
#
# # Fetch observed weather (hourly observations)
# observed_df = fetch_observed_weather(
#     spark=spark,
#     cache_path=OBS_CACHE_PATH,
#     start_date="2024-07-01",
#     end_date="2025-06-30",
# )
# ```
#

# %%
from weather_extraction import load_cached_weather, enrich_with_weather, WEATHER_FEATURE_COLS

FORECAST_CACHE_PATH =  f"{base}/forecast_weather_2024_07_2026_01.parquet"
OBS_CACHE_PATH      =  f"{base}/open_meteo_lausanne_observed_hourly_2024_07_2026_01.parquet"

forecast_df, observed_weather_df = load_cached_weather(spark, FORECAST_CACHE_PATH, OBS_CACHE_PATH)

print(f"Forecast rows: {forecast_df.count():,}")
print(f"Observed weather rows: {observed_weather_df.count():,}")

# %%
train_weather_enriched_df = enrich_with_weather(train_features_df,forecast_df,observed_weather_df)
calib_weather_enriched_df = enrich_with_weather(calib_features_df,forecast_df,observed_weather_df)
test_weather_enriched_df = enrich_with_weather(test_features_df,forecast_df,observed_weather_df)

print(f"Train weather-enriched rows: {train_weather_enriched_df.count():,}")
print(f"Calib weather-enriched rows: {calib_weather_enriched_df.count():,}")
print(f"Test weather-enriched rows:  {test_weather_enriched_df.count():,}")

# %%
print("train before:", train_features_df.count())
print("train after: ", train_weather_enriched_df.count())

print("calib before:", calib_features_df.count())
print("calib after: ", calib_weather_enriched_df.count())

print("test before:", test_features_df.count())
print("test after: ", test_weather_enriched_df.count())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Step 6: Special-day and event features
#
# Demand and traffic conditions can shift on public holidays, school holidays, and major
# local events. We use two small reference tables:
#
# - **Public holidays + Lausanne-region events** — official Vaud calendar plus manually
#   validated event dates (festivals, marathons, the 2025 Swiss Gymnastics Festival, etc.),
#   each tagged with a coarse intensity (1 = standard, 2 = large, 3 = major multi-day).
# - **Vaud school holiday ranges** — joined as a date-range lookup.
#
# Both tables and the join logic live in [`special_days.py`](special_days.py). Sources are
# manually curated rather than scraped. 
# Output: `train_final_features_df` / `test_final_features_df` with the new columns
# `is_special_day`, `special_day_name`, `special_day_type`, `special_day_intensity`,
# `is_school_holiday`, `school_holiday_name`.

# %%
from special_days import build_special_day_tables, add_special_day_features

special_days_df, school_holiday_ranges_df = build_special_day_tables(spark)

train_final_features_df = add_special_day_features(
    train_weather_enriched_df,
    special_days_df,
    school_holiday_ranges_df
)

calib_final_features_df = add_special_day_features(
    calib_weather_enriched_df,
    special_days_df,
    school_holiday_ranges_df
)
test_final_features_df = add_special_day_features(
    test_weather_enriched_df,
    special_days_df,
    school_holiday_ranges_df
)

print(f"Train final rows: {train_final_features_df.count():,}")
print(f"Calib final rows: {calib_final_features_df.count():,}")
print(f"Test final rows:  {test_final_features_df.count():,}")

# %%
print("train before:", train_weather_enriched_df.count())
print("train after: ", train_final_features_df.count())

print("calib before:", calib_weather_enriched_df.count())
print("calib after: ", calib_final_features_df.count())

print("test before:", test_weather_enriched_df.count())
print("test after: ", test_final_features_df.count())

# %%
from pyspark.sql import functions as F

for name, df in [
    ("train", train_final_features_df),
    ("calib", calib_final_features_df),
    ("test", test_final_features_df),
]:
    print(name)
    df.groupBy(
        "is_special_day",
        "special_day_type",
        "is_school_holiday"
    ).agg(
        F.count("*").alias("n_rows")
    ).orderBy(F.desc("n_rows")).show(30, truncate=False)


# %%
def add_special_bucket(df):
    return (
        df
        .withColumn(
            "special_bucket",
            F.when(F.col("special_day_type") == "public_holiday", F.lit("public_holiday"))
             .when(F.col("is_school_holiday") == 1, F.lit("school_holiday"))
             .otherwise(F.lit("normal_day"))
        )
    )


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Step 7: Final model feature definitions
#
# Group columns into label, identifiers (kept for diagnostics, not used as features), categorical features, numeric features, and excluded columns (target leakage / redundant identifiers / raw timestamps).

# %%
bad_obs_cols = [
    c for c in model_input_cols
    if c.endswith("_obs") and not c.endswith("_n_obs")
]

print("Suspicious observed columns:", bad_obs_cols)

# %%
model_input_cols = NUMERIC_COLS + CATEGORICAL_COLS

bad_model_cols = [c for c in EXCLUDE_COLS if c in model_input_cols]

print("Excluded columns used as model inputs:", bad_model_cols)

# %%
TRAIN_FEATURES_PATH = f"{base}/calibration_model_data/feature_splits/train_final_features_df.parquet"
CALIB_FEATURES_PATH = f"{base}/calibration_model_data/feature_splits/calib_final_features_df.parquet"
TEST_FEATURES_PATH  = f"{base}/calibration_model_data/feature_splits/test_final_features_df.parquet"

train_final_features_df = spark.read.parquet(TRAIN_FEATURES_PATH)
calib_final_features_df = spark.read.parquet(CALIB_FEATURES_PATH)
test_final_features_df  = spark.read.parquet(TEST_FEATURES_PATH)

# %%
from pyspark.sql import functions as F

LABEL_COL = "arrival_delay_seconds"

for name, df in [
    ("train", train_final_features_df),
    ("calib", calib_final_features_df),
    ("test", test_final_features_df),
]:
    print(name)
    df.select(
        F.count("*").alias("n_rows"),
        F.sum(F.col(LABEL_COL).isNull().cast("int")).alias("label_nulls"),
        F.min(LABEL_COL).alias("label_min"),
        F.expr(f"percentile_approx({LABEL_COL}, 0.5)").alias("label_median"),
        F.expr(f"percentile_approx({LABEL_COL}, 0.9)").alias("label_p90"),
        F.expr(f"percentile_approx({LABEL_COL}, 0.99)").alias("label_p99"),
        F.max(LABEL_COL).alias("label_max"),
        F.round(F.avg(LABEL_COL), 2).alias("label_mean"),
    ).show(truncate=False)

# %%
LABEL_COL = "arrival_delay_seconds"

MODEL_NUMERIC_COLS = NUMERIC_COLS
MODEL_CATEGORICAL_COLS = CATEGORICAL_COLS

# Safety check
model_input_cols = MODEL_NUMERIC_COLS + MODEL_CATEGORICAL_COLS

for name, df in [
    ("train", train_final_features_df),
    ("calib", calib_final_features_df),
    ("test", test_final_features_df),
]:
    missing_cols = [c for c in model_input_cols + [LABEL_COL] if c not in df.columns]
    print(name, "missing:", missing_cols)

# %% [markdown]
# ## Model Building

# %%
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor, DecisionTreeRegressor
from model_pipeline import build_pipeline, evaluate
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# %%
model_input_cols = NUMERIC_COLS + CATEGORICAL_COLS

bad_model_cols = [c for c in EXCLUDE_COLS if c in model_input_cols]
print("Excluded columns used as model inputs:", bad_model_cols)

# %%
feature_cols = MODEL_NUMERIC_COLS + MODEL_CATEGORICAL_COLS

# %%
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=LABEL_COL,
    predictionCol="pred_delay",
    maxDepth=8,
    maxBins=256,
    maxIter=100,
    stepSize=0.05,
    seed=42
)

pipe = build_pipeline(gbt,MODEL_NUMERIC_COLS,MODEL_CATEGORICAL_COLS)

print("Training GBT on train set...")
calib_model = pipe.fit(train_final_features_df)
print("Done.")

# %%
#load from cache:
from pyspark.ml import PipelineModel
EVAL_CALIB_MODEL_PATH = f"{base}/calibration_model_data/models/eval_calib_gbt_model"
calib_model = PipelineModel.load(EVAL_CALIB_MODEL_PATH)
print("Loaded calibration model.")

# %%
#calib_pred_df = (
 #   calib_model
  #  .transform(calib_final_features_df)
   # .withColumn("residual", F.col(LABEL_COL) - F.col("pred_delay"))
#)

#calib_pred_df.write.mode("overwrite").parquet(CALIB_PRED_PATH)


# %%
EVAL_CALIB_PRED_PATH = (
    f"{base}/calibration_model_data/evaluation_outputs/"
    "eval_calib_predictions_full.parquet"
)

calib_pred_df = spark.read.parquet(EVAL_CALIB_PRED_PATH)

print(calib_pred_df.count())
print("calib prediction cols:", len(calib_pred_df.columns))

# %%
calib_pred_df.select(
    F.count("*").alias("n_rows"),
    F.round(F.avg("residual"), 2).alias("mean_residual"),
    F.expr("percentile_approx(residual, 0.01)").alias("q01_residual"),
    F.expr("percentile_approx(residual, 0.05)").alias("q05_residual"),
    F.expr("percentile_approx(residual, 0.5)").alias("median_residual"),
    F.expr("percentile_approx(residual, 0.9)").alias("q90_residual"),
    F.expr("percentile_approx(residual, 0.95)").alias("q95_residual"),
    F.expr("percentile_approx(residual, 0.99)").alias("q99_residual"),
).show(truncate=False)

# %%
stop_hub_df = (
    train_base_df
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

stop_hub_df.groupBy("stop_hub_bucket").count().show()


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
calib_pred_grouped_df = (
    calib_pred_df
    .join(stop_hub_df, on="bpuic", how="left")
    .fillna({"stop_hub_bucket": "unknown_hub"})
)

calib_pred_grouped_df = (
    add_uncertainty_buckets(calib_pred_grouped_df))

print("rows:", calib_pred_grouped_df.count())

# %%
GROUP_LEVELS = {
    "L6_line_specific": [
        "pred_delay_bucket",
        "line_text",
        "transport_clean",
        "time_bucket",
        "day_bucket",
        "weather_bucket",
        "forecast_instability_bucket",
        "trip_stage_bucket",
        "stop_hub_bucket",
        "special_bucket",
    ],

    "L5_ultra_rich": [
        "pred_delay_bucket",
        "transport_clean",
        "time_bucket",
        "day_bucket",
        "weather_bucket",
        "forecast_instability_bucket",
        "line_risk_bucket",
        "trip_stage_bucket",
        "stop_hub_bucket",
        "special_bucket",
    ],

    "L4_rich": [
        "pred_delay_bucket",
        "transport_clean",
        "time_bucket",
        "day_bucket",
        "weather_bucket",
        "line_risk_bucket",
        "trip_stage_bucket",
        "stop_hub_bucket",
    ],

    "L3_operational": [
        "pred_delay_bucket",
        "transport_clean",
        "time_bucket",
        "weather_bucket",
        "line_risk_bucket",
        "trip_stage_bucket",
    ],

    "L2_context": [
        "pred_delay_bucket",
        "transport_clean",
        "time_bucket",
        "weather_bucket",
        "line_risk_bucket",
    ],

    "L1_core": [
        "pred_delay_bucket",
        "transport_clean",
        "time_bucket",
        "weather_bucket",
    ],

    "L0_simple": [
        "pred_delay_bucket",
        "transport_clean",
    ],
}

# %%
ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99]
MIN_GROUP_ROWS = 1000

config_rows = []

for level_name, group_cols in GROUP_LEVELS.items():
    config_rows.append((
        level_name,
        ",".join(group_cols),
        ",".join([str(a) for a in ALPHAS]),
        int(MIN_GROUP_ROWS)
    ))

config_df = spark.createDataFrame(
    config_rows,
    ["level_name", "group_cols_csv", "alphas_csv", "min_group_rows"]
)

# %%
CONFIG_PATH = (
    f"{base}/calibration_model_data/calibration_artifacts/"
    "calibration_config.parquet"
)

config_df = spark.read.parquet(CONFIG_PATH)

config_df.show(truncate=False)


# %%
def group_size_summary(df, group_cols, name):
    g = (
        df
        .withColumn("calib_group", F.concat_ws("__", *group_cols))
        .groupBy("calib_group")
        .agg(F.count("*").alias("n_rows"))
    )

    print(f"\n{name}")
    g.select(
        F.count("*").alias("n_groups"),
        F.min("n_rows").alias("min_group"),
        F.expr("percentile_approx(n_rows, 0.1)").alias("p10_group"),
        F.expr("percentile_approx(n_rows, 0.5)").alias("median_group"),
        F.expr("percentile_approx(n_rows, 0.9)").alias("p90_group"),
        F.max("n_rows").alias("max_group"),
    ).show(truncate=False)

    print("Smallest groups:")
    g.orderBy("n_rows").show(20, truncate=False)


for level_name, group_cols in GROUP_LEVELS.items():
    group_size_summary(calib_pred_grouped_df, group_cols, level_name)

# %%
MIN_GROUP_ROWS = 1000

ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.99]

def alpha_to_name(alpha):
    return f"q{int(round(alpha * 100)):02d}"


def make_group_residual_table(df, level_name, group_cols):
    group_key_col = f"{level_name}_group"

    df_with_key = df.withColumn(
        group_key_col,
        F.concat_ws("__", *group_cols)
    )

    exprs = [F.count("*").alias(f"{level_name}_n_rows")]

    for alpha in ALPHAS:
        qname = alpha_to_name(alpha)
        exprs.append(
            F.expr(
                f"percentile_approx(residual, {alpha}, 10000)"
            ).alias(f"{level_name}_residual_{qname}")
        )

    return (
        df_with_key
        .groupBy(group_key_col)
        .agg(*exprs)
    )


# %%
group_residual_tables = {}

for level_name, group_cols in GROUP_LEVELS.items():
    print("building:", level_name)

    group_residual_tables[level_name] = (
        make_group_residual_table(
            calib_pred_grouped_df,
            level_name,
            group_cols
        )
        .cache()
    )

    print(level_name, group_residual_tables[level_name].count())

# %%
global_residual_quantiles = {}

for alpha in ALPHAS:
    qname = alpha_to_name(alpha)

    residual_q = (
        calib_pred_grouped_df
        .agg(
            F.expr(f"percentile_approx(residual, {alpha}, 10000)").alias("q")
        )
        .collect()[0]["q"]
    )

    global_residual_quantiles[qname] = float(residual_q)

global_residual_quantiles

# %%
#CALIB_TABLES_PATH = f"{base}/calibration_residual_tables"

#for level_name, df in group_residual_tables.items():
   # out_path = f"{CALIB_TABLES_PATH}/{level_name}.parquet"
   # df.write.mode("overwrite").parquet(out_path)
   # print("Saved:", out_path)

# %%
#This is our fallback when no group level has enough calibration rows.
global_quantiles_rows = [
    (q, float(v))
    for q, v in global_residual_quantiles.items()
]

global_quantiles_df = spark.createDataFrame(
    global_quantiles_rows,
    ["quantile", "residual_quantile"]
)

#GLOBAL_QUANTILES_PATH = f"{base}/global_residual_quantiles.parquet"

#global_quantiles_df.write.mode("overwrite").parquet(GLOBAL_QUANTILES_PATH)

#print("Saved:", GLOBAL_QUANTILES_PATH)

# %%
spark.stop()
