# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 🚆 Robust Journey Planner — Interactive Demo
# ---
# This notebook demonstrates the robust route planner built using a
# **multi-criteria RAPTOR** algorithm with delay-model integration.
#
# **Features:**
# - Fastest Route / Latest Departure / Least Transfers / Least Walking / Safest Route
# - Configurable walking speed, max walking distance, extra transfer buffer
# - Minimum confidence threshold using calibrated delay predictions
# - Interactive map visualization of planned routes

# ## 1. Setup & Spark Session

# +
import os
import sys
import pwd
import warnings
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F

warnings.simplefilter(action="ignore", category=UserWarning)

username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS = os.getenv("HADOOP_FS", None)
groupName = "H1"

print(f"username={username}, group={groupName}")
# -

spark = (
    SparkSession.builder
    .appName(f"{username}-robust-planner")
    .config("spark.ui.port", randrange(4050, 4450, 5))
    .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path))
    .config(
        "spark.jars",
        f"{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar,"
        f"{hadoopFS}/data/com-490/jars/sedona-spark-shaded-3.5_2.13-1.7.1.jar,"
        f"{hadoopFS}/data/com-490/jars/geotools-wrapper-1.7.1-28.5.jar",
    )
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.iceberg.type", "hadoop")
    .config("spark.sql.catalog.iceberg.warehouse", f"{hadoopFS}/data/com-490/silver/")
    .config("spark.driver.memory", "8g")     
    .config("spark.executor.memory", "6g")
    .config("spark.executor.cores", "4")
    .config("spark.executor.instances", "4")
    .master("yarn")
    .getOrCreate()
)
spark.sparkContext

# ## 2. Precompute Transit Graphs for All Days
#
# We use `prepare()` to build RAPTOR graphs for all 7 days of the
# first week of February 2026 (Mon Feb 2 – Sun Feb 8).
# This takes a few minutes but only needs to run once per session.

# +
from src.route_planner import RobustJourneyPlanner

# Region UUIDs for Lausanne district and Ouest lausannois district
# (from iceberg.com490_iceberg.geo table)
REGION_UUIDS = [
    "a7a21b73-6ffe-4fbf-a635-6e2b961f3072",  # Lausanne
    "e168fd57-f57a-4075-a350-0dcfbb55147f",  # Ouest lausannois
]

planner = RobustJourneyPlanner(
    walking_speed_m_per_min=50.0,
    max_walk_m=500.0,
    min_transfer_sec=120,
    extra_transfer_sec=0,
    min_confidence=0.0,
    max_rounds=8,
)

# Precompute graphs for all 7 days + initialize delay model
planner.prepare(spark, regions=REGION_UUIDS)
# -

# ## 3. Interactive Route Planner UI

# +
from src.ui import create_interactive_ui

# Default stops (Lausanne Gare to EPFL if available)
default_source = "8501120"
default_target = "8501214"

create_interactive_ui(planner, default_source=default_source, default_target=default_target)
# -

# ## 4. Quick API Examples
#
# You can also use the planner programmatically.
# The `day` parameter selects the precomputed graph for that weekday.

# Example 1: Fastest route on a Wednesday
journeys = planner.plan(
    source=default_source,
    target=default_target,
    mode="fastest",
    day="wednesday",
    departure_time="12:30",
)
for j in journeys[:2]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

# Example 2: Safest route on a Friday with minimum 70% confidence
journeys = planner.plan(
    source=default_source,
    target=default_target,
    mode="safest",
    day="friday",
    departure_time="12:30",
    min_confidence=0.7,
)
for j in journeys[:2]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

# Example 3: Latest departure on a Sunday to arrive by 14:00
journeys = planner.plan(
    source=default_source,
    target=default_target,
    mode="latest_departure",
    day="sunday",
    arrival_time="14:00",
)
for j in journeys[:2]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

# ## 5. Stop the spark instance

# +
# spark.stop()
