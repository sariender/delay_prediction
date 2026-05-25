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

# # Route Planner Validation
#
# Validates **algorithm correctness** and **confidence sanity** by checking properties that must hold for the algorithm to be correct.

# ## Setup

# +
import os
import sys
import pwd
import warnings
import datetime as dt
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

# +
from src.route_planner import RobustJourneyPlanner

# Region UUIDs for Lausanne district and Ouest lausannois district
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

# Fixed for tests
lausanne_gare = "8501120"
epfl = "8501214"
# -

# ## Section 1: Algorithm Correctness

# ### 1.1 Forward RAPTOR: Lausanne -> EPFL at 12:30

# +
journeys_fwd = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30",
)

print("Earliest Arrival Routes:")
for j in journeys_fwd[:3]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

# Time consistency
for j in journeys_fwd:
    assert j.arrival_ts >= j.departure_ts
    for leg in j.legs:
        assert leg.arrival_ts >= leg.departure_ts

# Respects departure constraint
target_dep = dt.datetime(2026, 2, 4, 12, 30).timestamp()
for j in journeys_fwd:
    assert j.departure_ts >= target_dep

print(f"{len(journeys_fwd)} routes: all time-consistent, all depart ≥ 12:30")
# -

# ### 1.2 Reverse RAPTOR: Lausanne -> EPFL arrive by 12:50

# +
journeys_rev = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    arrival_time="12:50",
)

print("Latest Departure Routes:")
for j in journeys_rev[:3]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

target_arr = dt.datetime(2026, 2, 4, 12, 50).timestamp()
for j in journeys_rev:
    assert j.arrival_ts <= target_arr

print(f"{len(journeys_rev)} routes: all arrive ≤ 12:50")
# -

# ### 1.3 Forward/reverse symmetry
# The same physical journey should give the same confidence in both directions.

# +
fwd_best = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30",
    mode="all",
)[0]
print("Forward (dep=12:30):")
print(RobustJourneyPlanner.format_journey(fwd_best))
print()

arr_str = dt.datetime.fromtimestamp(fwd_best.arrival_ts).strftime("%H:%M")
rev_results = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    arrival_time=arr_str,
    mode="all",
)

def journey_signature(j):
    """Identity of a journey: trip_id + from_stop + to_stop + walk_distance per leg."""
    return tuple(
        (l.leg_type, l.trip_id, l.from_stop, l.to_stop, round(l.walk_distance_m))
        for l in j.legs
    )

fwd_sig = journey_signature(fwd_best)
rev_match = next((j for j in rev_results if journey_signature(j) == fwd_sig), None)

if rev_match is None:
    print(f"Reverse (arr={arr_str}): no identical journey found.")
    print("Forward and reverse picked different Pareto-optimal routes, symmetry assertion skipped.")
else:
    print(f"Reverse (arr={arr_str}, identical journey):")
    print(RobustJourneyPlanner.format_journey(rev_match))
    print()
    assert abs(fwd_best.confidence - rev_match.confidence) < 0.01
    print(f"Forward and reverse agree on identical journey: {fwd_best.confidence:.2%} = {rev_match.confidence:.2%}")

# -

# ### 1.4 Determinism: same query, same result

# +
j1 = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30"
)[0]

j2 = planner.plan(
    source=lausanne_gare, 
    target=epfl,
    day="wednesday", 
    departure_time="12:30"
)[0]

print("Run 1:")
print(RobustJourneyPlanner.format_journey(j1))
print()
print("Run 2:")
print(RobustJourneyPlanner.format_journey(j2))
print()

assert j1.arrival_ts == j2.arrival_ts
assert j1.num_transfers == j2.num_transfers
assert abs(j1.total_walk_m - j2.total_walk_m) < 1.0

print("Identical results across runs")
# -

# ### 1.5 Monotonicity: later departure -> later arrival

# +
print("Comparing arrival times for increasing departure times:\n")
arrivals = []
for t in ["08:00", "10:00", "12:00", "14:00", "16:00"]:
    j = planner.plan(
        source=lausanne_gare,
        target=epfl,
        day="wednesday",
        departure_time=t
    )
    
    if j:
        print(f"- Departure {t} -")
        print(RobustJourneyPlanner.format_journey(j[0]))
        print()
        arrivals.append(j[0].arrival_ts)

for a1, a2 in zip(arrivals[:-1], arrivals[1:]):
    assert a2 >= a1

print("Later departure never produces earlier arrival")
# -

# ### 1.6 Monotonicity: stricter max_walks_m -> equal or later arrival

# +
print("Comparing routes for increasingly tight walking limits:\n")
arrivals = []
for lim in [500, 300, 150, 50]:
    j = planner.plan(
        source=lausanne_gare,
        target=epfl,
        day="wednesday",
        departure_time="12:30",
        max_walk_m=lim,
    )
    if j:
        print(f"- max_walk_m = {lim} -")
        print(RobustJourneyPlanner.format_journey(j[0]))
        print()
        arrivals.append(j[0].arrival_ts)

for a1, a2 in zip(arrivals[:-1], arrivals[1:]):
    assert a2 >= a1, "Stricter walking limit produced an earlier arrival!"

print("Stricter walking limit never produces an earlier arrival")
# -

# ### 1.7 Monotonicity: slower walking speed -> equal or later arrival

# +
print("Comparing routes for decreasing walking speeds:\n")
arrivals = []
for speed in [80, 50, 30, 15]:
    j = planner.plan(
        source=lausanne_gare,
        target=epfl,
        day="wednesday",
        departure_time="12:30",
        mode="safest",
        walking_speed=speed,
    )
    if j:
        print(f"- walking_speed = {speed} m/min -")
        print(RobustJourneyPlanner.format_journey(j[0]))
        print()
        arrivals.append(j[0].arrival_ts)

for a1, a2 in zip(arrivals[:-1], arrivals[1:]):
    assert a2 >= a1, "Slower walking speed produced an earlier arrival!"

print("Slower walking speed never produces an earlier arrival")
# -

# ## Section 2: Confidence Sanity

# ### 2.1 Confidence is always a valid probability

# +
journeys = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday", 
    departure_time="12:30",
    max_results=10
)

print("Sample of Pareto-optimal routes:")
for j in journeys[:3]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

for j in journeys:
    assert 0.0 <= j.confidence <= 1.0

print(f"All {len(journeys)} routes have confidence in [0, 1]")
# -

# ### 2.2 Zero-transfer journeys -> 100% confidence

# +
direct = [j for j in journeys if j.num_transfers == 0]

print(f"Found {len(direct)} direct (0-transfer) routes:\n")
for j in direct[:3]:
    print(RobustJourneyPlanner.format_journey(j))
    print()

for j in direct:
    assert j.confidence == 1.0

print(f"All {len(direct)} direct routes have 100% confidence")
# -

# ### 2.3 Larger transfer buffer -> higher confidence

# +
print("Best route at increasing transfer buffers:")
confs = []
for b in [0, 60, 120, 300]:
    j = planner.plan(
        source=lausanne_gare,
        target=epfl,
        day="wednesday",
        departure_time="12:30",
        extra_transfer_sec=b
    )
    
    if j:
        print(f"- extra_transfer_sec = {b} -")
        print(RobustJourneyPlanner.format_journey(j[0]))
        print()
        confs.append(j[0].confidence)

for c1, c2 in zip(confs[:-1], confs[1:]):
    assert c2 >= c1 - 0.001

print("Larger transfer buffer yields higher or equal confidence")
# -

# ### 2.4 min_confidence filter works

# +
for threshold in [0.5, 0.7, 0.9]:
    js = planner.plan(
        source=lausanne_gare,
        target=epfl,
        day="wednesday",
        departure_time="12:30",
        min_confidence=threshold, 
        max_results=10
    )
    
    print(f"- min_confidence ≥ {threshold:.0%} ({len(js)} routes) -")
    for j in js[:2]:
        print(RobustJourneyPlanner.format_journey(j))
        print()
    for j in js:
        assert j.confidence >= threshold - 0.001

print("min_confidence filter correctly enforced at all thresholds")
# -

# ### 2.5 Confidence = product of per-transfer probabilities 

# +
# Reconstruct the confidence from scratch and compare with the reported value.
j = next(j for j in journeys if j.num_transfers >= 1)

print("Decomposing this journey:")
print(RobustJourneyPlanner.format_journey(j))
print()

product = 1.0
for i, leg in enumerate(j.legs):
    if leg.leg_type != "transit" or i == 0:
        continue

    prev_leg = j.legs[i - 1]
    if prev_leg.leg_type == "transit":
        spare = leg.departure_ts - prev_leg.arrival_ts - planner.min_transfer_sec - planner.extra_transfer_sec
        incoming = prev_leg
    elif prev_leg.leg_type == "walk":
        walk_sec = int(prev_leg.walk_distance_m / planner.walking_speed * 60)
        spare = leg.departure_ts - prev_leg.departure_ts - walk_sec - planner.min_transfer_sec - planner.extra_transfer_sec
        incoming = next((j.legs[k] for k in range(i - 2, -1, -1) if j.legs[k].leg_type == "transit"), None)
    else:
        continue

    if incoming is None:
        continue

    info = planner.graph.event_info.get((incoming.trip_id, incoming.to_stop), {})
    lookup = planner._delay_prob_fast if planner._delay_dict is not None else planner._delay_fn
    p = lookup(
        trip_id=incoming.trip_id,
        stop_id=info.get("stop_id", incoming.to_stop),
        date_value=info.get("date_value", ""),
        X=max(0, spare),
        scheduled_arrival_ts=info.get("arrival_timestamp"),
    )
    p = max(p, 0.001)
    print(f"  Transfer at {prev_leg.to_name}: spare={spare}s, P(make it)={p:.2%}")
    product *= p

print(f"\nReconstructed product:  {product:.4%}")
print(f"Reported confidence:    {j.confidence:.4%}")
assert abs(product - j.confidence) < 0.01
print("\n Reported confidence matches the independent-delay product")
# -
# ## Section 3: Mode Selection


# ### 3.1 mode="safest" returns the highest-confidence route first

# +
all_routes = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30",
    mode="all",
    max_results=10
)
safest = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30",
    mode="safest",
    max_results=10
)

print("Safest route returned first:")
print(RobustJourneyPlanner.format_journey(safest[0]))
print()

best_conf = max(j.confidence for j in all_routes)
assert safest[0].confidence == best_conf, (f"safest returned {safest[0].confidence:.2%} but best available is {best_conf:.2%}")
print(f"Safest route has highest confidence in Pareto front: {best_conf:.2%}")
# -

# ### 3.2 mode="least_transfers" returns the route with fewest transfers first

# +
least_tr = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30",
    mode="least_transfers", 
    max_results=10
)

print("Least-transfers route returned first:")
print(RobustJourneyPlanner.format_journey(least_tr[0]))
print()

min_transfers = min(j.num_transfers for j in all_routes)
assert least_tr[0].num_transfers == min_transfers, (f"least_transfers returned {least_tr[0].num_transfers} but min available is {min_transfers}")
print(f"Least-transfers route has fewest transfers in Pareto front: {min_transfers}")
# -

# ### 3.3 mode="least_walking" returns the route with the least walking first

# +
least_w = planner.plan(
    source=lausanne_gare,
    target=epfl,
    day="wednesday",
    departure_time="12:30",
    mode="least_walking", 
    max_results=10
)

print("Least-walking route returned first:")
print(RobustJourneyPlanner.format_journey(least_w[0]))
print()

min_walk = min(j.total_walk_m for j in all_routes)
assert abs(least_w[0].total_walk_m - min_walk) < 1.0, (f"least_walking returned {least_w[0].total_walk_m:.0f}m but min available is {min_walk:.0f}m")
print(f"Least-walking route has minimum walking in Pareto front: {min_walk:.0f}m")
# -

# ## Conclusion
# All algorithm-correctness, confidence-sanity, and mode-selection properties hold.

# ## Stop the spark instance

spark.stop()
