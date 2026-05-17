# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
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

# +
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
    .config("spark.executor.memory", "6g")
    .config("spark.executor.cores", "4")
    .config("spark.executor.instances", "4")
    .master("yarn")
    .getOrCreate()
)
spark.sparkContext
# -

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

# ## 4. Interactive Route Planner UI

# +
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go

# Build stop dropdown options
stops = planner.get_stops()
stop_options = [(f"{s['stop_name']} ({s['stop_id']})", s["stop_id"]) for s in stops]

# Default stops (Lausanne-Flon and Lausanne Gare if available)
default_source = stop_options[0][1] if stop_options else ""
default_target = stop_options[min(1, len(stop_options) - 1)][1] if stop_options else ""

# --- Widgets ---
w_source = widgets.Dropdown(
    options=stop_options,
    value=default_source,
    description="From:",
    layout=widgets.Layout(width="450px"),
    style={"description_width": "60px"},
)

w_target = widgets.Dropdown(
    options=stop_options,
    value=default_target,
    description="To:",
    layout=widgets.Layout(width="450px"),
    style={"description_width": "60px"},
)

w_day = widgets.Dropdown(
    options=[
        ("📅 Monday", "monday"),
        ("📅 Tuesday", "tuesday"),
        ("📅 Wednesday", "wednesday"),
        ("📅 Thursday", "thursday"),
        ("📅 Friday", "friday"),
        ("📅 Saturday", "saturday"),
        ("📅 Sunday", "sunday"),
    ],
    value="wednesday",
    description="Day:",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "60px"},
)

w_mode = widgets.Dropdown(
    options=[
        ("🚀 Fastest Route", "fastest"),
        ("⏰ Latest Departure", "latest_departure"),
        ("🔄 Least Transfers", "least_transfers"),
        ("🚶 Least Walking", "least_walking"),
        ("🛡️ Safest Route", "safest"),
        ("📋 All Pareto-Optimal", "all"),
    ],
    value="fastest",
    description="Mode:",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "60px"},
)

w_dep_hour = widgets.IntSlider(min=4, max=23, value=12, description="Hour:", layout=widgets.Layout(width="300px"))
w_dep_min = widgets.IntSlider(min=0, max=59, value=30, step=5, description="Min:", layout=widgets.Layout(width="300px"))

w_arr_hour = widgets.IntSlider(min=4, max=23, value=14, description="Arr Hour:", layout=widgets.Layout(width="300px"))
w_arr_min = widgets.IntSlider(min=0, max=59, value=0, step=5, description="Arr Min:", layout=widgets.Layout(width="300px"))

w_walk_speed = widgets.FloatSlider(
    min=20, max=100, value=50, step=5,
    description="Walk m/min:",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "100px"},
)

w_max_walk = widgets.IntSlider(
    min=0, max=500, value=500, step=50,
    description="Max walk (m):",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "100px"},
)

w_extra_transfer = widgets.IntSlider(
    min=0, max=300, value=0, step=30,
    description="Extra buffer (s):",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "100px"},
)

w_min_conf = widgets.FloatSlider(
    min=0.0, max=1.0, value=0.0, step=0.05,
    description="Min confidence:",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "100px"},
    readout_format=".0%",
)

w_button = widgets.Button(
    description="🔍 Find Routes",
    button_style="success",
    layout=widgets.Layout(width="200px", height="40px"),
)

w_output = widgets.Output()

# --- Layout ---
dep_time_box = widgets.HBox([w_dep_hour, w_dep_min], layout=widgets.Layout(margin="0 0 0 0"))
arr_time_box = widgets.HBox([w_arr_hour, w_arr_min], layout=widgets.Layout(margin="0 0 0 0"))

dep_label = widgets.HTML("<b>Departure time:</b>")
arr_label = widgets.HTML("<b>Arrival deadline (for 'Latest Departure' mode):</b>")

stops_box = widgets.VBox([w_source, w_target])
day_mode_box = widgets.VBox([w_day, w_mode])
time_box = widgets.VBox([dep_label, dep_time_box, arr_label, arr_time_box])
config_box = widgets.VBox([w_walk_speed, w_max_walk, w_extra_transfer, w_min_conf])

ui = widgets.VBox([
    widgets.HTML("<h2>🚆 Robust Journey Planner</h2>"),
    widgets.HBox([
        widgets.VBox([
            widgets.HTML("<h4>📍 Stops</h4>"), stops_box,
            widgets.HTML("<h4>🎯 Day & Mode</h4>"), day_mode_box,
        ]),
        widgets.VBox([
            widgets.HTML("<h4>🕐 Time</h4>"), time_box,
            widgets.HTML("<h4>⚙️ Settings</h4>"), config_box,
        ]),
    ]),
    w_button,
    w_output,
])


# --- Callback ---
def on_search(btn):
    w_output.clear_output()
    with w_output:
        mode = w_mode.value
        source = w_source.value
        target = w_target.value
        day = w_day.value

        dep_time = f"{w_dep_hour.value:02d}:{w_dep_min.value:02d}"
        arr_time = f"{w_arr_hour.value:02d}:{w_arr_min.value:02d}"

        print(f"Searching: {source} → {target}, day={day}, mode={mode}")

        try:
            journeys = planner.plan(
                source=source,
                target=target,
                mode=mode,
                day=day,
                departure_time=dep_time if mode != "latest_departure" else None,
                arrival_time=arr_time if mode == "latest_departure" else None,
                walking_speed=w_walk_speed.value,
                max_walk_m=w_max_walk.value,
                extra_transfer_sec=int(w_extra_transfer.value),
                min_confidence=w_min_conf.value,
                max_results=5,
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return

        if not journeys:
            print("❌ No routes found with the given constraints.")
            return

        # Display summary table
        summary_df = RobustJourneyPlanner.journeys_to_pandas(journeys)
        display(summary_df)

        # Display detailed legs
        for i, j in enumerate(journeys):
            print(f"\n{'='*60}")
            print(f"Option {i+1}:")
            print(RobustJourneyPlanner.format_journey(j))

        # Plot route on map
        try:
            fig = _plot_route_map(journeys[0], planner)
            fig.show()
        except Exception as e:
            print(f"(Map visualization skipped: {e})")


def _plot_route_map(journey, planner):
    """Plot the best journey on a map."""
    fig = go.Figure()

    # All stops as small gray dots
    all_lats = [s["lat"] for s in stops]
    all_lons = [s["lon"] for s in stops]
    all_names = [s["stop_name"] for s in stops]

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=all_lons, lat=all_lats,
        marker=dict(size=4, color="gray", opacity=0.4),
        text=all_names,
        hoverinfo="text",
        name="All stops",
        showlegend=False,
    ))

    # Route legs
    colors = {"transit": "#1E88E5", "walk": "#E53935"}
    for leg in journey.legs:
        from_stop = planner.graph.stops.get(leg.from_stop)
        to_stop = planner.graph.stops.get(leg.to_stop)
        if from_stop and to_stop:
            from datetime import datetime
            t_dep = datetime.fromtimestamp(leg.departure_ts).strftime("%H:%M")
            t_arr = datetime.fromtimestamp(leg.arrival_ts).strftime("%H:%M")
            color = colors.get(leg.leg_type, "#666")
            dash = "solid" if leg.leg_type == "transit" else "dot"
            label = leg.trip_id or "walk"

            fig.add_trace(go.Scattermapbox(
                mode="lines+markers",
                lon=[from_stop.lon, to_stop.lon],
                lat=[from_stop.lat, to_stop.lat],
                line=dict(width=4, color=color, dash=dash) if leg.leg_type == "walk" else dict(width=3, color=color),
                marker=dict(size=10, color=color),
                text=[f"{t_dep} {from_stop.stop_name}", f"{t_arr} {to_stop.stop_name}"],
                hoverinfo="text",
                name=f"{'🚌' if leg.leg_type == 'transit' else '🚶'} {label}",
            ))

    # Center map
    route_lats = []
    route_lons = []
    for leg in journey.legs:
        for sid in [leg.from_stop, leg.to_stop]:
            s = planner.graph.stops.get(sid)
            if s:
                route_lats.append(s.lat)
                route_lons.append(s.lon)

    center_lat = np.mean(route_lats) if route_lats else np.mean(all_lats)
    center_lon = np.mean(route_lons) if route_lons else np.mean(all_lons)

    from datetime import datetime
    dep = datetime.fromtimestamp(journey.departure_ts).strftime("%H:%M")
    arr = datetime.fromtimestamp(journey.arrival_ts).strftime("%H:%M")

    fig.update_layout(
        title=f"Route: {dep} → {arr} | {journey.num_transfers} transfers | {journey.confidence:.0%} confidence",
        mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=12),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


w_button.on_click(on_search)
display(ui)
# -

# ## 5. Quick API Examples
#
# You can also use the planner programmatically.
# The `day` parameter selects the precomputed graph for that weekday.

# +
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
# -

# +
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
# -

# +
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
# -
