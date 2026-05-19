import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from src.route_planner import RobustJourneyPlanner


def create_interactive_ui(planner, default_source="8501120", default_target="8501214"):
    """Create and display the interactive journey planner UI widgets."""
    # Build stop dropdown options
    stops = planner.get_stops()
    stop_options = [(f"{s['stop_name']} ({s['stop_id']})", s["stop_id"]) for s in stops]

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

    w_safest = widgets.Checkbox(
        value=False,
        description="🛡️ Safest Route",
        indent=False,
        layout=widgets.Layout(width="350px"),
    )
    w_least_walking = widgets.Checkbox(
        value=False,
        description="🚶 Least Walking",
        indent=False,
        layout=widgets.Layout(width="350px"),
    )
    w_least_transfers = widgets.Checkbox(
        value=False,
        description="🔄 Least Transfers",
        indent=False,
        layout=widgets.Layout(width="350px"),
    )

    w_time_type = widgets.Dropdown(
        options=[
            ("Depart at", "depart_at"),
            ("Arrive by", "arrive_by"),
        ],
        value="depart_at",
        description="Timing:",
        layout=widgets.Layout(width="170px"),
        style={"description_width": "60px"},
    )

    w_dep_hour = widgets.BoundedIntText(
        min=0, max=23, value=12,
        layout=widgets.Layout(width="50px"),
    )
    w_dep_min = widgets.BoundedIntText(
        min=0, max=59, value=30,
        layout=widgets.Layout(width="50px"),
    )

    w_arr_hour = widgets.BoundedIntText(
        min=0, max=23, value=14,
        layout=widgets.Layout(width="50px"),
    )
    w_arr_min = widgets.BoundedIntText(
        min=0, max=59, value=0,
        layout=widgets.Layout(width="50px"),
    )

    # Walk Speed
    w_walk_speed_title = widgets.HTML("<b>🚶 Walking Speed</b>")
    w_walk_speed = widgets.FloatSlider(
        min=20, max=100, value=50, step=5,
        layout=widgets.Layout(width="350px"),
    )
    w_walk_speed_help = widgets.HTML("<small style='color: #718096;'>Leisurely / slow pace (~3.0 km/h)</small>")

    def update_walk_speed_help(change):
        val = change["new"]
        if val <= 35:
            text = f"Very slow pace (~{val*0.06:.1f} km/h) - e.g., carrying heavy bags"
        elif val <= 60:
            text = f"Leisurely / slow pace (~{val*0.06:.1f} km/h)"
        elif val <= 85:
            text = f"Normal walking pace (~{val*0.06:.1f} km/h)"
        else:
            text = f"Fast / brisk pace (~{val*0.06:.1f} km/h)"
        w_walk_speed_help.value = f"<small style='color: #718096;'>{text}</small>"
    w_walk_speed.observe(update_walk_speed_help, names="value")

    walk_speed_box = widgets.VBox([w_walk_speed_title, w_walk_speed, w_walk_speed_help], layout=widgets.Layout(margin="5px 0 10px 0"))

    # Max Walk Distance
    w_max_walk_title = widgets.HTML("<b>📏 Max Walking Distance</b>")
    w_max_walk = widgets.IntSlider(
        min=0, max=500, value=500, step=50,
        layout=widgets.Layout(width="350px"),
    )
    w_max_walk_help = widgets.HTML("<small style='color: #718096;'>Up to 500 meters (approx. 10.0 minutes of walking)</small>")

    def update_max_walk_help(change):
        val = change["new"]
        speed = w_walk_speed.value
        minutes = val / speed if speed > 0 else 0
        w_max_walk_help.value = f"<small style='color: #718096;'>Up to {val} meters (approx. {minutes:.1f} minutes of walking at current speed)</small>"
    w_max_walk.observe(update_max_walk_help, names="value")
    w_walk_speed.observe(lambda change: update_max_walk_help({"new": w_max_walk.value}), names="value")

    max_walk_box = widgets.VBox([w_max_walk_title, w_max_walk, w_max_walk_help], layout=widgets.Layout(margin="5px 0 10px 0"))

    # Connection Buffer
    w_extra_transfer_title = widgets.HTML("<b>⏳ Connection Buffer</b>")
    w_extra_transfer = widgets.IntSlider(
        min=0, max=300, value=0, step=30,
        layout=widgets.Layout(width="350px"),
    )
    w_extra_transfer_help = widgets.HTML("<small style='color: #718096;'>No extra buffer (rely on standard transfer times)</small>")

    def update_extra_transfer_help(change):
        val = change["new"]
        if val == 0:
            text = "No extra buffer (rely on standard transfer times)"
        elif val < 60:
            text = f"Adds {val} seconds of extra cushion time at transfer points"
        else:
            m = val // 60
            s = val % 60
            time_str = f"{m} min" + (f" {s} sec" if s > 0 else "")
            text = f"Adds {time_str} of extra cushion time at transfer points"
        w_extra_transfer_help.value = f"<small style='color: #718096;'>{text}</small>"
    w_extra_transfer.observe(update_extra_transfer_help, names="value")

    extra_transfer_box = widgets.VBox([w_extra_transfer_title, w_extra_transfer, w_extra_transfer_help], layout=widgets.Layout(margin="5px 0 10px 0"))

    # Reliability Threshold
    w_min_conf_title = widgets.HTML("<b>🛡️ Minimum Reliability</b>")
    w_min_conf = widgets.FloatSlider(
        min=0.0, max=1.0, value=0.0, step=0.05,
        layout=widgets.Layout(width="350px"),
        readout_format=".0%",
    )
    w_min_conf_help = widgets.HTML("<small style='color: #718096;'>Show all routes, regardless of transfer risk</small>")

    def update_min_conf_help(change):
        val = change["new"]
        if val == 0.0:
            text = "Show all routes, regardless of transfer risk"
        elif val <= 0.5:
            text = f"Filter out highly risky routes (lower than {val:.0%} success chance)"
        elif val <= 0.8:
            text = f"Only show moderately reliable routes (at least {val:.0%} success chance)"
        else:
            text = f"Strict reliability: only show very safe connections (at least {val:.0%} success chance)"
        w_min_conf_help.value = f"<small style='color: #718096;'>{text}</small>"
    w_min_conf.observe(update_min_conf_help, names="value")

    min_conf_box = widgets.VBox([w_min_conf_title, w_min_conf, w_min_conf_help], layout=widgets.Layout(margin="5px 0 10px 0"))

    w_button = widgets.Button(
        description="🔍 Find Routes",
        button_style="success",
        layout=widgets.Layout(width="200px", height="40px", margin="15px 0 15px 0"),
    )

    w_output = widgets.Output()

    # --- Layout ---
    dep_time_box = widgets.HBox([w_dep_hour, widgets.HTML("<b style='padding: 0 4px; color: #4a5568;'>:</b>"), w_dep_min], layout=widgets.Layout(margin="0 0 0 5px", align_items="center"))
    arr_time_box = widgets.HBox([w_arr_hour, widgets.HTML("<b style='padding: 0 4px; color: #4a5568;'>:</b>"), w_arr_min], layout=widgets.Layout(margin="0 0 0 5px", align_items="center"))
    timing_row = widgets.HBox([w_time_type, dep_time_box, arr_time_box], layout=widgets.Layout(margin="0px", align_items="center"))

    # Show/hide container based on dropdown selection
    dep_time_box.layout.display = "flex"
    arr_time_box.layout.display = "none"

    def on_time_type_change(change):
        if change["new"] == "depart_at":
            w_dep_hour.value = w_arr_hour.value
            w_dep_min.value = w_arr_min.value
            dep_time_box.layout.display = "flex"
            arr_time_box.layout.display = "none"
        else:
            w_arr_hour.value = w_dep_hour.value
            w_arr_min.value = w_dep_min.value
            dep_time_box.layout.display = "none"
            arr_time_box.layout.display = "flex"

    w_time_type.observe(on_time_type_change, names="value")

    # Custom styles
    style_html = widgets.HTML("""
        <style>
            .custom-widget-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 20px !important;
                margin: 8px !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            }
            .widget-header-title {
                color: #2d3748 !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-weight: 600;
                border-bottom: 2px solid #edf2f7;
                padding-bottom: 8px;
                margin-top: 0px;
                margin-bottom: 15px;
            }
        </style>
    """)

    left_column_box = widgets.VBox([
        widgets.HTML("<h3 class='widget-header-title'>📍 Route Details</h3>"),
        w_source,
        w_target,
        w_day,
        timing_row,
        widgets.HTML("<b style='margin-top: 15px; display: block;'>Criteria:</b>"),
        w_safest,
        w_least_walking,
        w_least_transfers
    ])

    right_column_box = widgets.VBox([
        widgets.HTML("<h3 class='widget-header-title'>⚙️ Preferences</h3>"),
        walk_speed_box,
        max_walk_box,
        extra_transfer_box,
        min_conf_box
    ])

    left_column_box.add_class("custom-widget-card")
    right_column_box.add_class("custom-widget-card")

    ui = widgets.VBox([
        style_html,
        widgets.HTML("<h2 style='color: #2b6cb0; font-weight: 700; margin-left: 8px; margin-bottom: 10px;'>🚆 Robust Journey Planner</h2>"),
        widgets.HBox([
            left_column_box,
            right_column_box
        ], layout=widgets.Layout(width="100%")),
        w_button,
        w_output,
    ])

    # --- Callback ---
    def on_search(btn):
        w_output.clear_output()
        with w_output:
            selected_modes = []
            if w_safest.value:
                selected_modes.append("safest")
            if w_least_walking.value:
                selected_modes.append("least_walking")
            if w_least_transfers.value:
                selected_modes.append("least_transfers")

            source = w_source.value
            target = w_target.value
            day = w_day.value
            time_type = w_time_type.value

            dep_time = f"{w_dep_hour.value:02d}:{w_dep_min.value:02d}" if time_type == "depart_at" else None
            arr_time = f"{w_arr_hour.value:02d}:{w_arr_min.value:02d}" if time_type == "arrive_by" else None

            print(f"Searching: {source} → {target}, day={day}, modes={selected_modes}")

            try:
                journeys = planner.plan(
                    source=source,
                    target=target,
                    day=day,
                    departure_time=dep_time,
                    arrival_time=arr_time,
                    walking_speed=w_walk_speed.value,
                    max_walk_m=w_max_walk.value,
                    extra_transfer_sec=int(w_extra_transfer.value),
                    min_confidence=w_min_conf.value,
                    max_results=5,
                    modes=selected_modes,
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
                fig = _plot_route_map(journeys[0], planner, stops)
                display(go.FigureWidget(fig))
            except Exception as e:
                print(f"(Map visualization skipped: {e})")

    w_button.on_click(on_search)
    display(ui)


def _plot_route_map(journey, planner, stops):
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
            t_dep = datetime.fromtimestamp(leg.departure_ts).strftime("%H:%M")
            t_arr = datetime.fromtimestamp(leg.arrival_ts).strftime("%H:%M")
            color = colors.get(leg.leg_type, "#666")
            label = leg.trip_id or "walk"

            fig.add_trace(go.Scattermapbox(
                mode="lines+markers",
                lon=[from_stop.lon, to_stop.lon],
                lat=[from_stop.lat, to_stop.lat],
                line=dict(width=4, color=color) if leg.leg_type == "walk" else dict(width=3, color=color),
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
