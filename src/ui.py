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
        style=widgets.ButtonStyle(font_size="16px"),
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

            # Display confidence gauge + best route summary card
            display(HTML(_render_best_route_card(journeys[0])))

            # Display other options summary
            if len(journeys) > 1:
                display(HTML(_render_other_options_card(journeys[1:])))


            # Plot route on map
            try:
                fig = _plot_route_map(journeys[0], planner, stops)
                display(go.FigureWidget(fig))
            except Exception as e:
                print(f"(Map visualization skipped: {e})")

    w_button.on_click(on_search)
    display(ui)


def _render_other_options_card(journeys) -> str:
    """Return an HTML card summarising alternative journey options with expandable itineraries."""
    from datetime import datetime as _dt
    import random

    # Unique prefix to avoid ID collisions across multiple renders
    uid = random.randint(10000, 99999)

    rows_html = ""
    for i, j in enumerate(journeys):
        dep = _dt.fromtimestamp(j.departure_ts).strftime("%H:%M")
        arr = _dt.fromtimestamp(j.arrival_ts).strftime("%H:%M")
        dur = j.travel_time_seconds() / 60
        conf = j.confidence
        row_id = f"opt_{uid}_{i}"

        # Confidence badge color
        if conf >= 0.85:
            badge_bg, badge_color = "#dcfce7", "#15803d"
        elif conf >= 0.65:
            badge_bg, badge_color = "#ecfccb", "#4d7c0f"
        elif conf >= 0.45:
            badge_bg, badge_color = "#fef9c3", "#ca8a04"
        elif conf >= 0.25:
            badge_bg, badge_color = "#ffedd5", "#ea580c"
        else:
            badge_bg, badge_color = "#fee2e2", "#dc2626"

        # Leg summary icons
        leg_icons = ""
        for leg in j.legs:
            if leg.leg_type == "transit":
                leg_icons += "<span style='margin-right:2px;'>🚌</span>"
            else:
                leg_icons += "<span style='margin-right:2px;'>🚶</span>"

        # Build itinerary detail rows
        detail_legs = ""
        for leg in j.legs:
            t_dep = _dt.fromtimestamp(leg.departure_ts).strftime("%H:%M")
            t_arr = _dt.fromtimestamp(leg.arrival_ts).strftime("%H:%M")
            if leg.leg_type == "transit":
                icon = "🚌"
                detail = leg.trip_id or ""
                lbl_bg, lbl_color, lbl_border = "#eff6ff", "#1e40af", "#bfdbfe"
                badge_html = f"<span style='background:{lbl_bg};color:{lbl_color};border:1px solid {lbl_border};padding:2px 8px;border-radius:6px;font-size:11px;margin-left:6px;'>{detail}</span>" if detail else ""
            else:
                icon = "🚶"
                lbl_bg, lbl_color, lbl_border = "#fff7ed", "#9a3412", "#fed7aa"
                badge_html = f"<span style='background:{lbl_bg};color:{lbl_color};border:1px solid {lbl_border};padding:2px 8px;border-radius:6px;font-size:11px;margin-left:6px;'>{leg.walk_distance_m:.0f}m</span>"
            detail_legs += f"""
                <div style="display:flex;align-items:center;padding:4px 0;font-size:13px;">
                    <span style="margin-right:8px;">{icon}</span>
                    <span style="color:#334155;">{t_dep} {leg.from_name} → {t_arr} {leg.to_name}</span>
                    {badge_html}
                </div>"""

        rows_html += f"""
        <tr style="border-bottom:1px solid #f1f5f9;">
            <td style="padding:10px 8px;font-size:14px;color:#1e293b;font-weight:600;width:15%;white-space:nowrap;">{dep} → {arr}</td>
            <td style="padding:10px 8px;font-size:13px;color:#475569;width:14%;">{dur:.0f} min</td>
            <td style="padding:10px 8px;font-size:13px;color:#475569;text-align:center;width:13%;">{j.num_transfers}</td>
            <td style="padding:10px 8px;font-size:13px;color:#475569;text-align:center;width:14%;">{j.total_walk_m:.0f}m</td>
            <td style="padding:10px 8px;text-align:center;width:16%;">
                <span style="background:{badge_bg};color:{badge_color};padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">{conf:.0%}</span>
            </td>
            <td style="padding:10px 8px;font-size:13px;text-align:center;width:16%;">{leg_icons}</td>
            <td style="padding:10px 4px;text-align:center;">
                <button id="btn_{row_id}"
                    onclick="var d=document.getElementById('{row_id}');var b=document.getElementById('btn_{row_id}');if(d.style.display==='none'){{d.style.display='table-row';b.textContent='✕';}}else{{d.style.display='none';b.textContent='ℹ️';}}"
                    style="background:none;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;font-size:14px;padding:2px 8px;color:#475569;transition:background 0.15s;"
                    onmouseover="this.style.background='#f1f5f9'"
                    onmouseout="this.style.background='none'"
                >ℹ️</button>
            </td>
        </tr>
        <tr id="{row_id}" style="display:none;">
            <td colspan="7" style="padding:8px 16px 12px 32px;background:#ffffff;border-bottom:1px solid #e2e8f0;">
                {detail_legs}
            </td>
        </tr>"""

    return f"""
    <div style="
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        max-width: 700px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="font-weight:700;font-size:15px;color:#1e293b;margin-bottom:12px;">
            📋 Other Options
        </div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:2px solid #e2e8f0;">
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:left;width:15%;">Time</th>
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:left;width:14%;">Duration</th>
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:center;width:13%;">Transfers</th>
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:center;width:14%;">Walking</th>
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:center;width:16%;">Confidence</th>
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:center;width:16%;">Legs</th>
                    <th style="padding:6px 8px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;text-align:center;width:12%;">Info</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """


def _render_best_route_card(journey) -> str:
    """Return an HTML card combining a confidence gauge and route details."""
    from datetime import datetime as _dt

    pct = max(0.0, min(1.0, journey.confidence))
    position_pct = pct * 100

    if pct >= 0.85:
        label, label_color = "Very High", "#15803d"
    elif pct >= 0.65:
        label, label_color = "High", "#4d7c0f"
    elif pct >= 0.45:
        label, label_color = "Moderate", "#ca8a04"
    elif pct >= 0.25:
        label, label_color = "Low", "#ea580c"
    else:
        label, label_color = "Very Low", "#dc2626"

    dep = _dt.fromtimestamp(journey.departure_ts).strftime("%H:%M")
    arr = _dt.fromtimestamp(journey.arrival_ts).strftime("%H:%M")
    dur = journey.travel_time_seconds() / 60

    # Build legs HTML
    legs_html = ""
    for leg in journey.legs:
        t_dep = _dt.fromtimestamp(leg.departure_ts).strftime("%H:%M")
        t_arr = _dt.fromtimestamp(leg.arrival_ts).strftime("%H:%M")
        if leg.leg_type == "transit":
            icon = "🚌"
            detail = leg.trip_id or ""
            color = "#1e40af"
            bg = "#eff6ff"
            border = "#bfdbfe"
            line = f"{t_dep} {leg.from_name} → {t_arr} {leg.to_name}"
            badge = f"<span style='background:{bg};color:{color};border:1px solid {border};padding:2px 9px;border-radius:6px;font-size:12px;margin-left:8px;'>{detail}</span>" if detail else ""
        else:
            icon = "🚶"
            color = "#9a3412"
            bg = "#fff7ed"
            border = "#fed7aa"
            line = f"{t_dep} {leg.from_name} → {t_arr} {leg.to_name}"
            badge = f"<span style='background:{bg};color:{color};border:1px solid {border};padding:2px 9px;border-radius:6px;font-size:12px;margin-left:8px;'>{leg.walk_distance_m:.0f}m</span>"
        legs_html += f"""
        <div style="display:flex;align-items:center;padding:7px 0;border-bottom:1px solid #f1f5f9;font-size:14px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
            <span style="margin-right:10px;font-size:17px;">{icon}</span>
            <span style="color:#334155;">{line}</span>
            {badge}
        </div>"""

    return f"""
    <div style="
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        max-width: 640px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <!-- Title -->
        <div style="font-weight:700;font-size:16px;color:#1e293b;margin-bottom:14px;">
            🏆 Best Route
        </div>

        <!-- Itinerary (moved to top) -->
        <div style="margin-bottom:16px;">
            {legs_html}
        </div>

        <!-- Stats row -->
        <div style="display:flex;gap:10px;flex-wrap:nowrap;margin-bottom:16px;">
            <div style="flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 6px;text-align:center;">
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Departs</div>
                <div style="font-size:18px;font-weight:700;color:#1e293b;">{dep}</div>
            </div>
            <div style="flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 6px;text-align:center;">
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Arrives</div>
                <div style="font-size:18px;font-weight:700;color:#1e293b;">{arr}</div>
            </div>
            <div style="flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 6px;text-align:center;">
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Duration</div>
                <div style="font-size:18px;font-weight:700;color:#1e293b;">{dur:.0f} min</div>
            </div>
            <div style="flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 6px;text-align:center;">
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Transfers</div>
                <div style="font-size:18px;font-weight:700;color:#1e293b;">{journey.num_transfers}</div>
            </div>
            <div style="flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 6px;text-align:center;">
                <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Walking</div>
                <div style="font-size:18px;font-weight:700;color:#1e293b;">{journey.total_walk_m:.0f}m</div>
            </div>
        </div>

        <!-- Confidence gauge -->
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
            <span style="font-weight:600;font-size:13px;color:#334155;">🛡️ Confidence</span>
            <span style="font-weight:700;font-size:20px;color:{label_color};">{pct:.0%}</span>
        </div>
        <div style="position:relative;width:100%;height:12px;border-radius:7px;
            background:linear-gradient(to right,#ef4444 0%,#f97316 25%,#eab308 50%,#84cc16 75%,#22c55e 100%);
            box-shadow:inset 0 1px 3px rgba(0,0,0,0.15);">
            <div style="position:absolute;top:50%;left:{position_pct}%;transform:translate(-50%,-50%);
                width:20px;height:20px;border-radius:50%;background:#fff;border:3px solid #475569;
                box-shadow:0 2px 6px rgba(0,0,0,0.25);"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:5px;font-size:11px;color:#94a3b8;">
            <span>Risky</span>
            <span>Reliable</span>
        </div>
        <div style="text-align:center;margin-top:4px;font-size:12px;color:{label_color};font-weight:600;">{label} Confidence</div>
    </div>
    """


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
