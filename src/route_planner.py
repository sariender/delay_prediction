"""
RobustJourneyPlanner — high-level API for robust route planning.

Wraps the RAPTOR algorithm with delay model integration and
multi-criteria route selection.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.graph import TransitGraph, haversine_m
from src.raptor import (
    Journey,
    JourneyLeg,
    raptor_forward,
    raptor_reverse,
)


class RobustJourneyPlanner:
    """
    Robust journey planner with configurable options.

    Usage (Spark):
        planner = RobustJourneyPlanner()
        planner.build_graph(spark, target_date="2026-02-04")
        journeys = planner.plan(
            source="8592076",
            target="8501214",
            mode="fastest",
            departure_time="12:30",
        )

    Usage (pandas, for testing):
        planner = RobustJourneyPlanner()
        planner.build_graph_pandas(timetable_pdf)
        journeys = planner.plan(...)
    """

    # HDFS paths
    TIMETABLE_PATH = "/user/groups/com-490/H1/final/v1/input_from_data_side/timetable_february.parquet"
    TRANSFERS_PATH = "/user/groups/com-490/H1/final/v1/input_from_data_side/transfers.parquet"
    LOOKUP_PATH = "/user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet"

    # Map day-of-week name to a representative February 2026 date
    _DAY_TO_DATE = {
        "monday": "2026-02-02",
        "tuesday": "2026-02-03",
        "wednesday": "2026-02-04",
        "thursday": "2026-02-05",
        "friday": "2026-02-06",
        "saturday": "2026-02-07",
        "sunday": "2026-02-08",
    }

    def __init__(
        self,
        walking_speed_m_per_min: float = 50.0,
        max_walk_m: float = 500.0,
        min_transfer_sec: int = 120,
        extra_transfer_sec: int = 0,
        min_confidence: float = 0.0,
        max_rounds: int = 8,
    ):
        self.walking_speed = walking_speed_m_per_min
        self.max_walk_m = max_walk_m
        self.min_transfer_sec = min_transfer_sec
        self.extra_transfer_sec = extra_transfer_sec
        self.min_confidence = min_confidence
        self.max_rounds = max_rounds
        # Multi-day storage: day_name -> (TransitGraph, date_str)
        self._graphs: Dict[str, Tuple[TransitGraph, str]] = {}
        # Active graph for current query (set by plan())
        self.graph: Optional[TransitGraph] = None
        self._delay_fn: Optional[Callable] = None
        self._delay_dict: Optional[Dict] = None  # pre-collected lookup
        self._target_date: Optional[str] = None
        self._active_day: str = "wednesday"
        self._spark = None
        self._region_uuids: List[str] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def prepare(
        self,
        spark,
        regions: List[str],
        timetable_path: Optional[str] = None,
        transfers_path: Optional[str] = None,
    ):
        """
        Initialize the route planner for the given region UUIDs.

        Precomputes RAPTOR graphs for all 7 days of the first week of
        February 2026 (Mon Feb 2 – Sun Feb 8). The `plan()` method
        accepts a `day` parameter to select the appropriate graph.

        Also initializes the delay prediction model (lookup table is
        pre-collected into memory for fast routing).

        Parameters:
            spark: SparkSession
            regions: list of region UUID strings from iceberg.com490_iceberg.geo
        """
        self._spark = spark
        self._region_uuids = [str(r) for r in regions]

        tt_path = timetable_path or self.TIMETABLE_PATH
        tr_path = transfers_path or self.TRANSFERS_PATH

        timetable_df = spark.read.parquet(tt_path)
        try:
            transfers_df = spark.read.parquet(tr_path)
        except Exception:
            transfers_df = None

        # Build a graph for each day of the week
        for day_name, date_str in self._DAY_TO_DATE.items():
            print(f"Building graph for {day_name} ({date_str})...")
            g = TransitGraph()
            g.build_from_spark(
                timetable_df=timetable_df,
                transfers_df=transfers_df,
                target_date=date_str,
                walking_speed_m_per_min=self.walking_speed,
                max_walk_m=self.max_walk_m,
                min_transfer_seconds=self.min_transfer_sec,
            )
            self._graphs[day_name] = (g, date_str)
            print(f"  {day_name}: {len(g.stops)} stops, {len(g.routes)} routes")

        # Set default active graph (wednesday)
        self._set_active_day("wednesday")

        # Initialize delay model
        try:
            self.init_delay_model(spark)
        except Exception as e:
            print(f"Warning: delay model init failed ({e}), confidence will be 1.0")

        print(f"\n✅ Planner ready — {len(self._graphs)} daily graphs precomputed")
        return self

    def _set_active_day(self, day: str):
        """Switch the active graph to the given day of the week."""
        day = day.strip().lower()
        if day not in self._graphs:
            raise ValueError(
                f"No graph for '{day}'. Available: {list(self._graphs.keys())}"
            )
        self.graph, self._target_date = self._graphs[day]
        self._active_day = day

    def _pick_date_for_day(self, day: str) -> str:
        """Return a representative February 2026 date for the given weekday."""
        day = day.strip().lower()
        if day in self._DAY_TO_DATE:
            return self._DAY_TO_DATE[day]
        raise ValueError(f"Unknown day '{day}'. Use monday-sunday.")

    def build_graph(
        self,
        spark,
        target_date: str = "2026-02-04",
        timetable_path: Optional[str] = None,
        transfers_path: Optional[str] = None,
    ):
        """
        Build a single transit graph from HDFS parquet via Spark.

        For multi-day support, use prepare() instead.
        """
        self._spark = spark
        tt_path = timetable_path or self.TIMETABLE_PATH
        tr_path = transfers_path or self.TRANSFERS_PATH

        timetable_df = spark.read.parquet(tt_path)
        try:
            transfers_df = spark.read.parquet(tr_path)
        except Exception:
            transfers_df = None

        g = TransitGraph()
        g.build_from_spark(
            timetable_df=timetable_df,
            transfers_df=transfers_df,
            target_date=target_date,
            walking_speed_m_per_min=self.walking_speed,
            max_walk_m=self.max_walk_m,
            min_transfer_seconds=self.min_transfer_sec,
        )
        self.graph = g
        self._target_date = target_date

        # Also register in the multi-day dict
        for day_name, date_str in self._DAY_TO_DATE.items():
            if date_str == target_date:
                self._graphs[day_name] = (g, target_date)
                self._active_day = day_name
                break

        print(f"Graph built: {len(g.stops)} stops, {len(g.routes)} routes")
        return self

    def build_graph_pandas(self, timetable_pdf, transfers_pdf=None):
        """Build transit graph from pandas DataFrames (local testing)."""
        self.graph = TransitGraph()
        self.graph.build_from_pandas(
            timetable_pdf=timetable_pdf,
            transfers_pdf=transfers_pdf,
            walking_speed_m_per_min=self.walking_speed,
            max_walk_m=self.max_walk_m,
            min_transfer_seconds=self.min_transfer_sec,
        )
        return self

    def init_delay_model(self, spark, lookup_path: Optional[str] = None):
        """
        Initialize the delay prediction lookup for confidence computation.

        Pre-collects the lookup table into a Python dict keyed by
        (trip_id, bpuic, operating_day) for fast in-memory access
        during routing (avoids per-transfer Spark jobs).
        """
        from src.model.delay_lookup import (
            init_delay_oracle,
            delay_prob,
            prob_from_quantiles,
            QUANTILE_POINTS,
        )
        import pyspark.sql.functions as F

        path = lookup_path or self.LOOKUP_PATH
        lookup_df = init_delay_oracle(spark, path)
        self._delay_fn = delay_prob  # keep as fallback

        # Pre-collect into Python dict for fast per-transfer lookups
        print("Pre-collecting delay lookup into memory...")
        q_cols = [qc for _, qc in QUANTILE_POINTS]
        select_cols = ["trip_id", "bpuic", "operating_day", "scheduled_arrival_ts"] + q_cols
        rows = lookup_df.select(*select_cols).collect()

        self._delay_dict = {}
        for r in rows:
            key = (r["trip_id"], str(r["bpuic"]), str(r["operating_day"]))
            row_dict = r.asDict()
            self._delay_dict[key] = row_dict
            # Also key by (trip_id, bpuic, operating_day, arrival_ts) for exact match
            if r["scheduled_arrival_ts"] is not None:
                key2 = (r["trip_id"], str(r["bpuic"]), str(r["operating_day"]), str(r["scheduled_arrival_ts"]))
                self._delay_dict[key2] = row_dict

        print(f"Delay oracle initialized: {len(rows)} lookup rows pre-collected")
        return self

    def _delay_prob_fast(self, trip_id: str, stop_id: str, date_value: str,
                         X: float, scheduled_arrival_ts: Optional[str] = None) -> float:
        """
        Fast in-memory delay probability lookup.
        Falls back to Spark-based lookup if dict miss.
        """
        from src.model.delay_lookup import prob_from_quantiles

        if X <= 0:
            return 0.0

        bpuic = stop_id.split(":")[0]

        # Try exact match with arrival timestamp first
        if scheduled_arrival_ts is not None:
            key = (trip_id, bpuic, date_value, scheduled_arrival_ts)
            row = self._delay_dict.get(key)
            if row is not None:
                return prob_from_quantiles(row, X)

        # Fallback to (trip_id, bpuic, date)
        key = (trip_id, bpuic, date_value)
        row = self._delay_dict.get(key)
        if row is not None:
            return prob_from_quantiles(row, X)

        # Conservative fallback: unknown → unsafe
        return 0.0

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def _parse_time(self, time_str: str, day: Optional[str] = None) -> int:
        """
        Parse HH:MM or HH:MM:SS into epoch seconds.

        Uses the date corresponding to `day` (or the active day).
        """
        parts = time_str.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0

        # Determine the date to use
        if day is not None:
            date_str = self._pick_date_for_day(day)
        elif self._target_date:
            date_str = self._target_date
        else:
            date_str = "2026-02-04"  # default

        dt = datetime.fromisoformat(date_str)
        dt = dt.replace(hour=h, minute=m, second=s)
        return int(dt.timestamp())

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def compute_confidence(self, journey: Journey) -> float:
        """
        Compute route confidence as product of transfer success probabilities.

        For each transfer point, computes:
            X = spare_time = next_departure - arrival - walk_time - min_transfer - extra_buffer
        Then looks up P(delay <= X) from the pre-collected delay dict.
        Route confidence = product of all such probabilities.
        """
        if self._delay_fn is None and self._delay_dict is None:
            return 1.0  # No delay model → assume perfect

        confidence = 1.0
        for i, leg in enumerate(journey.legs):
            if leg.leg_type != "transit":
                continue

            # Check if there's a transfer BEFORE this transit leg
            if i == 0:
                continue  # First leg, no transfer risk

            prev_leg = journey.legs[i - 1]
            if prev_leg.leg_type == "transit":
                # Transit-to-transit transfer
                spare_time = leg.departure_ts - prev_leg.arrival_ts - self.min_transfer_sec - self.extra_transfer_sec
            elif prev_leg.leg_type == "walk":
                # Walk transfer — use configurable speed
                walk_sec = int(prev_leg.walk_distance_m / self.walking_speed * 60)
                spare_time = leg.departure_ts - prev_leg.departure_ts - walk_sec - self.min_transfer_sec - self.extra_transfer_sec
            else:
                continue

            # Find the incoming transit leg (the one whose delay risks the transfer)
            incoming = journey.legs[i - 1] if journey.legs[i - 1].leg_type == "transit" else None
            if incoming is None:
                # Walk before transit — find the transit leg before the walk
                for j in range(i - 2, -1, -1):
                    if journey.legs[j].leg_type == "transit":
                        incoming = journey.legs[j]
                        break

            if incoming is not None and incoming.trip_id is not None:
                info = self.graph.event_info.get((incoming.trip_id, incoming.to_stop), {})
                if info:
                    try:
                        # Use fast dict lookup if available
                        if self._delay_dict is not None:
                            p = self._delay_prob_fast(
                                trip_id=incoming.trip_id,
                                stop_id=info.get("stop_id", incoming.to_stop),
                                date_value=info.get("date_value", self._target_date or ""),
                                X=max(0, spare_time),
                                scheduled_arrival_ts=info.get("arrival_timestamp"),
                            )
                        else:
                            p = self._delay_fn(
                                trip_id=incoming.trip_id,
                                stop_id=info.get("stop_id", incoming.to_stop),
                                date_value=info.get("date_value", self._target_date),
                                X=max(0, spare_time),
                                scheduled_arrival_ts=info.get("arrival_timestamp"),
                            )
                        confidence *= max(p, 0.001)
                    except Exception:
                        confidence *= 0.5  # fallback

        return confidence

    def _annotate_confidence(self, journeys: List[Journey]) -> List[Journey]:
        """Add confidence scores to journeys."""
        for j in journeys:
            j.confidence = self.compute_confidence(j)
        return journeys

    # ------------------------------------------------------------------
    # Main planning API
    # ------------------------------------------------------------------

    def plan(
        self,
        source: str,
        target: str,
        mode: str = "fastest",
        departure_time: Optional[str] = None,
        arrival_time: Optional[str] = None,
        day: Optional[str] = None,
        min_confidence: Optional[float] = None,
        walking_speed: Optional[float] = None,
        max_walk_m: Optional[float] = None,
        extra_transfer_sec: Optional[int] = None,
        max_results: int = 5,
    ) -> List[Journey]:
        """
        Plan journeys between source and target.

        Modes:
            "fastest"       — earliest arrival (requires departure_time)
            "latest_departure" — latest departure to arrive on time (requires arrival_time)
            "least_transfers" — fewest transfers among Pareto-optimal routes
            "least_walking"  — least walking distance
            "safest"         — highest confidence route
            "all"            — return all Pareto-optimal routes

        Returns list of Journey objects sorted according to mode.
        """
        # Select the correct daily graph
        if day is not None:
            self._set_active_day(day)
        if self.graph is None:
            raise RuntimeError(
                "No graph available. Call prepare() or build_graph() first."
            )

        source = str(source)
        target = str(target)
        ws = walking_speed if walking_speed is not None else self.walking_speed
        mw = max_walk_m if max_walk_m is not None else self.max_walk_m
        et = extra_transfer_sec if extra_transfer_sec is not None else self.extra_transfer_sec
        mc = min_confidence if min_confidence is not None else self.min_confidence

        if mode in ("latest_departure",) and arrival_time is not None:
            # Reverse RAPTOR
            arr_ts = self._parse_time(arrival_time, day=day)
            journeys = raptor_reverse(
                graph=self.graph,
                target=target,
                arrival_deadline_ts=arr_ts,
                source=source,
                max_rounds=self.max_rounds,
                min_transfer_sec=self.min_transfer_sec,
                extra_transfer_sec=et,
                max_walk_m=mw,
                walking_speed_m_per_min=ws,
            )
        else:
            # Forward RAPTOR
            if departure_time is None:
                raise ValueError("departure_time required for forward search")
            dep_ts = self._parse_time(departure_time, day=day)
            journeys = raptor_forward(
                graph=self.graph,
                source=source,
                departure_ts=dep_ts,
                target=target,
                max_rounds=self.max_rounds,
                min_transfer_sec=self.min_transfer_sec,
                extra_transfer_sec=et,
                max_walk_m=mw,
                walking_speed_m_per_min=ws,
            )

        if not journeys:
            return []

        # Compute confidence for all routes
        self._annotate_confidence(journeys)

        # Filter by minimum confidence
        if mc > 0:
            journeys = [j for j in journeys if j.confidence >= mc]

        # Sort based on mode
        if mode == "fastest":
            journeys.sort(key=lambda j: j.arrival_ts)
        elif mode == "latest_departure":
            journeys.sort(key=lambda j: -j.departure_ts)
        elif mode == "least_transfers":
            journeys.sort(key=lambda j: (j.num_transfers, j.arrival_ts))
        elif mode == "least_walking":
            journeys.sort(key=lambda j: (j.total_walk_m, j.arrival_ts))
        elif mode == "safest":
            journeys.sort(key=lambda j: (-j.confidence, j.arrival_ts))
        else:  # "all"
            journeys.sort(key=lambda j: j.arrival_ts)

        return journeys[:max_results]

    # ------------------------------------------------------------------
    # Convenience: get stop list for UI dropdowns
    # ------------------------------------------------------------------

    def get_stops(self) -> List[Dict[str, Any]]:
        """Return list of stop dicts for UI, sorted by name."""
        if self.graph is None:
            return []
        stops = [
            {"stop_id": s.stop_id, "stop_name": s.stop_name, "lat": s.lat, "lon": s.lon}
            for s in self.graph.stops.values()
        ]
        stops.sort(key=lambda s: s["stop_name"])
        return stops

    # ------------------------------------------------------------------
    # Format journey for display
    # ------------------------------------------------------------------

    @staticmethod
    def format_journey(journey: Journey) -> str:
        """Format a journey into a readable string."""
        lines = []
        dep = datetime.fromtimestamp(journey.departure_ts).strftime("%H:%M")
        arr = datetime.fromtimestamp(journey.arrival_ts).strftime("%H:%M")
        mins = journey.travel_time_seconds() / 60
        lines.append(
            f"🕐 {dep} → {arr} ({mins:.0f} min) | "
            f"🔄 {journey.num_transfers} transfers | "
            f"🚶 {journey.total_walk_m:.0f}m walk | "
            f"✅ {journey.confidence:.0%} confidence"
        )
        for leg in journey.legs:
            t_dep = datetime.fromtimestamp(leg.departure_ts).strftime("%H:%M")
            t_arr = datetime.fromtimestamp(leg.arrival_ts).strftime("%H:%M")
            if leg.leg_type == "transit":
                lines.append(f"  🚌 {t_dep} {leg.from_name} → {t_arr} {leg.to_name} [{leg.trip_id}]")
            else:
                lines.append(f"  🚶 {t_dep} {leg.from_name} → {t_arr} {leg.to_name} ({leg.walk_distance_m:.0f}m)")
        return "\n".join(lines)

    @staticmethod
    def journeys_to_pandas(journeys: List[Journey]):
        """Convert journeys to a pandas DataFrame for display."""
        import pandas as pd
        rows = []
        for i, j in enumerate(journeys):
            dep = datetime.fromtimestamp(j.departure_ts).strftime("%H:%M:%S")
            arr = datetime.fromtimestamp(j.arrival_ts).strftime("%H:%M:%S")
            rows.append({
                "Option": i + 1,
                "Departure": dep,
                "Arrival": arr,
                "Duration (min)": round(j.travel_time_seconds() / 60, 1),
                "Transfers": j.num_transfers,
                "Walking (m)": round(j.total_walk_m),
                "Confidence": f"{j.confidence:.0%}",
                "Legs": len(j.legs),
            })
        return pd.DataFrame(rows)
