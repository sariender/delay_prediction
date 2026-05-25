"""
Graph data structures for the RAPTOR-based route planner.

Builds in-memory indexes from the February timetable parquet data
loaded via PySpark.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stop:
    stop_id: str
    stop_name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class StopEvent:
    """One scheduled stop on a trip, with times as epoch seconds."""
    stop_id: str
    arrival_ts: int        # unix-epoch seconds
    departure_ts: int      # unix-epoch seconds
    stop_sequence: int
    trip_id: str
    stop_name: str


@dataclass(frozen=True)
class FootPath:
    """Walking connection between two stops."""
    from_stop: str
    to_stop: str
    distance_m: float
    walk_seconds: int      # pre-computed at default speed


@dataclass
class Route:
    """
    A RAPTOR 'route' = an ordered sequence of stops served by ≥1 trip.
    All trips on the same route visit the same stops in the same order.
    """
    route_id: str
    stop_sequence: List[str]           # ordered stop_ids
    stop_index: Dict[str, int] = field(default_factory=dict)
    trips: List[List[StopEvent]] = field(default_factory=list)

    def __post_init__(self):
        self.stop_index = {s: i for i, s in enumerate(self.stop_sequence)}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Transit graph built from Spark DataFrames
# ---------------------------------------------------------------------------

class TransitGraph:
    """
    In-memory representation of the transit network for one operating day.
    """

    def __init__(self):
        self.stops: Dict[str, Stop] = {}
        self.footpaths: Dict[str, List[FootPath]] = defaultdict(list)
        self.routes: Dict[str, Route] = {}
        self.routes_at_stop: Dict[str, List[str]] = defaultdict(list)
        # trip_id -> route_id mapping
        self.trip_to_route: Dict[str, str] = {}
        # For delay model: trip_id+stop_id -> original row info
        self.event_info: Dict[Tuple[str, str], dict] = {}
        # stop_id -> explicit transfers from transfers.parquet:
        # (to_stop_id, transfer_seconds, geometric_distance_m)
        self.explicit_transfers: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
        self.trip_modes: Dict[str, str] = {}

    def build_from_spark(
        self,
        timetable_df,
        transfers_df,
        target_date: str,
        walking_speed_m_per_min: float = 50.0,
        max_walk_m: float = 500.0,
        min_transfer_seconds: int = 120,
    ):
        """
        Build the transit graph from Spark DataFrames.

        timetable_df: the February timetable parquet
        transfers_df: transfers parquet
        target_date: e.g. '2026-02-04' to filter to one day
        """
        import pyspark.sql.functions as F
        from datetime import datetime

        # --- 1. Filter timetable to target date ---
        day_df = timetable_df.filter(
            F.to_date(F.col("departure_timestamp")) == F.lit(target_date)
        )

        rows = day_df.collect()
        if not rows:
            raise ValueError(f"No timetable rows for date {target_date}")

        # --- 2. Build stops ---
        for r in rows:
            sid = str(r["stop_id"]).split(":")[0]
            if sid not in self.stops:
                self.stops[sid] = Stop(
                    stop_id=sid,
                    stop_name=r["stop_name"],
                    lat=float(r["stop_lat"]),
                    lon=float(r["stop_lon"]),
                )

        # --- 3. Group rows by trip_id, sort by stop_sequence ---
        trips_dict: Dict[str, List] = defaultdict(list)
        for r in rows:
            sid = str(r["stop_id"]).split(":")[0]
            arr_ts = r["arrival_timestamp"]
            dep_ts = r["departure_timestamp"]
            if arr_ts is None or dep_ts is None:
                continue
            arr_epoch = int(arr_ts.timestamp()) if isinstance(arr_ts, datetime) else int(datetime.fromisoformat(str(arr_ts)).timestamp())
            dep_epoch = int(dep_ts.timestamp()) if isinstance(dep_ts, datetime) else int(datetime.fromisoformat(str(dep_ts)).timestamp())

            evt = StopEvent(
                stop_id=sid,
                arrival_ts=arr_epoch,
                departure_ts=dep_epoch,
                stop_sequence=int(r["stop_sequence"]),
                trip_id=r["trip_id"],
                stop_name=r["stop_name"],
            )

            if r["trip_id"] not in self.trip_modes:
                try:
                    self.trip_modes[r["trip_id"]] = r["transport_clean"] or ""
                except Exception:
                    pass
                    
            trips_dict[r["trip_id"]].append(evt)

            # Store event info for delay model
            self.event_info[(r["trip_id"], sid)] = {
                "trip_id": r["trip_id"],
                "stop_id": r["stop_id"],
                "arrival_timestamp": str(arr_ts),
                "departure_timestamp": str(dep_ts),
                "date_value": target_date,
            }

        # Sort each trip by stop_sequence
        for tid in trips_dict:
            trips_dict[tid].sort(key=lambda e: e.stop_sequence)

        # --- 4. Group trips into RAPTOR routes ---
        # Two trips belong to the same route if they visit the same
        # sequence of stops.
        pattern_to_route: Dict[Tuple[str, ...], str] = {}
        route_counter = 0

        for tid, events in trips_dict.items():
            pattern = tuple(e.stop_id for e in events)
            if pattern not in pattern_to_route:
                rid = f"R{route_counter}"
                route_counter += 1
                pattern_to_route[pattern] = rid
                self.routes[rid] = Route(
                    route_id=rid,
                    stop_sequence=list(pattern),
                )
                for s in pattern:
                    self.routes_at_stop[s].append(rid)
            rid = pattern_to_route[pattern]
            self.routes[rid].trips.append(events)
            self.trip_to_route[tid] = rid

        # Sort trips within each route by departure time at first stop
        for route in self.routes.values():
            route.trips.sort(key=lambda evts: evts[0].departure_ts)

        # --- 5. Build walking footpaths ---
        stop_list = list(self.stops.values())
        for i in range(len(stop_list)):
            for j in range(i + 1, len(stop_list)):
                s1, s2 = stop_list[i], stop_list[j]
                d = haversine_m(s1.lat, s1.lon, s2.lat, s2.lon)
                if d <= max_walk_m:
                    walk_sec = max(1, int(math.ceil(d / walking_speed_m_per_min * 60)))
                    fp_ab = FootPath(s1.stop_id, s2.stop_id, d, walk_sec)
                    fp_ba = FootPath(s2.stop_id, s1.stop_id, d, walk_sec)
                    self.footpaths[s1.stop_id].append(fp_ab)
                    self.footpaths[s2.stop_id].append(fp_ba)

        # --- 6. Load explicit transfers ---
        if transfers_df is not None:
            try:
                t_rows = transfers_df.collect()
                for r in t_rows:
                    from_id = str(r["from_stop_id"]).split(":")[0]
                    to_id = str(r["to_stop_id"]).split(":")[0]
                    t_time = int(r["min_transfer_time"]) if r["min_transfer_time"] is not None else min_transfer_seconds
                    if from_id in self.stops and to_id in self.stops:
                        from_stop = self.stops[from_id]
                        to_stop = self.stops[to_id]
                        distance_m = haversine_m(from_stop.lat, from_stop.lon, to_stop.lat, to_stop.lon)
                        self.explicit_transfers[from_id].append((to_id, t_time, distance_m))
            except Exception:
                pass  # transfers may have different schema

        self._min_transfer_seconds = min_transfer_seconds
        return self

    def build_from_pandas(
        self,
        timetable_pdf,
        transfers_pdf=None,
        walking_speed_m_per_min: float = 50.0,
        max_walk_m: float = 500.0,
        min_transfer_seconds: int = 120,
    ):
        """
        Build graph from pandas DataFrames (already filtered to one day).
        Useful for local testing without Spark.
        """
        from datetime import datetime

        for _, r in timetable_pdf.iterrows():
            sid = str(r["stop_id"]).split(":")[0]
            if sid not in self.stops:
                self.stops[sid] = Stop(
                    stop_id=sid,
                    stop_name=r["stop_name"],
                    lat=float(r["stop_lat"]),
                    lon=float(r["stop_lon"]),
                )

        trips_dict: Dict[str, List] = defaultdict(list)
        for _, r in timetable_pdf.iterrows():
            sid = str(r["stop_id"]).split(":")[0]
            arr_ts = r["arrival_timestamp"]
            dep_ts = r["departure_timestamp"]
            if arr_ts is None or dep_ts is None:
                continue
            if isinstance(arr_ts, str):
                arr_ts = datetime.fromisoformat(arr_ts)
            if isinstance(dep_ts, str):
                dep_ts = datetime.fromisoformat(dep_ts)
            arr_epoch = int(arr_ts.timestamp())
            dep_epoch = int(dep_ts.timestamp())

            evt = StopEvent(
                stop_id=sid,
                arrival_ts=arr_epoch,
                departure_ts=dep_epoch,
                stop_sequence=int(r["stop_sequence"]),
                trip_id=r["trip_id"],
                stop_name=r["stop_name"],
            )
            trips_dict[r["trip_id"]].append(evt)

            self.event_info[(r["trip_id"], sid)] = {
                "trip_id": r["trip_id"],
                "stop_id": r["stop_id"],
                "arrival_timestamp": str(arr_ts),
                "departure_timestamp": str(dep_ts),
            }

        for tid in trips_dict:
            trips_dict[tid].sort(key=lambda e: e.stop_sequence)

        pattern_to_route: Dict[Tuple[str, ...], str] = {}
        route_counter = 0

        for tid, events in trips_dict.items():
            pattern = tuple(e.stop_id for e in events)
            if pattern not in pattern_to_route:
                rid = f"R{route_counter}"
                route_counter += 1
                pattern_to_route[pattern] = rid
                self.routes[rid] = Route(
                    route_id=rid,
                    stop_sequence=list(pattern),
                )
                for s in pattern:
                    self.routes_at_stop[s].append(rid)
            rid = pattern_to_route[pattern]
            self.routes[rid].trips.append(events)
            self.trip_to_route[tid] = rid

        for route in self.routes.values():
            route.trips.sort(key=lambda evts: evts[0].departure_ts)

        stop_list = list(self.stops.values())
        for i in range(len(stop_list)):
            for j in range(i + 1, len(stop_list)):
                s1, s2 = stop_list[i], stop_list[j]
                d = haversine_m(s1.lat, s1.lon, s2.lat, s2.lon)
                if d <= max_walk_m:
                    walk_sec = max(1, int(math.ceil(d / walking_speed_m_per_min * 60)))
                    fp_ab = FootPath(s1.stop_id, s2.stop_id, d, walk_sec)
                    fp_ba = FootPath(s2.stop_id, s1.stop_id, d, walk_sec)
                    self.footpaths[s1.stop_id].append(fp_ab)
                    self.footpaths[s2.stop_id].append(fp_ba)

        if transfers_pdf is not None:
            for _, r in transfers_pdf.iterrows():
                from_id = str(r["from_stop_id"]).split(":")[0]
                to_id = str(r["to_stop_id"]).split(":")[0]
                raw_time = r.get("min_transfer_time", None)
                if raw_time is None or (isinstance(raw_time, float) and math.isnan(raw_time)):
                    t_time = min_transfer_seconds
                else:
                    t_time = int(raw_time)
                if from_id in self.stops and to_id in self.stops:
                    from_stop = self.stops[from_id]
                    to_stop = self.stops[to_id]
                    distance_m = haversine_m(from_stop.lat, from_stop.lon, to_stop.lat, to_stop.lon)
                    self.explicit_transfers[from_id].append((to_id, t_time, distance_m))

        self._min_transfer_seconds = min_transfer_seconds
        return self
