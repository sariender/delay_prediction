"""
Multi-criteria RAPTOR algorithm for robust journey planning.

Supports:
- Forward RAPTOR (earliest arrival given departure time)
- Reverse RAPTOR (latest departure given arrival deadline)
- Pareto-optimal routes by (#transfers, arrival/departure time)
- Walking distance tracking
- Confidence computation via delay prediction model
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.graph import FootPath, Route, Stop, StopEvent, TransitGraph


INF = float("inf")

# ---------------------------------------------------------------------------
# Journey leg representation
# ---------------------------------------------------------------------------

@dataclass
class JourneyLeg:
    """One leg of a journey: either a transit ride or a walk."""
    leg_type: str                  # "transit" or "walk"
    from_stop: str
    to_stop: str
    from_name: str = ""
    to_name: str = ""
    departure_ts: int = 0
    arrival_ts: int = 0
    trip_id: Optional[str] = None
    route_id: Optional[str] = None
    walk_distance_m: float = 0.0
    # For delay model
    stop_id_raw: Optional[str] = None


@dataclass
class Journey:
    """A complete journey from origin to destination."""
    legs: List[JourneyLeg]
    departure_ts: int = 0
    arrival_ts: int = 0
    num_transfers: int = 0
    total_walk_m: float = 0.0
    confidence: float = 1.0

    def travel_time_seconds(self) -> int:
        return max(0, self.arrival_ts - self.departure_ts)

    def __repr__(self):
        from datetime import datetime
        dep = datetime.fromtimestamp(self.departure_ts).strftime("%H:%M:%S")
        arr = datetime.fromtimestamp(self.arrival_ts).strftime("%H:%M:%S")
        mins = self.travel_time_seconds() / 60
        return (
            f"Journey({dep}→{arr}, {mins:.0f}min, "
            f"{self.num_transfers} transfers, "
            f"{self.total_walk_m:.0f}m walk, "
            f"conf={self.confidence:.2%})"
        )


# ---------------------------------------------------------------------------
# RAPTOR label: best known arrival at a stop after k rounds
# ---------------------------------------------------------------------------

@dataclass
class Label:
    arrival_ts: int = 2**31        # best known arrival time
    departure_ts: int = 0          # departure time that achieved this
    trip_id: Optional[str] = None  # trip used to reach this stop
    board_stop: Optional[str] = None  # stop where we boarded the trip
    board_ts: int = 0              # time we boarded
    walk_from: Optional[str] = None  # if reached by walking
    walk_distance: float = 0.0    # cumulative walk distance so far
    total_walk: float = 0.0       # total walk over journey so far


# ---------------------------------------------------------------------------
# Forward RAPTOR
# ---------------------------------------------------------------------------

def _earliest_trip(route: Route, stop_idx: int, after_ts: int) -> Optional[int]:
    """
    Binary search for the earliest trip on `route` that departs from
    stop at position `stop_idx` at or after `after_ts`.
    Returns the trip index or None.
    """
    trips = route.trips
    lo, hi = 0, len(trips) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        dep = trips[mid][stop_idx].departure_ts
        if dep >= after_ts:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return result


def raptor_forward(
    graph: TransitGraph,
    source: str,
    departure_ts: int,
    target: Optional[str] = None,
    max_rounds: int = 8,
    min_transfer_sec: int = 120,
    extra_transfer_sec: int = 0,
    max_walk_m: float = 500.0,
    walking_speed_m_per_min: float = 50.0,
) -> List[Journey]:
    """
    Run forward RAPTOR from `source` at `departure_ts`.

    Returns Pareto-optimal journeys to `target` (or all stops if None),
    where Pareto dimensions are (arrival_time, num_transfers).
    """
    total_transfer_sec = min_transfer_sec + extra_transfer_sec

    # Lower bound: never accept arrivals before we even departed
    min_valid_arrival = departure_ts

    # labels[k][stop] = best Label reaching stop in exactly k transit legs
    labels: List[Dict[str, Label]] = [dict() for _ in range(max_rounds + 1)]
    # best_arrival[stop] = best arrival time across all rounds
    best_arrival: Dict[str, int] = defaultdict(lambda: 2**31)

    # Initialize source
    labels[0][source] = Label(arrival_ts=departure_ts, total_walk=0.0)
    best_arrival[source] = departure_ts

    # Apply initial footpaths from source
    for fp in graph.footpaths.get(source, []):
        if fp.distance_m <= max_walk_m:
            walk_sec = max(1, int(fp.distance_m / walking_speed_m_per_min * 60))
            arr = departure_ts + walk_sec
            if arr < best_arrival[fp.to_stop]:
                best_arrival[fp.to_stop] = arr
                labels[0][fp.to_stop] = Label(
                    arrival_ts=arr,
                    walk_from=source,
                    walk_distance=fp.distance_m,
                    total_walk=fp.distance_m,
                )

    # RAPTOR rounds
    for k in range(1, max_rounds + 1):
        labels[k] = {}
        marked: Set[str] = set()

        # Collect stops improved in previous round
        if k == 1:
            changed_stops = set(labels[0].keys())
        else:
            changed_stops = set(labels[k - 1].keys())

        if not changed_stops:
            break

        # Traverse routes
        routes_to_scan: Dict[str, int] = {}  # route_id -> earliest stop_idx to board
        for stop in changed_stops:
            for rid in graph.routes_at_stop.get(stop, []):
                route = graph.routes[rid]
                idx = route.stop_index.get(stop)
                if idx is not None:
                    if rid not in routes_to_scan or idx < routes_to_scan[rid]:
                        routes_to_scan[rid] = idx

        for rid, start_idx in routes_to_scan.items():
            route = graph.routes[rid]
            current_trip_idx: Optional[int] = None
            board_stop: Optional[str] = None
            board_ts: int = 0
            prev_walk: float = 0.0

            for si in range(start_idx, len(route.stop_sequence)):
                stop = route.stop_sequence[si]
                # Can we catch an earlier trip at this stop?
                arr_here = best_arrival[stop]
                if arr_here < 2**31:
                    depart_after = arr_here + total_transfer_sec if stop != source else arr_here
                    et = _earliest_trip(route, si, depart_after)
                    if et is not None and (current_trip_idx is None or et < current_trip_idx):
                        current_trip_idx = et
                        board_stop = stop
                        board_ts = route.trips[et][si].departure_ts
                        # Track walk distance from how we reached this stop
                        for kk in range(k - 1, -1, -1):
                            if stop in labels[kk]:
                                prev_walk = labels[kk][stop].total_walk
                                break

                if current_trip_idx is None:
                    continue

                # Propagate arrival at subsequent stops
                trip = route.trips[current_trip_idx]
                arr_ts = trip[si].arrival_ts
                # Guard against midnight-wrapping: reject arrivals before
                # the boarding time (i.e. timetable timestamp wrapped to
                # the same calendar day instead of the next).
                if arr_ts < board_ts:
                    continue
                if arr_ts < best_arrival[stop] and arr_ts >= min_valid_arrival:
                    best_arrival[stop] = arr_ts
                    labels[k][stop] = Label(
                        arrival_ts=arr_ts,
                        trip_id=trip[si].trip_id,
                        board_stop=board_stop,
                        board_ts=board_ts,
                        total_walk=prev_walk,
                    )
                    marked.add(stop)

        # Relax footpaths from newly marked stops
        new_marked = set()
        for stop in marked:
            arr = best_arrival[stop]
            curr_walk = 0.0
            if stop in labels[k]:
                curr_walk = labels[k][stop].total_walk

            for fp in graph.footpaths.get(stop, []):
                if curr_walk + fp.distance_m > max_walk_m:
                    continue
                walk_sec = max(1, int(fp.distance_m / walking_speed_m_per_min * 60))
                walk_arr = arr + walk_sec
                if walk_arr < best_arrival[fp.to_stop]:
                    best_arrival[fp.to_stop] = walk_arr
                    labels[k][fp.to_stop] = Label(
                        arrival_ts=walk_arr,
                        walk_from=stop,
                        walk_distance=fp.distance_m,
                        total_walk=curr_walk + fp.distance_m,
                    )
                    new_marked.add(fp.to_stop)

            # Also relax explicit transfers (from transfers.parquet)
            for to_stop, transfer_sec in graph.explicit_transfers.get(stop, []):
                transfer_arr = arr + transfer_sec
                if transfer_arr < best_arrival[to_stop]:
                    best_arrival[to_stop] = transfer_arr
                    labels[k][to_stop] = Label(
                        arrival_ts=transfer_arr,
                        walk_from=stop,
                        walk_distance=0.0,  # explicit transfers don't count as walking
                        total_walk=curr_walk,
                    )
                    new_marked.add(to_stop)

    # Reconstruct Pareto-optimal journeys
    if target is not None:
        return _reconstruct_journeys(graph, labels, source, target, departure_ts, walking_speed_m_per_min)
    else:
        # Return journeys to all reachable stops
        all_journeys = []
        for stop in best_arrival:
            if stop != source and best_arrival[stop] < 2**31:
                js = _reconstruct_journeys(graph, labels, source, stop, departure_ts, walking_speed_m_per_min)
                all_journeys.extend(js)
        return all_journeys


def _reconstruct_journeys(
    graph: TransitGraph,
    labels: List[Dict[str, Label]],
    source: str,
    target: str,
    departure_ts: int,
    walking_speed: float = 50.0,
) -> List[Journey]:
    """Reconstruct Pareto-optimal journeys from RAPTOR labels."""
    journeys = []
    best_arr = 2**31

    for k in range(len(labels)):
        if target not in labels[k]:
            continue
        label = labels[k][target]
        if label.arrival_ts >= best_arr:
            continue
        best_arr = label.arrival_ts

        # Trace back the path
        legs = []
        current_stop = target
        current_round = k

        while current_stop != source and current_round >= 0:
            lbl = None
            for kk in range(current_round, -1, -1):
                if current_stop in labels[kk]:
                    lbl = labels[kk][current_stop]
                    current_round = kk
                    break
            if lbl is None:
                break

            if lbl.walk_from is not None:
                # Walking leg
                from_stop = lbl.walk_from
                legs.append(JourneyLeg(
                    leg_type="walk",
                    from_stop=from_stop,
                    to_stop=current_stop,
                    from_name=graph.stops[from_stop].stop_name if from_stop in graph.stops else "",
                    to_name=graph.stops[current_stop].stop_name if current_stop in graph.stops else "",
                    departure_ts=lbl.arrival_ts - int(lbl.walk_distance / walking_speed * 60),
                    arrival_ts=lbl.arrival_ts,
                    walk_distance_m=lbl.walk_distance,
                ))
                current_stop = from_stop
            elif lbl.trip_id is not None:
                # Transit leg
                board_stop = lbl.board_stop or current_stop
                legs.append(JourneyLeg(
                    leg_type="transit",
                    from_stop=board_stop,
                    to_stop=current_stop,
                    from_name=graph.stops[board_stop].stop_name if board_stop in graph.stops else "",
                    to_name=graph.stops[current_stop].stop_name if current_stop in graph.stops else "",
                    departure_ts=lbl.board_ts,
                    arrival_ts=lbl.arrival_ts,
                    trip_id=lbl.trip_id,
                ))
                current_stop = board_stop
                current_round -= 1
            else:
                break

        legs.reverse()
        if legs:
            num_transit = sum(1 for l in legs if l.leg_type == "transit")
            total_walk = sum(l.walk_distance_m for l in legs if l.leg_type == "walk")
            journey = Journey(
                legs=legs,
                departure_ts=legs[0].departure_ts,
                arrival_ts=legs[-1].arrival_ts,
                num_transfers=max(0, num_transit - 1),
                total_walk_m=total_walk,
            )
            journeys.append(journey)

    return journeys


# ---------------------------------------------------------------------------
# Reverse RAPTOR (latest departure given arrival deadline)
# ---------------------------------------------------------------------------

def _latest_trip_arriving_before(route: Route, stop_idx: int, before_ts: int) -> Optional[int]:
    """Find the latest trip on route arriving at stop_idx before before_ts."""
    trips = route.trips
    lo, hi = 0, len(trips) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        arr = trips[mid][stop_idx].arrival_ts
        if arr <= before_ts:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def raptor_reverse(
    graph: TransitGraph,
    target: str,
    arrival_deadline_ts: int,
    source: Optional[str] = None,
    max_rounds: int = 8,
    min_transfer_sec: int = 120,
    extra_transfer_sec: int = 0,
    max_walk_m: float = 500.0,
    walking_speed_m_per_min: float = 50.0,
) -> List[Journey]:
    """
    Run reverse RAPTOR: find latest departure that arrives at `target`
    by `arrival_deadline_ts`.
    """
    total_transfer_sec = min_transfer_sec + extra_transfer_sec

    # Upper bound: never accept departures after the arrival deadline
    max_valid_departure = arrival_deadline_ts

    # labels[k][stop] = latest departure from this stop in k transit legs to target
    labels: List[Dict[str, Label]] = [dict() for _ in range(max_rounds + 1)]
    best_departure: Dict[str, int] = defaultdict(lambda: 0)

    # Initialize target
    labels[0][target] = Label(arrival_ts=arrival_deadline_ts, departure_ts=arrival_deadline_ts)
    best_departure[target] = arrival_deadline_ts

    # Apply footpaths TO target
    for fp in graph.footpaths.get(target, []):
        if fp.distance_m <= max_walk_m:
            walk_sec = max(1, int(fp.distance_m / walking_speed_m_per_min * 60))
            dep = arrival_deadline_ts - walk_sec
            if dep > best_departure.get(fp.to_stop, 0):
                best_departure[fp.to_stop] = dep
                labels[0][fp.to_stop] = Label(
                    departure_ts=dep,
                    arrival_ts=arrival_deadline_ts,
                    walk_from=target,
                    walk_distance=fp.distance_m,
                    total_walk=fp.distance_m,
                )

    # Also check incoming footpaths
    for stop_id, fps in graph.footpaths.items():
        for fp in fps:
            if fp.to_stop == target and fp.distance_m <= max_walk_m:
                walk_sec = max(1, int(fp.distance_m / walking_speed_m_per_min * 60))
                dep = arrival_deadline_ts - walk_sec
                if dep > best_departure.get(stop_id, 0):
                    best_departure[stop_id] = dep
                    labels[0][stop_id] = Label(
                        departure_ts=dep,
                        arrival_ts=arrival_deadline_ts,
                        walk_from=target,
                        walk_distance=fp.distance_m,
                        total_walk=fp.distance_m,
                    )

    for k in range(1, max_rounds + 1):
        labels[k] = {}
        marked: Set[str] = set()

        changed_stops = set(labels[k - 1].keys()) if k > 1 else set(labels[0].keys())
        if not changed_stops:
            break

        routes_to_scan: Dict[str, int] = {}
        for stop in changed_stops:
            for rid in graph.routes_at_stop.get(stop, []):
                route = graph.routes[rid]
                idx = route.stop_index.get(stop)
                if idx is not None:
                    if rid not in routes_to_scan or idx > routes_to_scan[rid]:
                        routes_to_scan[rid] = idx

        for rid, end_idx in routes_to_scan.items():
            route = graph.routes[rid]
            current_trip_idx: Optional[int] = None
            alight_stop: Optional[str] = None
            alight_ts: int = 0

            # Scan backwards
            for si in range(end_idx, -1, -1):
                stop = route.stop_sequence[si]
                dep_here = best_departure.get(stop, 0)
                if dep_here > 0 and (source is None or stop != source):
                    arrive_before = dep_here - total_transfer_sec if stop != target else dep_here
                    lt = _latest_trip_arriving_before(route, si, arrive_before)
                    if lt is not None and (current_trip_idx is None or lt > current_trip_idx):
                        current_trip_idx = lt
                        alight_stop = stop
                        alight_ts = route.trips[lt][si].arrival_ts

                if current_trip_idx is None:
                    continue

                trip = route.trips[current_trip_idx]
                dep_ts = trip[si].departure_ts
                # Guard against midnight-wrapping: reject departures after
                # the alight time (timestamp wrapped to wrong day).
                if dep_ts > alight_ts:
                    continue
                if dep_ts > best_departure.get(stop, 0) and dep_ts <= max_valid_departure:
                    best_departure[stop] = dep_ts
                    labels[k][stop] = Label(
                        departure_ts=dep_ts,
                        arrival_ts=alight_ts,
                        trip_id=trip[si].trip_id,
                        board_stop=alight_stop,
                        board_ts=alight_ts,
                    )
                    marked.add(stop)

        # Footpaths
        for stop in list(marked):
            dep = best_departure[stop]
            for fp in graph.footpaths.get(stop, []):
                if fp.distance_m > max_walk_m:
                    continue
                walk_sec = max(1, int(fp.distance_m / walking_speed_m_per_min * 60))
                buffer_sec = 0 if fp.to_stop == source else total_transfer_sec
                walk_dep = dep - walk_sec - buffer_sec
                if walk_dep > best_departure.get(fp.to_stop, 0):
                    best_departure[fp.to_stop] = walk_dep
                    labels[k][fp.to_stop] = Label(
                        departure_ts=walk_dep,
                        walk_from=stop,
                        walk_distance=fp.distance_m,
                    )

            # Explicit transfers (reverse: arriving at `stop` from a predecessor)
            for to_stop, transfer_sec in graph.explicit_transfers.get(stop, []):
                buffer_sec = 0 if to_stop == source else total_transfer_sec
                xfer_dep = dep - transfer_sec - buffer_sec
                if xfer_dep > best_departure.get(to_stop, 0):
                    best_departure[to_stop] = xfer_dep
                    labels[k][to_stop] = Label(
                        departure_ts=xfer_dep,
                        walk_from=stop,
                        walk_distance=0.0,
                    )

    # Reconstruct for source
    if source is not None:
        return _reconstruct_reverse_journeys(graph, labels, source, target, best_departure, walking_speed_m_per_min)
    return []


def _reconstruct_reverse_journeys(
    graph: TransitGraph,
    labels: List[Dict[str, Label]],
    source: str,
    target: str,
    best_departure: Dict[str, int],
    walking_speed: float = 50.0,
) -> List[Journey]:
    """Reconstruct journeys from reverse RAPTOR labels."""
    journeys = []
    best_dep = 0

    for k in range(len(labels)):
        if source not in labels[k]:
            continue
        label = labels[k][source]
        dep_ts = label.departure_ts
        if dep_ts <= best_dep:
            continue
        best_dep = dep_ts

        # Trace forward from source to target
        legs = []
        current_stop = source
        current_round = k

        while current_stop != target and current_round >= 0:
            lbl = None
            for kk in range(current_round, -1, -1):
                if current_stop in labels[kk]:
                    lbl = labels[kk][current_stop]
                    current_round = kk
                    break
            if lbl is None:
                break

            if lbl.walk_from is not None:
                legs.append(JourneyLeg(
                    leg_type="walk",
                    from_stop=current_stop,
                    to_stop=lbl.walk_from,
                    from_name=graph.stops.get(current_stop, Stop(current_stop, "", 0, 0)).stop_name,
                    to_name=graph.stops.get(lbl.walk_from, Stop(lbl.walk_from, "", 0, 0)).stop_name,
                    departure_ts=lbl.departure_ts,
                    arrival_ts=lbl.departure_ts + int(lbl.walk_distance / walking_speed * 60),
                    walk_distance_m=lbl.walk_distance,
                ))
                current_stop = lbl.walk_from
            elif lbl.trip_id is not None:
                dest = lbl.board_stop or target
                legs.append(JourneyLeg(
                    leg_type="transit",
                    from_stop=current_stop,
                    to_stop=dest,
                    from_name=graph.stops.get(current_stop, Stop(current_stop, "", 0, 0)).stop_name,
                    to_name=graph.stops.get(dest, Stop(dest, "", 0, 0)).stop_name,
                    departure_ts=lbl.departure_ts,
                    arrival_ts=lbl.board_ts,
                    trip_id=lbl.trip_id,
                ))
                current_stop = dest
                current_round -= 1
            else:
                break

        if legs:
            num_transit = sum(1 for l in legs if l.leg_type == "transit")
            total_walk = sum(l.walk_distance_m for l in legs if l.leg_type == "walk")
            journey = Journey(
                legs=legs,
                departure_ts=legs[0].departure_ts,
                arrival_ts=legs[-1].arrival_ts,
                num_transfers=max(0, num_transit - 1),
                total_walk_m=total_walk,
            )
            journeys.append(journey)

    return journeys
