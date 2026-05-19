# +
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

INF = datetime.max
MAX_ROUNDS = 8

StopId  = str
TripKey = str  
RouteId = str

QUANTILE_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50,
                   0.65, 0.80, 0.90, 0.95, 0.99]
QUANTILE_COLS   = ["q01", "q05", "q10", "q20", "q35", "q50",
                   "q65", "q80", "q90", "q95", "q99"]


def delay_prob(quantile_row: Dict, X: float) -> float:
    """P(delay <= X seconds) via linear interpolation between known quantiles"""
    qvals = [quantile_row[q] for q in QUANTILE_COLS]
    if X < qvals[0]:
        return 0.0
    if X >= qvals[-1]:
        return 1.0
    for i in range(len(qvals) - 1):
        if qvals[i] <= X < qvals[i + 1]:
            x0, x1 = qvals[i], qvals[i + 1]
            p0, p1 = QUANTILE_LEVELS[i], QUANTILE_LEVELS[i + 1]
            return p0 + (p1 - p0) * (X - x0) / (x1 - x0)
    return 1.0


@dataclass
class Leg:
    """One segment of a journey: trip ride or walk"""
    kind: str
    from_stop: StopId
    to_stop: StopId
    departure: datetime
    arrival: datetime
    trip_id: Optional[TripKey] = None


@dataclass
class Journey:
    """Complete A→B journey as ordered legs"""
    legs: List[Leg]

    @property
    def departure(self):
        return self.legs[0].departure
    @property
    def arrival(self):
        return self.legs[-1].arrival
    @property
    def num_transfers(self):
        return max(0, sum(1 for l in self.legs if l.kind == "trip") - 1)
    @property
    def walk_seconds(self):
        return sum((l.arrival - l.departure).total_seconds()
                   for l in self.legs if l.kind == "walk")

    def __repr__(self):
        header = (f"Journey: dep {self.departure} -> arr {self.arrival}, "
                  f"{self.num_transfers} transfers")
        lines = [header]
        for l in self.legs:
            label = l.trip_id if l.kind == "trip" else "WALK"
            lines.append(f"  {l.departure.strftime('%H:%M')} {l.from_stop} -> "
                         f"{l.arrival.strftime('%H:%M')} {l.to_stop} [{label}]")
        return "\n".join(lines)


@dataclass
class Trip:
    """One specific run (instance) of a route"""
    trip_key: TripKey
    events: List[Tuple[StopId, datetime, datetime, int]]


@dataclass
class Route:
    """RAPTOR route: all trips sharing the same stop pattern"""
    route_id: RouteId
    stop_sequence: List[StopId]
    stop_index: Dict[StopId, int]
    trips: List[Trip]


def build_routes_from_trips(trips_dict):
    """
    Group trip-instances into Routes by stop pattern
    Two trips share a route iff they visit identical stops in identical order

    Returns: (routes_dict, routes_at_stop_dict)
    """
    pattern_to_route = {}
    routes = {}
    routes_at_stop = defaultdict(list)
    counter = 0

    for trip_key, events in trips_dict.items():
        pattern = tuple(e[0] for e in events)
        if pattern not in pattern_to_route:
            rid = f"R{counter}"
            counter += 1
            pattern_to_route[pattern] = rid
            routes[rid] = Route(
                route_id=rid,
                stop_sequence=list(pattern),
                stop_index={s: i for i, s in enumerate(pattern)},
                trips=[],
            )
            for s in pattern:
                routes_at_stop[s].append(rid)
        rid = pattern_to_route[pattern]
        routes[rid].trips.append(Trip(trip_key=trip_key, events=events))

    # Sort trips within each route by departure at first stop
    for r in routes.values():
        r.trips.sort(key=lambda t: t.events[0][2])

    return routes, dict(routes_at_stop)


class Raptor:
    """RAPTOR planner with Q-grouped routes"""

    def __init__(self, routes, routes_at_stop, footpaths):
        self.routes = routes
        self.routes_at_stop = routes_at_stop
        self.footpaths = footpaths

    def query(self, source, target, departure_time, max_rounds=MAX_ROUNDS):
        """Earliest-arrival query. Returns Journey or None"""
        best_arrival = {source: departure_time}
        predecessor = {}
        marked = {source}

        for k in range(1, max_rounds + 1):
            new_marked = set()

            # Phase 1a: Build Q from marked stops
            # Q[route_id] = earliest-on-route boarding stop
            Q = {}
            for stop in marked:
                for rid in self.routes_at_stop.get(stop, []):
                    route = self.routes[rid]
                    if rid in Q:
                        if route.stop_index[stop] < route.stop_index[Q[rid]]:
                            Q[rid] = stop
                    else:
                        Q[rid] = stop

            target_best = best_arrival.get(target, INF)

            # Phase 1b: Traverse each route
            for rid, board_stop in Q.items():
                route = self.routes[rid]
                board_idx = route.stop_index[board_stop]
                board_arrival_time = best_arrival[board_stop]

                # Find earliest trip catchable at board_stop
                current_trip = None
                for trip in route.trips:
                    if trip.events[board_idx][2] >= board_arrival_time:
                        current_trip = trip
                        break
                if current_trip is None:
                    continue

                # Ride forward
                for i in range(board_idx + 1, len(route.stop_sequence)):
                    stop_id, arr_ts, dep_ts, _ = current_trip.events[i]

                    if arr_ts >= target_best:
                        break

                    if arr_ts < best_arrival.get(stop_id, INF):
                        board_time = current_trip.events[board_idx][2]
                        best_arrival[stop_id] = arr_ts
                        predecessor[stop_id] = (
                            board_stop, "trip", current_trip.trip_key,
                            board_time, arr_ts
                        )
                        new_marked.add(stop_id)
                        target_best = best_arrival.get(target, INF)

                    # If we know an earlier way to reach stop_id, try to switch to an earlier trip on this route
                    earlier_at_here = best_arrival.get(stop_id, INF)
                    if earlier_at_here < current_trip.events[i][2]:
                        for trip in route.trips:
                            if trip.events[i][2] >= earlier_at_here:
                                if trip.events[i][2] < current_trip.events[i][2]:
                                    current_trip = trip
                                    board_stop = stop_id
                                    board_idx = i
                                break

            # Phase 2: Footpaths
            for stop in list(new_marked):
                stop_arrival = best_arrival[stop]
                for neighbor, walk_sec in self.footpaths.get(stop, []):
                    new_arr = stop_arrival + timedelta(seconds=walk_sec)
                    if new_arr < best_arrival.get(neighbor, INF):
                        best_arrival[neighbor] = new_arr
                        predecessor[neighbor] = (
                            stop, "walk", None, stop_arrival, new_arr
                        )
                        new_marked.add(neighbor)

            if not new_marked:
                break
            marked = new_marked

        if target not in best_arrival or best_arrival[target] == INF:
            return None
        return self._reconstruct(source, target, predecessor)

    def query_range(self, source, target, arrive_before,
                    window_hours=2.0, step_minutes=5, max_rounds=MAX_ROUNDS,
                    lookup=None, min_confidence=0.0):
        """
        Multi-departure rRAPTOR with optional confidence filter.
        Returns journeys sorted latest-dep-first.
    
        Params:
            source, target: stop ids
            arrive_before: deadline
            window_hours: how far back to consider departures
            step_minutes: spacing between candidate departures
            max_rounds: cap on RAPTOR rounds
            lookup: delay-quantile lookup dict (required if min_confidence > 0)
            min_confidence: drop journeys with P(arrive on time) < this
        """
        earliest = arrive_before - timedelta(hours=window_hours)
        deps = []
        t = earliest
        while t < arrive_before:
            deps.append(t)
            t += timedelta(minutes=step_minutes)
    
        journeys = []
        for dep in deps:
            j = self.query(source, target, dep, max_rounds=max_rounds)
            if j is None or j.arrival > arrive_before:
                continue
            journeys.append(j)
    
        if lookup is not None and min_confidence > 0:
            journeys = [
                j for j in journeys
                if self.journey_confidence(j, lookup, arrive_before) >= min_confidence
            ]
    
        unique = self._remove_duplicates(journeys)
        return sorted(unique, key=lambda j: j.departure, reverse=True)

    def journey_confidence(self, journey, lookup, arrive_before):
        """
        P(journey arrives by arrive_before)
        Risky points: each inter-trip transfer + final arrival vs deadline
        Independence assumed
        """
        confidence = 1.0
        legs = journey.legs
        trip_indices = [i for i, l in enumerate(legs) if l.kind == "trip"]

        # 1. Transfers between consecutive trip-legs
        for k in range(len(trip_indices) - 1):
            i = trip_indices[k]
            j = trip_indices[k + 1]
            curr = legs[i]
            nxt = legs[j]
            prev_of_nxt = legs[j - 1]
            spare = (nxt.departure - prev_of_nxt.arrival).total_seconds()
            p = self._delay_prob_for_leg(curr, lookup, spare)
            if p is not None:
                confidence *= p

        # 2. Final arrival on time
        if trip_indices:
            last_trip = legs[trip_indices[-1]]
            margin = (arrive_before - journey.arrival).total_seconds()
            p_target = self._delay_prob_for_leg(last_trip, lookup, margin)
            if p_target is not None:
                confidence *= p_target

        return confidence

    def _delay_prob_for_leg(self, leg, lookup, spare):
        if leg.trip_id is None:
            return None
        trip_id_raw = leg.trip_id.split("#")[0]
        try:
            bpuic = int(leg.to_stop)
        except ValueError:
            return None
        date = leg.arrival.date()
        qrow = lookup.get((trip_id_raw, bpuic, date))
        if qrow is None:
            return None
        return delay_prob(qrow, spare)

    def _remove_duplicates(self, journeys):
        seen = set()
        result = []
        for j in journeys:
            key = (j.departure, j.arrival,
                   tuple(l.trip_id for l in j.legs if l.kind == "trip"))
            if key not in seen:
                seen.add(key)
                result.append(j)
        return result

    def _reconstruct(self, source, target, predecessor):
        legs = []
        current = target
        while current != source:
            prev_stop, kind, trip_id, dep, arr = predecessor[current]
            legs.append(Leg(
                kind=kind, from_stop=prev_stop, to_stop=current,
                departure=dep, arrival=arr, trip_id=trip_id,
            ))
            current = prev_stop
        legs.reverse()
        return Journey(legs=legs)
