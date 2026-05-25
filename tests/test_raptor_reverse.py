from __future__ import annotations

from src.graph import FootPath, Route, Stop, StopEvent, TransitGraph
from src.raptor import raptor_reverse


def ts(hour: int, minute: int) -> int:
    return hour * 3600 + minute * 60


def make_graph(*stop_ids: str) -> TransitGraph:
    graph = TransitGraph()
    for stop_id in stop_ids:
        graph.stops[stop_id] = Stop(stop_id, stop_id, 0.0, 0.0)
    return graph


def add_trip(
    graph: TransitGraph,
    route_id: str,
    trip_id: str,
    stop_times: list[tuple[str, int, int]],
) -> None:
    route = Route(route_id=route_id, stop_sequence=[stop_id for stop_id, _, _ in stop_times])
    route.trips.append([
        StopEvent(
            stop_id=stop_id,
            arrival_ts=arrival_ts,
            departure_ts=departure_ts,
            stop_sequence=i,
            trip_id=trip_id,
            stop_name=stop_id,
        )
        for i, (stop_id, arrival_ts, departure_ts) in enumerate(stop_times)
    ])
    graph.routes[route_id] = route
    graph.trip_modes[trip_id] = "train"
    for stop_id in route.stop_sequence:
        graph.routes_at_stop[stop_id].append(route_id)


def add_walk(graph: TransitGraph, a: str, b: str, seconds: int) -> None:
    # Tests use walking_speed_m_per_min=60, so distance_m equals seconds.
    graph.footpaths[a].append(FootPath(a, b, float(seconds), seconds))
    graph.footpaths[b].append(FootPath(b, a, float(seconds), seconds))


def best(journeys):
    assert journeys
    return max(journeys, key=lambda journey: journey.departure_ts)


def test_reverse_source_walk_to_first_transit_keeps_boarding_buffer():
    graph = make_graph("O", "B", "T")
    add_walk(graph, "O", "B", 300)
    add_trip(graph, "R1", "trip1", [
        ("B", ts(9, 0), ts(9, 0)),
        ("T", ts(9, 10), ts(9, 10)),
    ])

    journey = best(raptor_reverse(
        graph,
        target="T",
        source="O",
        arrival_deadline_ts=ts(9, 15),
        min_transfer_sec=120,
        max_walk_m=1000,
        walking_speed_m_per_min=60,
    ))

    assert journey.departure_ts == ts(8, 53)
    assert [(leg.leg_type, leg.departure_ts, leg.arrival_ts) for leg in journey.legs] == [
        ("walk", ts(8, 53), ts(8, 58)),
        ("transit", ts(9, 0), ts(9, 10)),
    ]


def test_reverse_transit_to_final_walk_does_not_apply_transfer_buffer():
    graph = make_graph("A", "B", "T")
    add_walk(graph, "B", "T", 300)
    add_trip(graph, "R1", "trip1", [
        ("A", ts(9, 0), ts(9, 0)),
        ("B", ts(9, 8), ts(9, 8)),
    ])

    journey = best(raptor_reverse(
        graph,
        target="T",
        source="A",
        arrival_deadline_ts=ts(9, 13),
        min_transfer_sec=120,
        max_walk_m=1000,
        walking_speed_m_per_min=60,
    ))

    assert [(leg.leg_type, leg.departure_ts, leg.arrival_ts) for leg in journey.legs] == [
        ("transit", ts(9, 0), ts(9, 8)),
        ("walk", ts(9, 8), ts(9, 13)),
    ]


def test_reverse_transit_walk_transit_applies_buffer_once():
    graph = make_graph("A", "B", "C", "T")
    add_walk(graph, "B", "C", 300)
    add_trip(graph, "R1", "trip1", [
        ("A", ts(8, 45), ts(8, 45)),
        ("B", ts(8, 53), ts(8, 53)),
    ])
    add_trip(graph, "R2", "trip2", [
        ("C", ts(9, 0), ts(9, 0)),
        ("T", ts(9, 10), ts(9, 10)),
    ])

    journey = best(raptor_reverse(
        graph,
        target="T",
        source="A",
        arrival_deadline_ts=ts(9, 15),
        min_transfer_sec=120,
        max_walk_m=1000,
        walking_speed_m_per_min=60,
    ))

    assert [(leg.leg_type, leg.departure_ts, leg.arrival_ts) for leg in journey.legs] == [
        ("transit", ts(8, 45), ts(8, 53)),
        ("walk", ts(8, 53), ts(8, 58)),
        ("transit", ts(9, 0), ts(9, 10)),
    ]


def test_reverse_uses_incoming_explicit_transfer_duration_before_transit():
    graph = make_graph("A", "B", "C", "T")
    graph.explicit_transfers["B"].append(("C", 180))
    add_trip(graph, "R1", "trip1", [
        ("A", ts(8, 45), ts(8, 45)),
        ("B", ts(8, 55), ts(8, 55)),
    ])
    add_trip(graph, "R2", "trip2", [
        ("C", ts(9, 0), ts(9, 0)),
        ("T", ts(9, 10), ts(9, 10)),
    ])

    journey = best(raptor_reverse(
        graph,
        target="T",
        source="A",
        arrival_deadline_ts=ts(9, 15),
        min_transfer_sec=120,
        max_walk_m=1000,
        walking_speed_m_per_min=60,
    ))

    legs = [
        (leg.leg_type, leg.departure_ts, leg.arrival_ts, leg.walk_duration_sec)
        for leg in journey.legs
    ]
    assert legs == [
        ("transit", ts(8, 45), ts(8, 55), None),
        ("walk", ts(8, 55), ts(8, 58), 180),
        ("transit", ts(9, 0), ts(9, 10), None),
    ]
