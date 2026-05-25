from __future__ import annotations

from src.graph import FootPath, Route, Stop, StopEvent, TransitGraph
from src.raptor import raptor_forward


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


def test_forward_does_not_board_from_same_round_improvement():
    graph = make_graph("O", "B", "T")
    graph.footpaths["O"].append(FootPath("O", "B", 1740.0, 1740))
    add_trip(graph, "R0", "trip0", [
        ("O", ts(12, 30), ts(12, 30)),
        ("B", ts(12, 40), ts(12, 40)),
    ])
    add_trip(graph, "R1", "trip1", [
        ("B", ts(12, 49), ts(12, 49)),
        ("T", ts(13, 1), ts(13, 1)),
    ])

    journeys = raptor_forward(
        graph,
        source="O",
        target="T",
        departure_ts=ts(12, 30),
        max_rounds=3,
        min_transfer_sec=0,
        max_walk_m=2000,
        walking_speed_m_per_min=60,
    )

    assert journeys
    journey = min(journeys, key=lambda j: j.arrival_ts)
    assert [(leg.leg_type, leg.from_stop, leg.to_stop) for leg in journey.legs] == [
        ("transit", "O", "B"),
        ("transit", "B", "T"),
    ]
    for prev_leg, next_leg in zip(journey.legs[:-1], journey.legs[1:]):
        assert prev_leg.arrival_ts <= next_leg.departure_ts


def test_forward_initial_walk_must_arrive_before_boarding():
    graph = make_graph("O", "B", "T")
    graph.footpaths["O"].append(FootPath("O", "B", 1740.0, 1740))
    add_trip(graph, "R1", "trip1", [
        ("B", ts(12, 49), ts(12, 49)),
        ("T", ts(13, 1), ts(13, 1)),
    ])
    add_trip(graph, "R2", "trip2", [
        ("B", ts(13, 5), ts(13, 5)),
        ("T", ts(13, 17), ts(13, 17)),
    ])

    journeys = raptor_forward(
        graph,
        source="O",
        target="T",
        departure_ts=ts(12, 30),
        max_rounds=3,
        min_transfer_sec=0,
        max_walk_m=2000,
        walking_speed_m_per_min=60,
    )

    assert journeys
    journey = min(journeys, key=lambda j: j.arrival_ts)
    assert [(leg.leg_type, leg.departure_ts, leg.arrival_ts) for leg in journey.legs] == [
        ("walk", ts(12, 30), ts(12, 59)),
        ("transit", ts(13, 5), ts(13, 17)),
    ]


def test_forward_explicit_transfer_counts_toward_walking_limit():
    graph = make_graph("O", "B", "C", "T")
    graph.explicit_transfers["B"].append(("C", 180, 96.0))
    add_trip(graph, "R0", "trip0", [
        ("O", ts(12, 30), ts(12, 30)),
        ("B", ts(12, 35), ts(12, 35)),
    ])
    add_trip(graph, "R1", "trip1", [
        ("C", ts(12, 40), ts(12, 40)),
        ("T", ts(12, 46), ts(12, 46)),
    ])

    assert not raptor_forward(
        graph,
        source="O",
        target="T",
        departure_ts=ts(12, 30),
        max_rounds=3,
        min_transfer_sec=0,
        max_walk_m=50,
        walking_speed_m_per_min=60,
    )

    journeys = raptor_forward(
        graph,
        source="O",
        target="T",
        departure_ts=ts(12, 30),
        max_rounds=3,
        min_transfer_sec=0,
        max_walk_m=100,
        walking_speed_m_per_min=60,
    )

    assert journeys
    journey = min(journeys, key=lambda j: j.arrival_ts)
    assert journey.total_walk_m == 96.0
    assert [(leg.leg_type, leg.from_stop, leg.to_stop, leg.walk_distance_m) for leg in journey.legs] == [
        ("transit", "O", "B", 0.0),
        ("walk", "B", "C", 96.0),
        ("transit", "C", "T", 0.0),
    ]
