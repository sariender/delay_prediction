# Route Planner — Specification & Implementation Notes

## User-Facing Options

The route planner should give the following options to users:
- Fastest Route (only pick departure)
- Latest leaving route to make it on time (only pick arrival)
- Least Transfers
- Least Walking
- Speed of walking configurable
- Safest route (trip w/ highest confidence as defined by the delay prediction model)
- Extra time for connections configurable
- Minimum trip confidence as defined by the delay prediction model

---

## Algorithm: Multi-Criteria RAPTOR with Robustness

### Core Algorithm
We use a **modified RAPTOR** (Round-Based Public Transit Routing Protocol) which is well-suited because:
1. It naturally finds Pareto-optimal routes in terms of (arrival time, #transfers)
2. Each "round" adds exactly one transit leg → easy to enumerate by transfer count
3. Walkable connections are handled as foot-paths between rounds
4. Can be run in **reverse** for "latest departure given arrival deadline" queries

### Multi-Criteria Extensions
- **Fastest route**: Forward RAPTOR from departure time, return route with earliest arrival
- **Latest departure**: Reverse RAPTOR from arrival deadline, return route with latest departure
- **Least transfers**: Return Pareto-front route with fewest rounds (k)
- **Least walking**: Track cumulative walking distance; prefer routes with less walking among Pareto-optimal set
- **Safest route**: After finding candidate routes, compute confidence using delay model; return highest confidence route
- **Min confidence filter**: Prune routes whose confidence < Q threshold

### Transfer / Connection Rules
- Minimum 120s transfer time at same station (per assignment spec)
- Walking transfers: 1 min per 50m at default speed (configurable)
- Maximum total walking distance: 500m (configurable)
- Extra connection buffer: user-configurable seconds added to transfer time

### Robustness / Confidence
For each transfer in a route, compute spare time:
```
X = next_departure_time - arrival_time - walking_time - min_transfer_time - extra_buffer
```
Query `delay_prob(trip_id, stop_id, date, X, scheduled_arrival_ts)` → P(delay ≤ X)

Route confidence = product of all transfer probabilities (independence assumption).

---

## Data Sources

| Data | Path | Description |
|------|------|-------------|
| February timetable | `/user/groups/com-490/H1/final/v1/input_from_data_side/timetable_february.parquet` | Pre-joined stop events with timestamps |
| Transfers | `/user/groups/com-490/H1/final/v1/input_from_data_side/transfers.parquet` | Platform transfer times |
| Delay lookup | `/user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet` | Precomputed delay quantiles |

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/route_planner.py` | Core `RobustJourneyPlanner` class with RAPTOR algorithm |
| `src/model/delay_lookup.py` | Delay probability interface (existing) |
| `routing_demo.ipynb` | Interactive notebook with ipywidgets UI |

---

## Status

- [x] Requirements analysis
- [x] Algorithm design (RAPTOR + robustness)
- [x] Core RAPTOR implementation (`src/raptor.py`)
- [x] Reverse RAPTOR for latest-departure
- [x] Multi-criteria Pareto front
- [x] Delay model integration
- [x] Walking distance tracking
- [x] Graph builder (`src/graph.py`)
- [x] High-level planner API (`src/route_planner.py`)
- [x] Interactive notebook UI (`routing_demo.py`)
- [ ] Validation & testing