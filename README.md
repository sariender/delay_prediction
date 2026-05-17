# COM-490 Final Project — Robust Routing Delay Lookup

This repository contains the delay-prediction interface used by the robust routing part of the COM-490 final project.

The routing system does **not** call the delay model live. Instead, delay distributions are precomputed offline for the February demo timetable and saved as a lookup table. During routing, our robust router uses this lookup to estimate the probability that a vehicle arrives within the available spare time.

---

## Main idea

For each scheduled stop event in the February demo timetable, we precompute:

- a point delay prediction: `pred_delay`
- calibrated delay quantiles: `q01`, `q05`, `q10`, `q20`, `q35`, `q50`, `q65`, `q80`, `q90`, `q95`, `q99`

The router then asks:

```text
Given spare time X seconds, what is P(delay <= X)?
```

The function `delay_prob(...)` answers this by reading the quantiles from the lookup table and interpolating between them.

### Method documentation

The detailed step-by-step explanation of the delay prediction, residual calibration, backoff strategy, and robust-routing probability interface is available here:

[Delay Prediction and Calibration Logic](docs/delay_prediction_calibration_logic_updated.md)

---

## Main router interface

The router should use:

```python
from src.model.delay_lookup import init_delay_oracle, delay_prob
```

Initialize the lookup once:

```python
LOOKUP_PATH = "/user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet"

init_delay_oracle(spark, LOOKUP_PATH)
```

Then call:

```python
p = delay_prob(
    trip_id=trip_id,
    stop_id=stop_id,
    date_value=date_value,
    X=X,
    scheduled_arrival_ts=scheduled_arrival_ts,
)
```

where:

| Argument | Meaning |
|---|---|
| `trip_id` | Timetable trip identifier |
| `stop_id` | Raw timetable stop id, such as `"8592015:0:10001"`, or clean `bpuic` |
| `date_value` | Operating day, for example `"2026-02-08"` |
| `X` | Spare time in seconds after subtracting walking time and minimum transfer time |
| `scheduled_arrival_ts` | Scheduled arrival timestamp; recommended to avoid ambiguous lookup matches |

The function returns:

```text
P(delay <= X)
```

For example, if `X = 180`, the function returns the probability that the vehicle arrives with delay less than or equal to 180 seconds.

---

## How the router uses it

For each risky transfer, the router computes:

```text
X = next_departure_time - scheduled_arrival_time - walking_time - minimum_transfer_time
```

Then it calls:

```python
p = delay_prob(
    trip_id=trip_id,
    stop_id=stop_id,
    date_value=date_value,
    X=X,
    scheduled_arrival_ts=scheduled_arrival_ts,
)
```

For a route with multiple risky points, the route confidence can be computed as:

```python
route_confidence = p1 * p2 * ...
```

under the independence assumption used by the routing logic.

A route can then be kept if:

```python
route_confidence >= Q
```

where `Q` is the user-requested confidence level.

---

## Lookup table

The main routing lookup is stored at:

```python
LOOKUP_PATH = "/user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet"
```

Each row corresponds to one scheduled stop event:

```text
operating_day + trip_id + bpuic + scheduled_arrival_ts
```

The table contains:

| Column | Meaning |
|---|---|
| `operating_day` | Service date |
| `trip_id` | Timetable trip identifier |
| `bpuic` | Clean stop identifier |
| `stop_name` | Stop name |
| `scheduled_arrival_ts` | Scheduled arrival timestamp |
| `line_text` | Public line number/name |
| `transport_clean` | Clean transport type |
| `pred_delay` | Point prediction of delay in seconds |
| `q01` ... `q99` | Calibrated delay quantiles in seconds |
| `chosen_calibration_level_q90` | Backoff calibration level used for the q90 residual correction |
| `line_seen_in_training` | Whether the line appeared in the historical training data |
| `line_id_source` | Whether the line id came from historical lookup or fallback logic |

The quantiles are used to approximate:

```text
P(delay <= X)
```

by interpolating between the closest quantile levels.

---

## Example

```python
from src.model.delay_lookup import init_delay_oracle, delay_prob

LOOKUP_PATH = "/user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet"

init_delay_oracle(spark, LOOKUP_PATH)

p = delay_prob(
    trip_id="61.TA.92-64-F-j26-1.1.H",
    stop_id="8592015:0:10001",
    date_value="2026-02-08",
    X=180,
    scheduled_arrival_ts="2026-02-08 12:24:00",
)

print(p)
```

This returns the estimated probability that the vehicle delay is at most 180 seconds.

---

## Calling from the raw February timetable

The raw February timetable has `stop_id` values such as:

```text
8592015:0:10001
```

The delay lookup module converts this internally to clean `bpuic`:

```text
8592015
```

If you are working directly with rows from:

```python
/user/groups/com-490/H1/final/v1/input_from_data_side/timetable_february.parquet
```

you can use:

```python
from src.model.delay_lookup import init_delay_oracle, delay_prob_from_timetable_row

LOOKUP_PATH = "/user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet"

init_delay_oracle(spark, LOOKUP_PATH)

row = (
    feb_raw_df
    .filter("arrival_timestamp is not null")
    .limit(1)
    .first()
)

p = delay_prob_from_timetable_row(row, X=180)

print(p)
```

---

## Project repository layout

```text
project_root/
├── README.md
│
├── docs/
│   └── delay_prediction_calibration_logic_updated.md
│
├── src/
│   ├── __init__.py
│   └── model/
│       ├── __init__.py
│       └── delay_lookup.py
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── final_model_training.ipynb
│   ├── february_lookup_generation.ipynb
│   └── validation_experiments.ipynb
│
├── legacy_notebooks/
│   └── old_experiments/
│
└── reports/
    └── figures/
```

### Main source file

```text
src/model/delay_lookup.py
```

This file exposes the routing interface:

```python
delay_prob(trip_id, stop_id, date_value, X, scheduled_arrival_ts=None)
```

It reads the precomputed February lookup and returns:

```text
P(delay <= X)
```

where `X` is spare time in seconds after subtracting walking time and minimum transfer time.

### Repository folders

| Folder | Purpose |
|---|---|
| `src/` | Source code used by the router and delay lookup module |
| `src/model/delay_lookup.py` | Lookup-based delay probability interface |
| `models/` | Lightweight model documentation or small metadata files|
| `notebooks/` | Final notebooks used for training, lookup generation, and validation |
| `legacy_notebooks/` | Older exploratory notebooks kept for reference only |
| `reports/figures/` | Exported figures for reports or slides |

---

## HDFS artifact layout

Base path:

```python
BASE = "/user/groups/com-490/H1/final/v1"
```

```text
/user/groups/com-490/H1/final/v1/
├── input_from_data_side/
│   ├── timetable_february.parquet
│   ├── transfers.parquet
│   ├── trip_stop_events.parquet
│   └── trip_stop_events_norm.parquet
│
├── features_and_training_data/
│   ├── final_full_data.parquet
│   └── full_data_w_o_hist_weather_specialdays.parquet
│
├── trained_models/
│   └── prod_gbt_model_full_data/
│
├── calibration_model_data/
│   ├── feature_splits/
│   │   ├── train_final_features_df.parquet
│   │   ├── calib_final_features_df.parquet
│   │   └── test_final_features_df.parquet
│   │
│   ├── models/
│   │   └── eval_calib_gbt_model/
│   │
│   ├── calibration_artifacts/
│   │   ├── calibration_config.parquet
│   │   ├── calibration_residual_tables/
│   │   └── global_residual_quantiles.parquet
│   │
│   └── evaluation_outputs/
│       ├── eval_calib_predictions_full.parquet
│       ├── eval_test_predictions_full.parquet
│       ├── eval_test_quantiles_backoff.parquet
│       ├── eval_coverage_backoff.parquet
│       ├── coverage_backoff.parquet
│       └── test_quantiles_backoff.parquet
│
├── weather_data/
│   ├── forecast/
│   │   ├── forecast_weather_2024_07_2026_01.parquet
│   │   └── forecast_weather_2026_02.parquet
│   │
│   └── observed/
│       └── open_meteo_lausanne_observed_hourly_2024_07_2026_01.parquet
│
└── route_demo_outputs/
    ├── predictions/
    │   └── feb_predictions_full.parquet
    │
    ├── quantiles/
    │   └── feb_quantiles_backoff.parquet
    │
    └── lookup/
        └── route_demo_lookup_february.parquet
```

---

## Important paths

Repository source file:

```text
src/model/delay_lookup.py
```

HDFS paths:

```python
TIMETABLE_PATH = f"{BASE}/input_from_data_side/timetable_february.parquet"

LOOKUP_PATH = f"{BASE}/route_demo_outputs/lookup/route_demo_lookup_february.parquet"

FINAL_MODEL_PATH = f"{BASE}/trained_models/prod_gbt_model_full_data"

FULL_DATA_PATH = f"{BASE}/features_and_training_data/final_full_data.parquet"

CALIB_TABLES_PATH = f"{BASE}/calibration_model_data/calibration_artifacts/calibration_residual_tables"

GLOBAL_QUANTILES_PATH = f"{BASE}/calibration_model_data/calibration_artifacts/global_residual_quantiles.parquet"

CONFIG_PATH = f"{BASE}/calibration_model_data/calibration_artifacts/calibration_config.parquet"

FEB_WEATHER_PATH = f"{BASE}/weather_data/forecast/forecast_weather_2026_02.parquet"
```

---

## Evaluation files vs routing files

The router should use:

```text
route_demo_outputs/lookup/route_demo_lookup_february.parquet
```

The following files are for validation/reporting, not for routing:

```text
calibration_model_data/evaluation_outputs/eval_test_quantiles_backoff.parquet
calibration_model_data/evaluation_outputs/eval_coverage_backoff.parquet
```

These evaluation files were used to check whether the calibrated quantiles achieve the intended empirical coverage.

---

## Notes

- The February lookup uses forecast weather features only.
- Observed weather is not used for February prediction.
- The model is not called live during routing.
- Routing uses the precomputed lookup for speed.
- If no lookup row is found, the current fallback in `delay_prob(...)` returns `0.0` conservatively.