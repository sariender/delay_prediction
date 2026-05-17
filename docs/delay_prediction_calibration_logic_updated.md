# Delay Prediction and Calibration Logic

This document explains the delay-prediction and calibration pipeline used for the robust routing component.

The goal is to provide the router with the following interface:

```python
delay_prob(trip_id, stop_id, date_value, X, scheduled_arrival_ts=None)
```

The function returns:

```text
P(delay <= X)
```

where `X` is the usable spare time in seconds.

For a transfer, `X` is computed as:

```text
X = next_departure_time
    - scheduled_arrival_time
    - walking_time
    - minimum_transfer_time
```

In other words, `X` is the spare time left after accounting for the time needed to walk between stops/platforms and the minimum transfer buffer.

---

## 1. Objective

For robust routing, a single point prediction is not enough.

The router needs to answer questions such as:

```text
If I have X seconds of usable spare time, what is the probability that I still catch the transfer?
```

Therefore, the output of the delay component is not only:

```text
pred_delay
```

but also a calibrated delay distribution represented by quantiles:

```text
q01, q05, q10, q20, q35, q50, q65, q80, q90, q95, q99
```

These quantiles allow the router to estimate:

```text
P(delay <= X)
```

for any transfer slack `X`.

---

## 2. Historical labeled data

The pipeline starts from historical labeled delay data.

Each row represents a scheduled stop event with an observed arrival delay.

The target variable is:

```text
arrival_delay_seconds
```

This is the difference between actual and scheduled arrival time:

```text
arrival_delay_seconds = actual_arrival_time - scheduled_arrival_time
```

Each row also contains features such as:

- calendar and scheduled-time features
- transport type
- line information
- stop and trip position features
- historical delay statistics
- forecast weather features
- special-day features

This labeled historical data is used for training, calibration, and testing.

---

## 3. Train / calibration / test split

The historical data is split temporally into three parts:

```text
train
calibration
test
```

Each split has a different role:

| Split | Purpose |
|---|---|
| Train | Train the point delay model |
| Calibration | Learn how wrong the model tends to be |
| Test | Evaluate whether calibrated quantiles are reliable |

This separation is important because calibration and evaluation should not be performed on the same data.

---

## 4. Point delay model

A GBT regression model is trained on the training split.

The model learns:

```text
features -> predicted delay
```

For each row, it produces:

```text
pred_delay
```

Example:

```text
pred_delay = 73 seconds
```

This is the model's best guess for the delay.

However, robust routing needs uncertainty, not only a best guess. For example, if a passenger has 150 seconds of spare time, the router needs to know:

```text
What is the probability that the delay is less than or equal to 150 seconds?
```

A single point prediction cannot answer that.

---

## 5. Point-model performance on the calibration split

Before applying the residual calibration layer, we evaluated the point delay model on the calibration split.

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Calibration | 194.32 sec | 110.64 sec | 0.1184 |

These metrics evaluate only the point prediction `pred_delay`.


---

## 6. Calibration residuals

After training the point model, we apply it to the calibration data.

For each calibration row, we have:

```text
actual_delay
pred_delay
```

Then we compute the residual:

```text
residual = actual_delay - pred_delay
```

The residual tells us how wrong the model was.

Example 1:

```text
actual_delay = 180 sec
pred_delay   = 120 sec
residual     = 60 sec
```

The model underpredicted the delay by 60 seconds.

Example 2:

```text
actual_delay = 40 sec
pred_delay   = 100 sec
residual     = -60 sec
```

The model overpredicted the delay by 60 seconds.

So the calibration set is used to learn the model's historical mistakes.

---

## 7. Calibration residual distribution

The calibration residual distribution is important because these residuals become the data-driven safety margins.

The key quantity is:

```text
residual = actual_delay - pred_delay
```

Positive residuals mean the model underpredicted the delay. These are important for robust routing because they represent cases where the vehicle arrived later than expected.

### Residual and absolute-error summary

| Metric | Value |
|---|---:|
| Number of calibration rows | 7,630,465 |
| Mean residual | -10.41 sec |
| Median residual | -28.40 sec |
| Residual q80 | 59.79 sec |
| Residual q90 | 140.49 sec |
| Residual q95 | 249.26 sec |
| Residual q99 | 666.25 sec |
| Absolute error p50 | 69.84 sec |
| Absolute error p80 | 157.70 sec |
| Absolute error p90 | 235.47 sec |
| Absolute error p95 | 330.78 sec |
| Absolute error p99 | 685.33 sec |

### Interpretation

The median residual is negative:

```text
median_residual = -28.40 sec
```

This means that, at the median, the model slightly overpredicts delay.

However, the upper residual quantiles are positive and large:

```text
residual_q90 = 140.49 sec
residual_q95 = 249.26 sec
residual_q99 = 666.25 sec
```

This means that in the calibration data:

- 90% of residuals are below about 140 seconds
- 95% of residuals are below about 249 seconds
- 99% of residuals are below about 666 seconds

These upper residual quantiles become safety margins.

For example, if a new row has:

```text
pred_delay = 70 sec
residual_q90 = 140 sec
```

then:

```text
q90 = pred_delay + residual_q90
q90 = 70 + 140 = 210 sec
```

```text
The model's best guess is 70 seconds of delay, but a 90%-safe calibrated delay estimate is about 210 seconds.
```

---

## 8. Residual error bands

It is also useful to look at residuals in bands.

| Residual band | Number of rows | Share |
|---|---:|---:|
| `< -300 sec` | 174,764 | 2.29% |
| `-300 to -180 sec` | 482,914 | 6.33% |
| `-180 to -60 sec` | 2,066,509 | 27.08% |
| `-60 to 60 sec` | 3,383,059 | 44.34% |
| `60 to 180 sec` | 945,977 | 12.40% |
| `180 to 300 sec` | 281,800 | 3.69% |
| `300 to 600 sec` | 202,107 | 2.65% |
| `> 600 sec` | 93,335 | 1.22% |

### Interpretation

Most calibration residuals are close to zero:

```text
-60 to 60 sec: 44.34%
```

This means that for many rows, the point model is within about one minute of the actual delay.

However, there is still a meaningful upper tail:

```text
60 to 180 sec: 12.40%
180 to 300 sec: 3.69%
300 to 600 sec: 2.65%
> 600 sec: 1.22%
```

These positive residual bands are important for robust routing because they represent cases where the actual delay was larger than the model prediction.

This is why calibrated upper quantiles such as q90, q95, and q99 are needed. They provide data-driven safety margins for these underprediction cases.

---

## 9. Residuals as data-driven safety margins

The simplest calibration approach would be to calculate residual quantiles globally.

For example:

```text
global_residual_q90 = 90th percentile of all calibration residuals
```

Then, for a new prediction:

```text
q90 = pred_delay + global_residual_q90
```

This means:

```text
Use one universal 90% safety margin for all predictions.
```

However, this is too simple because uncertainty is not the same in every context.

For example:

- late-night buses may be more uncertain than off-peak metros
- wet or snowy weather may require larger margins
- historically risky lines may need larger margins
- later stops may accumulate more delay than origin stops

Therefore, instead of using only one global residual distribution, we estimate context-specific residual distributions.

---

## 10. Operational uncertainty buckets

The function `add_uncertainty_buckets(df)` converts raw features into interpretable operational buckets.

These buckets define what "similar situations" means for calibration.

The main buckets are:

```text
pred_delay_bucket
time_bucket
day_bucket
weather_bucket
forecast_instability_bucket
line_risk_bucket
trip_stage_bucket
special_bucket
stop_hub_bucket
```

These bucket variables are used to group calibration rows before estimating residual quantiles.

---

## 11. Bucket definitions

### `pred_delay_bucket`

Groups rows by the model's point prediction.

| Bucket | Rule | Meaning |
|---|---:|---|
| `low_pred_delay` | `pred_delay < 60` | Model expects less than 1 minute delay |
| `medium_pred_delay` | `60 <= pred_delay < 180` | Model expects 1-3 minutes delay |
| `high_pred_delay` | `180 <= pred_delay < 420` | Model expects 3-7 minutes delay |
| `very_high_pred_delay` | `pred_delay >= 420` | Model expects more than 7 minutes delay |

This matters because the model's error may behave differently when the predicted delay is already high.

### `time_bucket`

Groups rows by time-of-day context.

| Bucket | Meaning |
|---|---|
| `morning_peak` | Morning peak period |
| `evening_peak` | Evening peak period |
| `late_night` | Late-night service |
| `offpeak` | All other times |

Delay uncertainty can differ between peak hours, off-peak periods, and late-night services.

### `day_bucket`

Separates weekday and weekend services.

| Bucket | Meaning |
|---|---|
| `weekday` | Monday-Friday |
| `weekend` | Saturday/Sunday |

Weekend services may have different delay patterns from weekdays.

### `weather_bucket`

Groups rows by forecast weather severity.

| Bucket | Rule | Meaning |
|---|---:|---|
| `snow` | `snowfall_fcst_d1 > 0` | Snow forecast |
| `heavy_wet` | `precipitation_fcst_d1 >= 2.0` | Strong precipitation |
| `light_wet` | `precipitation_fcst_d1 > 0.1` | Light precipitation |
| `windy` | `wind_gusts_10m_fcst_d1 > 50` | Windy conditions |
| `dry` | otherwise | No relevant adverse weather |

This allows the calibration layer to learn larger uncertainty margins under difficult weather conditions.

### `forecast_instability_bucket`

Captures whether the forecast changed substantially between D-2 and D-1.

| Bucket | Meaning |
|---|---|
| `unstable_forecast` | Forecast changed substantially between D-2 and D-1 |
| `stable_forecast` | Forecast was relatively stable |

This is useful because unstable forecasts may indicate higher uncertainty in the future operating condition.

### `line_risk_bucket`

Uses historical line-hour risk:

```text
hist_line_hour_p90_delay
```

| Bucket | Rule | Meaning |
|---|---:|---|
| `low_line_risk` | `hist_line_hour_p90_delay < 180` | Historically low-risk line-hour |
| `medium_line_risk` | `180 <= hist_line_hour_p90_delay < 420` | Moderate historical delay risk |
| `high_line_risk` | `hist_line_hour_p90_delay >= 420` | Historically high-risk line-hour |

This captures whether a line is historically delay-prone at that time of day.

### `trip_stage_bucket`

Describes where the stop is located within the trip.

| Bucket | Meaning |
|---|---|
| `origin` | First observed stop |
| `early` | Early part of the trip |
| `middle` | Middle part of the trip |
| `late` | Late part of the trip |
| `terminal` | Last observed stop |

This matters because delay can accumulate along a route.

### `special_bucket`

Captures special calendar context.

| Bucket | Meaning |
|---|---|
| `public_holiday` | Public holiday |
| `school_holiday` | School holiday |
| `special_day` | Other special event/day |
| `normal_day` | Regular day |

Special days may create unusual traffic or passenger patterns.

### `stop_hub_bucket`

This bucket is created separately from historical data.

For each stop, we count how many distinct lines serve it:

```text
n_lines_at_stop
```

Then we classify the stop:

| Bucket | Rule | Meaning |
|---|---:|---|
| `major_hub` | `n_lines_at_stop >= 8` | Highly connected stop |
| `medium_hub` | `3 <= n_lines_at_stop < 8` | Medium-size transfer point |
| `local_stop` | `n_lines_at_stop < 3` | Local stop |

We assume that this captures whether the stop is a major transfer hub or a local stop.

---

## 12. Creating `stop_hub_bucket`

The stop hub bucket is created from historical data:

```python
stop_hub_full_df = (
    full_base_df
    .groupBy("bpuic")
    .agg(
        F.countDistinct("line_text").alias("n_lines_at_stop"),
        F.countDistinct("transport_clean").alias("n_transport_types_at_stop"),
        F.count("*").alias("n_events_at_stop")
    )
    .withColumn(
        "stop_hub_bucket",
        F.when(F.col("n_lines_at_stop") >= 8, F.lit("major_hub"))
         .when(F.col("n_lines_at_stop") >= 3, F.lit("medium_hub"))
         .otherwise(F.lit("local_stop"))
    )
    .select("bpuic", "stop_hub_bucket")
)
```

This means:

```text
For each stop, count how many different lines serve it.
Then classify the stop as a local stop, medium hub, or major hub.
```

This is joined to the February prediction rows using `bpuic`.

---

## 13. Building calibration groups

After creating the buckets, we combine them into group keys.

For example, a rich group may look like:

```text
medium_pred_delay
+ Bus
+ evening_peak
+ light_wet
+ high_line_risk
+ middle
+ medium_hub
+ normal_day
+ line 64 (Lausanne, Chevreuils)
```

This group represents calibration rows that are operationally similar to the new prediction row.

For each group, residual quantiles are calculated from the calibration set:

```text
residual_q01
residual_q05
residual_q10
...
residual_q90
residual_q95
residual_q99
```

These are our data-driven safety margins.

---

## 14. Backoff calibration

Very specific groups are useful only if they contain enough calibration rows.

If a group has too few examples, its residual quantiles are unreliable.

Therefore, the calibration logic uses a backoff strategy:

```text
L6_line_specific
-> L5_ultra_rich
-> L4_rich
-> L3_operational
-> L2_context
-> L1_core
-> L0_simple
-> global
```
As a baseline fallback, we also compute global residual quantiles from all calibration rows. 

| Quantile | Global residual quantile |
|---|---:|
| q01 | -378.41 sec |
| q05 | -228.12 sec |
| q10 | -167.26 sec |
| q20 | -108.94 sec |
| q35 | -61.72 sec |
| q50 | -28.40 sec |
| q65 | 5.87 sec |
| q80 | 59.79 sec |
| q90 | 140.49 sec |
| q95 | 249.26 sec |
| q99 | 666.25 sec |

For example, the global q90 residual is 140.49 seconds. 

If a new row has:

```text
pred_delay = 70 sec

```
The algorithm starts from the most specific level (L6).

If that group has enough calibration rows (n=1000), it uses its residual quantiles.

If not, it backs off to broader levels.

If no group is sufficiently supported, it uses the global residual distribution.

---

## 15. `chosen_calibration_level_q90`

Our lookup also includes the diagnostic column:

```text
chosen_calibration_level_q90
```

This tells us which calibration level was used for the q90 residual correction.

Example:

```text
chosen_calibration_level_q90 = L6_line_specific
```

means that the q90 safety margin was estimated using a highly specific, line-level calibration group.

Example:

```text
chosen_calibration_level_q90 = global
```

means that no specific group had enough calibration examples, so the global residual distribution was used.

This column is not the prediction itself. It is our debugging column.

---

## 16. Creating calibrated quantiles

For a new prediction row, we first obtain:

```text
pred_delay
```

Then we add the selected residual quantiles.

The core formula is:

```text
calibrated qα = pred_delay + residual_qα
```

Example:

```text
pred_delay   = 73 sec
residual_q90 = 83 sec

q90 = 73 + 83 = 156 sec
```

```text
The model's best guess is 73 seconds of delay.
Based on similar past model errors, a 90%-safe delay estimate is 156 seconds.
```

So:

| Quantile | Meaning |
|---|---|
| `q50` | Typical/median calibrated delay |
| `q80` | Safer delay estimate |
| `q90` | Conservative delay estimate |
| `q95` | More conservative delay estimate |
| `q99` | Very conservative delay estimate |

---

## 17. Test-set validation

After calibration, the calibrated quantiles are evaluated on the held-out test set.

For example, for q90, we check:

```text
actual_delay <= q90
```

If the calibration is good, this should be true for about 90% of test rows.

Similarly:

| Quantile | Expected empirical coverage |
|---|---:|
| `q50` | about 50% |
| `q80` | about 80% |
| `q90` | about 90% |
| `q95` | about 95% |
| `q99` | about 99% |

This is why we kept the evaluation artifacts exist:

```text
eval_test_quantiles_backoff.parquet
eval_coverage_backoff.parquet
```

(These files are used for validation and reporting, not for routing).

---

## 18. Final model for February

After validating the method, the final point model is trained on all available labeled historical data.

This final model is then used to predict the February demo timetable.

The February timetable has no actual observed delay labels, so it is not used to measure accuracy. It is used for inference.

For each February scheduled stop event:

```text
February row -> features -> pred_delay -> calibrated q01...q99
```

---

## 19. Final route lookup table

The final lookup table is:

```text
route_demo_lookup_february.parquet
```

Each row corresponds to one scheduled stop event in the February demo timetable.

The lookup key is:

```text
operating_day + trip_id + bpuic + scheduled_arrival_ts
```

The prediction columns are:

```text
pred_delay
q01, q05, q10, q20, q35, q50, q65, q80, q90, q95, q99
```

So the table maps:

```text
trip-stop-time -> predicted delay distribution
```

The router does not call the model live. It reads this precomputed lookup.

---

## 20. February lookup quality check

For the February route-demo lookup, the summary is:

| Metric | Value |
|---|---:|
| Number of rows | 3,071,095 |
| Missing `pred_delay` | 0 |
| Missing `q50` | 0 |
| Missing `q90` | 0 |
| Median `pred_delay` | 77.67 sec |
| Median `q50` | 48.26 sec |
| Median `q90` | 211.67 sec |
| Median `q95` | 293.94 sec |
| Median `q99` | 592.64 sec |

Interpretation:

- The lookup contains 3.07M February scheduled stop events.
- There are no missing point predictions or key calibrated quantiles.
- The median point prediction is about 78 seconds.
- The median q90 is about 212 seconds.
- The median q99 is about 593 seconds.

This shows that the calibrated quantiles provide increasingly conservative delay estimates around the point prediction.

---

## 21. Backoff-level distribution for February

After considering the bucket definitions, the February lookup mostly uses rich calibration groups.

| Calibration level | Share |
|---|---:|
| L6_line_specific | 35.399% |
| L5_ultra_rich | 32.196% |
| L4_rich | 13.619% |
| L3_operational | 6.641% |
| L2_context | 2.714% |
| L1_core | 2.048% |
| L0_simple | 0.358% |
| global | 7.027% |

The rich levels L4-L6 cover around 81.2% of February rows.

This means that most February predictions use context-specific uncertainty estimates rather than only a generic global safety margin.

The global fallback is still used for around 7.0% of rows, which is expected for rare or unusual timetable/weather/line combinations.

---

## 22. Example lookup row

Example February row:

| Field | Value |
|---|---:|
| Line | 64 |
| Stop | Lausanne, Chevreuils |
| Scheduled arrival | 2026-02-08 12:24 |
| `pred_delay` | 73.46 sec |
| `q50` | 60.92 sec |
| `q80` | 118.88 sec |
| `q90` | 155.78 sec |
| `q95` | 190.17 sec |
| `q99` | 269.48 sec |
| `chosen_calibration_level_q90` | L6_line_specific |

Interpretation:

The point model predicts around 73 seconds of delay.

The calibrated q90 is around 156 seconds, meaning that in similar calibrated cases, the actual delay is expected to be below this value about 90% of the time.

Because `chosen_calibration_level_q90 = L6_line_specific`, the q90 safety margin came from the most specific line-level calibration group.

---

## 23. From quantiles to probability

The router does not directly ask:

```text
What is q90?
```

It asks:

```text
Given X seconds of usable spare time, what is P(delay <= X)?
```

The function `delay_prob(...)` answers this question.

It does the following:

```text
1. Find the matching lookup row
2. Read q01...q99
3. Compare X to the quantiles
4. Interpolate between the closest quantile levels
5. Return P(delay <= X)
```

Example:

```text
q80 = 120 sec
q90 = 160 sec
X   = 150 sec
```

Since 150 is between q80 and q90:

```text
P(delay <= 150) is between 0.80 and 0.90
```

With linear interpolation:

```text
P(delay <= 150) ≈ 0.875
```

---

## 24. Use in robust routing

At a transfer, the router computes:

```text
X = next_departure_time
    - scheduled_arrival_time
    - walking_time
    - minimum_transfer_time
```

Then it calls:

```python
p = delay_prob(trip_id, stop_id, date_value, X, scheduled_arrival_ts)
```

This gives the probability that the transfer is feasible.

For a route with multiple risky points:

```text
route_confidence = p1 * p2 * ...
```

under the independence assumption.

The route is kept if:

```text
route_confidence >= Q
```

where `Q` is the user-requested confidence.

---

## 25. Short summary

The full pipeline is:

```text
historical labeled data
-> train point delay model
-> predict calibration set
-> compute residuals
-> create operational buckets
-> estimate context-specific residual quantiles
-> validate coverage on test set
-> train final model on all historical data
-> predict February timetable
-> add calibrated residual quantiles
-> save lookup table
-> router uses lookup to compute P(delay <= X)
```

The core formula is:

```text
calibrated qα = pred_delay + residual_qα
```

The core routing interface is:

```text
delay_prob(...) = P(delay <= X)
```

The intuition is:

```text
point model = best guess
residual quantiles = data-driven safety margins
calibrated quantiles = best guess + safety margin
delay_prob = compare spare time against calibrated quantiles
```