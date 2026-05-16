# +
# src/model/delay_lookup.py

from datetime import date, datetime
from typing import Any, Dict, Optional

from pyspark.sql import SparkSession, functions as F


LOOKUP_DF = None

QUANTILE_POINTS = [
    (0.01, "q01"),
    (0.05, "q05"),
    (0.10, "q10"),
    (0.20, "q20"),
    (0.35, "q35"),
    (0.50, "q50"),
    (0.65, "q65"),
    (0.80, "q80"),
    (0.90, "q90"),
    (0.95, "q95"),
    (0.99, "q99"),
]


def init_delay_oracle(
    spark: SparkSession,
    lookup_path: str,
    cache: bool = True,
):
    """
    Load the precomputed delay lookup once.

    lookup_path example:
    /user/groups/com-490/H1/final/v1/route_demo_outputs/lookup/route_demo_lookup_february.parquet"
    """
    global LOOKUP_DF

    LOOKUP_DF = spark.read.parquet(lookup_path)

    if cache:
        LOOKUP_DF = LOOKUP_DF.cache()
        LOOKUP_DF.count()

    return LOOKUP_DF


def clean_bpuic(stop_id: Any) -> int:
    """
    Convert raw timetable stop_id to clean bpuic.

    Examples:
    "8592015:0:10001" -> 8592015
    8592015           -> 8592015
    """
    if stop_id is None:
        raise ValueError("stop_id cannot be None")

    if isinstance(stop_id, int):
        return stop_id

    stop_id_str = str(stop_id)
    return int(stop_id_str.split(":")[0])


def normalize_timestamp(ts: Any) -> datetime:
    """
    Convert timestamp-like value to Python datetime.
    """
    if ts is None:
        raise ValueError("scheduled_arrival_ts / arrival_timestamp cannot be None")

    if isinstance(ts, datetime):
        return ts

    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", ""))

    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def normalize_date(d: Any, scheduled_arrival_ts: Optional[Any] = None) -> date:
    """
    Convert date-like value to Python date.
    If date is missing, derive it from scheduled_arrival_ts.
    """
    if d is None:
        if scheduled_arrival_ts is None:
            raise ValueError("Either date or scheduled_arrival_ts must be provided")
        return normalize_timestamp(scheduled_arrival_ts).date()

    if isinstance(d, date) and not isinstance(d, datetime):
        return d

    if isinstance(d, datetime):
        return d.date()

    if isinstance(d, str):
        return date.fromisoformat(d[:10])

    raise TypeError(f"Unsupported date type: {type(d)}")


def prob_from_quantiles(row_dict: Dict[str, Any], X: float) -> float:
    """
    Approximate P(delay <= X) from calibrated delay quantiles.
    X is spare time in seconds.
    """
    if X <= 0:
        return 0.0

    points = []

    for p, qcol in QUANTILE_POINTS:
        value = row_dict.get(qcol)
        if value is not None:
            points.append((p, float(value)))

    if not points:
        return 0.0

    # Keep probability order and enforce monotone delay values.
    monotone_points = []
    running_max_delay = None

    for p, delay_value in points:
        if running_max_delay is None:
            running_max_delay = delay_value
        else:
            running_max_delay = max(running_max_delay, delay_value)

        monotone_points.append((p, running_max_delay))

    # Below q01: at most 1% probability.
    if X <= monotone_points[0][1]:
        return monotone_points[0][0]

    # Above q99: at least 99% probability. We return 0.99 conservatively.
    if X >= monotone_points[-1][1]:
        return monotone_points[-1][0]

    # Linear interpolation between neighboring quantile anchors.
    for (p_low, delay_low), (p_high, delay_high) in zip(
        monotone_points[:-1],
        monotone_points[1:],
    ):
        if delay_low <= X <= delay_high:
            if delay_high == delay_low:
                return p_high

            weight = (X - delay_low) / (delay_high - delay_low)
            return float(p_low + weight * (p_high - p_low))

    return 0.0


def delay_quantiles(
    trip_id: str,
    stop_id: Any,
    date_value: Any,
    scheduled_arrival_ts: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the lookup row with pred_delay and q01...q99.

    stop_id can be raw timetable stop_id or clean bpuic.
    date_value can be Python date or YYYY-MM-DD string.
    scheduled_arrival_ts is strongly recommended to avoid ambiguity.
    """
    global LOOKUP_DF

    if LOOKUP_DF is None:
        raise RuntimeError(
            "LOOKUP_DF is not initialized. Call init_delay_oracle(spark, lookup_path) first."
        )

    bpuic = clean_bpuic(stop_id)
    operating_day = normalize_date(date_value, scheduled_arrival_ts)

    cond = (
        (F.col("trip_id") == F.lit(trip_id))
        & (F.col("bpuic") == F.lit(bpuic))
        & (F.col("operating_day") == F.lit(operating_day))
    )

    if scheduled_arrival_ts is not None:
        ts = normalize_timestamp(scheduled_arrival_ts)
        cond = cond & (F.col("scheduled_arrival_ts") == F.lit(ts))

    rows = (
        LOOKUP_DF
        .filter(cond)
        .select(
            "operating_day",
            "trip_id",
            "bpuic",
            "stop_name",
            "scheduled_arrival_ts",
            "line_text",
            "transport_clean",
            "pred_delay",
            "q01", "q05", "q10", "q20", "q35", "q50",
            "q65", "q80", "q90", "q95", "q99",
            "chosen_calibration_level_q90",
        )
        .limit(2)
        .collect()
    )

    if len(rows) == 0:
        return None

    if len(rows) > 1 and scheduled_arrival_ts is None:
        raise ValueError(
            "Ambiguous lookup: multiple rows found. "
            "Pass scheduled_arrival_ts / arrival_timestamp as an additional key."
        )

    return rows[0].asDict()


def delay_prob(
    trip_id: str,
    stop_id: Any,
    date_value: Any,
    X: float,
    scheduled_arrival_ts: Optional[Any] = None,
) -> float:
    """
    Return P(delay <= X).

    trip_id: str
    stop_id: raw timetable stop_id or clean bpuic
    date_value: operating_day, e.g. '2026-02-08'
    X: spare time in seconds after walking + min transfer time are subtracted
    scheduled_arrival_ts: recommended, from timetable arrival_timestamp
    """
    if X <= 0:
        return 0.0

    q = delay_quantiles(
        trip_id=trip_id,
        stop_id=stop_id,
        date_value=date_value,
        scheduled_arrival_ts=scheduled_arrival_ts,
    )

    if q is None:
        # Conservative fallback: if we cannot find the row, treat it as unsafe.
        return 0.0

    return prob_from_quantiles(q, X)


def delay_prob_from_timetable_row(row: Any, X: float) -> float:
    """
    Convenience wrapper when Mauro calls directly from timetable_february rows.

    Expects raw timetable columns:
    - trip_id
    - stop_id
    - arrival_timestamp
    """
    return delay_prob(
        trip_id=row["trip_id"],
        stop_id=row["stop_id"],
        date_value=row["arrival_timestamp"].date(),
        X=X,
        scheduled_arrival_ts=row["arrival_timestamp"],
    )


def add_lookup_keys_from_timetable_df(df):
    """
    For vectorized Spark joins: convert raw timetable_february schema
    into lookup-compatible keys.
    """
    return (
        df
        .withColumn("bpuic", F.split(F.col("stop_id"), ":").getItem(0).cast("int"))
        .withColumn("scheduled_arrival_ts", F.col("arrival_timestamp"))
        .withColumn("operating_day", F.to_date(F.col("arrival_timestamp")))
    )
# -


