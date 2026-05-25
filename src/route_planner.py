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


def mode_emoji(mode):
    """Return the emoji representing a transit mode."""
    return {
        "Bus": "🚌",
        "T": "🚊", # tram
        "M": "🚇", # metro
        "R": "🚆", # train
        "NJ": "🌙", # night train
        "BAT": "⛴️", # boat
        # Train categories that pass through as raw route_desc
        "IC": "🚆",
        "ICE": "🚆",
        "IR": "🚆",
        "RE": "🚆",
        "S": "🚆",
        "SN": "🚆",
        "TGV": "🚆",
        "EC": "🚆",
        "RJX": "🚆",
        "TER": "🚆",
        "PE": "🚆",
        "EXT": "🚆",
    }.get(mode, "🚌")


class RobustJourneyPlanner:
    """
    Robust journey planner with configurable options.

    Usage (Spark):
        planner = RobustJourneyPlanner()
        planner.build_graph(spark, target_date="2026-02-04")
        journeys = planner.plan(
            source="8592076",
            target="8501214",
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

        timetable_df, transfers_df = self._load_or_fetch_data(spark, timetable_path, transfers_path)

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

    def _load_or_fetch_data(
        self,
        spark,
        timetable_path: Optional[str] = None,
        transfers_path: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """
        Load timetable and transfers data from group cache if available,
        otherwise fall back to default paths or fetch from raw Iceberg tables.
        """
        # Determine the group name from the paths, default to "H1"
        group_name = "H1"
        for path_opt in [timetable_path, transfers_path, self.TIMETABLE_PATH, self.TRANSFERS_PATH]:
            if path_opt and "/user/groups/com-490/" in path_opt:
                parts = path_opt.split("/")
                if len(parts) > 4:
                    group_name = parts[4]
                    break

        # If regions are specified, we use/generate regional cache under group folder
        if self._region_uuids:
            import hashlib
            regions_hash = hashlib.md5("_".join(sorted(self._region_uuids)).encode()).hexdigest()
            cache_dir = f"/user/groups/com-490/{group_name}/final/v1/precomputed"
            cache_tt_path = f"{cache_dir}/timetable_{regions_hash}.parquet"
            cache_tr_path = f"{cache_dir}/transfers_{regions_hash}.parquet"

            # 1. Try to load from cache
            try:
                print(f"Checking if precomputed tables exist under group folder: {cache_dir}...")
                timetable_df = spark.read.parquet(cache_tt_path)
                try:
                    transfers_df = spark.read.parquet(cache_tr_path)
                except Exception:
                    transfers_df = None
                print("Precomputed tables found and loaded from cache.")
                return timetable_df, transfers_df
            except Exception:
                print("Precomputed tables not found in cache. Resolving fallback...")

            # 2. Try loading from default/provided paths and save to cache
            timetable_df = None
            transfers_df = None
            try:
                tt_path = timetable_path or self.TIMETABLE_PATH
                tr_path = transfers_path or self.TRANSFERS_PATH
                print(f"Loading from paths: {tt_path}")
                timetable_df = spark.read.parquet(tt_path)
                try:
                    transfers_df = spark.read.parquet(tr_path)
                except Exception:
                    transfers_df = None
                
                print(f"Persisting loaded tables to cache: {cache_dir}...")
                try:
                    timetable_df.write.mode("overwrite").parquet(cache_tt_path)
                    if transfers_df is not None:
                        transfers_df.write.mode("overwrite").parquet(cache_tr_path)
                except Exception as e_write:
                    print(f"Warning: could not write to cache path ({e_write})")
                return timetable_df, transfers_df
            except Exception as e_default:
                # 3. Fetch from raw Iceberg tables
                print(f"Could not load tables from default paths ({e_default}). Generating from raw Iceberg tables...")
                timetable_df, transfers_df = self._fetch_raw_timetable_and_transfers(spark, self._region_uuids)

                print(f"Persisting fetched tables to cache: {cache_dir}...")
                try:
                    timetable_df.write.mode("overwrite").parquet(cache_tt_path)
                    if transfers_df is not None:
                        transfers_df.write.mode("overwrite").parquet(cache_tr_path)
                except Exception as e_write:
                    print(f"Warning: could not write to cache path ({e_write})")
                return timetable_df, transfers_df

        # Fallback if no regions are specified (load directly from paths)
        tt_path = timetable_path or self.TIMETABLE_PATH
        tr_path = transfers_path or self.TRANSFERS_PATH
        timetable_df = spark.read.parquet(tt_path)
        try:
            transfers_df = spark.read.parquet(tr_path)
        except Exception:
            transfers_df = None
        return timetable_df, transfers_df

    def _fetch_raw_timetable_and_transfers(self, spark, regions: List[str]):
        """
        Fetch timetable and transfers from raw Iceberg tables, filtering by region UUIDs.
        """
        from sedona.spark import SedonaContext
        spark = SedonaContext.create(spark)

        # 1. Load agency text
        print("Loading agency data...")
        df_agency = spark.read.options(header=True).csv(
            "/data/com-490/bronze/sbb/agency/year=2026/month=01/day=31/agency.txt"
        )
        df_agency.createOrReplaceTempView("agency")

        # 2. Load transfers CSV
        print("Loading transfers data...")
        try:
            transfers_df = spark.read.csv(
                "/data/com-490/bronze/sbb/transfers/year=2026/month=01/day=31/transfers.txt", 
                header=True, 
                inferSchema=True
            )
        except Exception:
            transfers_df = None

        # 3. Create operator mapping view
        operator_mapping = spark.table("iceberg.sbb.istdaten") \
            .select("operator_abrv", "operator_id", "operator_name") \
            .distinct()
        operator_mapping.createOrReplaceTempView("op_map")

        # 4. Filter stop times by shapes
        print(f"Filtering stops and trips for region UUIDs: {regions}...")
        formatted_regions = ", ".join([f"'{r}'" for r in regions])
        
        trip_stop_events_df = spark.sql(f"""
        SELECT DISTINCT
         st.trip_id, st.stop_id, split_part(s.stop_id, ':', 1) AS cleaned_stop_id, s.stop_name, s.stop_lat, s.stop_lon, s.location_type, st.arrival_time, st.departure_time,
         st.stop_sequence, t.service_id, t.trip_short_name, ist.operator_id, ist.operator_name,
         c.monday, c.tuesday, c.wednesday, c.thursday, c.friday, c.saturday, c.sunday,
         c.start_date, c.end_date, r.route_short_name, 
         CASE 
                WHEN r.route_desc = 'EN' THEN 'NJ'
                WHEN r.route_desc IN ('EXB', 'KB', 'RUB', 'TX') THEN 'Bus'
                WHEN r.route_desc IN ('BP', 'FAE') THEN 'BAT'
                WHEN r.route_desc IN ('GB', 'PB', 'SL') THEN 'T'
                WHEN r.route_desc IN ('FUN', 'ASC') THEN 'M'
                WHEN r.route_desc IN ('ZUG', 'EST', 'ARZ', 'IRE') THEN 'R'
                ELSE r.route_desc 
            END AS transport_clean
        FROM iceberg.sbb.stop_times st
         INNER JOIN iceberg.sbb.stops s 
            ON s.stop_id = st.stop_id AND s.pub_date = '2026-01-31' 
        INNER JOIN iceberg.sbb.trips t 
            ON st.trip_id = t.trip_id AND t.pub_date = '2026-01-31'
        INNER JOIN iceberg.sbb.calendar c 
            ON c.service_id = t.service_id AND c.pub_date = '2026-01-31'
        INNER JOIN iceberg.sbb.routes r 
            ON r.route_id = t.route_id AND r.pub_date = '2026-01-31'
        INNER JOIN agency a 
            ON a.agency_id = r.agency_id
         LEFT JOIN op_map ist 
            ON a.agency_name = ist.operator_name
         JOIN iceberg.geo.shapes g
            ON ST_Contains(ST_GeomFromWKB(g.wkb_geometry), ST_Point(s.stop_lon, s.stop_lat))
        WHERE 
         st.pub_date = '2026-01-31' AND
         st.stop_id LIKE '85%' AND 
         g.uuid IN ({formatted_regions})
        """)

        trip_stop_events_df.createOrReplaceTempView("base_trips")

        # 5. Build February timetable
        print("Expanding timetable calendar for February 2026...")
        timetable_df = spark.sql("""
        WITH date_range AS (
            SELECT explode(sequence(to_date('2026-02-01'), to_date('2026-02-28'))) AS target_date
        ),
        calendar_expanded AS (
            SELECT 
                d.target_date,
                t.*,
                lower(date_format(d.target_date, 'EEEE')) as day_name
            FROM date_range d
            CROSS JOIN base_trips t
            WHERE d.target_date BETWEEN t.start_date AND t.end_date
        ),
        exceptions AS (
            SELECT service_id, exception_date as ex_date, exception_type 
            FROM iceberg.sbb.calendar_dates
            WHERE exception_date BETWEEN '2026-02-01' AND '2026-02-28'
        )
        SELECT DISTINCT
            to_timestamp(
                concat(
                    date_add(ce.target_date, CASE WHEN CAST(split(ce.arrival_time, ':')[0] AS INT) >= 24 THEN 1 ELSE 0 END),
                    ' ',
                    CASE 
                        WHEN CAST(split(ce.arrival_time, ':')[0] AS INT) >= 24 
                        THEN concat(lpad(CAST(split(ce.arrival_time, ':')[0] AS INT) - 24, 2, '0'), ':', split(ce.arrival_time, ':')[1], ':', split(ce.arrival_time, ':')[2])
                        ELSE ce.arrival_time 
                    END
                )
            ) AS arrival_timestamp,

            to_timestamp(
                concat(
                    date_add(ce.target_date, CASE WHEN CAST(split(ce.departure_time, ':')[0] AS INT) >= 24 THEN 1 ELSE 0 END),
                    ' ',
                    CASE 
                        WHEN CAST(split(ce.departure_time, ':')[0] AS INT) >= 24 
                        THEN concat(lpad(CAST(split(ce.departure_time, ':')[0] AS INT) - 24, 2, '0'), ':', split(ce.departure_time, ':')[1], ':', split(ce.departure_time, ':')[2])
                        ELSE ce.departure_time 
                    END
                )
            ) AS departure_timestamp,

            ce.trip_id,
            ce.stop_id,
            ce.stop_name,
            ce.stop_lat,
            ce.stop_lon,
            ce.stop_sequence,
            ce.service_id,
            ce.trip_short_name,
            ce.operator_id,
            ce.operator_name,
            ce.route_short_name,
            ce.transport_clean,
            CAST(NULL AS TIMESTAMP) AS predicted_arrival_time,
            CAST(NULL AS TIMESTAMP) AS predicted_departure_time

        FROM calendar_expanded ce
        LEFT JOIN exceptions ex 
            ON ce.service_id = ex.service_id AND ce.target_date = ex.ex_date
        WHERE 
            (
                (
                    (ce.day_name = 'monday' AND ce.monday = 1) OR
                    (ce.day_name = 'tuesday' AND ce.tuesday = 1) OR
                    (ce.day_name = 'wednesday' AND ce.wednesday = 1) OR
                    (ce.day_name = 'thursday' AND ce.thursday = 1) OR
                    (ce.day_name = 'friday' AND ce.friday = 1) OR
                    (ce.day_name = 'saturday' AND ce.saturday = 1) OR
                    (ce.day_name = 'sunday' AND ce.sunday = 1)
                )
                AND (ex.exception_type IS NULL OR ex.exception_type != 2)
            )
            OR (ex.exception_type = 1)
        """)

        # Clean up temporary views
        spark.catalog.dropTempView("agency")
        spark.catalog.dropTempView("op_map")
        spark.catalog.dropTempView("base_trips")

        return timetable_df, transfers_df

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
        timetable_df, transfers_df = self._load_or_fetch_data(spark, timetable_path, transfers_path)

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
                walk_sec = prev_leg.walk_duration_sec
                if walk_sec is None:
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
        mode: str = "all",
        departure_time: Optional[str] = None,
        arrival_time: Optional[str] = None,
        day: Optional[str] = None,
        min_confidence: Optional[float] = None,
        walking_speed: Optional[float] = None,
        max_walk_m: Optional[float] = None,
        extra_transfer_sec: Optional[int] = None,
        max_results: int = 5,
        modes: Optional[List[str]] = None,
    ) -> List[Journey]:
        """
        Plan journeys between source and target.

        Modes:
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

        if arrival_time is not None:
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
                raise ValueError("departure_time or arrival_time must be provided")
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

        # Support both single mode and multiple modes
        if modes is None:
            if mode is not None and mode != "all":
                modes = [mode]
            else:
                modes = []

        valid_modes = [m for m in modes if m in ["safest", "least_transfers", "least_walking"]]

        # Sort based on criteria hierarchy
        if valid_modes:
            def sort_key(j):
                key = []
                for m in valid_modes:
                    if m == "safest":
                        key.append(-j.confidence)
                    elif m == "least_transfers":
                        key.append(j.num_transfers)
                    elif m == "least_walking":
                        key.append(j.total_walk_m)
                # Tie breaker: fastest
                if arrival_time is not None:
                    key.append(-j.departure_ts)
                else:
                    key.append(j.arrival_ts)
                return tuple(key)
            journeys.sort(key=sort_key)
        else:
            if arrival_time is not None:
                journeys.sort(key=lambda j: -j.departure_ts)
            else:
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
                icon = mode_emoji(leg.transport_mode)
                lines.append(f"  {icon} {t_dep} {leg.from_name} → {t_arr} {leg.to_name} [{leg.trip_id}]")   
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
