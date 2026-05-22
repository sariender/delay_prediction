"""Special-day and event features for the SBB delay model.

Two reference tables drive these features:
  1. SPECIAL_DAYS          — public holidays + manually validated Lausanne-region events
  2. SCHOOL_HOLIDAY_RANGES — Vaud school holiday date ranges within the modeling window

Sources are official Lausanne/Vaud holiday calendars + manually validated event dates.
Kept small and reliable rather than scraped.

Public API:
  - build_special_day_tables(spark) → (special_days_df, school_holiday_ranges_df)
  - add_special_day_features(features_df, special_days_df, school_holiday_ranges_df)
"""

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import Window


# (operating_day, special_day_name, special_day_type, special_day_intensity)
SPECIAL_DAYS = [
    # -------------------------
    # Public holidays / official special days
    # -------------------------
    ("2024-08-01", "swiss_national_day", "public_holiday", 1),
    ("2024-09-16", "federal_fast_monday", "public_holiday", 1),
    ("2024-12-24", "christmas_eve", "special_day", 1),
    ("2024-12-25", "christmas_day", "public_holiday", 1),
    ("2024-12-31", "new_years_eve", "special_day", 1),
    ("2025-01-01", "new_years_day", "public_holiday", 1),
    ("2025-01-02", "berchtolds_day", "public_holiday", 1),
    ("2025-04-18", "good_friday", "public_holiday", 1),
    ("2025-04-21", "easter_monday", "public_holiday", 1),
    ("2025-05-29", "ascension_day", "public_holiday", 1),
    ("2025-06-09", "whit_monday", "public_holiday", 1),

    # -------------------------
    # Major Lausanne-region events
    # -------------------------
    ("2024-07-02", "festival_de_la_cite", "festival", 1),
    ("2024-07-03", "festival_de_la_cite", "festival", 1),
    ("2024-07-04", "festival_de_la_cite", "festival", 1),
    ("2024-07-05", "festival_de_la_cite", "festival", 1),
    ("2024-07-06", "festival_de_la_cite", "festival", 1),
    ("2024-07-07", "festival_de_la_cite", "festival", 1),

    ("2024-10-27", "lausanne_marathon", "sports_event", 2),

    ("2025-04-04", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-05", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-06", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-07", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-08", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-09", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-10", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-11", "cully_jazz_festival", "festival_region", 1),
    ("2025-04-12", "cully_jazz_festival", "festival_region", 1),

    ("2025-05-03", "20km_lausanne", "sports_event", 2),
    ("2025-05-04", "20km_lausanne", "sports_event", 2),

    ("2025-05-05", "bdfil_lausanne", "festival", 1),
    ("2025-05-06", "bdfil_lausanne", "festival", 1),
    ("2025-05-07", "bdfil_lausanne", "festival", 1),
    ("2025-05-08", "bdfil_lausanne", "festival", 1),
    ("2025-05-09", "bdfil_lausanne", "festival", 1),
    ("2025-05-10", "bdfil_lausanne", "festival", 1),
    ("2025-05-11", "bdfil_lausanne", "festival", 1),
    ("2025-05-12", "bdfil_lausanne", "festival", 1),
    ("2025-05-13", "bdfil_lausanne", "festival", 1),
    ("2025-05-14", "bdfil_lausanne", "festival", 1),
    ("2025-05-15", "bdfil_lausanne", "festival", 1),
    ("2025-05-16", "bdfil_lausanne", "festival", 1),
    ("2025-05-17", "bdfil_lausanne", "festival", 1),
    ("2025-05-18", "bdfil_lausanne", "festival", 1),

    ("2025-05-09", "balelec_festival", "festival", 2),

    ("2025-06-07", "miam_festival", "festival", 1),
    ("2025-06-08", "miam_festival", "festival", 1),
    ("2025-06-09", "miam_festival", "festival", 1),

    ("2025-06-12", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-13", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-14", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-15", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-16", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-17", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-18", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-19", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-20", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-21", "swiss_gymnastics_festival", "major_sports_event", 3),
    ("2025-06-22", "swiss_gymnastics_festival", "major_sports_event", 3),

    # -------------------------
    # 2026 — public holidays (Canton de Vaud, verified against vd.ch)
    # -------------------------
    ("2026-01-01", "new_years_day",       "public_holiday", 1),
    ("2026-01-02", "berchtolds_day",      "public_holiday", 1),
    ("2026-04-03", "good_friday",         "public_holiday", 1),
    ("2026-04-06", "easter_monday",       "public_holiday", 1),
    ("2026-05-14", "ascension_day",       "public_holiday", 1),
    ("2026-05-25", "whit_monday",         "public_holiday", 1),
    ("2026-08-01", "swiss_national_day",  "public_holiday", 1),
    ("2026-09-21", "federal_fast_monday", "public_holiday", 1),
    ("2026-12-24", "christmas_eve",       "special_day",    1),
    ("2026-12-25", "christmas_day",       "public_holiday", 1),
    ("2026-12-31", "new_years_eve",       "special_day",    1),

    # -------------------------
    # 2026 — major Lausanne-region events (all dates verified, see source per block)
    # -------------------------
    # Cully Jazz Festival, 43rd ed. — verified cullyjazz.ch (Apr 10-18)
    ("2026-04-10", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-11", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-12", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-13", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-14", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-15", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-16", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-17", "cully_jazz_festival", "festival_region", 1),
    ("2026-04-18", "cully_jazz_festival", "festival_region", 1),

    # 20km de Lausanne, 44th ed. (~38.5k runners, road closures) — verified lausanne.ch
    ("2026-04-24", "20km_lausanne", "sports_event", 2),
    ("2026-04-25", "20km_lausanne", "sports_event", 2),
    ("2026-04-26", "20km_lausanne", "sports_event", 2),

    # BDFIL, 20th ed. (Lausanne train-station district) — verified bdfil.ch (Apr 27 - May 10)
    ("2026-04-27", "bdfil_lausanne", "festival", 1),
    ("2026-04-28", "bdfil_lausanne", "festival", 1),
    ("2026-04-29", "bdfil_lausanne", "festival", 1),
    ("2026-04-30", "bdfil_lausanne", "festival", 1),
    ("2026-05-01", "bdfil_lausanne", "festival", 1),
    ("2026-05-02", "bdfil_lausanne", "festival", 1),
    ("2026-05-03", "bdfil_lausanne", "festival", 1),
    ("2026-05-04", "bdfil_lausanne", "festival", 1),
    ("2026-05-05", "bdfil_lausanne", "festival", 1),
    ("2026-05-06", "bdfil_lausanne", "festival", 1),
    ("2026-05-07", "bdfil_lausanne", "festival", 1),
    ("2026-05-08", "bdfil_lausanne", "festival", 1),
    ("2026-05-09", "bdfil_lausanne", "festival", 1),
    ("2026-05-10", "bdfil_lausanne", "festival", 1),

    # Balélec, EPFL campus (~15k attendees) — verified balelec.ch (May 1)
    ("2026-05-01", "balelec_festival", "festival", 2),

    # Miam Festival, central Lausanne (Pentecost weekend) — verified lausanneatable.ch (May 23-25)
    ("2026-05-23", "miam_festival", "festival", 1),
    ("2026-05-24", "miam_festival", "festival", 1),
    ("2026-05-25", "miam_festival", "festival", 1),

    # Festival de la Cité, 54th ed. — verified festivalcite.ch (Jun 30 - Jul 5)
    ("2026-06-30", "festival_de_la_cite", "festival", 1),
    ("2026-07-01", "festival_de_la_cite", "festival", 1),
    ("2026-07-02", "festival_de_la_cite", "festival", 1),
    ("2026-07-03", "festival_de_la_cite", "festival", 1),
    ("2026-07-04", "festival_de_la_cite", "festival", 1),
    ("2026-07-05", "festival_de_la_cite", "festival", 1),

    # Lausanne Marathon (~13.6k runners) — verified worldsmarathons.com (Oct 25)
    ("2026-10-25", "lausanne_marathon", "sports_event", 2),
]

SPECIAL_DAYS_COLS = [
    "operating_day",
    "special_day_name",
    "special_day_type",
    "special_day_intensity",
]


# (holiday_start, holiday_end, school_holiday_name)
SCHOOL_HOLIDAY_RANGES = [
    ("2024-07-01", "2024-08-18", "summer_holiday_2024_partial"),
    ("2024-10-12", "2024-10-27", "autumn_holiday_2024"),
    ("2024-12-21", "2025-01-05", "winter_holiday_2024_2025"),
    ("2025-02-15", "2025-02-23", "sport_holiday_2025"),
    ("2025-04-12", "2025-04-27", "easter_holiday_2025"),
    ("2025-05-29", "2025-06-01", "ascension_holiday_2025"),
    ("2025-06-09", "2025-06-09", "whit_monday_2025"),
    ("2025-06-28", "2025-06-30", "summer_holiday_2025_partial"),

    # 2026 — Canton de Vaud school holidays (verified against vd.ch)
    ("2025-12-20", "2026-01-04", "winter_holiday_2025_2026"),
    ("2026-02-14", "2026-02-22", "sport_holiday_2026"),
    ("2026-04-03", "2026-04-19", "easter_holiday_2026"),
    ("2026-05-14", "2026-05-17", "ascension_holiday_2026"),
    ("2026-05-25", "2026-05-25", "whit_monday_2026"),
    ("2026-06-27", "2026-08-16", "summer_holiday_2026"),
    ("2026-10-10", "2026-10-25", "autumn_holiday_2026"),
    ("2026-12-24", "2027-01-10", "winter_holiday_2026_2027"),
]

SCHOOL_HOLIDAY_COLS = ["holiday_start", "holiday_end", "school_holiday_name"]


def build_special_day_tables(spark):
    """Build the two reference DataFrames used by `add_special_day_features`.

    Returns (special_days_df, school_holiday_ranges_df), both cached.
    """
    special_days_pdf = pd.DataFrame(SPECIAL_DAYS, columns=SPECIAL_DAYS_COLS)
    special_days_df = (
        spark.createDataFrame(special_days_pdf)
        .withColumn("operating_day", F.to_date("operating_day"))
        .dropDuplicates(["operating_day", "special_day_name"])
    )

    # Some dates have >1 event (e.g. 2026-05-01: BDFIL + Balélec). A plain
    # left-join on operating_day would then duplicate every feature row on
    # that date. Collapse to exactly one row per day: keep the highest
    # special_day_intensity, breaking ties deterministically by name so the
    # result is reproducible across runs.
    _one_per_day = Window.partitionBy("operating_day").orderBy(
        F.col("special_day_intensity").desc(),
        F.col("special_day_name").asc(),
    )
    special_days_df = (
        special_days_df
        .withColumn("_rn", F.row_number().over(_one_per_day))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .cache()
    )

    school_holiday_pdf = pd.DataFrame(SCHOOL_HOLIDAY_RANGES, columns=SCHOOL_HOLIDAY_COLS)
    school_holiday_ranges_df = (
        spark.createDataFrame(school_holiday_pdf)
        .withColumn("holiday_start", F.to_date("holiday_start"))
        .withColumn("holiday_end", F.to_date("holiday_end"))
        .cache()
    )

    return special_days_df, school_holiday_ranges_df


def add_special_day_features(features_df, special_days_df, school_holiday_ranges_df):
    """Join special-day and school-holiday flags onto `features_df`.

    Adds: special_day_name, special_day_type, special_day_intensity, is_special_day,
          school_holiday_name, is_school_holiday.
    Drops the helper columns holiday_start / holiday_end after the range join.
    """
    df_events = (
        features_df
        .join(special_days_df, on="operating_day", how="left")
        .withColumn(
            "is_special_day",
            F.when(F.col("special_day_name").isNotNull(), 1).otherwise(0),
        )
    )

    df_with_school = (
        df_events
        .join(
            school_holiday_ranges_df,
            (df_events.operating_day >= school_holiday_ranges_df.holiday_start)
            & (df_events.operating_day <= school_holiday_ranges_df.holiday_end),
            how="left",
        )
        .withColumn(
            "is_school_holiday",
            F.when(F.col("school_holiday_name").isNotNull(), 1).otherwise(0),
        )
    )

    return (
        df_with_school
        .fillna({
            "special_day_name": "none",
            "special_day_type": "none",
            "special_day_intensity": 0,
            "school_holiday_name": "none",
            "is_school_holiday": 0,
        })
        .drop("holiday_start", "holiday_end")
    )
