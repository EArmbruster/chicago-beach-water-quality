"""Step 6: fix the two genuine bugs, and clear now-stale outputs.

1. precip_max_in_window_per_station summed inside its CTE, so the outer
   PIVOT ... USING MAX took the max of a single value -> the _max_ columns were
   byte-identical to the _sum_ ones and peak intensity did not exist.
2. groin_onshore_* went NaN whenever no observation in the window was onshore,
   colliding with the min_coverage NaN that means "not enough data". Two
   opposite meanings sharing one encoding.
"""
import sys
from pathlib import Path
import nbformat

REPO = Path("/Users/emily/Documents/chicago-beach-water-quality")
NB = REPO / "Final_feature_building.ipynb"

PRECIP_CELL = "03f1c8c5"
WIND_CELL = "5c9b64ec"

PRECIP_SRC = '''def bucket_precip_per_station(precip_df, ts_col, value_col, station_col,
                              bucket_minutes, stations):
    if stations is None:
        station_filter = ""
    else:
        quoted = ", ".join(f"'{s}'" for s in stations)
        station_filter = f"AND {station_col} IN ({quoted})"

    sql = f"""
        SELECT
            time_bucket(INTERVAL '{bucket_minutes} minutes', {ts_col}) AS ts,
            {station_col} AS station,
            MAX({value_col}) AS precip_mm
        FROM precip_df
        WHERE {ts_col} IS NOT NULL
          AND {value_col} IS NOT NULL
          {station_filter}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return duckdb.sql(sql).df()


def precip_sum_in_window_per_station(readings, bucketed, window_hours, source_label):
    """Total precipitation in the lookback window, per station.

    The join is strictly backward-looking (ts > sample_ts - window AND
    ts <= sample_ts), so no future information leaks into a sample.
    """
    sql = f"""
        WITH summed AS (
            SELECT
                r.sample_ts,
                b.station,
                COALESCE(SUM(b.precip_mm), 0) AS precip_mm
            FROM readings r
            LEFT JOIN bucketed b
              ON b.ts >  r.sample_ts - INTERVAL {window_hours} HOUR
             AND b.ts <= r.sample_ts
            GROUP BY r.sample_ts, b.station
        )
        PIVOT summed
            ON station
            USING SUM(precip_mm)
            GROUP BY sample_ts
    """
    out = duckdb.sql(sql).df()
    rename = {
        col: f"{source_label}_{col}_precip_sum_{window_hours}h"
        for col in out.columns if col != "sample_ts"
    }
    return out.rename(columns=rename)


def precip_max_in_window_per_station(readings, bucketed, window_hours, source_label):
    """Peak single-bucket intensity in the lookback window, per station.

    Genuinely different information from the sum: 20 mm as a cloudburst and
    20 mm as six hours of drizzle produce very different runoff, and only the
    peak separates them.

    FIXED: the CTE aggregate is now MAX, not SUM. Summing inside the CTE
    collapsed each (sample_ts, station) to one row, so the outer
    PIVOT ... USING MAX took the max of a single value and these columns came
    out identical to the _sum_ ones -- peak intensity was absent from the
    feature set entirely.

    Bucket width sets the unit of intensity: COC buckets are 60 min (peak
    hourly), USGS 5 min (peak 5-minute). Comparable across samples within a
    source, not between sources.
    """
    sql = f"""
        WITH peaked AS (
            SELECT
                r.sample_ts,
                b.station,
                COALESCE(MAX(b.precip_mm), 0) AS precip_mm
            FROM readings r
            LEFT JOIN bucketed b
              ON b.ts >  r.sample_ts - INTERVAL {window_hours} HOUR
             AND b.ts <= r.sample_ts
            GROUP BY r.sample_ts, b.station
        )
        PIVOT peaked
            ON station
            USING MAX(precip_mm)
            GROUP BY sample_ts
    """
    out = duckdb.sql(sql).df()
    rename = {
        col: f"{source_label}_{col}_precip_max_{window_hours}h"
        for col in out.columns if col != "sample_ts"
    }
    return out.rename(columns=rename)


def build_precip_features_for_source(readings, precip_df, ts_col, value_col,
                                     station_col, source_label, bucket_minutes,
                                     stations, sum_windows_hours, max_windows_hours):
    bucketed = bucket_precip_per_station(
        precip_df, ts_col, value_col, station_col, bucket_minutes, stations
    )
    out = readings[["sample_ts"]].copy()
    for hours in sum_windows_hours:
        feat = precip_sum_in_window_per_station(readings, bucketed, hours, source_label)
        out = out.merge(feat, on="sample_ts", how="left")
    for hours in max_windows_hours:
        feat = precip_max_in_window_per_station(readings, bucketed, hours, source_label)
        out = out.merge(feat, on="sample_ts", how="left")
    return out'''

WIND_SRC = '''def build_wind_features(readings, chii2_df, facing_az_deg, groin_length_m,
                        windows_hours=[6, 12, 24], tolerance_deg=20,
                        min_coverage=0.5, use_geo_features=True):
    """Wind features from CHII2, decomposed relative to the beach's facing.

    Onshore/alongshore projection follows Madani & Seth (2020); facing azimuth
    and groin length come from Mattheus et al. (2022) via beach_geomorphology.
    """
    facing_rad = np.deg2rad(facing_az_deg)
    tol = int(tolerance_deg)

    wind = chii2_df[["timestamp_central", "WSPD", "WDIR"]].copy()
    wind = wind.dropna(subset=["WSPD"])

    wdir_rad = np.deg2rad(wind["WDIR"])
    # WDIR is the direction wind comes FROM, so negate to get where it blows TO
    wind["u"] = -wind["WSPD"] * np.sin(wdir_rad)
    wind["v"] = -wind["WSPD"] * np.cos(wdir_rad)
    wind["onshore"] = wind["u"] * np.sin(facing_rad) + wind["v"] * np.cos(facing_rad)
    wind["alongshore"] = wind["u"] * np.cos(facing_rad) - wind["v"] * np.sin(facing_rad)

    delta = (wind["WDIR"] - facing_az_deg + 180) % 360 - 180
    wind["is_onshore"] = (delta.abs() <= tolerance_deg).astype(int)
    wind.loc[wind["WDIR"].isna(), "is_onshore"] = np.nan

    samples = readings[["sample_ts"]].drop_duplicates()
    out = samples.copy()

    for wh in windows_hours:
        expected_obs = wh * 6          # CHII2 reports every 10 minutes
        min_obs = int(expected_obs * min_coverage)

        # FIXED: the mean onshore wind speed is COALESCEd to 0.
        # Previously AVG(CASE WHEN is_onshore = 1 ...) returned NULL when no
        # observation in the window was onshore, making the whole product NULL.
        # That collided with the min_coverage NULL below, which means something
        # entirely different ("not enough observations to judge"). One encoding,
        # two opposite meanings -- and the median imputer then filled the
        # "no onshore wind" case with a TYPICAL onshore value.
        #
        # Now: NULL means only "insufficient coverage"; a genuine absence of
        # onshore wind gives 0, which is what zero groin-onshore exposure is.
        # (AVG(is_onshore) is already 0 in that case, so the product is 0.)
        groin_sql = f"""
            CASE WHEN COUNT(w.is_onshore) >= {min_obs}
                THEN {groin_length_m}
                    * AVG(w.is_onshore)
                    * COALESCE(
                        AVG(CASE WHEN w.is_onshore = 1 THEN w.WSPD ELSE NULL END),
                        0)
                END
                AS groin_onshore_{wh}h_tol{tol}
        """ if use_geo_features else ""

        agg = duckdb.query(f"""
            SELECT
                s.sample_ts,
                CASE WHEN COUNT(w.WSPD) >= {min_obs}
                    THEN AVG(w.WSPD) END AS chii2_wspd_mean_{wh}h,
                CASE WHEN COUNT(w.WSPD) >= {min_obs}
                    THEN MAX(w.WSPD) END AS chii2_wspd_max_{wh}h,
                CASE WHEN COUNT(w.onshore) >= {min_obs}
                    THEN AVG(w.onshore) END AS chii2_onshore_mean_{wh}h,
                CASE WHEN COUNT(w.alongshore) >= {min_obs}
                    THEN AVG(w.alongshore) END AS chii2_alongshore_mean_{wh}h
                {(',' + groin_sql) if use_geo_features else ''}
            FROM samples s
            LEFT JOIN wind w
                ON w.timestamp_central > s.sample_ts - INTERVAL '{wh} hours'
               AND w.timestamp_central <= s.sample_ts
            GROUP BY s.sample_ts
        """).df()

        out = out.merge(agg, on="sample_ts", how="left")

    return out'''

nb = nbformat.read(NB, as_version=4)
found = set()
for c in nb.cells:
    if c.get("id") == PRECIP_CELL:
        c.source = PRECIP_SRC
        found.add(PRECIP_CELL)
    elif c.get("id") == WIND_CELL:
        c.source = WIND_SRC
        found.add(WIND_CELL)

missing = {PRECIP_CELL, WIND_CELL} - found
if missing:
    print("!! cells not found:", missing)
    sys.exit(1)

# Code changed -> every stored output is now stale. Clear them.
cleared = 0
for c in nb.cells:
    if c.cell_type == "code":
        if c.get("outputs"):
            cleared += 1
        c["outputs"] = []
        c["execution_count"] = None

nbformat.write(nb, NB)
print(f"patched {len(found)} cells, cleared outputs on {cleared} cells")
print(f"{NB} -> {NB.stat().st_size:,} bytes")
