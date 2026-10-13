"""Step 3 checks: run_experiment writes all four files, recipe.json round-trips,
load_experiments returns one group per run directory.

Uses a scratchpad out_dir so nothing is written into the repo.
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd

REPO = "/Users/emily/Documents/chicago-beach-water-quality"
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, REPO)

import pipeline_draft as P
from beach_geomorphology import get_geomorphology_df
from station_locations import get_beach_coords

dp = f"{REPO}/data/prepared_data/"
dataframes = {
    "beach_coc_precip_df": pd.read_parquet(dp + "COC_precip.parquet"),
    "usgs_precip_df": pd.read_parquet(dp + "usgs_precip.parquet"),
    "chii2_wind_df": pd.read_parquet(dp + "chii2_noaa.parquet"),
}
readings_df = pd.read_parquet(dp + "dna_readings.parquet")
beach_geo_df = get_geomorphology_df()
beach_coords = get_beach_coords(readings_df)

RECIPE = {
    "label": "smoke_test",
    "target_beach": "montrose",
    "min_year": 2021,
    "target": {"threshold": 1000},
    "features": {
        "precip": {
            "enabled": True,
            "sources": [
                {"df_name": "beach_coc_precip_df", "ts_col": "timestamp_central",
                 "value_col": "coc_mmprecip_sum1h", "station_col": "station_name",
                 "label": "coc", "bucket_minutes": 60,
                 "stations": ['foster', 'oak_street', '63rd_street']},
                {"df_name": "usgs_precip_df", "ts_col": "timestamp_central",
                 "value_col": "usgs_mmprecip_sum5m", "station_col": "station_name",
                 "label": "usgs", "bucket_minutes": 5,
                 "stations": ['south_shore', 'hamilton_park_at_chicago',
                              'ping_tom_park_at_chicago', 'oakton_college_at_skokie',
                              'winnetka_golf_club_at_winnetka', 'cicero',
                              'north_austin', 'lake_view']},
            ],
            "sum_windows_hours": [6, 24, 48, 72],
            "max_windows_hours": [6],
        },
        "wind": {"enabled": True, "use_geo_features": True, "df_name": "chii2_wind_df",
                 "windows_hours": [6, 12, 24], "tolerance_deg": 20, "min_coverage": 0.5},
        "distance": {"enabled": False}, "temp": {"enabled": False}, "sewer": {"enabled": False},
    },
}

OUT = SCRATCH / "experiments_test"

# montrose survives the guard. 'foster' does not: it is one of the five
# Mattheus beaches absent from the baseline results (foster, north_avenue,
# oak_street, oakwood, osterman), i.e. dropped for too few positives.
SKIP_EXPECTED = "foster"
run = P.run_experiment(
    RECIPE, beaches=["montrose", SKIP_EXPECTED],
    dataframes=dataframes, readings_df=readings_df, beach_geo_df=beach_geo_df,
    beach_coords=beach_coords, out_dir=str(OUT), repo_root=REPO,
)

print("\n--- files written ---")
expected = {"recipe.json", "results.csv", "sweeps.csv", "meta.json"}
actual = {p.name for p in run.run_dir.iterdir()}
print(sorted(actual))
assert expected <= actual, f"MISSING: {expected - actual}"

print("\n--- recipe.json round-trips ---")
loaded = json.loads((run.run_dir / "recipe.json").read_text())
assert loaded == RECIPE, "recipe.json does not round-trip to an equal dict"
print("OK: loaded == RECIPE")

print("\n--- meta.json ---")
meta = json.loads((run.run_dir / "meta.json").read_text())
for k in ["run_id", "recipe_sha256", "notebook_sha256", "git_commit", "git_dirty",
          "beaches_requested", "beaches_modelled", "beaches_skipped", "elapsed_sec"]:
    v = meta.get(k)
    if isinstance(v, str) and len(v) > 24:
        v = v[:16] + "..."
    print(f"  {k}: {v}")
assert meta["beaches_skipped"] == [SKIP_EXPECTED], meta["beaches_skipped"]
assert meta["notebook_sha256"], "notebook hash not captured"
print("OK: skip guard recorded, notebook hash captured")

print("\n--- summary() ---")
print(run.summary().to_string(index=False))

print("\n--- load_experiments ---")
allruns = P.load_experiments(out_dir=str(OUT))
print("rows:", len(allruns), "| run_ids:", allruns["run_id"].unique().tolist())
assert "run_id" in allruns.columns and "filename" not in allruns.columns
print("\n*** STEP 3 CHECKS PASSED ***")
