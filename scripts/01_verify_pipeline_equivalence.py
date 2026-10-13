"""Fast equivalence check: does pipeline_draft reproduce the baseline exactly?

Runs the draft pipeline for a subset of beaches and compares the resulting rows
against the Step 0 baseline CSV. A full notebook run takes ~67 min; this checks
the logic in a few minutes so Step 4's full run is confirmation, not debugging.
"""
import copy
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

# ── data ──────────────────────────────────────────────────────────────
dp = f"{REPO}/data/prepared_data/"
beach_coc_precip_df = pd.read_parquet(dp + "COC_precip.parquet")
usgs_precip_df = pd.read_parquet(dp + "usgs_precip.parquet")
readings_df = pd.read_parquet(dp + "dna_readings.parquet")
chii2_wind_df = pd.read_parquet(dp + "chii2_noaa.parquet")

dataframes = {
    "beach_coc_precip_df": beach_coc_precip_df,
    "usgs_precip_df": usgs_precip_df,
    "chii2_wind_df": chii2_wind_df,
}
beach_geo_df = get_geomorphology_df()
beach_coords = get_beach_coords(readings_df)

# ── the recipe exactly as the notebook defines it (cell 9) ────────────
RECIPE = {
    "target_beach": "montrose",
    "min_year": 2021,
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
        "wind": {
            "enabled": True, "use_geo_features": True, "df_name": "chii2_wind_df",
            "windows_hours": [6, 12, 24], "tolerance_deg": 20, "min_coverage": 0.5,
        },
        "distance": {"enabled": False},
        "temp": {"enabled": False},
        "sewer": {"enabled": False},
    },
}

RECIPE_WIND_ONLY = copy.deepcopy(RECIPE)
RECIPE_WIND_ONLY["label"] = "wind_only"
RECIPE_WIND_ONLY["features"]["wind"]["use_geo_features"] = False

RECIPE_WIND_GEO = copy.deepcopy(RECIPE)
RECIPE_WIND_GEO["label"] = "wind_geo"
RECIPE_WIND_GEO["features"]["wind"]["use_geo_features"] = True

# ── subset: a mix of beaches that survived the guard ──────────────────
SUBSET = ["montrose", "leone", "rainbow"]

beach_recipes = (
    [(b, copy.deepcopy(RECIPE_WIND_ONLY)) for b in SUBSET] +
    [(b, copy.deepcopy(RECIPE_WIND_GEO)) for b in SUBSET]
)

results, sweeps, resolved = P.run_pipeline_for_many(
    beach_recipes, dataframes, readings_df, beach_geo_df,
    beach_coords=beach_coords,
)

print("\nresolved stations (should be the full declared lists in 'all' mode):")
for k, v in list(resolved.items())[:2]:
    print(" ", k, {src: len(names) for src, names in v.items()})

# ── compare against baseline ──────────────────────────────────────────
base = pd.read_csv(SCRATCH / "baseline" / "master_results.csv")

KEY = ["beach", "recipe_label", "model"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "true_neg", "false_pos", "false_neg", "true_pos",
           "n_features", "n_train_pos", "n_test_pos"]

new = results[KEY + METRICS].sort_values(KEY).reset_index(drop=True)
old = base[base["beach"].isin(SUBSET)][KEY + METRICS].sort_values(KEY).reset_index(drop=True)

print(f"\ncomparing {len(new)} new rows against {len(old)} baseline rows")

if len(new) != len(old):
    print("!! ROW COUNT MISMATCH")
    print("new keys:", new[KEY].apply(tuple, axis=1).tolist())
    print("old keys:", old[KEY].apply(tuple, axis=1).tolist())
    sys.exit(1)

mismatches = []
for col in METRICS:
    a, b = new[col], old[col]
    if pd.api.types.is_numeric_dtype(a):
        same = ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
    else:
        same = a == b
    if not same.all():
        for i in (~same).to_numpy().nonzero()[0]:
            mismatches.append((tuple(new.loc[i, KEY]), col, b[i], a[i]))

if mismatches:
    print(f"\n!! {len(mismatches)} MISMATCHES (key, column, baseline, draft):")
    for m in mismatches[:25]:
        print("   ", m)
    sys.exit(1)

print("\n*** EXACT MATCH on all metrics for all subset rows ***")
print(new.to_string(index=False))
