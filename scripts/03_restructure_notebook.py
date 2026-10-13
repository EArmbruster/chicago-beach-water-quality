"""Step 2: rebuild Final_feature_building.ipynb into the agreed sections.

Pipeline code is lifted verbatim from pipeline_draft.py (which passed the
equivalence check against the Step 0 baseline) by splitting on its section
banners, so no code is retyped and transcription errors are impossible.

Writes the notebook with NO outputs -- it is meant to be run fresh
(Restart & Run All), which is also what Step 4 requires.
"""
import json
import re
import sys
from pathlib import Path

import nbformat

REPO = Path("/Users/emily/Documents/chicago-beach-water-quality")
SCRATCH = Path(__file__).parent
DRAFT = SCRATCH / "pipeline_draft.py"
TARGET = REPO / "Final_feature_building.ipynb"

# ── split the validated draft on its banners ──────────────────────────
text = DRAFT.read_text()
BANNER = re.compile(
    r"^# ═+\n# (?P<name>.+?)\n# ═+\n", re.M
)
parts, names = [], []
matches = list(BANNER.finditer(text))
for i, m in enumerate(matches):
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    names.append(m.group("name").strip())
    parts.append(text[start:end].strip("\n"))

SEC = dict(zip(names, parts))
expected = ["FeatureSet", "Target readings", "Station resolution",
            "Precip features   (logic unchanged from the original notebook)",
            "Wind features   (logic unchanged from the original notebook)",
            "Assemble", "Target, split, metrics",
            "Models  (factory dict: every call yields a fresh, unfitted estimator)",
            "Per-beach pipeline", "Experiment log"]
missing = [e for e in expected if e not in SEC]
if missing:
    print("MISSING SECTIONS:", missing)
    print("FOUND:", names)
    sys.exit(1)
print(f"parsed {len(SEC)} sections from pipeline_draft.py")

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = []

# ══════════════════════════════════════════════════════════════════════
cells.append(md("""# Chicago Beach Water Quality — Feature Building & Modeling

Predicting E. coli exceedances at Chicago beaches from precipitation, wind, and
beach geomorphology.

## How this notebook is organised

| Section | Contains |
|---|---|
| **Setup** | Imports, data loading, timestamp health checks |
| **Reference data** | Beach geomorphology, beach + station coordinates |
| **Pipeline** | *Definitions only.* No execution, no output. |
| **Recipes** | Experiment configs as plain dicts |
| **Runs** | `run_experiment(...)` — one call per recipe |
| **Compare** | Reads `experiments/` from disk, not kernel state |
| **Figures** | Report plots |

**The rule: nothing in the Pipeline section produces output.** No prints, no
module-level assignments. If a cell there has an `Out[]`, it belongs in Runs.

This exists because the pipeline previously had two code paths — an interactive
one built on montrose globals, and the per-beach functions — which silently
disagreed once a recipe changed. Every run now goes through `run_experiment`,
and each run is written to `experiments/<run_id>/` with the exact recipe that
produced it, so results survive re-execution instead of living in cell output.

**Before any run you intend to report: Restart & Run All.**"""))

# ── Setup ─────────────────────────────────────────────────────────────
cells.append(md("## Setup"))

cells.append(code("%reload_ext autoreload\n%autoreload 2"))

cells.append(code('''import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

duckdb.execute("SET enable_progress_bar = false")  # quiet the run bars

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Project modules
from beach_geomorphology import get_geomorphology_df
from station_locations import (
    get_beach_coords, get_station_df, station_distances_km, validate_station_names,
)
from wind_diagnostics import plot_wind_diagnostics, plot_uv_agreement'''))

cells.append(code('''# ─── Load prepared data ─────────────────────────────────────────────────
load_prepared_datapath = "data/prepared_data/"

# precipitation
beach_coc_precip_df = pd.read_parquet(load_prepared_datapath + "COC_precip.parquet")
usgs_precip_df      = pd.read_parquet(load_prepared_datapath + "usgs_precip.parquet")
# bacteria readings
readings_df         = pd.read_parquet(load_prepared_datapath + "dna_readings.parquet")
# wind
beach_coc_wind_df   = pd.read_parquet(load_prepared_datapath + "COC_wind.parquet")
chii2_wind_df       = pd.read_parquet(load_prepared_datapath + "chii2_noaa.parquet")

# The recipe refers to dataframes by STRING NAME; this dict supplies the objects.
# That indirection is what keeps a recipe JSON-serialisable and portable.
dataframes = {
    "beach_coc_precip_df": beach_coc_precip_df,
    "usgs_precip_df":      usgs_precip_df,
    "chii2_wind_df":       chii2_wind_df,
}'''))

cells.append(code('''def check_timestamps(df, ts_col, name):
    """Quick health check on a timestamp column before using it."""
    n_total = len(df)
    n_null  = df[ts_col].isna().sum()
    print(f"\\n{name}:")
    print(f"  Total rows:       {n_total:,}")
    print(f"  Null/NaT rows:    {n_null:,} ({n_null/n_total:.1%})")
    print(f"  Min timestamp:    {df[ts_col].min()}")
    print(f"  Max timestamp:    {df[ts_col].max()}")
    print(f"  Dtype:            {df[ts_col].dtype}")
    print(f"  Timezone:         {getattr(df[ts_col].dt, 'tz', None)}")


check_timestamps(beach_coc_precip_df, "timestamp_central", "COC precip")
check_timestamps(usgs_precip_df,      "timestamp_central", "USGS precip")
check_timestamps(readings_df,         "timestamp_central", "DNA readings")
check_timestamps(beach_coc_wind_df,   "timestamp_central", "COC wind")
check_timestamps(chii2_wind_df,       "timestamp_central", "CHII2 wind")'''))

# ── Reference data ────────────────────────────────────────────────────
cells.append(md("""## Reference data

Beach geomorphology from Mattheus et al. (2022); beach coordinates from the
readings parquet; station coordinates from `station_locations.py`.

Beach coordinates deliberately come from `dna_readings.parquet` (tracked in git)
rather than being duplicated in a module, so there is one source of truth. The
station CSVs *are* gitignored, which is why those coordinates live in
`station_locations.py` instead of being read at runtime."""))

cells.append(code('''beach_geo_df = get_geomorphology_df()

# Only model beaches that have Mattheus data
MATTHEUS_BEACHES = beach_geo_df[beach_geo_df["mattheus_id"].notna()].index.tolist()

# beach -> (lat, lon), used by distance-based station selection
beach_coords = get_beach_coords(readings_df)

print(f"Beaches with Mattheus geomorphology ({len(MATTHEUS_BEACHES)}):")
print(MATTHEUS_BEACHES)
print(f"\\nBeaches with coordinates: {len(beach_coords)}")
print(f"Precip stations registered: {len(get_station_df())}")

beach_geo_df.loc[MATTHEUS_BEACHES,
    ["lake_interface_az_deg", "facing_az_deg",
     "downstream_groin_length_m", "lake_interface_length_m", "shape_index"]]'''))

# ── Pipeline ──────────────────────────────────────────────────────────
cells.append(md("""## Pipeline — definitions only

**Nothing in this section produces output.** These cells define the feature
builders, the modelling harness, and the experiment logger. Running them should
print nothing and assign nothing at module level.

Feature naming convention: `{source}_{variable}_{operation}_{window}` —
e.g. `coc_foster_precip_sum_24h`. Changing one thing about a feature should
change exactly one slot in its name."""))

cells.append(code(SEC["FeatureSet"]))
cells.append(code(SEC["Target readings"]))
cells.append(code(SEC["Station resolution"]))
cells.append(code(SEC["Precip features   (logic unchanged from the original notebook)"]))
cells.append(code(SEC["Wind features   (logic unchanged from the original notebook)"]))
cells.append(code(SEC["Assemble"]))
cells.append(code(SEC["Target, split, metrics"]))
cells.append(code(SEC["Models  (factory dict: every call yields a fresh, unfitted estimator)"]))
cells.append(code(SEC["Per-beach pipeline"]))
cells.append(code(SEC["Experiment log"]))

# ── Recipes ───────────────────────────────────────────────────────────
cells.append(md("""## Recipes

A recipe is a plain dict: fully JSON-serialisable, so it can be stored next to
the results it produced.

`station_selection.mode`:
- `"all"` — use the declared `stations` list (original behaviour)
- `"radius"` — every station within `max_km` of the beach, `min_stations` floor
- `"knn"` — the `k` nearest stations"""))

cells.append(code('''RECIPE = {
    "label": "all_stations",
    "target_beach": "montrose",     # overridden per beach by run_experiment
    "min_year": 2021,

    "target": {"threshold": 1000},  # E. coli exceedance cutoff

    "features": {
        "precip": {
            "enabled": True,
            "station_selection": {"mode": "all"},
            "sources": [
                {"df_name": "beach_coc_precip_df",
                 "ts_col": "timestamp_central",
                 "value_col": "coc_mmprecip_sum1h",
                 "station_col": "station_name",
                 "label": "coc",
                 "bucket_minutes": 60,
                 "stations": ['foster', 'oak_street', '63rd_street']},
                {"df_name": "usgs_precip_df",
                 "ts_col": "timestamp_central",
                 "value_col": "usgs_mmprecip_sum5m",
                 "station_col": "station_name",
                 "label": "usgs",
                 "bucket_minutes": 5,
                 "stations": ['south_shore', 'hamilton_park_at_chicago',
                              'ping_tom_park_at_chicago', 'oakton_college_at_skokie',
                              'winnetka_golf_club_at_winnetka', 'cicero',
                              'north_austin', 'lake_view']},
            ],
            "sum_windows_hours": [6, 24, 48, 72],
            "max_windows_hours": [6],
        },

        "wind": {
            "enabled": True,
            "use_geo_features": True,
            "df_name": "chii2_wind_df",
            "windows_hours": [6, 12, 24],
            "tolerance_deg": 20,
            "min_coverage": 0.5,
        },

        # Declared but not yet implemented by assemble_features.
        "distance": {"enabled": False},
        "temp":     {"enabled": False},
        "sewer":    {"enabled": False},
    },
}'''))

cells.append(code('''# ─── Does beach geometry help? wind alone vs wind + groin interaction ───
# These differ by exactly the three groin_onshore_* columns (70 vs 67 features).
RECIPE_WIND_ONLY = copy.deepcopy(RECIPE)
RECIPE_WIND_ONLY["label"] = "wind_only"
RECIPE_WIND_ONLY["features"]["wind"]["use_geo_features"] = False

RECIPE_WIND_GEO = copy.deepcopy(RECIPE)
RECIPE_WIND_GEO["label"] = "wind_geo"
RECIPE_WIND_GEO["features"]["wind"]["use_geo_features"] = True'''))

# ── Runs ──────────────────────────────────────────────────────────────
cells.append(md("""## Runs

Every run — including a single-beach spot check — goes through `run_experiment`,
so there is no second code path that can drift. Each call writes
`experiments/<timestamp>_<label>/` containing `recipe.json`, `results.csv`,
`sweeps.csv`, and `meta.json`."""))

cells.append(code('''# Smoke test: one beach, same entry point as a full run.
# Relabelled so it lands in its own run directory and cannot be mistaken for
# a real wind_geo result in the comparison below.
SMOKE_RECIPE = copy.deepcopy(RECIPE_WIND_GEO)
SMOKE_RECIPE["label"] = "smoke_test"

smoke = run_experiment(
    SMOKE_RECIPE, beaches=["montrose"],
    dataframes=dataframes, readings_df=readings_df,
    beach_geo_df=beach_geo_df, beach_coords=beach_coords,
)
smoke.summary()'''))

cells.append(code('''run_wind_only = run_experiment(
    RECIPE_WIND_ONLY, beaches=MATTHEUS_BEACHES,
    dataframes=dataframes, readings_df=readings_df,
    beach_geo_df=beach_geo_df, beach_coords=beach_coords,
)
run_wind_only.summary()'''))

cells.append(code('''run_wind_geo = run_experiment(
    RECIPE_WIND_GEO, beaches=MATTHEUS_BEACHES,
    dataframes=dataframes, readings_df=readings_df,
    beach_geo_df=beach_geo_df, beach_coords=beach_coords,
)
run_wind_geo.summary()'''))

# ── Compare ───────────────────────────────────────────────────────────
cells.append(md("""## Compare

Reads from `experiments/` on disk, so this shows every run ever recorded —
including from previous sessions — rather than only what is in the kernel."""))

cells.append(code('''all_results = load_experiments()
print(f"{len(all_results)} result rows across {all_results['run_id'].nunique()} runs\\n")
print(all_results.groupby(['run_id', 'recipe_label']).size().to_string())'''))

cells.append(code('''# ─── Reduce to the LATEST run of each recipe ────────────────────────────
# Re-running the notebook appends new run directories rather than overwriting,
# so without this the comparison would silently mix runs of the same recipe.
# run_id is timestamp-prefixed, so lexical sort == chronological.
latest_run_ids = (
    all_results.sort_values("run_id").groupby("recipe_label")["run_id"].last()
)
current = all_results[all_results["run_id"].isin(latest_run_ids)]
current = current[~current["recipe_label"].str.startswith("smoke")]

print("Comparing:")
print(latest_run_ids.to_string())'''))

cells.append(code('''# ─── Best model per beach x recipe, by PR-AUC ───────────────────────────
nd = current[current["model"] != "dummy"]
best = (
    nd.loc[nd.groupby(["beach", "recipe_label"])["pr_auc"].idxmax()]
      [["beach", "recipe_label", "model", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "n_features", "n_test_pos"]]
      .sort_values(["beach", "recipe_label"])
)
print(best.to_string(index=False))'''))

cells.append(code('''# ─── PR-AUC: wind only vs wind + geometry ───────────────────────────────
pivot = (
    nd.groupby(["beach", "recipe_label"])["pr_auc"].max()
      .unstack("recipe_label")
      .dropna(how="all")
)

if {"wind_only", "wind_geo"} <= set(pivot.columns):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(pivot))
    width = 0.35
    ax.bar([i - width/2 for i in x], pivot["wind_only"], width,
           label="wind only", alpha=0.8, color="steelblue")
    ax.bar([i + width/2 for i in x], pivot["wind_geo"], width,
           label="wind + geo", alpha=0.8, color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Best PR-AUC (across all models)")
    ax.set_title("PR-AUC: wind only vs wind + beach geometry")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
else:
    print("Need both wind_only and wind_geo runs recorded to compare.")
    print("have:", sorted(pivot.columns))'''))

# ── Figures ───────────────────────────────────────────────────────────
cells.append(md("## Figures for report"))

cells.append(code('''# from readings_plots import plot_readings_timeseries, plot_exceedance_rates
#
# fig5 = plot_readings_timeseries(readings_df, MATTHEUS_BEACHES)
# fig6 = plot_exceedance_rates(readings_df, MATTHEUS_BEACHES)

# Wind diagnostics (CHII2 vs City of Chicago stations)
# fig1 = plot_wind_diagnostics(beach_coc_wind_df, chii2_wind_df)
# fig2 = plot_uv_agreement(beach_coc_wind_df, chii2_wind_df)'''))

# ══════════════════════════════════════════════════════════════════════
nb = nbformat.v4.new_notebook(cells=cells)

# Preserve kernel/language metadata from the existing notebook
old = nbformat.read(TARGET, as_version=4)
nb.metadata = old.metadata

nbformat.write(nb, TARGET)

n_code = sum(1 for c in cells if c.cell_type == "code")
n_md = sum(1 for c in cells if c.cell_type == "markdown")
print(f"wrote {TARGET}")
print(f"  {len(cells)} cells ({n_code} code, {n_md} markdown), no outputs")
print(f"  size: {TARGET.stat().st_size:,} bytes (was 325,909)")
