# Chicago Beach Water Quality Prediction

Classification to predict bacterial contamination (E. coli) at Chicago's public beaches using precipitation, wind, and beach geomorphology features.

## Motivation

Predicting E. coli levels that allow for planning ahead for safe swimming.

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| [City of Chicago Beach Lab Data](https://data.cityofchicago.org/) | DNA-based bacteria readings (qPCR) | 2015-2025 |
| [City of Chicago Weather Stations](https://data.cityofchicago.org/) | Hourly precipitation and wind at beach stations | 2015-2026 |
| [USGS Water Resources](https://waterdata.usgs.gov/) | 5-minute precipitation from gauge network | 2021-2025 |
| [NOAA CHII2 Buoy](https://www.ndbc.noaa.gov/station_page.php?station=chii2) | 10-minute wind speed/direction from Chicago crib | 2021-2025 |
| [MWRD GeoHub](https://geohub.mwrd.org/pages/cso) | Combined Sewer Overflow events | 2021-2025 |
| Mattheus et al. (2022) | Beach geomorphology: facing azimuth, groin lengths, shape index | Static |

## Methodology

Everything downstream of the prepared parquet files lives in
`Final_feature_building.ipynb`, organised as:

| Section | Contains |
|---|---|
| **Setup** | Imports, data loading, timestamp health checks |
| **Reference data** | Beach geomorphology, beach + station coordinates |
| **Pipeline** | *Definitions only.* No execution, no output. |
| **Recipes** | Experiment configs as plain dicts |
| **Runs** | `run_experiment(...)` — one call per recipe |
| **Compare** | Reads `experiments/` from disk, not kernel state |
| **Figures** | Report plots |

The rule that keeps this honest: **nothing in the Pipeline section produces
output**, and every run — including a one-beach spot check — goes through
`run_experiment`. There is no second, interactive code path that can drift from
the one that produces reported numbers.

### Recipes

A recipe is a plain, JSON-serialisable dict specifying data sources, station
selection, time windows, feature types, and the exceedance threshold. It refers
to dataframes by *string name*; a `dataframes` dict supplies the objects. That
indirection is what lets a recipe be stored next to the results it produced.

```python
RECIPE = {
    "label": "all_stations",
    "min_year": 2021,
    "target": {"threshold": 1000},          # E. coli exceedance cutoff
    "features": {
        "precip": {
            "enabled": True,
            "station_selection": {"mode": "all"},
            "sources": [...],               # COC hourly, USGS 5-minute
            "sum_windows_hours": [6, 24, 48, 72],
            "max_windows_hours": [6],
        },
        "wind": {"enabled": True, "use_geo_features": True,
                 "df_name": "chii2_wind_df", "windows_hours": [6, 12, 24],
                 "tolerance_deg": 20, "min_coverage": 0.5},
        "distance": {"enabled": False},     # declared, not yet implemented
        "temp":     {"enabled": False},
        "sewer":    {"enabled": False},
    },
}
```

Recipe variants are built with `copy.deepcopy` and a single edited key, so two
compared runs differ by exactly the thing under test.

### Feature Engineering

Feature naming convention: `{source}_{variable}_{operation}_{window}` — e.g.
`coc_foster_precip_sum_24h`. Changing one thing about a feature changes exactly
one slot in its name.

- **Precipitation:** cumulative rainfall (`sum`) and peak intensity (`max`) per
  station over 6h/24h/48h/72h lookback windows. Window joins are strictly
  backward-looking, so no future information leaks into a sample.
- **Wind:** mean/max speed plus onshore and alongshore components decomposed
  against each beach's facing azimuth (Madani & Seth, 2020). Windows with less
  than `min_coverage` of expected observations are NaN rather than partial.
- **Geomorphology interaction:** groin-adjusted onshore wind exposure combining
  downstream groin length with onshore wind fraction and speed
  (Mattheus et al., 2022).

Two structural guarantees:

- `assemble_features` returns a **`FeatureSet`** carrying a *positive*
  `feature_cols` list decided at build time. Features are never derived
  subtractively ("everything except these columns"), which would silently
  promote any new column — an id, a year, a CSO event key — into the model.
- `FeatureSet.resolved` records which stations each source actually
  contributed. Under distance-based selection the feature set is a function of
  recipe *and* beach, not recipe alone, so it has to be recorded to be audited.

**Station selection** (`station_selection.mode`):

| mode | behaviour |
|---|---|
| `"all"` | use the recipe's declared `stations` list; names validated so a typo fails loudly |
| `"radius"` | every station within `max_km`, nearest first, with a `min_stations` floor |
| `"knn"` | the `k` nearest stations |

Upcoming: Combined Sewer Overflow features — event date, duration window,
outfall location, and reach of the overflow.

### Classification Models

- `dummy` (most-frequent baseline), logistic regression (median-impute →
  standardise → L2, class-weighted), HistGradientBoosting, XGBoost, LightGBM
- Chronological train/test split (last 20% held out) — no future leakage
- Beaches are **skipped** rather than reported when the split leaves too few
  positives (`min_train_pos=10`, `min_test_pos=3`)
- Threshold sweeps per model, recorded to `sweeps.csv`, for the
  precision-recall tradeoff on rare exceedance events
- Primary metric is **PR-AUC**, since the base rate is ~7% across modelled
  beaches (3-11% per beach) and accuracy is uninformative at that prevalence

### Experiment logging

Each `run_experiment(...)` call writes `experiments/<YYYYMMDD_HHMMSS>_<label>/`:

```
recipe.json    the exact config that produced this run (canonical JSON)
results.csv    one row per beach x model, at the default 0.5 threshold
sweeps.csv     one row per beach x model x decision threshold
meta.json      provenance: recipe + notebook sha256, git commit and dirty flag,
               resolved stations per beach, skipped beaches, timing, packages
```

Runs are never overwritten — re-running a recipe appends a new directory — so
results survive re-execution instead of living in cell output. The Compare
section reads these files back with `load_experiments()` and reduces to the
latest run per recipe label. See [experiments/README.md](experiments/README.md).

## Key Results

Comparable prior work on these beaches: https://github.com/Chicago/clear-water (2017).

Results remain weak. Best PR-AUC per beach (any model, `wind_geo` recipe,
2021+, threshold 1000, 70 features):

| Beach | Best model | PR-AUC | ROC-AUC | Test base rate |
|---|---|---|---|---|
| montrose | lgbm | 0.44 | 0.88 | 0.089 |
| ohio_street | lgbm | 0.38 | 0.70 | 0.168 |
| 63rd_street | logistic | 0.34 | 0.73 | 0.119 |
| rainbow | logistic | 0.29 | 0.53 | 0.058 |
| margaret_t_burroughs_31st | lgbm | 0.17 | 0.48 | 0.081 |
| leone | hgb | 0.16 | 0.70 | 0.039 |
| south_shore | logistic | 0.13 | 0.61 | 0.049 |
| 57th_street | xgb | 0.04 | 0.47 | 0.030 |

Five of the thirteen Mattheus beaches (osterman, foster, north_avenue,
oak_street, oakwood) were skipped for too few positives in the split.

Recipe comparison, mean best PR-AUC across beaches:

| Recipe | Mean PR-AUC |
|---|---|
| wind_only | 0.254 |
| wind_geo / all_stations | 0.244 |
| radius_d15 | 0.223 |
| radius_d10 | 0.213 |

Two readings of this:

- **Beach geometry does not currently help.** `wind_only` and `wind_geo` differ
  by exactly the three `groin_onshore_*` columns, and adding them does not
  improve mean PR-AUC.
- **The distance-limited recipes are confounded, not clean tests.** The
  `63rd_street` City of Chicago gauge effectively stopped reporting after 2022,
  and it is the nearest COC station for the southern beaches — so `radius_d10`
  selects a near-empty station and discards the two working ones. The drop is
  at least partly data availability, not distance.

## Continuing Work

- Sewer overflow features (`features.sewer`, declared but not yet implemented)
- Handling the `63rd_street` station outage before the distance-selection
  comparison can be trusted
- Pooled / multi-beach models, given how few positives each beach has on its own

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── CSO/                        # Combined sewer overflow event synopses
│   └── prepared_data/              # Processed parquet files ready for modeling
│       ├── COC_precip.parquet
│       ├── COC_wind.parquet
│       ├── chii2_noaa.parquet
│       ├── dna_readings.parquet
│       └── usgs_precip.parquet
│
├── experiments/                    # One directory per run, written by run_experiment
│   ├── README.md
│   └── <YYYYMMDD_HHMMSS>_<label>/  # recipe.json, results.csv, sweeps.csv, meta.json
│
├── Final_feature_building.ipynb    # Main analysis: pipeline, recipes, runs, comparison
├── Final_plots.ipynb               # Report figures
├── Precip_temp_wind_EDA.ipynb      # Exploratory data analysis
│
├── bouy_readings.ipynb             # NOAA buoy data processing
├── pathogen_readings.ipynb         # Bacteria reading processing
├── USGS_concat_cleaning.ipynb      # USGS precipitation processing
├── beach_weather_stations.ipynb    # City of Chicago weather processing
├── combined_sewer_overflow_pluslocks.ipynb  # CSO data processing
│
├── beach_geomorphology.py          # Beach geometry data (Mattheus et al.)
├── station_locations.py            # Station coordinates, distances, name validation
├── readings_plots.py               # Bacteria reading visualizations
├── wind_diagnostics.py             # Wind data diagnostic plots
└── wind_report_figs.py             # Publication-quality wind figures
```

Beach coordinates come from `dna_readings.parquet` (tracked in git) rather than
being duplicated in a module, so there is one source of truth. The station CSVs
*are* gitignored, which is why those coordinates live in `station_locations.py`.

Notebook output is stripped on commit via `nbstripout` (see `.gitattributes`),
so diffs show code rather than output blobs — another reason results are written
to `experiments/` instead of being read out of cell output.

## How to Reproduce

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data:** The `data/prepared_data/` directory contains pre-processed parquet files. To rebuild from raw sources, run the processing notebooks in order:
   - `pathogen_readings.ipynb` (bacteria readings)
   - `bouy_readings.ipynb` (NOAA wind data)
   - `USGS_concat_cleaning.ipynb` (USGS precipitation)
   - `beach_weather_stations.ipynb` (City of Chicago weather)

3. **Run the main analysis:** `Final_feature_building.ipynb`. Before any run you
   intend to report: **Restart & Run All.** Each run writes a new
   `experiments/<run_id>/` directory; nothing is overwritten.

4. **Compare runs** without re-running anything:
   ```python
   all_results = load_experiments()        # every run ever recorded
   load_experiments(what="sweeps")         # threshold sweeps instead
   ```

## References

- Mattheus, C.R. et al. (2022). Geomorphological classification of Chicago's lakefront beaches. *Journal of Great Lakes Research*.
- Madani, M. & Seth, R. (2020). Wind-driven transport modeling for beach water quality.
