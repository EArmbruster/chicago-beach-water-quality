# Experiment log

One directory per run, written by `run_experiment(...)` in
`Final_feature_building.ipynb`. Runs are never overwritten — re-running a recipe
appends a new directory, so results survive re-execution instead of living in
notebook cell output.

```
experiments/<YYYYMMDD_HHMMSS>_<recipe label>/
    recipe.json    the exact config that produced this run (canonical JSON)
    results.csv    one row per beach x model, at the default 0.5 threshold
    sweeps.csv     one row per beach x model x decision threshold
    meta.json      provenance (see below)
```

## meta.json

| field | why it's there |
|---|---|
| `recipe_sha256` | hash of the canonical recipe — identical configs hash identically |
| `notebook_sha256` | hash of the notebook that ran it; catches "a builder was edited after this run" |
| `git_commit`, `git_dirty` | which commit produced it, and whether the tree was clean |
| `resolved_stations` | which precip stations each beach actually used (recipe + beach, not recipe alone) |
| `beaches_skipped` | dropped by the positive-count guard (`min_train_pos=10`, `min_test_pos=3`) |
| `elapsed_sec`, `packages`, `python` | runtime and environment |

## Reading it back

```python
all_results = load_experiments()            # every run ever recorded
load_experiments(what="sweeps")             # threshold sweeps instead
```

Re-running a recipe appends rather than replaces, so reduce to the most recent
run per label before comparing — the Compare section of the notebook does this:

```python
latest = all_results.sort_values("run_id").groupby("recipe_label")["run_id"].last()
current = all_results[all_results["run_id"].isin(latest)]
```

## Runs from 2026-08-14 predate two feature bug fixes

Runs dated `20260814_*` were produced **before** these were fixed, and their
feature values differ from later runs of the same recipe:

- `precip_*_max_*h` summed inside its CTE, so the outer `PIVOT ... USING MAX`
  took the max of a single value. Those columns were identical to their `_sum_`
  counterparts — peak intensity was absent from the feature set entirely.
- `groin_onshore_*` went NaN whenever no observation in the window was onshore,
  colliding with the `min_coverage` NaN meaning "not enough data". For the
  logistic pipeline the median imputer then filled genuine calm periods with a
  *typical onshore* value. Non-null coverage at 6h was 167/503; after the fix,
  475/503.

They are kept as the historical record. Distinguish them by `notebook_sha256`
in `meta.json`, or by the date prefix.

## Known data caveat

The `63rd_street` City of Chicago weather station effectively stopped reporting
after 2022 (1 row in 2023, 11 in 2024, 6 in 2025, against ~8,400/yr for
`foster` and `oak_street`). Because the train/test split is chronological, this
falls in the test window. It also affects distance-based station selection: for
the southern beaches it is the nearest COC gauge, so `radius_d10` selects a
station with almost no data and discards the two working ones. Treat the
`radius_*` runs as confounded by availability, not as a clean test of distance.
