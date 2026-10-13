# Pipeline restructure, feature fixes, and distance-based station selection

What changed between `main` and `pipeline-restructure`, why, and how each claim
was checked. Every number here is reproducible from files in this repository —
`scripts/` holds the verification record, `experiments/` holds the runs.

---

## Summary

Three independent things, bundled in one branch:

1. **Two feature bugs fixed.** `precip_max` was a byte-identical duplicate of
   `precip_sum`; `groin_onshore` conflated "no onshore wind" with "insufficient
   data". These are the only changes that alter reported numbers.
2. **The pipeline was restructured** so that one code path produces every
   result. Verified to change nothing: 80 rows × 13 metrics identical to a
   baseline captured from the pre-restructure notebook.
3. **Distance-based station selection added** as a new capability, declared off
   by default. The hypothesis it was built to test failed, and the test is
   confounded; both are recorded rather than buried.

Supporting changes: `station_locations.py`, the `experiments/` run log,
dependency pins, and nbstripout.

---

## 1. Two feature bugs

### `precip_max_in_window_per_station` — peak intensity did not exist

Before:

```sql
WITH summed AS (
    SELECT r.sample_ts, b.station,
           COALESCE(SUM(b.precip_mm), 0) AS precip_mm      -- SUM
    FROM readings r
    LEFT JOIN bucketed b
      ON b.ts >  r.sample_ts - INTERVAL {window_hours} HOUR
     AND b.ts <= r.sample_ts
    GROUP BY r.sample_ts, b.station
)
PIVOT summed ON station USING MAX(precip_mm) GROUP BY sample_ts
```

The CTE's `GROUP BY` already collapses each `(sample_ts, station)` to one row.
Taking `MAX` of a single value returns that value, so every `_max_` column came
out byte-identical to its `_sum_` twin — all 11 of them. The outer aggregate had
been changed from `SUM` to `MAX`; the inner one had not.

After: the CTE aggregate is `MAX` (`COALESCE(MAX(b.precip_mm), 0)`, CTE renamed
`peaked`). Nothing else changed.

Why it matters: 20 mm falling as a cloudburst and 20 mm as six hours of drizzle
produce very different runoff, and only peak intensity separates them. The model
had cumulative rainfall twice under two names, and no intensity variable at all.

Bucket width sets the unit: COC buckets are 60 min (peak hourly), USGS 5 min
(peak 5-minute). Comparable across samples within a source, not between sources.

### `groin_onshore_*` — one NULL encoding carrying two meanings

Before:

```sql
CASE WHEN COUNT(w.is_onshore) >= {min_obs}
    THEN {groin_length_m}
       * AVG(w.is_onshore)
       * AVG(CASE WHEN w.is_onshore = 1 THEN w.WSPD ELSE NULL END)
    END AS groin_onshore_{wh}h_tol{tol}
```

In a window where the wind was never onshore, the inner `CASE` is `NULL` on every
row, so `AVG` returns `NULL`. In SQL `0 * NULL` is `NULL`, not `0` — so the whole
product is `NULL` even though `AVG(w.is_onshore)` is legitimately `0`.

The outer `CASE` *also* yields `NULL`, for an unrelated reason: fewer than
`min_obs` observations in the window, meaning "not enough data to judge".

One encoding, two opposite facts. Downstream, the median imputer could not
distinguish them and filled both with the median — a *typical onshore exposure*
value. Genuinely calm windows were assigned a confident, wrong number.

After: the third factor is wrapped in `COALESCE(..., 0)`. `NULL` now means only
insufficient coverage; a real absence of onshore wind gives `0`.

Verified in `scripts/07_verify_feature_fixes.py`:

- 0 of 11 `_max_` columns still identical to their `_sum_` twin
- `groin_onshore` non-null coverage at 6h: 167/503 → 475/503, of which 311 are
  correctly `0`
- the remaining NaN pattern now matches `chii2_wspd_mean_6h` exactly, which is
  what it should be if `NULL` means coverage and nothing else

### How the two differ

| | `precip_max` | `groin_onshore` |
|---|---|---|
| Failure | silent duplication | semantic collision |
| Produced | a plausible wrong value | a missing value, then an imputed one |
| Nature | one aggregate in the wrong place | a flaw in how absence was encoded |
| Caught by | comparing `_max_` to `_sum_` columns | checking what `NULL` could mean |

The first destroyed information; the second invented it. That asymmetry explains
the measured effect below: duplicated features are merely redundant and the tree
models tolerate them, whereas the groin fix changed what the feature *meant*.

### Measured effect

Mean PR-AUC across all paired `(beach, model)` rows, excluding the dummy
baseline (`scripts/08_run_and_compare_fixes.py`):

| recipe | pre-fix | post-fix | Δ |
|---|---|---|---|
| `wind_only` | 0.1912 | 0.1895 | −0.002 |
| `wind_geo` | 0.1922 | 0.1772 | −0.015 |
| `all_stations` | 0.1922 | 0.1772 | −0.015 |
| `radius_d10` | 0.1719 | 0.1639 | −0.008 |
| `radius_d15` | 0.1745 | 0.1741 | −0.000 |

`wind_only` has no groin features, so it isolates the `precip_max` fix alone —
roughly zero effect. The geo recipes add the groin fix, at −0.015. Both sit
inside the noise at 64 test positives.

**The substantive consequence:** post-fix, `wind_only` (0.2538) scores *above*
`wind_geo` (0.2440), reversing the earlier reading. The previous comparison was
contaminated by `groin_onshore` being mostly NaN. There is currently no evidence
the Mattheus groin interaction carries signal.

> **Two aggregations appear in this repository.** The table above averages every
> `(beach, model)` row (`all_stations` post-fix = 0.1772). The README's recipe
> table averages only the best model per beach (same run = 0.2440). Both are
> correct; they are different aggregations, not a contradiction.

---

## 2. Why the pipeline was restructured

### What the pre-restructure notebook did

The pre-restructure notebook is preserved at
`scripts/reference/Final_feature_building.PRE_RESTRUCTURE.ipynb`, and the
committed version is on `main` (40 cells, 34 code, 319 stored outputs). Three
problems, all visible in that file:

**Dead code layered over live code.** Cell 14 is 139 lines, 84 of them commented
out — an older `build_wind_features` directly above the live one. Cell 15
(`assemble_features`) is 51% commented. Cells 5, 35 and 38 are entirely
commented out. With two versions of a function in one cell, which one ran is a
question about scrollback.

**Execution interleaved with definitions.** Cell 16 runs
`features = assemble_features(readings_df, RECIPE, dataframes)` at the top
level — and cells 20–26 then go on to *define* `add_target`, `time_split`,
`evaluate`, `run_all_models` and `threshold_sweep`. The notebook only works in
one specific execution order.

**Three separate code paths to the same numbers.** `RECIPE["target_beach"]` is
`montrose`, so cell 16 builds montrose features; cell 30 builds them again as a
spot check; cell 31 calls `run_pipeline_for_many`, which builds them a third
time. Beyond the waste, a number read off cell 16 or 30 did not have to agree
with what cell 31 reported, and nothing would have noticed. Results lived in
cell output — lost on kernel restart, overwritten on re-run, not comparable
between runs.

### What replaced it

Nine named sections: Setup, Reference data, Pipeline, Recipes, Runs, Compare,
Figures. The load-bearing rules:

- **The Pipeline section produces no output.** It contains definitions only —
  verifiable by parsing cells 9–18, which hold no expression statements.
- **One entry point.** Every run goes through `run_experiment(...)`, including
  one-beach spot checks. There is no second interactive path that can drift from
  the one producing reported numbers.
- **Recipes are plain JSON-serialisable dicts** that reference dataframes by
  string name, so a recipe can be stored next to the results it produced.
  Variants are built with `copy.deepcopy` and a single edited key, so two
  compared runs differ by exactly the thing under test.
- **Every run persists to `experiments/<run_id>/`** — `recipe.json`,
  `results.csv`, `sweeps.csv`, `meta.json`. Runs append; nothing is overwritten.
- **The Compare section reads from disk**, not from kernel state.

`meta.json` records the recipe hash, the notebook hash, the git commit and dirty
flag, which stations each beach actually resolved to, which beaches the
positive-count guard dropped, timings, Python version and package versions.

Side effects: file size 325,909 → 49,946 bytes, and a full run went from 4,047 s
to 1,110 s, because the duplicate montrose builds are gone.

### How "it changed nothing" was verified

| step | what it did | result |
|---|---|---|
| `00` | ran the **pre-restructure** notebook cold, saved results as a baseline | 80 result rows, 448 sweep rows, 4,047 s |
| `01` | rewritten pipeline, 3 beaches × 2 recipes, compared to baseline | exact match, 30 rows × 13 metrics |
| `02` | checked the run log writes and reads back correctly | passed |
| `03` | built the new notebook by parsing `reference/pipeline_draft.py` section banners, lifting verbatim the code that passed `01` | 33 cells (25 code, 8 markdown) |
| `04` | ran the restructured notebook end to end, compared to the `00` baseline | **80 rows × 13 metrics, every value identical**, 1,110 s |

Step 00 also established something that was genuinely uncertain: that the
original notebook *could* run top to bottom in a cold kernel.

Note the ordering — both feature bugs were found in steps 06–08, *after* the
restructure. `precip_max` surfaced by comparing `_max_` against `_sum_` across a
single clean feature set; `groin_onshore` by checking coverage against a known
NaN pattern. Neither check is practical when the same features are built three
times by three code paths. The restructure did not fix the bugs; it made them
findable.

---

## 3. Distance-based station selection

### What it adds

`resolve_stations(beach, source, selection, beach_coords, declared)` decides
which stations of a source feed a beach, in three modes:

| mode | behaviour |
|---|---|
| `"all"` | the recipe's declared station list, unchanged; names validated so a typo fails loudly rather than quietly shrinking the feature set |
| `"radius"` | every station within `max_km`, nearest first, with a `min_stations` floor |
| `"knn"` | the `k` nearest stations |

The declared list gives every beach all 11 gauges regardless of distance —
Rainbow Beach is weighted on Winnetka, 42 km north, the same as a gauge 600 m
away. Against roughly 400 training rows that is mostly dilution. Radius mode
tests whether removing the distant gauges helps.

`min_stations` is a floor applied **per source**: if the radius excludes
everything for a source, the nearest station is kept anyway, since a tight
radius would otherwise produce a source with no columns at all. So for a beach
whose nearest gauge of some source lies beyond `max_km`, the effective radius is
larger than the label suggests. What each beach actually resolved to is recorded
per run in `meta.json` under `resolved_stations`.

This capability is **off by default**: the base recipe uses
`station_selection: {"mode": "all"}` with `distance: {"enabled": False}`, and the
radius recipes are `copy.deepcopy` variants differing only in
`station_selection`. Step 05 confirmed the control still equalled the baseline,
which is what proves the resolver is transparent when it is not filtering.

Recipes defined: `radius_d5` (`max_km: 5`), `radius_d10`, `radius_d15`. The d5
recipe is defined and diagnosable but has not been run; `experiments/` contains
no `radius_d5` directory.

### What it found

Feature counts fall as intended, confirmed from the committed runs:

| recipe | features per beach |
|---|---|
| `all_stations` | 70 |
| `radius_d15` | 35–55 |
| `radius_d10` | 30–35 |

Performance does not improve. Mean of the best model per beach, from the latest
committed run of each recipe:

| recipe | mean PR-AUC |
|---|---|
| `wind_only` | 0.254 |
| `wind_geo` / `all_stations` | 0.244 |
| `radius_d15` | 0.223 |
| `radius_d10` | 0.213 |

Pre-fix, step 05 measured `all_stations` 0.271 against `radius_d10` 0.236 and
`radius_d15` 0.219, with paired win rates of 13/32 and 15/32 — coin flips.

**The test is confounded and should not be read as a clean result on distance.**
The `63rd_street` City of Chicago gauge effectively stopped reporting after 2022
— 1 row in 2023, 11 in 2024, 6 in 2025, against roughly 8,400 per year for
`foster` and `oak_street`. Dry hours *are* recorded as zeros, so this is sensor
loss rather than an encoding artifact. It is also the nearest COC gauge to the
southern beaches, and the train/test split is chronological, so the outage lands
squarely in the test window. `radius_d10` therefore hands those beaches a dead
station while discarding the two working ones. The drop is at least partly data
availability, not distance.

---

## 4. Supporting changes

**`station_locations.py`** (new, 239 lines) — coordinates for the precipitation
stations, with distance helpers and name validation. The values are baked into
the module rather than read at runtime because both source CSVs are excluded by
`.gitignore`, so any code reading them breaks on a fresh clone. Beach
coordinates are deliberately *not* duplicated here: they come from
`dna_readings.parquet`, which is tracked, via `get_beach_coords(readings_df)`,
so there is one source of truth for where a beach is.

**`experiments/`** — the run log described above, with its own README.

**Dependency pins** — `requirements.txt` now pins exact versions. Two
requirements were previously undeclared: `pyarrow`, although every data load
goes through `pd.read_parquet`, and `ipywidgets`, without which the Setup cell's
`duckdb.execute("SET enable_progress_bar = false")` raises
`InvalidInputException`. A fresh `pip install -r requirements.txt` would
previously have failed on the first cell.

**nbstripout** — strips notebook outputs on commit so diffs show code changes.
Enable once per clone: `nbstripout --install --attributes .gitattributes`.

**`scripts/`** — the verification record is now tracked, so the evidence behind
this branch ships with it.

---

## 5. How to verify this yourself

Fastest check, requiring nothing but the repository (about a minute): the
committed Oct 7 six-recipe sweep reproduces the README's recipe table to three
decimal places on all five recipes. Different day, different working tree, same
numbers.

```python
import pandas as pd, glob, os
for label in ["wind_only","wind_geo","all_stations","radius_d10","radius_d15"]:
    d = sorted(glob.glob(f"experiments/*_{label}"))[-1]
    r = pd.read_csv(os.path.join(d, "results.csv"))
    nd = r[r["model"] != "dummy"]
    print(label, round(nd.groupby("beach")["pr_auc"].max().mean(), 3))
```

Other checks that stand on committed files alone:

- `scripts/reference/baseline_master_results.csv` is the step 00 baseline, 80
  rows, captured from the pre-restructure notebook. Any future claim that a
  refactor is a no-op should be checked against it.
- Every `experiments/*/meta.json` carries a `recipe_sha256`; it is the SHA-256
  of `json.dumps(recipe, indent=2, sort_keys=True, default=str)`. All 22
  committed runs verify.
- The Pipeline section containing definitions only can be confirmed by parsing
  the notebook: cells 9–18 contain no expression statements.

---

## 6. Known limitations

- **Three scripts will not re-run as written.** They reference a scratch
  directory that no longer exists. `07` asserts `sum >= max` with `.all()` over
  columns containing NaN, and `NaN >= NaN` is False — the invariant does hold,
  verified separately on non-null rows with 0 violations across all 11 station
  pairs. `04` errored *after* its comparison completed. See `scripts/README.md`.
- **The radius experiment is confounded** by the `63rd_street` outage described
  above, and is not a clean test of distance.
- **Five of the thirteen Mattheus beaches are dropped** by the positive-count
  guard (`min_train_pos=10`, `min_test_pos=3`): `osterman`, `foster`,
  `north_avenue`, `oak_street`, `oakwood`.
- **Results remain weak.** Best PR-AUC per beach ranges from 0.44 (montrose) to
  0.04 (57th_street) against test base rates of 0.03–0.17.
- **`cicero` USGS rows are duplicated exactly 2× in 2024–2025** — identical
  values, same `usgs_id`, every timestamp. No feature impact, because
  `bucket_precip_per_station` collapses them with
  `MAX(...) GROUP BY (bucket, station)`. `north_austin` also lost roughly 30% of
  2023.
- **`features.sewer`, `features.temp` and `features.distance` are declared but
  not implemented.** Combined sewer overflow is not being pursued: only one CSO
  event (2023) was found to have reached Lake Michigan in this period.
