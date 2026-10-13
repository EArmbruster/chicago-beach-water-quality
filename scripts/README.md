# scripts/ — local working records (not tracked in git)

This directory is listed in `.gitignore`. Nothing in here is committed or
published; it exists so the evidence behind the August 2026 pipeline
restructure survives past the session that produced it.

Everything here is a **record**, not part of the pipeline. Deleting the whole
directory breaks nothing — `Final_feature_building.ipynb`,
`station_locations.py`, and `experiments/` are self-sufficient.

---

## ⚠️ Read this before running anything

These scripts were written against a temporary scratch directory that **no
longer exists**. Near the top of each one you will find something like:

```python
SCRATCH = Path(__file__).parent
# or a literal /private/tmp/claude-501/.../scratchpad path
```

Any script referencing that literal path will fail until you repoint it at this
directory. They are preserved verbatim rather than rewritten, so what you read
is exactly what ran and produced the results described below — editing them to
"fix" the paths would mean the record no longer matches what was executed.

They also assume:
- the working directory is the repo root
- the `analytics` conda env (`/opt/miniconda3/envs/analytics/bin/python3`)
- `MPLBACKEND=Agg` for headless plotting

Several take **20–60 minutes** because they execute the full notebook.

---

## The scripts, in the order they were run

### `00_capture_baseline.py` — *the reference everything was checked against*

Executed the **original** notebook (pre-restructure) in a fresh kernel and saved
`master_results` / `master_sweeps` to CSV. Reads the repo notebook, appends a
save cell to an **in-memory copy**, and executes that copy with the kernel's
working directory pinned to the repo root so relative data paths still resolve.
The repo notebook was never modified.

**Why it mattered:** without a baseline, "the restructure changed nothing" is an
assertion. With one, it's testable. It also proved the original notebook could
run top-to-bottom in a cold kernel — genuinely uncertain at the time, because
cells 20–26 depended on execution order.

**Result:** ran clean in 4,047 s (67 min). 80 result rows, 448 sweep rows.
Output preserved as `reference/baseline_master_results.csv`.

---

### `01_verify_pipeline_equivalence.py` — *the cheap gate*

Ran the rewritten pipeline for 3 beaches × 2 recipes and compared every metric
against the baseline, row by row.

**Why it mattered:** a full notebook run costs an hour. This checks the logic in
about five minutes, so the expensive run at step 04 became confirmation rather
than debugging.

**Result:** exact match on 30 rows × 13 metrics. That single check validated
all of: `assemble_features` taking `beach_geo_df` explicitly, `FeatureSet` with
a positive `feature_cols` list, the model factory replacing `clone()`, the
threshold moving into the recipe, and `resolve_stations` in `"all"` mode.

---

### `02_verify_experiment_log.py` — *does the run log actually work*

Checked that `run_experiment()` writes all four files, that `recipe.json`
round-trips to an equal dict, that `meta.json` captures the notebook hash / git
commit / dirty flag, that the positive-count guard is recorded, and that
`load_experiments()` reads runs back with a `run_id`.

**Result:** passed. Note the first attempt failed on a **wrong assumption in the
test**, not in the code — it assumed `57th_street` would be skipped by the
guard, when in fact `foster` is (8 train / 1 test positives). The five beaches
the guard drops are `osterman`, `foster`, `north_avenue`, `oak_street`,
`oakwood`.

---

### `03_restructure_notebook.py` — *the rewrite itself*

Rebuilt `Final_feature_building.ipynb` into
Setup / Reference / Pipeline / Recipes / Runs / Compare.

Rather than retyping code into cells, it **parses `reference/pipeline_draft.py`
by its section banners** and lifts each block verbatim. Since that draft is the
exact code that passed step 01, transcription errors were impossible.

**Result:** 33 cells (25 code, 8 markdown), 325,909 → 49,946 bytes.

---

### `04_run_and_verify_no_op.py` — *the primary gate*

Executed the restructured notebook end to end and compared the resulting
`experiments/` output against the step 00 baseline.

**Result: passed — 80 rows × 13 metrics, every value identical.** This is the
proof that the restructure changed structure and nothing else. Took 1,110 s
(18.5 min), notably faster than the original 67 min because the old notebook
rebuilt montrose features three separate times.

*(The script itself errored after the comparison on a `sys.path` mistake of
mine — the notebook execution and the comparison logic were both fine.)*

---

### `05_run_radius_experiments.py` — *distance-based station selection*

Added `radius_d10` / `radius_d15` recipes and ran them against the
`all_stations` control, with two consistency checks: `all_stations` must still
equal the baseline (proving the resolver is transparent when not filtering),
and feature counts must shrink as predicted.

**Result:** both checks passed. Feature counts fell 70 → 30–35 (d=10) and
35–55 (d=15). **The hypothesis failed** — mean PR-AUC `all_stations` 0.271 vs
`radius_d10` 0.236 vs `radius_d15` 0.219, with paired win rates of 13/32 and
15/32, i.e. coin flips.

---

### `06_apply_feature_fixes.py` — *the two real bugs*

Patched two cells by id and cleared all stored outputs (changing pipeline code
makes existing outputs stale).

1. **`precip_max_in_window_per_station`** summed inside its CTE, collapsing each
   `(sample_ts, station)` to one row, so the outer `PIVOT ... USING MAX` took
   the max of a single value. All 11 `_max_` columns were byte-identical to
   their `_sum_` twins — peak rainfall intensity did not exist in the feature
   set at all.
2. **`groin_onshore_*`** returned NULL when no observation in the window was
   onshore, colliding with the `min_coverage` NULL meaning "not enough data".
   Two opposite meanings, one encoding — and the median imputer filled genuine
   calm periods with a *typical onshore* value.

---

### `07_verify_feature_fixes.py` — *did the fixes do what they claim*

Execs the notebook's **own** Setup/Reference/Pipeline cells (not a copy), builds
montrose features, and checks both fixes.

**Result:** 0 of 11 `_max_` columns still identical to their `_sum_` twin;
`sum >= max` holds on every non-null row. `groin_onshore` non-null coverage at
6h went 167/503 → 475/503, of which 311 are now correctly `0`, and its NaN
pattern now matches `chii2_wspd_mean_6h` exactly.

**One assertion in this script is wrong** and will fail if re-run: it asserts
`sum >= max` with `.all()` over columns containing NaN, and `NaN >= NaN` is
False in pandas. The invariant genuinely holds — verified separately on
non-null rows only, 0 violations across all 11 station pairs.

---

### `08_run_and_compare_fixes.py` — *what the fixes cost*

Re-ran the whole notebook post-fix and paired every `(beach, model)` against the
pre-fix run of the same recipe.

**Result:** the fixes did **not** improve performance.

| recipe | pre | post | Δ |
|---|---|---|---|
| wind_only | 0.1912 | 0.1895 | −0.002 |
| wind_geo | 0.1922 | 0.1772 | −0.015 |
| all_stations | 0.1922 | 0.1772 | −0.015 |
| radius_d10 | 0.1719 | 0.1639 | −0.008 |
| radius_d15 | 0.1745 | 0.1741 | −0.000 |

`wind_only` has no groin features, so it isolates the `precip_max` fix alone
(≈ 0 effect); the geo recipes add the groin fix (−0.015). Both sit inside the
noise at 64 test positives.

**The substantive consequence:** post-fix, `wind_only` (0.2538) now scores
*above* `wind_geo` (0.2440), reversing the earlier reading. The old comparison
was contaminated by `groin_onshore` being mostly NaN. There is currently no
evidence the Mattheus groin interaction carries signal.

---

## `reference/`

| file | what it is |
|---|---|
| `baseline_master_results.csv` | Step 00 baseline, 80 rows. **The regression reference.** Any future claim that a refactor is a no-op should be checked against this. |
| `baseline_master_sweeps.csv` | matching threshold sweeps, 448 rows |
| `Final_feature_building.PRE_RESTRUCTURE.ipynb` | your notebook exactly as it was before the restructure, 39 cells, 325,909 bytes. It had uncommitted changes at the time, so **this is the only copy** — `git` cannot recover it. |
| `pipeline_draft.py` | the pipeline as a flat module, the version that passed step 01. Superseded by the notebook cells; kept because `03_restructure_notebook.py` parses it, and because it is the cleanest single-file view of the whole pipeline. |
| `mwrd_collection_system_om_manual_2026.txt` | extracted text of the 212-page MWRD Collection System O&M Manual, from the earlier CSO investigation. Searchable; the source PDF is under gitignored `literature/`. Contains Appendix E (which beaches MWRD samples after each backflow structure) and Appendix F (the CSO outfall list). |

## `logs/`

Raw stdout from the long runs: `baseline_run.log`, `step4.log`, `step5.log`,
`step6.log`. Useful mainly for timings and the per-beach skip messages.

---

## Findings recorded nowhere else

Two data observations came out of this work and are **not** captured in the
notebook or `experiments/`:

**The `63rd_street` City of Chicago weather station stopped reporting after
2022** — 1 row in 2023, 11 in 2024, 6 in 2025, against ~8,400/yr for `foster`
and `oak_street`. Dry hours *are* recorded as zeros, so this is genuine sensor
loss, not an encoding artifact. Because the train/test split is chronological it
lands squarely in the test window, and because it is the nearest COC gauge to
the southern beaches, `radius_d10` hands them a dead station while discarding
the two working ones. **The radius experiment is confounded by availability and
is not a clean test of distance.**

**`cicero` USGS rows are duplicated exactly 2× in 2024–2025** (identical values,
same `usgs_id`, every timestamp) — 87,546 rows against ~43,772 for every other
gauge-year. Almost certainly a double-append in `USGS_concat_cleaning.ipynb`.
**No feature impact**, because `bucket_precip_per_station` does
`MAX(...) GROUP BY (bucket, station)` and collapses them. `north_austin` also
lost ~30% of 2023.
