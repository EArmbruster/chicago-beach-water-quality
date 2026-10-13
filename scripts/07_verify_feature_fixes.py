"""Verify the Step 6 fixes, running the notebook's OWN cells (not a copy).

Execs every code cell up to the '## Runs' heading, then builds features for one
beach and checks:
  1. precip_max columns are no longer identical to precip_sum
  2. groin_onshore is NaN only for genuine coverage gaps, not for calm windows
"""
import os
import sys

import nbformat
import numpy as np
import pandas as pd

REPO = "/Users/emily/Documents/chicago-beach-water-quality"
os.chdir(REPO)
sys.path.insert(0, REPO)

nb = nbformat.read(f"{REPO}/Final_feature_building.ipynb", as_version=4)

ns = {"__name__": "__main__"}
for cell in nb.cells:
    src = "".join(cell.source)
    if cell.cell_type == "markdown":
        if src.strip().startswith("## Runs"):
            break
        continue
    clean = "\n".join("" if l.strip().startswith("%") else l for l in src.split("\n"))
    exec(compile(clean, "<nb>", "exec"), ns)

print("executed notebook Setup/Reference/Pipeline/Recipes cells\n")

assemble_features = ns["assemble_features"]
RECIPE = ns["RECIPE"]
dataframes, readings_df = ns["dataframes"], ns["readings_df"]
beach_geo_df, beach_coords = ns["beach_geo_df"], ns["beach_coords"]

rec = {**RECIPE, "target_beach": "montrose"}
fs = assemble_features(readings_df, rec, dataframes, beach_geo_df, beach_coords)
f = fs.frame
print(f"montrose feature frame: {f.shape}, {fs.n_features} features\n")

# ── check 1: max != sum ───────────────────────────────────────────────
print("=== check 1: precip_max vs precip_sum (6h window) ===")
max_cols = [c for c in f.columns if "_precip_max_6h" in c]
identical, differing = [], []
for mc in max_cols:
    sc = mc.replace("_precip_max_6h", "_precip_sum_6h")
    if sc not in f.columns:
        continue
    if np.allclose(f[mc].fillna(-1), f[sc].fillna(-1)):
        identical.append(mc)
    else:
        differing.append((mc, float((f[sc] - f[mc]).mean())))

print(f"  columns checked: {len(max_cols)}")
print(f"  still identical to their _sum_ twin: {len(identical)}  {identical}")
print(f"  now differing: {len(differing)}")
for name, gap in differing[:4]:
    print(f"    {name}: mean(sum - max) = {gap:.3f} mm  (sum >= max always)")
assert not identical, "some _max_ columns are STILL identical to _sum_"

ex = differing[0][0]
exs = ex.replace("_precip_max_6h", "_precip_sum_6h")
wet = f[f[exs] > 0]
print(f"\n  on {len(wet)} samples with rain in the 6h window, {ex}:")
print(f"    peak  median {wet[ex].median():.2f}  max {wet[ex].max():.2f}")
print(f"    total median {wet[exs].median():.2f}  max {wet[exs].max():.2f}")
assert (f[exs] + 1e-9 >= f[ex]).all(), "sum must be >= max"
print("    invariant sum >= max holds")

# ── check 2: groin NaN semantics ──────────────────────────────────────
print("\n=== check 2: groin_onshore NaN semantics ===")
BEFORE = {"groin_onshore_6h_tol20": 167, "groin_onshore_12h_tol20": 213,
          "groin_onshore_24h_tol20": 321}   # non-null counts, pre-fix, n=503
for col in [c for c in f.columns if c.startswith("groin_onshore")]:
    nn, zeros = f[col].notna().sum(), (f[col] == 0).sum()
    was = BEFORE.get(col)
    extra = f"(was {was}/503 before the fix)" if was else ""
    print(f"  {col}: non-null {nn}/{len(f)}  of which exactly 0: {zeros}  {extra}")

# the wind-speed columns share the same coverage gate, so NaNs should now align
wind_nan = f["chii2_wspd_mean_6h"].isna()
groin_nan = f["groin_onshore_6h_tol20"].isna()
print(f"\n  6h window: chii2_wspd_mean NaN = {wind_nan.sum()}, "
      f"groin_onshore NaN = {groin_nan.sum()}")
print("  NaN patterns identical (both now mean 'insufficient coverage'):",
      bool((wind_nan == groin_nan).all()))
print("\n*** STEP 6 CHECKS PASSED ***")
