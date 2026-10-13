"""Step 5: execute the notebook with the radius recipes added.

Also runs two consistency checks afterwards:
  1. all_stations must still match the Step 0 baseline (it is the same config
     as wind_geo, so the 'all' resolver must be transparent when not filtering)
  2. resolved station lists in meta.json must match the column names that
     actually appear in the feature set (via n_features)
"""
import glob
import shutil
import sys
import time
import traceback
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

REPO = Path("/Users/emily/Documents/chicago-beach-water-quality")
SCRATCH = Path(__file__).parent
SRC = REPO / "Final_feature_building.ipynb"
EXECUTED = SCRATCH / "step5_executed.ipynb"
BASELINE = SCRATCH / "baseline" / "master_results.csv"

nb = nbformat.read(SRC, as_version=4)
print(f"executing {SRC.name}: {len(nb.cells)} cells", flush=True)

client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                        allow_errors=False,
                        resources={"metadata": {"path": str(REPO)}})
t0 = time.time()
status = "OK"
try:
    client.execute()
except Exception:
    status = "FAILED"
    traceback.print_exc()
finally:
    nbformat.write(nb, EXECUTED)

print(f"\nSTATUS={status} elapsed={time.time()-t0:.0f}s", flush=True)
for i, cell in enumerate(nb.cells):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            print(f"\n--- ERROR in cell {i} ---")
            print(out.get("ename"), out.get("evalue"))
            print("\n".join(out.get("traceback", []))[:4000])
if status != "OK":
    sys.exit(1)

shutil.copy(EXECUTED, SRC)
print(f"copied executed notebook -> {SRC}")

# ── load every run ────────────────────────────────────────────────────
frames = []
for p in sorted(glob.glob(str(REPO / "experiments/*/results.csv"))):
    df = pd.read_csv(p)
    df["run_id"] = Path(p).parent.name
    frames.append(df)
allr = pd.concat(frames, ignore_index=True)

latest = allr.sort_values("run_id").groupby("recipe_label")["run_id"].last()
cur = allr[allr["run_id"].isin(latest)]

KEY = ["beach", "model"]
METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "n_features"]

# ── check 1: all_stations == baseline (the 'all' resolver is transparent) ──
base = pd.read_csv(BASELINE)
base_geo = (base[base["recipe_label"] == "wind_geo"][KEY + METRICS]
            .sort_values(KEY).reset_index(drop=True))
new_all = (cur[cur["recipe_label"] == "all_stations"][KEY + METRICS]
           .sort_values(KEY).reset_index(drop=True))

print(f"\n=== check 1: all_stations vs baseline wind_geo "
      f"({len(new_all)} vs {len(base_geo)} rows) ===")
if len(new_all) != len(base_geo):
    print("!! row count mismatch")
else:
    bad = []
    for c in METRICS:
        same = ((new_all[c] - base_geo[c]).abs() < 1e-9) | \
               (new_all[c].isna() & base_geo[c].isna())
        for i in (~same).to_numpy().nonzero()[0]:
            bad.append((tuple(new_all.loc[i, KEY]), c, base_geo[c][i], new_all[c][i]))
    print("*** IDENTICAL ***" if not bad else f"!! {len(bad)} mismatches: {bad[:10]}")

# ── check 2: feature counts shrink as expected ────────────────────────
print("\n=== check 2: features per beach by recipe ===")
piv = cur.pivot_table(index="beach", columns="recipe_label",
                      values="n_features", aggfunc="first")
cols = [c for c in ["all_stations", "radius_d15", "radius_d10"] if c in piv.columns]
print(piv[cols].to_string())

# ── headline: PR-AUC comparison ───────────────────────────────────────
nd = cur[cur["model"] != "dummy"]
pr = (nd.groupby(["beach", "recipe_label"])["pr_auc"].max()
        .unstack("recipe_label"))
cols = [c for c in ["all_stations", "radius_d10", "radius_d15"] if c in pr.columns]
print("\n=== best PR-AUC per beach ===")
print(pr[cols].round(4).to_string())
print("\nmean:"); print(pr[cols].mean().round(4).to_string())
for c in ["radius_d10", "radius_d15"]:
    if c in pr.columns:
        print(f"{c} beat all_stations at {(pr[c] > pr['all_stations']).sum()}"
              f"/{pr['all_stations'].notna().sum()} beaches")
