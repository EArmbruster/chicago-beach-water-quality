"""Step 4: execute the restructured notebook and prove it reproduces the baseline.

Executes a COPY (an interruption cannot corrupt the repo notebook) with the
kernel cwd pinned to the repo root, so experiments/ is written into the repo as
intended. On success the executed copy -- outputs and all -- is copied back over
the repo notebook, which is what a Restart & Run All would have produced.
"""
import os
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
EXECUTED = SCRATCH / "step4_executed.ipynb"
BASELINE = SCRATCH / "baseline" / "master_results.csv"

nb = nbformat.read(SRC, as_version=4)
print(f"executing {SRC.name}: {len(nb.cells)} cells", flush=True)

client = NotebookClient(
    nb, timeout=5400, kernel_name="python3", allow_errors=False,
    resources={"metadata": {"path": str(REPO)}},
)

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

# ── compare experiments/ output against the Step 0 baseline ───────────
sys.path.insert(0, str(SCRATCH))
import pipeline_draft as P

os.chdir(REPO)
allr = P.load_experiments(out_dir="experiments")
# keep only the two real recipes; drop the smoke test
allr = allr[allr["recipe_label"].isin(["wind_only", "wind_geo"])]

base = pd.read_csv(BASELINE)

KEY = ["beach", "recipe_label", "model"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
           "true_neg", "false_pos", "false_neg", "true_pos",
           "n_features", "n_train_pos", "n_test_pos"]

new = allr[KEY + METRICS].sort_values(KEY).reset_index(drop=True)
old = base[KEY + METRICS].sort_values(KEY).reset_index(drop=True)

print(f"\ncomparing {len(new)} new rows vs {len(old)} baseline rows")
if len(new) != len(old):
    print("!! ROW COUNT MISMATCH")
    nk = set(map(tuple, new[KEY].values)); ok = set(map(tuple, old[KEY].values))
    print("  only in new:", sorted(nk - ok))
    print("  only in baseline:", sorted(ok - nk))
    sys.exit(1)

mismatches = []
for col in METRICS:
    a, b = new[col], old[col]
    same = ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
    if not same.all():
        for i in (~same).to_numpy().nonzero()[0]:
            mismatches.append((tuple(new.loc[i, KEY]), col, b[i], a[i]))

if mismatches:
    print(f"\n!! {len(mismatches)} MISMATCHES (key, column, baseline, new):")
    for m in mismatches[:30]:
        print("   ", m)
    sys.exit(1)

print("\n*** STEP 4 PASSED: restructured notebook reproduces the baseline exactly ***")
print(f"    {len(new)} rows x {len(METRICS)} metrics, all identical")

shutil.copy(EXECUTED, SRC)
print(f"\ncopied executed notebook (with outputs) -> {SRC}")
