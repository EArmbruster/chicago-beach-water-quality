"""Step 0: execute Final_feature_building.ipynb in a fresh kernel and capture
master_results / master_sweeps as the regression baseline.

The repo notebook is NOT modified. We read it, append a save cell to an in-memory
copy, and execute that copy with the kernel's cwd pinned to the repo root so the
notebook's relative data paths ("data/prepared_data/...") still resolve.
"""
import os
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

REPO = "/Users/emily/Documents/chicago-beach-water-quality"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(SCRATCH, "baseline")
SRC = os.path.join(REPO, "Final_feature_building.ipynb")
EXECUTED = os.path.join(SCRATCH, "baseline_executed.ipynb")

os.makedirs(BASELINE, exist_ok=True)

SAVE_CELL = f"""
import pathlib as _pl
_out = _pl.Path(r"{BASELINE}")
_out.mkdir(parents=True, exist_ok=True)
master_results.to_csv(_out / "master_results.csv", index=False)
master_sweeps.to_csv(_out / "master_sweeps.csv", index=False)
print("BASELINE_SAVED",
      "results", master_results.shape,
      "sweeps", master_sweeps.shape)
print("BASELINE_COLUMNS", sorted(master_results.columns.tolist()))
"""

nb = nbformat.read(SRC, as_version=4)
print(f"read {SRC}: {len(nb.cells)} cells", flush=True)
nb.cells.append(nbformat.v4.new_code_cell(SAVE_CELL))

client = NotebookClient(
    nb,
    timeout=3600,            # per-cell
    kernel_name="python3",
    allow_errors=False,      # stop on first error so we see it clearly
    resources={"metadata": {"path": REPO}},   # kernel cwd = repo root
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
    print(f"\nwrote executed copy -> {EXECUTED}", flush=True)

elapsed = time.time() - t0
print(f"\nSTATUS={status} elapsed={elapsed:.1f}s", flush=True)

# Surface the last cell's stdout regardless of status, and any error tracebacks.
for i, cell in enumerate(nb.cells):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            print(f"\n--- ERROR in cell {i} ---", flush=True)
            print(out.get("ename"), out.get("evalue"), flush=True)
            print("\n".join(out.get("traceback", []))[:3000], flush=True)
        elif out.get("output_type") == "stream" and "BASELINE_" in "".join(out.get("text", "")):
            print("\n--- save cell ---", flush=True)
            print("".join(out["text"]), flush=True)

sys.exit(0 if status == "OK" else 1)
