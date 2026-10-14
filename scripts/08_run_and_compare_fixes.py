"""Re-run the notebook after the Step 6 fixes and compare pre-fix vs post-fix.

Each recipe label now has two runs on disk: the older (pre-fix) and the new
(post-fix). They are distinguished by run_id ordering, and cross-checked with
notebook_sha256 from meta.json.
"""
import glob
import json
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
EXECUTED = SCRATCH / "step6_executed.ipynb"

pre_existing = {p for p in glob.glob(str(REPO / "experiments/*"))}

nb = nbformat.read(SRC, as_version=4)
print(f"executing {SRC.name}: {len(nb.cells)} cells", flush=True)
client = NotebookClient(nb, timeout=7200, kernel_name="python3", allow_errors=False,
                        resources={"metadata": {"path": str(REPO)}})
t0 = time.time(); status = "OK"
try:
    client.execute()
except Exception:
    status = "FAILED"; traceback.print_exc()
finally:
    nbformat.write(nb, EXECUTED)
print(f"\nSTATUS={status} elapsed={time.time()-t0:.0f}s", flush=True)
for i, cell in enumerate(nb.cells):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            print(f"\n--- ERROR cell {i} ---"); print(out.get("ename"), out.get("evalue"))
            print("\n".join(out.get("traceback", []))[:4000])
if status != "OK":
    sys.exit(1)
shutil.copy(EXECUTED, SRC)
print(f"copied executed notebook -> {SRC}")

# ── gather runs, tag pre/post fix ─────────────────────────────────────
rows = []
for p in sorted(glob.glob(str(REPO / "experiments/*/results.csv"))):
    d = pd.read_csv(p)
    rd = Path(p).parent
    d["run_id"] = rd.name
    d["is_new"] = str(rd) not in pre_existing
    rows.append(d)
a = pd.concat(rows, ignore_index=True)
a = a[a.model != "dummy"]

print("\n=== runs on disk ===")
print(a.groupby(["recipe_label", "is_new"])["run_id"].agg(["nunique", "first"]).to_string())

LAB = ["all_stations", "wind_only", "wind_geo", "radius_d10", "radius_d15"]
print("\n=== mean PR-AUC, pre-fix vs post-fix ===")
out = []
for lab in LAB:
    sub = a[a.recipe_label == lab]
    if sub.is_new.nunique() < 2:
        continue
    old = sub[~sub.is_new]; new = sub[sub.is_new]
    # pair on (beach, model)
    m = old.merge(new, on=["beach", "model"], suffixes=("_pre", "_post"))
    out.append({
        "recipe": lab, "n_pairs": len(m),
        "pr_auc_pre": m.pr_auc_pre.mean(), "pr_auc_post": m.pr_auc_post.mean(),
        "delta": m.pr_auc_post.mean() - m.pr_auc_pre.mean(),
        "better": int((m.pr_auc_post > m.pr_auc_pre).sum()),
    })
if out:
    print(pd.DataFrame(out).round(4).to_string(index=False))

print("\n=== best PR-AUC per beach, post-fix ===")
new = a[a.is_new]
pr = new.groupby(["beach", "recipe_label"])["pr_auc"].max().unstack("recipe_label")
cols = [c for c in LAB if c in pr.columns]
print(pr[cols].round(4).to_string())
print("\nmean:"); print(pr[cols].mean().round(4).to_string())
