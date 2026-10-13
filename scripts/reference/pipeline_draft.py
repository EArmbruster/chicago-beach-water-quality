"""DRAFT of the restructured notebook Pipeline section.

Developed here for fast iteration; transplanted into notebook cells in Step 2.
Section banners map 1:1 onto the cells they become.

INVARIANT FOR STEP 4: the feature-building and modelling logic must be
behaviourally identical to the current notebook. The two known bugs
(precip_max == precip_sum; groin_onshore NaN semantics) are PRESERVED here
deliberately -- they are Step 6. Do not "fix" anything in this file.
"""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

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

from station_locations import station_distances_km, validate_station_names


# ══════════════════════════════════════════════════════════════════════
# FeatureSet
# ══════════════════════════════════════════════════════════════════════

IDENTITY_COLS = ("Beach", "reading", "sample_ts")


@dataclass
class FeatureSet:
    """What assemble_features built, and what went into it.

    `feature_cols` is a POSITIVE list, decided at build time. The old code
    derived features subtractively (everything except NON_FEATURE_COLS), which
    silently promotes any new column -- an id, a year, a CSO event key -- into
    the model. Recording the list at the moment of construction removes that
    class of bug.

    `resolved` records which stations each source actually contributed, so a
    run can be audited after the fact. With distance-based selection (Step 5)
    the recipe alone no longer determines the feature set: it is recipe+beach.
    """
    frame: pd.DataFrame
    feature_cols: list[str]
    resolved: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


# ══════════════════════════════════════════════════════════════════════
# Target readings
# ══════════════════════════════════════════════════════════════════════

def get_target_readings(readings, target_beach, min_year):
    sifted_readings = duckdb.query("""
        SELECT
            Beach,
            reading,
            timestamp_central AS sample_ts
        FROM readings
        WHERE Beach = ? AND year >= ?
        ORDER BY sample_ts
    """, params=[target_beach, min_year]).df()
    return sifted_readings


# ══════════════════════════════════════════════════════════════════════
# Station resolution
# ══════════════════════════════════════════════════════════════════════

def resolve_stations(beach, source, selection, beach_coords, declared=None):
    """Decide which stations of `source` feed `beach`.

    Modes:
      "all"    -> the recipe's declared list, unchanged (current behaviour)
      "radius" -> every station within `max_km`, nearest first
      "knn"    -> the `k` nearest stations

    `min_stations` is a floor applied to radius mode: if the threshold excludes
    everything, keep the nearest anyway. Without it a tight radius silently
    produces a source with no columns at all.

    Declared names are validated even in "all" mode so a typo in a recipe fails
    loudly rather than quietly shrinking the feature set.
    """
    mode = selection.get("mode", "all")

    if mode == "all":
        if declared is None:
            raise ValueError(
                f"station_selection mode 'all' needs an explicit station list "
                f"for source {source!r}"
            )
        return validate_station_names(declared, source)

    dists = station_distances_km(beach, source, beach_coords)

    if mode == "radius":
        max_km = selection["max_km"]
        keep = sorted([s for s, km in dists.items() if km <= max_km],
                      key=lambda s: dists[s])
        floor = selection.get("min_stations", 1)
        if len(keep) < floor:
            keep = sorted(dists, key=lambda s: dists[s])[:floor]
        return keep

    if mode == "knn":
        k = selection["k"]
        return sorted(dists, key=lambda s: dists[s])[:k]

    raise ValueError(
        f"Unknown station_selection mode {mode!r}; expected 'all', 'radius' or 'knn'"
    )


# ══════════════════════════════════════════════════════════════════════
# Precip features   (logic unchanged from the original notebook)
# ══════════════════════════════════════════════════════════════════════

def bucket_precip_per_station(precip_df, ts_col, value_col, station_col,
                              bucket_minutes, stations):
    if stations is None:
        station_filter = ""
    else:
        quoted = ", ".join(f"'{s}'" for s in stations)
        station_filter = f"AND {station_col} IN ({quoted})"

    sql = f"""
        SELECT
            time_bucket(INTERVAL '{bucket_minutes} minutes', {ts_col}) AS ts,
            {station_col} AS station,
            MAX({value_col}) AS precip_mm
        FROM precip_df
        WHERE {ts_col} IS NOT NULL
          AND {value_col} IS NOT NULL
          {station_filter}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return duckdb.sql(sql).df()


def precip_sum_in_window_per_station(readings, bucketed, window_hours, source_label):
    sql = f"""
        WITH summed AS (
            SELECT
                r.sample_ts,
                b.station,
                COALESCE(SUM(b.precip_mm), 0) AS precip_mm
            FROM readings r
            LEFT JOIN bucketed b
              ON b.ts >  r.sample_ts - INTERVAL {window_hours} HOUR
             AND b.ts <= r.sample_ts
            GROUP BY r.sample_ts, b.station
        )
        PIVOT summed
            ON station
            USING SUM(precip_mm)
            GROUP BY sample_ts
    """
    out = duckdb.sql(sql).df()
    rename = {
        col: f"{source_label}_{col}_precip_sum_{window_hours}h"
        for col in out.columns if col != "sample_ts"
    }
    return out.rename(columns=rename)


def precip_max_in_window_per_station(readings, bucketed, window_hours, source_label):
    # NOTE (Step 6): this does not actually compute a max. The CTE already
    # collapses to one row per (sample_ts, station) via SUM, so the outer
    # PIVOT ... USING MAX takes the max of a single value -- making these
    # columns identical to the _sum_ ones. Preserved verbatim so Step 4 can
    # prove the restructure is a no-op. Fix separately.
    sql = f"""
        WITH summed AS (
            SELECT
                r.sample_ts,
                b.station,
                COALESCE(SUM(b.precip_mm), 0) AS precip_mm
            FROM readings r
            LEFT JOIN bucketed b
              ON b.ts >  r.sample_ts - INTERVAL {window_hours} HOUR
             AND b.ts <= r.sample_ts
            GROUP BY r.sample_ts, b.station
        )
        PIVOT summed
            ON station
            USING MAX(precip_mm)
            GROUP BY sample_ts
    """
    out = duckdb.sql(sql).df()
    rename = {
        col: f"{source_label}_{col}_precip_max_{window_hours}h"
        for col in out.columns if col != "sample_ts"
    }
    return out.rename(columns=rename)


def build_precip_features_for_source(readings, precip_df, ts_col, value_col,
                                     station_col, source_label, bucket_minutes,
                                     stations, sum_windows_hours, max_windows_hours):
    bucketed = bucket_precip_per_station(
        precip_df, ts_col, value_col, station_col, bucket_minutes, stations
    )
    out = readings[["sample_ts"]].copy()
    for hours in sum_windows_hours:
        feat = precip_sum_in_window_per_station(readings, bucketed, hours, source_label)
        out = out.merge(feat, on="sample_ts", how="left")
    for hours in max_windows_hours:
        feat = precip_max_in_window_per_station(readings, bucketed, hours, source_label)
        out = out.merge(feat, on="sample_ts", how="left")
    return out


# ══════════════════════════════════════════════════════════════════════
# Wind features   (logic unchanged from the original notebook)
# ══════════════════════════════════════════════════════════════════════

def build_wind_features(readings, chii2_df, facing_az_deg, groin_length_m,
                        windows_hours=[6, 12, 24], tolerance_deg=20,
                        min_coverage=0.5, use_geo_features=True):
    facing_rad = np.deg2rad(facing_az_deg)
    tol = int(tolerance_deg)

    wind = chii2_df[["timestamp_central", "WSPD", "WDIR"]].copy()
    wind = wind.dropna(subset=["WSPD"])

    wdir_rad = np.deg2rad(wind["WDIR"])
    wind["u"] = -wind["WSPD"] * np.sin(wdir_rad)
    wind["v"] = -wind["WSPD"] * np.cos(wdir_rad)
    wind["onshore"] = wind["u"] * np.sin(facing_rad) + wind["v"] * np.cos(facing_rad)
    wind["alongshore"] = wind["u"] * np.cos(facing_rad) - wind["v"] * np.sin(facing_rad)

    delta = (wind["WDIR"] - facing_az_deg + 180) % 360 - 180
    wind["is_onshore"] = (delta.abs() <= tolerance_deg).astype(int)
    wind.loc[wind["WDIR"].isna(), "is_onshore"] = np.nan

    samples = readings[["sample_ts"]].drop_duplicates()
    out = samples.copy()

    for wh in windows_hours:
        expected_obs = wh * 6
        min_obs = int(expected_obs * min_coverage)

        # NOTE (Step 6): the inner AVG(... WHEN is_onshore = 1 ...) is NULL when
        # no observation in the window was onshore, so this column is NaN rather
        # than 0 for calm/offshore windows. Preserved verbatim; fix separately.
        groin_sql = f"""
            CASE WHEN COUNT(w.is_onshore) >= {min_obs}
                THEN {groin_length_m}
                    * AVG(w.is_onshore)
                    * AVG(CASE WHEN w.is_onshore = 1 THEN w.WSPD ELSE NULL END)
                END
                AS groin_onshore_{wh}h_tol{tol}
        """ if use_geo_features else ""

        agg = duckdb.query(f"""
            SELECT
                s.sample_ts,
                CASE WHEN COUNT(w.WSPD) >= {min_obs}
                    THEN AVG(w.WSPD) END AS chii2_wspd_mean_{wh}h,
                CASE WHEN COUNT(w.WSPD) >= {min_obs}
                    THEN MAX(w.WSPD) END AS chii2_wspd_max_{wh}h,
                CASE WHEN COUNT(w.onshore) >= {min_obs}
                    THEN AVG(w.onshore) END AS chii2_onshore_mean_{wh}h,
                CASE WHEN COUNT(w.alongshore) >= {min_obs}
                    THEN AVG(w.alongshore) END AS chii2_alongshore_mean_{wh}h
                {(',' + groin_sql) if use_geo_features else ''}
            FROM samples s
            LEFT JOIN wind w
                ON w.timestamp_central > s.sample_ts - INTERVAL '{wh} hours'
               AND w.timestamp_central <= s.sample_ts
            GROUP BY s.sample_ts
        """).df()

        out = out.merge(agg, on="sample_ts", how="left")

    return out


# ══════════════════════════════════════════════════════════════════════
# Assemble
# ══════════════════════════════════════════════════════════════════════

def assemble_features(readings, recipe, dataframes, beach_geo_df,
                      beach_coords=None) -> FeatureSet:
    """Build the feature frame for one beach.

    `beach_geo_df` is now an explicit argument. It used to be read from a
    module-level global, which quietly broke the recipe's portability promise:
    the recipe was serialisable but a run still depended on ambient state.
    """
    beach = recipe["target_beach"]
    readings = get_target_readings(readings, beach, recipe["min_year"])
    out = readings.copy()
    feats = recipe["features"]
    resolved: dict[str, list[str]] = {}

    if feats["precip"]["enabled"]:
        cfg = feats["precip"]
        selection = cfg.get("station_selection", {"mode": "all"})
        for source in cfg["sources"]:
            df = dataframes[source["df_name"]]
            stations = resolve_stations(
                beach=beach,
                source=source["label"],
                selection=selection,
                beach_coords=beach_coords or {},
                declared=source.get("stations"),
            )
            resolved[source["label"]] = list(stations)
            f = build_precip_features_for_source(
                readings=readings,
                precip_df=df,
                ts_col=source["ts_col"],
                value_col=source["value_col"],
                station_col=source["station_col"],
                source_label=source["label"],
                bucket_minutes=source["bucket_minutes"],
                stations=stations,
                sum_windows_hours=cfg["sum_windows_hours"],
                max_windows_hours=cfg["max_windows_hours"],
            )
            out = out.merge(f, on="sample_ts", how="left")

    if feats.get("wind", {}).get("enabled", False):
        cfg = feats["wind"]
        geo = beach_geo_df.loc[beach]
        f = build_wind_features(
            readings=readings,
            chii2_df=dataframes[cfg["df_name"]],
            facing_az_deg=geo["facing_az_deg"],
            groin_length_m=geo["downstream_groin_length_m"],
            windows_hours=cfg["windows_hours"],
            tolerance_deg=cfg.get("tolerance_deg", 20),
            min_coverage=cfg.get("min_coverage", 0.5),
            use_geo_features=cfg.get("use_geo_features", True),
        )
        out = out.merge(f, on="sample_ts", how="left")

    feature_cols = [c for c in out.columns if c not in IDENTITY_COLS]
    return FeatureSet(frame=out, feature_cols=feature_cols, resolved=resolved)


# ══════════════════════════════════════════════════════════════════════
# Target, split, metrics
# ══════════════════════════════════════════════════════════════════════

def add_target(df, threshold=1000):
    df = df.copy()
    df["exceeds"] = (df["reading"] >= threshold).astype(int)
    return df


def time_split(df, ts_col="sample_ts", test_frac=0.2):
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def split_X_y(df, feature_cols, target_col="exceeds"):
    """Select features POSITIVELY from the recorded list."""
    return df[feature_cols], df[target_col]


def evaluate(model, X_train, y_train, X_test, y_test, model_name="model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        "true_neg": tn, "false_pos": fp, "false_neg": fn, "true_pos": tp,
    }


def threshold_sweep(model, X_train, y_train, X_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    model.fit(X_train, y_train)
    if not hasattr(model, "predict_proba"):
        return None
    y_proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Models  (factory dict: every call yields a fresh, unfitted estimator)
# ══════════════════════════════════════════════════════════════════════

MODEL_FACTORIES = {
    "dummy": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic": lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "hgb": lambda: HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
    "xgb": lambda: XGBClassifier(random_state=42, scale_pos_weight=1, eval_metric="logloss"),
    "lgbm": lambda: LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1),
}


def run_all_models(models, X_train, y_train, X_test, y_test):
    results = []
    for name, model in models.items():
        results.append(evaluate(model, X_train, y_train, X_test, y_test, model_name=name))
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
# Per-beach pipeline
# ══════════════════════════════════════════════════════════════════════

def run_pipeline_for_beach(beach_name, recipe, dataframes, readings_df, beach_geo_df,
                           model_factories=None, beach_coords=None,
                           test_frac=0.2, sweep_thresholds=None,
                           min_train_pos=10, min_test_pos=3):
    model_factories = model_factories or MODEL_FACTORIES
    recipe = {**recipe, "target_beach": beach_name}
    recipe_label = recipe.get("label", "default")
    threshold = recipe.get("target", {}).get("threshold", 1000)

    fs = assemble_features(readings_df, recipe, dataframes, beach_geo_df, beach_coords)
    features = add_target(fs.frame, threshold=threshold)

    train_df, test_df = time_split(features, test_frac=test_frac)

    n_train_pos = int(train_df["exceeds"].sum())
    n_test_pos = int(test_df["exceeds"].sum())
    if n_train_pos < min_train_pos or n_test_pos < min_test_pos:
        print(f"  [skip] {beach_name} ({recipe_label}): {n_train_pos} train pos, {n_test_pos} test pos")
        return None, None, fs

    X_train, y_train = split_X_y(train_df, fs.feature_cols)
    X_test, y_test = split_X_y(test_df, fs.feature_cols)

    local_models = {}
    for name, factory in model_factories.items():
        m = factory()
        if name == "xgb":
            m.set_params(scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1))
        local_models[name] = m

    model_results = run_all_models(local_models, X_train, y_train, X_test, y_test)
    model_results["beach"] = beach_name
    model_results["recipe_label"] = recipe_label
    model_results["n_features"] = X_train.shape[1]
    model_results["n_train_pos"] = n_train_pos
    model_results["n_test_pos"] = n_test_pos

    sweep_rows = []
    for name, model in local_models.items():
        if name == "dummy":
            continue
        sweep = threshold_sweep(model, X_train, y_train, X_test, y_test, sweep_thresholds)
        if sweep is not None:
            sweep["model"] = name
            sweep["beach"] = beach_name
            sweep["recipe_label"] = recipe_label
            sweep_rows.append(sweep)

    sweep_results = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    return model_results, sweep_results, fs


def run_pipeline_for_many(beach_recipes, dataframes, readings_df, beach_geo_df, **kwargs):
    all_model_results, all_sweep_results = [], []
    resolved_by_beach = {}

    for beach_name, recipe in beach_recipes:
        recipe_label = recipe.get("label", "default")
        print(f"Running: beach={beach_name}, recipe={recipe_label}")
        model_res, sweep_res, fs = run_pipeline_for_beach(
            beach_name=beach_name, recipe=recipe, dataframes=dataframes,
            readings_df=readings_df, beach_geo_df=beach_geo_df, **kwargs,
        )
        if fs is not None:
            resolved_by_beach[f"{beach_name}|{recipe_label}"] = fs.resolved
        if model_res is not None:
            all_model_results.append(model_res)
        if sweep_res is not None and not sweep_res.empty:
            all_sweep_results.append(sweep_res)

    master_model_results = pd.concat(all_model_results, ignore_index=True) if all_model_results else pd.DataFrame()
    master_sweep_results = pd.concat(all_sweep_results, ignore_index=True) if all_sweep_results else pd.DataFrame()
    return master_model_results, master_sweep_results, resolved_by_beach


# ══════════════════════════════════════════════════════════════════════
# Experiment log
# ══════════════════════════════════════════════════════════════════════

try:                                   # running as a module
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:                      # running inside a notebook cell
    REPO_ROOT = Path.cwd()
NOTEBOOK_NAME = "Final_feature_building.ipynb"


@dataclass
class ExperimentRun:
    """One (recipe -> results) pair, already written to disk."""
    run_id: str
    run_dir: Path
    recipe: dict
    results: pd.DataFrame
    sweeps: pd.DataFrame
    meta: dict

    def summary(self, top_n: int = 1) -> pd.DataFrame:
        """Best non-dummy model per beach by PR-AUC."""
        if self.results.empty:
            return self.results
        nd = self.results[self.results["model"] != "dummy"]
        idx = nd.groupby("beach")["pr_auc"].idxmax()
        cols = ["beach", "recipe_label", "model", "precision", "recall",
                "f1", "roc_auc", "pr_auc", "n_features", "n_test_pos"]
        return nd.loc[idx, cols].sort_values("pr_auc", ascending=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git_state(repo: Path) -> dict:
    """Commit and dirty flag, best-effort. Never raises."""
    def _run(args):
        try:
            out = subprocess.run(args, cwd=repo, capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
    }


def canonical_recipe_json(recipe: dict) -> str:
    """Stable JSON for hashing and storage.

    sort_keys so logically-identical recipes hash identically regardless of the
    order keys happened to be written in; default=str so a stray non-JSON value
    degrades to a string rather than blowing up a completed run at write time.
    """
    return json.dumps(recipe, indent=2, sort_keys=True, default=str)


def run_experiment(recipe, beaches, dataframes, readings_df, beach_geo_df,
                   model_factories=None, beach_coords=None,
                   out_dir="experiments", repo_root=None, **kwargs) -> ExperimentRun:
    """Run one recipe across beaches and persist (recipe -> results) to disk.

    This is the single entry point: the notebook's interactive per-recipe cells
    and any batch loop both go through here, so displayed numbers cannot drift
    from what was actually run. Results survive re-execution because they are
    written to experiments/<run_id>/ rather than living in cell output.
    """
    import copy as _copy

    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    label = recipe.get("label", "default")
    beaches = list(beaches)

    t0 = time.time()
    beach_recipes = [(b, _copy.deepcopy(recipe)) for b in beaches]
    results, sweeps, resolved = run_pipeline_for_many(
        beach_recipes, dataframes, readings_df, beach_geo_df,
        model_factories=model_factories, beach_coords=beach_coords, **kwargs,
    )
    elapsed = time.time() - t0

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{label}"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    recipe_json = canonical_recipe_json(recipe)
    (run_dir / "recipe.json").write_text(recipe_json)
    results.to_csv(run_dir / "results.csv", index=False)
    sweeps.to_csv(run_dir / "sweeps.csv", index=False)

    modelled = sorted(results["beach"].unique()) if not results.empty else []
    skipped = [b for b in beaches if b not in modelled]

    meta = {
        "run_id": run_id,
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 1),
        "recipe_sha256": _sha256_text(recipe_json),
        # Code provenance: with the pipeline living in notebook cells, this is
        # what catches "a builder was edited but this run predates the edit".
        "notebook_sha256": _sha256_file(repo_root / NOTEBOOK_NAME),
        **_git_state(repo_root),
        "python": platform.python_version(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "duckdb": duckdb.__version__,
        },
        "beaches_requested": beaches,
        "beaches_modelled": modelled,
        "beaches_skipped": skipped,   # dropped by the positive-count guard
        "resolved_stations": resolved,
        "n_result_rows": int(len(results)),
        "n_sweep_rows": int(len(sweeps)),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"\n[experiment] {run_id}  ->  {run_dir}")
    print(f"  {len(modelled)}/{len(beaches)} beaches modelled"
          + (f", skipped: {skipped}" if skipped else "")
          + f", {elapsed:.0f}s")

    return ExperimentRun(run_id=run_id, run_dir=run_dir, recipe=recipe,
                         results=results, sweeps=sweeps, meta=meta)


def load_experiments(out_dir="experiments", what="results") -> pd.DataFrame:
    """Load every past run's results (or sweeps) into one frame.

    Reads from disk rather than kernel state, so this shows every experiment
    ever run -- including from earlier sessions -- not just what is in memory.
    """
    pattern = str(Path(out_dir) / "*" / f"{what}.csv")
    matches = sorted(Path(out_dir).glob(f"*/{what}.csv"))
    if not matches:
        return pd.DataFrame()
    df = duckdb.sql(
        f"SELECT * FROM read_csv('{pattern}', filename=true, union_by_name=true)"
    ).df()
    # filename -> run_id, so runs are distinguishable after concatenation
    df["run_id"] = df["filename"].map(lambda p: Path(p).parent.name)
    return df.drop(columns=["filename"])
