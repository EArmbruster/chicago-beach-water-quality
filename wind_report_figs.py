"""
Wind report figures using CHII2 buoy data (summers 2021-2026).

Three figures suitable for a report or paper:
  1. Wind rose — direction and speed distribution
  2. Monthly wind speed — seasonal pattern across summers
  3. Onshore wind frequency per beach — connects data to model feature

Usage:
    from wind_report_figs import (
        plot_wind_rose,
        plot_monthly_wind_speed,
        plot_onshore_frequency,
    )
    from beach_geomorphology import get_geomorphology_df

    beach_geo_df = get_geomorphology_df()
    MATTHEUS_BEACHES = beach_geo_df[beach_geo_df["mattheus_id"].notna()].index.tolist()

    fig1 = plot_wind_rose(chii2_wind_df)
    fig2 = plot_monthly_wind_speed(chii2_wind_df)
    fig3 = plot_onshore_frequency(chii2_wind_df, beach_geo_df, MATTHEUS_BEACHES)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MIN_YEAR = 2021
MAX_YEAR = 2026


def _summer_chii2(chii2_df):
    """Filter CHII2 to summers 2021-2026, drop rows missing WSPD."""
    df = chii2_df[
        (chii2_df["timestamp_central"].dt.year >= MIN_YEAR) &
        (chii2_df["timestamp_central"].dt.year <= MAX_YEAR) &
        (chii2_df["timestamp_central"].dt.month.between(5, 9))
    ].copy()
    df = df.dropna(subset=["WSPD"])
    return df


def plot_wind_rose(chii2_df, n_dir_bins=16, speed_bins=None):
    """
    Figure 1: Wind rose showing direction and speed distribution.

    The standard meteorological figure — where wind comes FROM and
    how fast. Bar length = frequency, bar color = speed range.

    Args:
        chii2_df:    CHII2 wind dataframe.
        n_dir_bins:  Number of compass directions (16 = every 22.5°).
        speed_bins:  Speed thresholds in m/s. Default [0, 3, 6, 9, 999].
    """
    if speed_bins is None:
        speed_bins = [0, 3, 6, 9, 999]

    df = _summer_chii2(chii2_df)
    df = df.dropna(subset=["WDIR"])

    # Bin directions
    bin_width = 360 / n_dir_bins
    df["dir_bin"] = (
        ((df["WDIR"] + bin_width / 2) % 360) // bin_width * bin_width
    )

    # Speed categories
    speed_labels = []
    for i in range(len(speed_bins) - 1):
        lo, hi = speed_bins[i], speed_bins[i + 1]
        label = f"{lo}–{hi} m/s" if hi < 999 else f">{lo} m/s"
        speed_labels.append(label)
        df[f"spd_{i}"] = (
            (df["WSPD"] >= lo) & (df["WSPD"] < hi)
        ).astype(int)

    n_total = len(df)
    dir_vals = np.deg2rad(np.arange(0, 360, bin_width))
    colors = cm.YlOrRd(np.linspace(0.2, 0.9, len(speed_labels)))

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    bottoms = np.zeros(n_dir_bins)
    for i, (label, color) in enumerate(zip(speed_labels, colors)):
        counts = df.groupby("dir_bin")[f"spd_{i}"].sum()
        freqs = np.array([
            counts.get(b, 0) / n_total * 100
            for b in np.arange(0, 360, bin_width)
        ])
        ax.bar(
            dir_vals, freqs, width=np.deg2rad(bin_width) * 0.9,
            bottom=bottoms, color=color, label=label, alpha=0.9,
        )
        bottoms += freqs

    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=10)
    ax.set_ylabel("Frequency (%)", labelpad=30, fontsize=9)
    ax.set_title(
        f"CHII2 Buoy Wind Rose\n"
        f"Summers {MIN_YEAR}–{MAX_YEAR} (May–Sep, n={n_total:,})",
        pad=20, fontsize=11,
    )
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.35, -0.05),
        fontsize=9, title="Wind speed",
    )
    plt.tight_layout()
    return fig


def plot_monthly_wind_speed(chii2_df):
    """
    Figure 2: Monthly mean wind speed across all summers.

    Shows the within-season pattern — useful for explaining
    why some time windows matter more than others.
    Each summer is a thin line; thick line is the overall mean.
    """
    df = _summer_chii2(chii2_df)
    df["year"] = df["timestamp_central"].dt.year
    df["month"] = df["timestamp_central"].dt.month

    # Monthly mean per year
    monthly = (
        df.groupby(["year", "month"])["WSPD"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Overall monthly mean and std across all years
    overall = (
        df.groupby("month")["WSPD"]
        .agg(["mean", "std"])
        .reset_index()
    )

    month_labels = ["May", "Jun", "Jul", "Aug", "Sep"]
    months = [5, 6, 7, 8, 9]
    years = sorted(df["year"].unique())

    fig, ax = plt.subplots(figsize=(8, 4))

    # Individual years
    for yr in years:
        sub = monthly[monthly["year"] == yr]
        sub = sub.set_index("month").reindex(months)
        ax.plot(
            months, sub["mean"].values,
            color="steelblue", alpha=0.35, linewidth=1.2,
            label=str(yr) if yr == years[0] else None,
        )

    # Overall mean ± 1 std
    ov = overall.set_index("month").reindex(months)
    ax.plot(months, ov["mean"].values, color="steelblue",
            linewidth=2.5, label="Mean all years")
    ax.fill_between(
        months,
        ov["mean"].values - ov["std"].values,
        ov["mean"].values + ov["std"].values,
        color="steelblue", alpha=0.15, label="±1 std",
    )

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(
        f"CHII2 Buoy Monthly Mean Wind Speed\n"
        f"Summers {MIN_YEAR}–{MAX_YEAR}",
        fontsize=11,
    )
    # Add individual year labels as a note
    ax.text(
        0.98, 0.05,
        f"Thin lines = individual summers ({', '.join(str(y) for y in years)})",
        transform=ax.transAxes, fontsize=8, ha="right", color="steelblue", alpha=0.7,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_onshore_frequency(
    chii2_df,
    beach_geo_df,
    beaches,
    tolerance_deg=20,
):
    """
    Figure 3: Fraction of summer hours with onshore wind per beach.

    Onshore = CHII2 WDIR within ±tolerance of each beach's facing direction.
    Shows which beaches are most frequently exposed to lake-driven wind
    — directly connects CHII2 data to the groin interaction feature.

    Beaches ordered north to south (by Mattheus beach number).
    """
    df = _summer_chii2(chii2_df)
    df = df.dropna(subset=["WDIR"])

    # Compute onshore fraction per beach
    records = []
    for beach in beaches:
        if beach not in beach_geo_df.index:
            continue
        facing = beach_geo_df.loc[beach, "facing_az_deg"]
        if pd.isna(facing):
            continue
        delta = (df["WDIR"] - facing + 180) % 360 - 180
        frac = (delta.abs() <= tolerance_deg).mean()
        mattheus_id = beach_geo_df.loc[beach, "mattheus_id"]
        records.append({
            "beach": beach,
            "frac_onshore": frac,
            "facing_az_deg": facing,
            "mattheus_id": mattheus_id,
        })

    result = (
        pd.DataFrame(records)
        .sort_values("mattheus_id")  # north to south per Mattheus numbering
    )

    # Expected baseline: tolerance_deg * 2 / 360
    baseline = tolerance_deg * 2 / 360

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(result))
    bars = ax.bar(
        x, result["frac_onshore"] * 100,
        color="steelblue", alpha=0.8, width=0.6,
    )
    ax.axhline(
        baseline * 100, color="red", linestyle="--", linewidth=1.2,
        label=f"Random baseline (±{tolerance_deg}° = {baseline*100:.1f}%)",
    )

    # Label facing angle on each bar
    for i, row in enumerate(result.itertuples()):
        ax.text(
            i, row.frac_onshore * 100 + 0.3,
            f"{row.facing_az_deg:.0f}°",
            ha="center", va="bottom", fontsize=7, color="steelblue",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        result["beach"].str.replace("_", "\n"),
        rotation=0, fontsize=8, ha="center",
    )
    ax.set_ylabel("Hours with onshore wind (%)")
    ax.set_title(
        f"Frequency of Onshore Wind by Beach\n"
        f"CHII2 Buoy, Summers {MIN_YEAR}–{MAX_YEAR}, "
        f"tolerance ±{tolerance_deg}° (north → south)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(result["frac_onshore"].max() * 100 * 1.2, baseline * 100 * 2))
    plt.tight_layout()
    return fig
