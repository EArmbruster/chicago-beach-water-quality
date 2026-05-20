"""
Simple bacteria readings plots for the final 13 Mattheus beaches.

Usage in notebook:
    from readings_plots import plot_readings_timeseries, plot_exceedance_rates

    fig1 = plot_readings_timeseries(readings_df, MATTHEUS_BEACHES)
    fig2 = plot_exceedance_rates(readings_df, MATTHEUS_BEACHES)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_readings_timeseries(readings_df, beaches, threshold=1000, min_year=2021):
    """
    Plot 5: Faceted time series of bacteria readings per beach.
    One panel per beach, stacked vertically. Y-axis is log10(reading).
    Red dots = exceedances, gray dots = safe. Dashed line at threshold.
    """
    df = readings_df[
        (readings_df["Beach"].isin(beaches)) &
        (readings_df["timestamp_central"].dt.year >= min_year)
    ].copy()

    df["log_reading"] = np.log10(df["reading"].clip(lower=1))
    df["exceeds"] = df["reading"] >= threshold

    # Sort beaches by exceedance rate (highest first)
    rates = df.groupby("Beach")["exceeds"].mean().sort_values(ascending=False)
    ordered_beaches = [b for b in rates.index if b in beaches]

    n = len(ordered_beaches)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    log_thresh = np.log10(threshold)

    for ax, beach in zip(axes, ordered_beaches):
        sub = df[df["Beach"] == beach].sort_values("timestamp_central")
        safe = sub[~sub["exceeds"]]
        over = sub[sub["exceeds"]]

        ax.scatter(safe["timestamp_central"], safe["log_reading"],
                   s=8, c="gray", alpha=0.4, label="safe")
        ax.scatter(over["timestamp_central"], over["log_reading"],
                   s=15, c="red", alpha=0.7, label="exceedance")
        ax.axhline(log_thresh, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

        n_total = len(sub)
        n_over = len(over)
        rate = n_over / n_total * 100 if n_total > 0 else 0
        ax.set_ylabel(beach, fontsize=8, rotation=0, ha="right", va="center")
        ax.text(0.99, 0.85, f"{n_over}/{n_total} ({rate:.0f}%)",
                transform=ax.transAxes, fontsize=7, ha="right")
        ax.set_ylim(0, 5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="y", labelsize=7)

    axes[0].set_title("Bacteria readings by beach (log₁₀ scale, red = exceedance ≥ 1000 CCE)")
    axes[-1].set_xlabel("Date")

    # Add y-axis scale reference on first panel
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(["1", "10", "100", "1k", "10k"], fontsize=7)

    plt.tight_layout()
    return fig


def plot_exceedance_rates(readings_df, beaches, threshold=1000, min_year=2021):
    """
    Plot 6: Exceedance rate bar chart per beach.
    Ordered by rate. Shows fraction of samples exceeding threshold.
    """
    df = readings_df[
        (readings_df["Beach"].isin(beaches)) &
        (readings_df["timestamp_central"].dt.year >= min_year)
    ].copy()

    df["exceeds"] = df["reading"] >= threshold

    stats = (
        df.groupby("Beach")
        .agg(
            n_total=("exceeds", "size"),
            n_exceed=("exceeds", "sum"),
        )
        .assign(rate=lambda x: x["n_exceed"] / x["n_total"])
        .sort_values("rate", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(stats))
    bars = ax.barh(y, stats["rate"], color="steelblue", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(stats.index, fontsize=9)
    ax.set_xlabel("Exceedance rate (fraction of samples ≥ 1000 CCE)")
    ax.set_title("Exceedance rates by beach (2021+)")

    # Add count labels on bars
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row["rate"] + 0.005, i,
                f"{int(row['n_exceed'])}/{int(row['n_total'])}",
                va="center", fontsize=8)

    ax.set_xlim(0, stats["rate"].max() * 1.2)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig
