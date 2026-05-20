"""
Diagnostic plots for comparing wind data across stations.

Drop this in the feature notebook after the wind dataframes are loaded.
Concatenates COC and CHII2 dfs, then produces a 4-panel figure:
  1. Coverage over time per station (when does each have data?)
  2. Wind speed distribution per station
  3. Wind direction polar plot per station (where wind comes FROM)
  4. Wind speed time series (overlapping period only, daily means)

Usage:
    plot_wind_diagnostics(beach_coc_wind_df, chii2_wind_df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_wind_diagnostics(coc_df, chii2_df, year_range=(2021, 2026)):
    """
    Generate diagnostic plots comparing wind data across stations.

    Args:
        coc_df: COC weather df (foster, oak_street, 63rd_street).
                Expected cols: station_name, WSPD, GST, WDIR, timestamp_central.
        chii2_df: CHII2 buoy df.
                  Expected cols: station_name, WSPD, GST, WDIR, timestamp_central.
        year_range: (min_year, max_year) inclusive, for filtering.
    """
    # Combine into one long df for easier plotting
    cols = ["station_name", "WSPD", "GST", "WDIR", "timestamp_central"]
    all_wind = pd.concat([coc_df[cols], chii2_df[cols]], ignore_index=True)

    # Filter to the years we care about
    all_wind = all_wind[
        (all_wind["timestamp_central"].dt.year >= year_range[0]) &
        (all_wind["timestamp_central"].dt.year <= year_range[1])
    ].copy()

    # Filter to summer only (May-Sep), since that's the modeling window
    all_wind = all_wind[all_wind["timestamp_central"].dt.month.between(5, 9)].copy()

    stations = sorted(all_wind["station_name"].unique())
    print(f"Stations: {stations}")
    print(f"Date range (summer months only): "
          f"{all_wind['timestamp_central'].min()} to "
          f"{all_wind['timestamp_central'].max()}")
    print(f"Rows per station:")
    print(all_wind.groupby("station_name").size().to_string())

    # Use a consistent color per station
    colors = plt.cm.tab10(np.linspace(0, 1, len(stations)))
    color_map = dict(zip(stations, colors))

    fig = plt.figure(figsize=(14, 10))

    # ----- Panel 1: Coverage over time (top left, wide) -----
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    for i, st in enumerate(stations):
        sub = all_wind[all_wind["station_name"] == st]
        # Resample to daily counts to avoid plotting 250k points
        daily = sub.set_index("timestamp_central").resample("D").size()
        # Plot as a horizontal "rug" — y position = station index, marker only when data
        present = daily[daily > 0].index
        ax1.scatter(present, [i] * len(present), s=2, color=color_map[st], alpha=0.6)
    ax1.set_yticks(range(len(stations)))
    ax1.set_yticklabels(stations)
    ax1.set_title("Data coverage by station (summer months only, daily presence)")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.3)

    # ----- Panel 2: Wind speed distribution (middle left) -----
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    for st in stations:
        sub = all_wind[all_wind["station_name"] == st]
        wspd = sub["WSPD"].dropna()
        if len(wspd) > 0:
            ax2.hist(
                wspd, bins=40, alpha=0.4, label=st,
                color=color_map[st], density=True,
            )
    ax2.set_xlabel("Wind speed (m/s)")
    ax2.set_ylabel("Density")
    ax2.set_title("Wind speed distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ----- Panel 3: Wind direction polar plot (middle right) -----
    ax3 = plt.subplot2grid((3, 2), (1, 1), projection="polar")
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)  # clockwise (compass convention)

    for st in stations:
        sub = all_wind[all_wind["station_name"] == st]
        wdir = sub["WDIR"].dropna()
        if len(wdir) == 0:
            continue
        # Histogram of directions in 10-degree bins
        bins = np.linspace(0, 360, 37)  # 36 bins of 10 degrees
        counts, edges = np.histogram(wdir, bins=bins)
        # Plot as a line (one polygon per station)
        bin_centers = np.deg2rad((edges[:-1] + edges[1:]) / 2)
        # Normalize to density so stations with different sample counts compare
        density = counts / counts.sum()
        # Close the polygon
        theta = np.concatenate([bin_centers, [bin_centers[0]]])
        rho = np.concatenate([density, [density[0]]])
        ax3.plot(theta, rho, label=st, color=color_map[st], linewidth=1.5)

    ax3.set_title("Wind direction distribution (where wind comes FROM)", pad=20)
    ax3.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # ----- Panel 4: Daily mean wind speed time series (bottom, wide) -----
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    for st in stations:
        sub = all_wind[all_wind["station_name"] == st]
        if len(sub) == 0:
            continue
        daily_mean = (
            sub.set_index("timestamp_central")["WSPD"]
            .resample("D").mean()
            .dropna()
        )
        ax4.plot(
            daily_mean.index, daily_mean.values,
            label=st, color=color_map[st], alpha=0.7, linewidth=0.8,
        )
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Daily mean wind speed (m/s)")
    ax4.set_title("Daily mean wind speed (summer months)")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_uv_agreement(coc_df, chii2_df, year_range=(2021, 2026)):
    """
    Scatter plot showing whether stations agree on u/v wind components
    at the same timestamps. Strong positive correlation = stations see
    the same wind. Use this to decide if you really need all 4 sources.

    Compares each COC station against CHII2 (the buoy as ground truth).
    """
    # Compute u, v from speed and direction
    # Convention: u = east-component, v = north-component
    # WDIR is direction wind comes FROM (meteorological convention)
    # so wind blowing toward East has WDIR=270 (from West)
    def add_uv(df):
        df = df.copy()
        rad = np.deg2rad(df["WDIR"])
        df["u"] = -df["WSPD"] * np.sin(rad)  # east component
        df["v"] = -df["WSPD"] * np.cos(rad)  # north component
        return df

    cols = ["station_name", "WSPD", "GST", "WDIR", "timestamp_central"]
    coc = add_uv(coc_df[cols])
    chii2 = add_uv(chii2_df[cols])

    # Filter
    for d in (coc, chii2):
        mask = (
            (d["timestamp_central"].dt.year >= year_range[0])
            & (d["timestamp_central"].dt.year <= year_range[1])
            & (d["timestamp_central"].dt.month.between(5, 9))
        )
        d.drop(d.index[~mask], inplace=True)

    # Resample CHII2 to hourly (it's 10-min) so it joins with COC (hourly)
    chii2_hourly = (
        chii2.set_index("timestamp_central")[["u", "v", "WSPD"]]
        .resample("h").mean()
        .reset_index()
    )

    coc_stations = sorted(coc["station_name"].unique())
    fig, axes = plt.subplots(2, len(coc_stations), figsize=(4 * len(coc_stations), 7))
    if len(coc_stations) == 1:
        axes = axes.reshape(2, 1)

    for j, st in enumerate(coc_stations):
        coc_st = coc[coc["station_name"] == st][
            ["timestamp_central", "u", "v"]
        ].rename(columns={"u": "u_coc", "v": "v_coc"})
        merged = coc_st.merge(
            chii2_hourly.rename(columns={"u": "u_chii2", "v": "v_chii2"}),
            on="timestamp_central", how="inner",
        )
        merged = merged.dropna(subset=["u_coc", "u_chii2", "v_coc", "v_chii2"])

        if len(merged) == 0:
            for k in range(2):
                axes[k, j].text(0.5, 0.5, "no overlap", ha="center", va="center")
                axes[k, j].set_title(f"{st}")
            continue

        # u-component
        ax = axes[0, j]
        ax.scatter(merged["u_chii2"], merged["u_coc"], s=2, alpha=0.3)
        lim = max(abs(merged["u_chii2"]).max(), abs(merged["u_coc"]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=0.8)
        r_u = merged[["u_coc", "u_chii2"]].corr().iloc[0, 1]
        ax.set_title(f"{st}\nu (east-component), r={r_u:.2f}, n={len(merged)}")
        ax.set_xlabel("CHII2 u (m/s)")
        ax.set_ylabel(f"{st} u (m/s)")
        ax.grid(True, alpha=0.3)

        # v-component
        ax = axes[1, j]
        ax.scatter(merged["v_chii2"], merged["v_coc"], s=2, alpha=0.3)
        lim = max(abs(merged["v_chii2"]).max(), abs(merged["v_coc"]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=0.8)
        r_v = merged[["v_coc", "v_chii2"]].corr().iloc[0, 1]
        ax.set_title(f"v (north-component), r={r_v:.2f}")
        ax.set_xlabel("CHII2 v (m/s)")
        ax.set_ylabel(f"{st} v (m/s)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig