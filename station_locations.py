"""
Coordinates for the precipitation stations used as feature sources.

Why this is a module and not a CSV read
---------------------------------------
Both source files are excluded by .gitignore (`data/**/*.csv`), so any code that
reads them at runtime breaks on a fresh clone of this repo. Baking the values in
here, with provenance recorded, is the same approach `beach_geomorphology.py`
takes for the Mattheus values.

Beach coordinates are deliberately NOT duplicated here. They live in
`data/prepared_data/dna_readings.parquet`, which IS tracked in git and is
already loaded by the pipeline. Use `get_beach_coords(readings_df)` instead, so
there is exactly one source of truth for where a beach is.

Sources
-------
COC weather stations
    data/Beach_Water_and_Weather_Sensor_Locations_20260423.csv
    City of Chicago open data portal, export dated 2026-04-23.
    Rows where "Sensor Type" == "Weather" (the "Water" rows are beach water
    sensors, not weather stations, and are not precip sources).

USGS rain gauges
    data/fav_USGS_location.csv
    Columns `y` (latitude) and `x` (longitude), NAD83.
    Only the eight `RAIN GAGE AT ...` sites are included. That file also lists
    stream gauges (North Branch Chicago River, CSSC, etc.) which measure
    discharge, not precipitation, and are not feature sources.

Naming
------
The dict keys are the slugs already used in the RECIPE station lists
(`foster`, `south_shore`, ...), so a recipe can be resolved without any
name translation. Each record keeps `source_name` -- the verbatim name in the
source file -- so a value can be traced back to its origin.

Convention: coordinates are (latitude, longitude) in decimal degrees, WGS84 /
NAD83 (the difference is well under a metre here and irrelevant at the
kilometre scale these distances are used for).
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


StationRecord = dict


STATION_LOCATIONS: dict[str, dict[str, StationRecord]] = {
    # ── City of Chicago beach weather stations ──────────────────────────────
    # Hourly precipitation. All three sit on the lakefront.
    "coc": {
        "foster": {
            "lat": 41.976464,
            "lon": -87.647525,
            "source_name": "Foster Weather Station",
            "site_id": None,
        },
        "oak_street": {
            "lat": 41.901997,
            "lon": -87.622817,
            "source_name": "Oak Street Weather Station",
            "site_id": None,
        },
        "63rd_street": {
            "lat": 41.780992,
            "lon": -87.572619,
            "source_name": "63rd Street Weather Station",
            "site_id": None,
        },
    },

    # ── USGS rain gauges ────────────────────────────────────────────────────
    # 5-minute precipitation. Mostly inland, spread across the city and
    # north suburbs -- which is why distance-based selection matters.
    "usgs": {
        "south_shore": {
            "lat": 41.757500,
            "lon": -87.543889,
            "source_name": "RAIN GAGE AT SOUTH SHORE, IL",
            "site_id": "414527087323801",
        },
        "hamilton_park_at_chicago": {
            "lat": 41.761667,
            "lon": -87.635833,
            "source_name": "RAIN GAGE AT HAMILTON PARK AT CHICAGO, IL",
            "site_id": "414542087380901",
        },
        "ping_tom_park_at_chicago": {
            "lat": 41.855000,
            "lon": -87.636667,
            "source_name": "RAIN GAGE AT PING TOM PARK AT CHICAGO, IL",
            "site_id": "415118087381201",
        },
        "cicero": {
            "lat": 41.865278,
            "lon": -87.755278,
            "source_name": "RAIN GAGE AT CICERO, IL",
            "site_id": "415155087451901",
        },
        "north_austin": {
            "lat": 41.916111,
            "lon": -87.772778,
            "source_name": "RAIN GAGE AT NORTH AUSTIN, IL",
            "site_id": "415458087462201",
        },
        "lake_view": {
            "lat": 41.949167,
            "lon": -87.639167,
            "source_name": "RAIN GAGE AT LAKE VIEW, IL",
            "site_id": "415657087382101",
        },
        "oakton_college_at_skokie": {
            "lat": 42.021944,
            "lon": -87.747500,
            "source_name": "RAIN GAGE AT OAKTON COLLEGE AT SKOKIE, IL",
            "site_id": "420119087445101",
        },
        "winnetka_golf_club_at_winnetka": {
            "lat": 42.107222,
            "lon": -87.756389,
            "source_name": "RAIN GAGE AT WINNETKA GOLF CLUB AT WINNETKA, IL",
            "site_id": "420626087452301",
        },
    },
}


EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points.

    Great-circle rather than planar because the beaches span ~0.4 degrees of
    latitude; a flat approximation is fine here but costs nothing to avoid.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def get_station_df(source: str | None = None) -> pd.DataFrame:
    """Station coordinates as a DataFrame indexed by station slug.

    Args:
        source: "coc" or "usgs" to restrict to one source; None for all,
                in which case a `source` column distinguishes them.

    Mirrors `beach_geomorphology.get_geomorphology_df()`.
    """
    if source is not None:
        _require_source(source)
        sources = {source: STATION_LOCATIONS[source]}
    else:
        sources = STATION_LOCATIONS

    rows = []
    for src, stations in sources.items():
        for slug, rec in stations.items():
            rows.append({"station": slug, "source": src, **rec})

    df = pd.DataFrame(rows).set_index("station")
    return df


def get_beach_coords(readings_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Map beach name -> (lat, lon), read from the readings dataframe.

    Beach coordinates are constant per beach in `dna_readings.parquet`, so the
    first non-null row for each beach is taken. Beaches missing coordinates are
    omitted rather than returned as NaN -- a caller asking for a distance to a
    beach with no location should get a KeyError, not a silent NaN that
    propagates into a comparison and quietly evaluates False.
    """
    coords = (
        readings_df.dropna(subset=["Latitude", "Longitude"])
        .groupby("Beach")[["Latitude", "Longitude"]]
        .first()
    )
    return {b: (row.Latitude, row.Longitude) for b, row in coords.iterrows()}


def validate_station_names(names: Iterable[str], source: str) -> list[str]:
    """Check station slugs against the registry, raising on anything unknown.

    Raises rather than skipping because a silently dropped station is
    indistinguishable from a station legitimately filtered out by a distance
    threshold: both just produce a smaller feature set with no error. A typo in
    a recipe should fail loudly.

    Returns the names as a list so this can be used inline.
    """
    _require_source(source)
    known = STATION_LOCATIONS[source]
    names = list(names)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise KeyError(
            f"Unknown {source!r} station(s): {sorted(unknown)}. "
            f"Known: {sorted(known)}"
        )
    return names


def station_distances_km(
    beach: str,
    source: str,
    beach_coords: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Distance from one beach to every station of a source, in km.

    Returns {station_slug: distance_km}, unsorted.
    """
    _require_source(source)
    if beach not in beach_coords:
        raise KeyError(
            f"No coordinates for beach {beach!r}. "
            f"Known: {sorted(beach_coords)}"
        )
    blat, blon = beach_coords[beach]
    return {
        slug: haversine_km(blat, blon, rec["lat"], rec["lon"])
        for slug, rec in STATION_LOCATIONS[source].items()
    }


def _require_source(source: str) -> None:
    if source not in STATION_LOCATIONS:
        raise KeyError(
            f"Unknown source {source!r}. Known: {sorted(STATION_LOCATIONS)}"
        )
