"""
Per-beach geomorphic features from Mattheus, Braun & Theuerkauf (2022).

Citation:
    Mattheus, C.R., Braun, K.N., Theuerkauf, E.J. (2022).
    Great lakes urban pocket-beach dynamics: A GIS-based analysis of
    infrastructure-design influences on geomorphic development.
    Journal of Great Lakes Research, 48(1), 68-83.
    DOI: 10.1016/j.jglr.2021.10.020

Source values:
    - embayment_area_m2, embayment_perimeter_m, shape_index,
      lake_interface_length_m, lake_interface_az_deg,
      downstream_groin_length_m, downstream_groin_az_deg
        --> Table 3, "Embayment characteristics by beach #"
    - behavioral_type
        --> Figure 9, "Synthesis of study insights into a general,
            conceptual classification of major geomorphic pocket-beach types"
        - "type1": NE-facing system with no protection to north
                   (open lead-up; counterclockwise rotation under lake-level rise).
                   Example given in paper: Foster (#12).
        - "type2": NE-facing system with a groin to the north shielding it
                   (clockwise rotation under lake-level rise).
                   Example given: 31st Street (#17).
        - "type3": E-facing system with both distal ends enclosed
                   (passive inundation, no rotation).
                   Example given: Clark St. Beach (#4).
        - None: beach not classified by Mattheus, or not in the paper.

Beach key matching (NEEDS USER VERIFICATION):
    The keys in BEACH_GEOMORPHOLOGY below are the names used in the user's
    `Beach` column. The Mattheus paper uses different names for some beaches.
    Confident matches are those where Mattheus's UTM coordinates agree closely
    with the beach's known location.

    Beaches in user's data NOT in Mattheus 2022a (Table 1):
        - 12th_street: Pocket beach on Northerly Island (downtown).
                       Not included in Mattheus 22-beach set.
        - calumet:     Far southern, near IL-IN border.
                       Not included in Mattheus 22-beach set.
        - margaret_t_burroughs_31st: Likely the same physical beach as
                       31st_street (Mattheus #17). Verify with lat/lon.

    Behavioral type assignments below for non-example beaches are MY
    INFERENCES based on the paper's described patterns, not explicit
    paper assignments. Mark as "type1_inferred", "type2_inferred", etc.
    if you want to flag this distinction in modeling.

Convention notes:
    - Azimuths are clockwise from North in degrees (compass convention).
    - lake_interface_az_deg is the STRIKE of the line between groin termini.
      To get a beach's facing direction (the direction it "looks" toward
      the open lake), use add_facing_direction() below.
    - Mattheus paper says "Almost all systems face the NE quadrant, with
      strike values expressing the lake interface ranging between 107 and 192
      (clockwise from N). The NW-facing Ohio St. Beach (#16) is a notable
      exception". This confirms the strike convention.
"""

from typing import Optional

import pandas as pd


# Type alias for clarity in IDE
BeachRecord = dict


BEACH_GEOMORPHOLOGY: dict[str, BeachRecord] = {
    # ============================================================
    # NORTH-SIDE BEACHES (Rogers Park to Uptown)
    # ============================================================

    # marion_mahony_griffin_jarvis: Likely Mattheus's "Jarvis" area beach.
    # Mattheus does not name a Jarvis beach explicitly in Table 1. The
    # northernmost beaches he covers are Lighthouse (#1), Lincoln (#3),
    # Clark St. (#4), Greenwood (#6), Lee St. (#7), South Boulevard (#8),
    # Loyola (#9), and Columbia/Helen Doria (#10). These are mostly in
    # Evanston (UTM northing > 4,650,000 m, roughly latitude > 42.0).
    # Jarvis Beach is at ~42.016 N, which is in the southern part of
    # Mattheus's range. NEEDS VERIFICATION via lat/lon comparison.
    "marion_mahony_griffin_jarvis": {
        "mattheus_id": None,  # not confidently matched
        "embayment_area_m2": None,
        "embayment_perimeter_m": None,
        "shape_index": None,
        "lake_interface_length_m": None,
        "lake_interface_az_deg": None,
        "downstream_groin_length_m": None,
        "downstream_groin_az_deg": None,
        "behavioral_type": None,
        "notes": "not confidently matched to Mattheus 2022a; verify lat/lon",
    },
    #doesn't profile this beach either
    "hartigan_albion": {
        "mattheus_id": None,  # not confidently matched
        "embayment_area_m2": None,
        "embayment_perimeter_m": None,
        "shape_index": None,
        "lake_interface_length_m": None,
        "lake_interface_az_deg": None,
        "downstream_groin_length_m": None,
        "downstream_groin_az_deg": None,
        "behavioral_type": None,
        "notes": "not confidently matched to Mattheus 2022a; verify lat/lon",
    },

    # leone: just north of loyola beach 
    "leone": {
        "mattheus_id": 9,
        "embayment_area_m2": 46746,
        "embayment_perimeter_m": 1359,
        "shape_index": 1.77,
        "lake_interface_length_m": 450,
        "lake_interface_az_deg": 138,
        "downstream_groin_length_m": 159,
        "downstream_groin_az_deg": 65,
        "behavioral_type": None,  # not assigned by Mattheus; verify from Fig 5d
        "notes": "tentative match to Mattheus #9 Loyola Beach; verify lat/lon",
    },

    # hartigan_albion: Possibly Mattheus's #10 "Columbia/Helen Doria" or
    # one of the unnamed beaches in that cluster. Hartigan Beach is in
    # Rogers Park around N. Albion Ave. NEEDS VERIFICATION.
    # "loyola": {
    #     "mattheus_id": 10,
    #     "embayment_area_m2": 76032,
    #     "embayment_perimeter_m": 1261,
    #     "shape_index": 1.29,
    #     "lake_interface_length_m": 437,
    #     "lake_interface_az_deg": 152,
    #     "downstream_groin_length_m": 243,
    #     "downstream_groin_az_deg": 73,
    #     "behavioral_type": None,
    #     "notes": "tentative match to Mattheus #10 Columbia/Helen Doria; verify lat/lon",
    # },

    # osterman: Mattheus's #11 "Kathy Osterman Beach/Hollywood Beach".
    # Confident match (named explicitly in Mattheus Table 1).
    "osterman": {
        "mattheus_id": 11,
        "embayment_area_m2": 124940,
        "embayment_perimeter_m": 1807,
        "shape_index": 1.44,
        "lake_interface_length_m": 638,
        "lake_interface_az_deg": 142,
        "downstream_groin_length_m": 170,
        "downstream_groin_az_deg": 21,
        "behavioral_type": None,
        "notes": "confident match",
    },

    # foster: Mattheus's #12 "Foster Beach". Confident match.
    # Type 1 example given in Figure 9. Rotated 21 degrees counterclockwise
    # over 2012-2019 (the largest rotation in the dataset).
    "foster": {
        "mattheus_id": 12,
        "embayment_area_m2": 208774,
        "embayment_perimeter_m": 2344,
        "shape_index": 1.45,
        "lake_interface_length_m": 986,
        "lake_interface_az_deg": 165,
        "downstream_groin_length_m": 308,
        "downstream_groin_az_deg": 82,
        "behavioral_type": "type1",
        "notes": "confident match; Type 1 example in Mattheus Fig 9",
    },

    # montrose: Mattheus's #13 "Montrose Beach". Confident match.
    # Largest beach in Chicago. Lakefill peninsula moved shoreline ~1.5 km
    # lakeward in 1930s.
    "montrose": {
        "mattheus_id": 13,
        "embayment_area_m2": 836590,
        "embayment_perimeter_m": 4531,
        "shape_index": 1.40,
        "lake_interface_length_m": 1598,
        "lake_interface_az_deg": 133,
        "downstream_groin_length_m": 509,
        "downstream_groin_az_deg": 6,
        "behavioral_type": None,  # not classified; excluded from rotation analysis
        "notes": ("confident match; excluded from rotation analysis due to "
                  "inundation and groin scour"),
    },

    # ============================================================
    # DOWNTOWN BEACHES
    # ============================================================

    # north_avenue: Mattheus's #14 "North Avenue Beach". Confident match.
    "north_avenue": {
        "mattheus_id": 14,
        "embayment_area_m2": 428704,
        "embayment_perimeter_m": 3378,
        "shape_index": 1.46,
        "lake_interface_length_m": 1347,
        "lake_interface_az_deg": 151,
        "downstream_groin_length_m": 313,
        "downstream_groin_az_deg": 46,
        "behavioral_type": None,
        "notes": ("confident match; partial 2012 LiDAR coverage so excluded "
                  "from volumetric analysis in Mattheus"),
    },

    # oak_street: Mattheus's #15 "Oak Street Beach". Confident match.
    # Highest shape index in dataset (1.79) - high width-to-depth ratio.
    "oak_street": {
        "mattheus_id": 15,
        "embayment_area_m2": 75552,
        "embayment_perimeter_m": 1747,
        "shape_index": 1.79,
        "lake_interface_length_m": 803,
        "lake_interface_az_deg": 147,
        "downstream_groin_length_m": 237,
        "downstream_groin_az_deg": 110,
        "behavioral_type": None,
        "notes": "confident match; shape index endmember (highest at 1.79)",
    },

    # ohio_street: Mattheus's #16 "Ohio Street Beach". Confident match.
    # The only NW-facing system in the paper (interface az = 238 deg).
    # Located on north side of Navy Pier, faces into harbor.
    "ohio_street": {
        "mattheus_id": 16,
        "embayment_area_m2": 51145,
        "embayment_perimeter_m": 1073,
        "shape_index": 1.34,
        "lake_interface_length_m": 423,
        "lake_interface_az_deg": 238,
        "downstream_groin_length_m": 506,
        "downstream_groin_az_deg": 39,
        "behavioral_type": None,
        "notes": ("confident match; only NW-facing system in Mattheus dataset; "
                  "partial 2012 LiDAR coverage"),
    },

    # 12th_street: NOT IN MATTHEUS 2022a.
    # Pocket beach on east side of Northerly Island, near Adler Planetarium.
    # All geomorphic features unavailable.
    "12th_street": {
        "mattheus_id": None,
        "embayment_area_m2": None,
        "embayment_perimeter_m": None,
        "shape_index": None,
        "lake_interface_length_m": None,
        "lake_interface_az_deg": None,
        "downstream_groin_length_m": None,
        "downstream_groin_az_deg": None,
        "behavioral_type": None,
        "notes": "not in Mattheus 2022a (inner harbor pocket beach)",
    },

    # ============================================================
    # SOUTH-SIDE BEACHES
    # ============================================================

    # 31st_street: Mattheus's #17 "31st Street Beach". Confident match.
    # Type 2 example given in Figure 9. The only beach to rotate clockwise
    # (+10 degrees) over 2012-2019. TAKEN OUT BECAUSE ITS A DUPLICATE AND MARGARET IS IN THE READINGS DATA
    # "31st_street": {
    #     "mattheus_id": 17,
    #     "embayment_area_m2": 36341,
    #     "embayment_perimeter_m": 800,
    #     "shape_index": 1.18,
    #     "lake_interface_length_m": 214,
    #     "lake_interface_az_deg": 110,
    #     "downstream_groin_length_m": 262,
    #     "downstream_groin_az_deg": 53,
    #     "behavioral_type": "type2",
    #     "notes": ("confident match; Type 2 example in Mattheus Fig 9; "
    #               "only clockwise-rotating beach in dataset"),
    # },

    # margaret_t_burroughs_31st: Likely the same physical beach as 31st_street, CONFIRMED by eye with maps
    # renamed in 2022 to honor Margaret T. Burroughs (founder of DuSable Museum).
    # Same geomorphic features as 31st_street.
    #CONFIRMED by eye with maps
    "margaret_t_burroughs_31st": {
        "mattheus_id": 17,
        "embayment_area_m2": 36341,
        "embayment_perimeter_m": 800,
        "shape_index": 1.18,
        "lake_interface_length_m": 214,
        "lake_interface_az_deg": 110,
        "downstream_groin_length_m": 262,
        "downstream_groin_az_deg": 53,
        "behavioral_type": "type2",
        "notes": "renamed version of 31st_street; Type 2 example in Mattheus Fig 9; only clockwise-rotating beach in dataset",
    },

    # oakwood: Mattheus's #18 "Oakwood Beach". Confident match.
    "oakwood": {
        "mattheus_id": 18,
        "embayment_area_m2": 46076,
        "embayment_perimeter_m": 982,
        "shape_index": 1.29,
        "lake_interface_length_m": 309,
        "lake_interface_az_deg": 138,
        "downstream_groin_length_m": 173,
        "downstream_groin_az_deg": 29,
        "behavioral_type": None,
        "notes": "confident match",
    },

    # 57th_street: Mattheus's #19 "57th St. Beach". Confident match.
    "57th_street": {
        "mattheus_id": 19,
        "embayment_area_m2": 64417,
        "embayment_perimeter_m": 1282,
        "shape_index": 1.43,
        "lake_interface_length_m": 548,
        "lake_interface_az_deg": 140,
        "downstream_groin_length_m": 362,
        "downstream_groin_az_deg": 116,
        "behavioral_type": None,
        "notes": "confident match",
    },

    # 63rd_street: Mattheus's #20 "63rd Street Beach". Confident match.
    # Most spherical container shape (lowest SI at 1.12).
    "63rd_street": {
        "mattheus_id": 20,
        "embayment_area_m2": 411068,
        "embayment_perimeter_m": 2539,
        "shape_index": 1.12,
        "lake_interface_length_m": 741,
        "lake_interface_az_deg": 124,
        "downstream_groin_length_m": 519,
        "downstream_groin_az_deg": 32,
        "behavioral_type": None,
        "notes": "confident match; shape index endmember (lowest at 1.12)",
    },

    # south_shore: Mattheus's #21 "South Shore Beach". Confident match.
    "south_shore": {
        "mattheus_id": 21,
        "embayment_area_m2": 36067,
        "embayment_perimeter_m": 871,
        "shape_index": 1.29,
        "lake_interface_length_m": 286,
        "lake_interface_az_deg": 107,
        "downstream_groin_length_m": 263,
        "downstream_groin_az_deg": 42,
        "behavioral_type": None,
        "notes": "confident match",
    },

    # rainbow: Mattheus's #22 "Rainbow Beach". Confident match.
    # Most southerly beach in the paper. Same lake_interface_az as
    # South Shore (107), but different downstream groin orientation.
    "rainbow": {
        "mattheus_id": 22,
        "embayment_area_m2": 226535,
        "embayment_perimeter_m": 2230,
        "shape_index": 1.32,
        "lake_interface_length_m": 635,
        "lake_interface_az_deg": 107,
        "downstream_groin_length_m": 314,
        "downstream_groin_az_deg": 292,
        "behavioral_type": None,
        "notes": "confident match",
    },

    # calumet: NOT IN MATTHEUS 2022a.
    # Far south Chicago, near IL-IN border, around 95th St.
    # All geomorphic features unavailable.
    "calumet": {
        "mattheus_id": None,
        "embayment_area_m2": None,
        "embayment_perimeter_m": None,
        "shape_index": None,
        "lake_interface_length_m": None,
        "lake_interface_az_deg": None,
        "downstream_groin_length_m": None,
        "downstream_groin_az_deg": None,
        "behavioral_type": None,
        "notes": "not in Mattheus 2022a (south of paper's southern boundary)",
    },
}


def get_geomorphology_df() -> pd.DataFrame:
    """
    Return Mattheus geomorphology features as a DataFrame.

    Returns:
        DataFrame with one row per beach, indexed by beach key
        (matching the `Beach` column in readings_df). Columns include
        all features in BEACH_GEOMORPHOLOGY plus derived features.
    """
    df = pd.DataFrame.from_dict(BEACH_GEOMORPHOLOGY, orient="index")
    df.index.name = "Beach"
    df = add_facing_direction(df)
    return df


def add_facing_direction(geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `facing_az_deg` column derived from the lake interface azimuth.

    Mattheus reports `lake_interface_az_deg` as the strike of the line
    between the groin termini (the "mouth" of the beach embayment).

    For Chicago beaches, the lake is generally to the EAST or SOUTHEAST of
    the strike line. Adding 90 degrees (clockwise from N) rotates the strike
    to point lakeward, giving the facing direction.

    Sanity check from the paper:
        - Foster Beach (#12) interface_az = 165 deg (SSE strike).
          Adding 90 -> 255 deg (WSW). That's wrong (would point inland).
        - Subtracting 90 -> 75 deg (ENE). That points lakeward. CORRECT.

    So the convention is: facing = (strike - 90) mod 360, NOT +90.

    Exception: Ohio Street (#16) has interface_az = 238 (WSW strike).
        - 238 - 90 = 148 (SSE). Ohio St faces into harbor on north side
          of Navy Pier, so a SSE-ish facing is plausible.
        - Mattheus calls it "NW-facing" though, which would be ~315.
          That doesn't match either convention cleanly. Inner-harbor
          beaches may need manual override.

    NOTE: Verify this convention with one or two beaches you know well
    before relying on facing_az_deg as a feature.
    """
    geo_df = geo_df.copy()
    interface_az = geo_df["lake_interface_az_deg"]
    geo_df["facing_az_deg"] = (interface_az - 90) % 360
    return geo_df


def list_unmatched_beaches() -> list[str]:
    """Return beach keys that have no Mattheus data (all NaN features)."""
    return [
        key for key, rec in BEACH_GEOMORPHOLOGY.items()
        if rec["mattheus_id"] is None
    ]


def list_inferred_types() -> list[str]:
    """
    Return beach keys whose behavioral_type is None.

    Most beaches in Mattheus aren't explicitly labeled with a Type
    (only Foster, 31st St, and Clark St are named in Fig 9 as examples).
    Other beaches' types must be inferred from their rotation behavior
    in Fig 5d, OR left as None.
    """
    return [
        key for key, rec in BEACH_GEOMORPHOLOGY.items()
        if rec.get("behavioral_type") is None
        and rec.get("mattheus_id") is not None
    ]