# Chicago Beach Water Quality Prediction

Classification to predict bacterial contamination (E. coli) at Chicago's public beaches using precipitation, wind, and beach geomorphology features.

## Motivation

Predicting E. coli levels that allow for planning ahead for safe swimming.

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| [City of Chicago Beach Lab Data](https://data.cityofchicago.org/) | DNA-based bacteria readings (qPCR) | 2015-2025 |
| [City of Chicago Weather Stations](https://data.cityofchicago.org/) | Hourly precipitation and wind at beach stations | 2015-2026 |
| [USGS Water Resources](https://waterdata.usgs.gov/) | 5-minute precipitation from gauge network | 2021-2025 |
| [NOAA CHII2 Buoy](https://www.ndbc.noaa.gov/station_page.php?station=chii2) | 10-minute wind speed/direction from Chicago crib | 2021-2025 |
| [MWRD GeoHub](https://geohub.mwrd.org/pages/cso) | Combined Sewer Overflow events | 2021-2025 |
| Mattheus et al. (2022) | Beach geomorphology: facing azimuth, groin lengths, shape index | Static |

## Methodology

### Feature Engineering

Features are built using a modular recipe configuration pattern, with each experiment specifying the data sources, time windows, and feature types as a portable dictionary:

Precipitation features: Cumulative rainfall (sum) and peak intensity (max) at multiple stations over 6h, 24h, 48h, and 72h lookback windows
Wind features: Mean/max wind speed, onshore and alongshore wind components decomposed relative to each beach's facing direction (Madani & Seth, 2020)
Geomorphology interaction: Groin-adjusted onshore wind exposure combining downstream groin length with onshore wind fraction and speed (Mattheus et al., 2022)

### Classification Models

- Logistic Regression (baseline, L2-regularized, class-weighted)
- HistGradientBoosting, XGBoost, LightGBM
- Evaluated with time-based train/test splits (no future leakage)
- Threshold sweeps to optimize precision-recall tradeoff for rare exceedance events (~9% base rate)

## Key Results

Key results can be compared to the 2017 modeling of these same beaches: https://github.com/Chicago/clear-water
Current classification results are poor, with the beaches that have the best results giving around a 35 percent error. 
Adding Geomorphology features does not decrease classification errors.

## Continuing Work

Continuing work could be done to add more precipitation coverage or test the impact of water temperature on modeling results. Sewer overflow is no longer being pursued after only on sewer overflow event was found to have reached lake michigan (2023) for this case.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── prepared_data/          # Processed parquet files ready for modeling
│       ├── COC_precip.parquet
│       ├── COC_wind.parquet
│       ├── chii2_noaa.parquet
│       ├── dna_readings.parquet
│       └── usgs_precip.parquet
│
├── Final_feature_building.ipynb    # Main analysis: feature engineering + modeling
├── Final_plots.ipynb               # Report figures
├── Precip_temp_wind_EDA.ipynb      # Exploratory data analysis
│
├── bouy_readings.ipynb             # NOAA buoy data processing
├── pathogen_readings.ipynb         # Bacteria reading processing
├── USGS_concat_cleaning.ipynb      # USGS precipitation processing
├── beach_weather_stations.ipynb    # City of Chicago weather processing
├── combined_sewer_overflow_pluslocks.ipynb  # CSO data processing
│
├── beach_geomorphology.py          # Beach geometry data (Mattheus et al.)
├── precip_plots.py                 # Precipitation visualization utilities
├── readings_plots.py               # Bacteria reading visualizations
├── wind_diagnostics.py             # Wind data diagnostic plots
└── wind_report_figs.py             # Publication-quality wind figures
```

## How to Reproduce

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data:** The `data/prepared_data/` directory contains pre-processed parquet files. To rebuild from raw sources, run the processing notebooks in order:
   - `pathogen_readings.ipynb` (bacteria readings)
   - `bouy_readings.ipynb` (NOAA wind data)
   - `USGS_concat_cleaning.ipynb` (USGS precipitation)
   - `beach_weather_stations.ipynb` (City of Chicago weather)

3. **Run the main analysis:**
   - `Final_feature_building.ipynb` — feature engineering and model evaluation

## References

- Mattheus, C.R. et al. (2022). Geomorphological classification of Chicago's lakefront beaches. *Journal of Great Lakes Research*.
- Madani, M. & Seth, R. (2020). Wind-driven transport modeling for beach water quality.

