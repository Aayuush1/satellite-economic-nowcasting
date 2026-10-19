# Remote Sensing Economic Nowcasting Pipeline (Nigeria Admin-1 & Admin-2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible, production-grade remote sensing and spatial econometric pipeline to nowcast subnational economic welfare (poverty headcount rates) across Nigeria's **37 Admin-1 States** and **775 Admin-2 Local Government Areas (LGAs)** using multi-sensor Earth Observation (EO) data.

---

## 1. Methodological Architecture

Subnational surveys (e.g., National Bureau of Statistics / Multidimensional Poverty Index) are costly, infrequent, and rarely available at fine administrative granularities. This pipeline bridges this gap by fusing multi-sensor satellite imagery with spatial econometric models:

```text
+-----------------------------------------------------------------------------------+
|                           Google Earth Engine Ingestion                           |
|  - NOAA VIIRS DNB (Annual Median Nighttime Radiance, 500m)                        |
|  - MODIS Terra MOD13Q1 (16-Day NDVI Annual Mean, 250m)                            |
|  - Copernicus Sentinel-2 L2A (Dry-Season BOA Reflectance: NDBI, MNDWI, BSI, 10m)  |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                           Server-Side Spatial Reduction                           |
|  - FAO GAUL Level-1 (37 States) & Level-2 (775 LGAs)                              |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                        Econometric & Machine Learning Engine                      |
|  - Spatial Leave-One-Geopolitical-Zone-Out Cross-Validation                       |
|  - OLS, Ridge, Random Forest Baselines (R² ≈ 0.81 under Spatial CV)               |
|  - Spatial Lag Model (SAR via Queen Contiguity W, Moran's I = 0.734, p < 0.0001)  |
|  - Balanced Panel Fixed Effects Estimation (2018–2023, N=37, T=6)                 |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                                 Delivery & Outputs                                |
|  - Subnational 775-LGA Economic Welfare Predictions                               |
|  - Interactive Web GIS Choropleth (Leaflet / Folium + VIIRS Tile Layers)          |
+-----------------------------------------------------------------------------------+
```

---

## 2. Key Empirical Findings

1. **Physical Remote Sensing Signals:**
   * $\log(\text{VIIRS Nighttime Radiance})$ correlates inversely with poverty headcount ($r = -0.63, p < 0.0001$).
   * MODIS NDVI shows strong agricultural productivity mitigation ($r = -0.74, p < 0.0001$).
2. **Spatial Autocorrelation & Generalization:**
   * **Global Moran's $I = 0.734$ ($p < 0.0001$):** Strong spatial clustering confirms that standard random cross-validation yields overly optimistic performance due to spatial leakage.
   * **Spatial Leave-One-Geopolitical-Zone-Out CV:** The multi-sensor model achieves a spatial out-of-region $R^2 \approx 0.81$ with an RMSE of $\sim 10.6$ percentage points.
3. **Macro Structural Shifts (Panel 2018–2023):**
   * Panel fixed-effects models reject naive pooling ($F = 89.04, p < 0.0001$).
   * Captures major structural shocks, notably gas flaring reduction mandates in the Niger Delta (Bayelsa $-55\%$, Akwa Ibom $-42\%$).

---

## 3. Repository Structure

```text
├── config/
│   └── pipeline.yaml               # Declarative sensor, boundary, and model parameters
├── data/
│   ├── processed/                  # Cleaned multi-sensor matrices & nowcast outputs
│   └── raw/                        # Ground-truth benchmarks & macro indicators
├── docs/                           # Publication figures & interactive HTML maps
├── notebooks/                      # Exploratory spatial analysis & modeling notebooks
│   ├── 01_first_image.ipynb
│   ├── 02_target_boundaries.ipynb
│   ├── 03_extract_features.ipynb
│   ├── 04_exploratory_analysis.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_lga_spatial_analysis.ipynb
│   ├── 07_multisensor_modeling.ipynb
│   └── 08_panel_econometrics.ipynb
├── src/
│   ├── data/                       # Earth Engine spatial feature extractors
│   │   ├── extract_satellite_features.py
│   │   ├── extract_sentinel2_features.py
│   │   ├── extract_lga_features.py
│   │   ├── extract_panel_features.py
│   │   └── build_dataset.py
│   ├── models/                     # Spatial CV, SAR, and inference pipelines
│   │   ├── nowcast_lga.py
│   │   └── spatial_econometrics.py
│   └── visualization/              # Folium web GIS rendering
│       └── visualize_lga_map.py
├── main.py                         # Unified master CLI orchestrator
└── requirements.txt                # Pinned virtual environment dependencies
```

---

## 4. Setup and Quickstart

### Prerequisites

* Python 3.10+
* Google Earth Engine account, authenticated (`gcloud` or `earthengine authenticate`)

### Installation

```bash
git clone https://github.com/Aayuush1/satellite-economic-nowcasting.git
cd satellite-economic-nowcasting
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### CLI Execution

Run modular steps or execute end-to-end:

```bash
# Display all options
python main.py --help

# Execute spatial autocorrelation tests (Moran's I & SAR Spatial Lag)
python main.py --spatial-test

# Generate granular 775-LGA poverty predictions and compile interactive map
python main.py --nowcast-lga --visualize

# Execute full pipeline end-to-end
python main.py --run-all
```