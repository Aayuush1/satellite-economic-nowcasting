"""
Extracts multi-sensor remote sensing features across Nigerian administrative boundaries
using Google Earth Engine.
Features:
- NOAA VIIRS Annual Median Nighttime Radiance
- MODIS Terra 16-Day NDVI Annual Mean
"""
from pathlib import Path
import ee
import pandas as pd

PROJECT_ID = 'satellite-economic-nowcasting'

def extract_features(year: int = 2023) -> pd.DataFrame:
    print(f"Initializing Earth Engine (Project: {PROJECT_ID})...")
    ee.Initialize(project=PROJECT_ID)

    # 1. Load Admin-1 Boundaries (Nigeria States)
    states = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
    )
    print("Loaded 37 state boundaries from FAO GAUL.")

    # 2. Process Nighttime Lights (NOAA VIIRS)
    print(f"Aggregating NOAA VIIRS Nighttime Radiance for {year}...")
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .select('avg_rad')
        .median()
        .rename('viirs_mean_radiance')
    )

    # 3. Process Vegetation Index (MODIS 250m 16-day NDVI)
    print(f"Aggregating MODIS NDVI for {year}...")
    modis_ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .select('NDVI')
        .mean()
        .multiply(0.0001)  # Scale factor for MODIS NDVI
        .rename('modis_mean_ndvi')
    )

    # 4. Combine Bands into a Single Multimodal Image
    composite = viirs.addBands(modis_ndvi)

    # 5. Spatial Reduction over Administrative Polygons
    print("Computing zonal statistics across all states...")
    reduced_features = composite.reduceRegions(
        collection=states,
        reducer=ee.Reducer.mean(),
        scale=1000  # 1km aggregation scale
    )

    # 6. Extract tabular attributes
    extracted = reduced_features.reduceColumns(
        reducer=ee.Reducer.toList(3),
        selectors=['ADM1_NAME', 'viirs_mean_radiance', 'modis_mean_ndvi']
    ).getInfo()['list']

    df = pd.DataFrame(
        extracted,
        columns=['state_name', 'viirs_mean_radiance', 'modis_mean_ndvi']
    )
    df['year'] = year
    return df

if __name__ == '__main__':
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_features = extract_features(year=2023)
    out_file = out_dir / "nigeria_satellite_features_2023.csv"
    df_features.to_csv(out_file, index=False)

    print(f"\n✓ Successfully extracted features for {len(df_features)} states.")
    print(f"Saved to: {out_file}\n")
    print(df_features.sort_values('viirs_mean_radiance', ascending=False).head(10))