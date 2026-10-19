"""
Extracts multi-sensor Earth Observation features across all 775 Local Government
Areas (Admin-2 / LGAs) in Nigeria using Google Earth Engine.
Features:
- NOAA VIIRS Annual Median Nighttime Radiance
- MODIS Terra 16-Day NDVI Annual Mean
"""
from pathlib import Path
import ee
import pandas as pd

PROJECT_ID = 'satellite-economic-nowcasting'

def extract_lga_features(year: int = 2023) -> pd.DataFrame:
    print(f"Initializing Earth Engine (Project: {PROJECT_ID})...")
    ee.Initialize(project=PROJECT_ID)

    # 1. Load Nigeria Admin-2 (LGAs) boundaries from FAO GAUL
    lgas = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
    )
    total_units = lgas.size().getInfo()
    print(f"Loaded {total_units} LGA boundaries from FAO GAUL.")

    # 2. Ingest and aggregate NOAA VIIRS Nighttime Radiance for the target year
    print(f"Aggregating NOAA VIIRS Nighttime Radiance for {year}...")
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .select('avg_rad')
        .median()
        .rename('viirs_mean_radiance')
    )

    # 3. Ingest and aggregate MODIS NDVI (250m resolution)
    print(f"Aggregating MODIS NDVI for {year}...")
    modis_ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .select('NDVI')
        .mean()
        .multiply(0.0001)
        .rename('modis_mean_ndvi')
    )

    # 4. Composite bands into a multi-sensor asset
    composite = viirs.addBands(modis_ndvi)

    # 5. Compute zonal statistics over all 775 LGAs
    print("Executing server-side zonal reduction across 775 LGAs (scale=1000m)...")
    reduced = composite.reduceRegions(
        collection=lgas,
        reducer=ee.Reducer.mean(),
        scale=1000,
        tileScale=4  # Prevents Earth Engine memory exhaustion on large polygons
    )

    # 6. Extract tabular properties
    print("Formatting extracted tabular attributes...")
    extracted = reduced.reduceColumns(
        reducer=ee.Reducer.toList(5),
        selectors=['ADM1_NAME', 'ADM2_NAME', 'ADM2_CODE', 'viirs_mean_radiance', 'modis_mean_ndvi']
    ).getInfo()['list']

    df = pd.DataFrame(
        extracted,
        columns=['state_name', 'lga_name', 'lga_code', 'viirs_mean_radiance', 'modis_mean_ndvi']
    )
    df['year'] = year
    
    # Handle occasional edge water/coastal nulls by state median imputation
    df['viirs_mean_radiance'] = df.groupby('state_name')['viirs_mean_radiance'].transform(
        lambda s: s.fillna(s.median())
    )
    df['modis_mean_ndvi'] = df.groupby('state_name')['modis_mean_ndvi'].transform(
        lambda s: s.fillna(s.median())
    )
    
    return df

if __name__ == '__main__':
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_lga = extract_lga_features(year=2023)
    out_file = out_dir / "nigeria_lga_satellite_features_2023.csv"
    df_lga.to_csv(out_file, index=False)

    print(f"\n✓ Successfully extracted features for {len(df_lga)} LGAs.")
    print(f"Saved to: {out_file}\n")
    print(df_lga.head(10).to_string(index=False))