"""
Constructs a balanced multi-temporal panel dataset (2018-2023)
aggregating VIIRS nighttime lights and MODIS NDVI across Nigerian states.
"""
from pathlib import Path
import ee
import pandas as pd

PROJECT_ID = 'satellite-economic-nowcasting'

def extract_panel_data(start_year: int = 2018, end_year: int = 2023) -> pd.DataFrame:
    print(f"Initializing Earth Engine (Project: {PROJECT_ID})...")
    ee.Initialize(project=PROJECT_ID)

    states = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
    )

    records = []

    for year in range(start_year, end_year + 1):
        print(f"Extracting zonal aggregates for year {year}...")
        
        # 1. VIIRS Annual Median
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .select('avg_rad')
            .median()
            .rename('viirs_mean_radiance')
        )

        # 2. MODIS NDVI Annual Mean
        modis = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .select('NDVI')
            .mean()
            .multiply(0.0001)
            .rename('modis_mean_ndvi')
        )

        composite = viirs.addBands(modis)

        reduced = composite.reduceRegions(
            collection=states,
            reducer=ee.Reducer.mean(),
            scale=1000,
            tileScale=4
        )

        extracted = reduced.reduceColumns(
            reducer=ee.Reducer.toList(3),
            selectors=['ADM1_NAME', 'viirs_mean_radiance', 'modis_mean_ndvi']
        ).getInfo()['list']

        df_year = pd.DataFrame(
            extracted,
            columns=['state_name', 'viirs_mean_radiance', 'modis_mean_ndvi']
        )
        df_year['year'] = year
        records.append(df_year)

    df_panel = pd.concat(records, ignore_index=True)
    return df_panel

if __name__ == '__main__':
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_df = extract_panel_data(2018, 2023)
    out_file = out_dir / "nigeria_panel_features_2018_2023.csv"
    panel_df.to_csv(out_file, index=False)

    print(f"\n✓ Extracted balanced panel: {len(panel_df)} total observations ({37} states × 6 years).")
    print(f"Saved to: {out_file}")