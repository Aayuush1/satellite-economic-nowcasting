"""
Extracts Sentinel-2 Level-2A multi-spectral indices (NDBI, BSI, MNDWI)
across all 37 Nigerian states using Google Earth Engine cloud masking.
"""
from pathlib import Path
import ee
import pandas as pd

PROJECT_ID = 'satellite-economic-nowcasting'

def mask_s2_clouds(image):
    """Masks clouds and cirrus using the S2 QA60 bitmask band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)

def extract_sentinel2_features(year: int = 2023) -> pd.DataFrame:
    print(f"Initializing Earth Engine (Project: {PROJECT_ID})...")
    ee.Initialize(project=PROJECT_ID)

    # 1. Load Nigerian State Boundaries
    states = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
    )

    # 2. Filter Sentinel-2 L2A collection (Dry season median for clear observations)
    print(f"Filtering Sentinel-2 L2A Harmonized collection for {year}...")
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(f'{year}-01-01', f'{year}-04-30')  # Dry season minimizes cloud contamination
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds)
        .median()
    )

    # 3. Calculate Normalized Indices
    # NDBI: (B11 - B8) / (B11 + B8)
    ndbi = s2.normalizedDifference(['B11', 'B8']).rename('s2_mean_ndbi')

    # MNDWI: (B3 - B11) / (B3 + B11)
    mndwi = s2.normalizedDifference(['B3', 'B11']).rename('s2_mean_mndwi')

    # BSI: ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    bsi = s2.expression(
        '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
        {
            'B11': s2.select('B11'),
            'B4': s2.select('B4'),
            'B8': s2.select('B8'),
            'B2': s2.select('B2')
        }
    ).rename('s2_mean_bsi')

    composite = ndbi.addBands(mndwi).addBands(bsi)

    # 4. Zonal reduction over states
    print("Computing zonal statistics across all states (scale=500m)...")
    reduced = composite.reduceRegions(
        collection=states,
        reducer=ee.Reducer.mean(),
        scale=500,
        tileScale=4
    )

    extracted = reduced.reduceColumns(
        reducer=ee.Reducer.toList(4),
        selectors=['ADM1_NAME', 's2_mean_ndbi', 's2_mean_mndwi', 's2_mean_bsi']
    ).getInfo()['list']

    df = pd.DataFrame(
        extracted,
        columns=['state_name', 's2_mean_ndbi', 's2_mean_mndwi', 's2_mean_bsi']
    )
    df['year'] = year
    return df

if __name__ == '__main__':
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_s2 = extract_sentinel2_features(year=2023)
    out_file = out_dir / "nigeria_sentinel2_features_2023.csv"
    df_s2.to_csv(out_file, index=False)

    print(f"\n✓ Successfully extracted Sentinel-2 indices for {len(df_s2)} states.")
    print(f"Saved to: {out_file}\n")
    print(df_s2.head(10).to_string(index=False))
    