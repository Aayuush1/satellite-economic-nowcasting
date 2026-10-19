"""
Merges ground-truth poverty benchmarks with multi-sensor satellite features:
- NOAA VIIRS Nighttime Radiance (500m)
- MODIS Terra NDVI (250m)
- Copernicus Sentinel-2 L2A (NDBI, MNDWI, BSI) (10-20m)
"""
from pathlib import Path
import numpy as np
import pandas as pd

def build_multisensor_dataset(year: int = 2023) -> pd.DataFrame:
    data_dir = Path("data/processed")
    
    # 1. Load existing modeling dataset (State boundaries + Ground Truth + VIIRS + MODIS)
    base_file = data_dir / "nigeria_modeling_dataset.csv"
    if not base_file.exists():
        raise FileNotFoundError(f"Missing {base_file}")
    df_base = pd.read_csv(base_file)
    
    # 2. Load Sentinel-2 features
    s2_file = data_dir / f"nigeria_sentinel2_features_{year}.csv"
    if not s2_file.exists():
        raise FileNotFoundError(f"Missing {s2_file}. Run extract_sentinel2_features.py first.")
    df_s2 = pd.read_csv(s2_file)
    
    # 3. Merge on state_name
    df_merged = pd.merge(
        df_base,
        df_s2[['state_name', 's2_mean_ndbi', 's2_mean_mndwi', 's2_mean_bsi']],
        on='state_name',
        how='left'
    )
    
    # Check for any missing values
    if df_merged.isna().sum().sum() > 0:
        print("Imputing missing values with column medians...")
        df_merged = df_merged.fillna(df_merged.median(numeric_only=True))
        
    out_path = data_dir / "nigeria_multisensor_dataset.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"✓ Consolidated multi-sensor dataset saved to: {out_path}")
    print(f"Dataset Shape: {df_merged.shape}")
    print("\nSample Preview:")
    print(df_merged[['state_name', 'viirs_mean_radiance', 'modis_mean_ndvi', 's2_mean_ndbi', 'poverty_rate_pct']].head())
    
    return df_merged

if __name__ == '__main__':
    build_multisensor_dataset(year=2023)