"""
Merges Earth Observation features (VIIRS Nighttime Radiance, MODIS NDVI)
with ground-truth targets (MPI poverty rates) across Nigerian states.
"""
from pathlib import Path
import pandas as pd

def build_modeling_dataset(year: int = 2023) -> pd.DataFrame:
    data_dir = Path("data/processed")
    
    features_path = data_dir / f"nigeria_satellite_features_{year}.csv"
    targets_path = data_dir / "nigeria_state_ground_truth.csv"
    
    if not features_path.exists() or not targets_path.exists():
        raise FileNotFoundError("Missing feature or target CSV files in data/processed/.")
    
    df_features = pd.read_csv(features_path)
    df_targets = pd.read_csv(targets_path)
    
    # Merge on state_name
    df_merged = pd.merge(df_features, df_targets, on="state_name", how="inner")
    
    # Select and order primary modeling columns
    columns = [
        "state_name",
        "state_code",
        "year",
        "viirs_mean_radiance",
        "modis_mean_ndvi",
        "poverty_rate_pct"
    ]
    df_final = df_merged[columns].sort_values("poverty_rate_pct")
    
    out_file = data_dir / "nigeria_modeling_dataset.csv"
    df_final.to_csv(out_file, index=False)
    
    print(f"✓ Successfully built modeling dataset with {len(df_final)} states.")
    print(f"Saved to: {out_file}\n")
    print(df_final.head(10).to_string(index=False))
    return df_final

if __name__ == "__main__":
    build_modeling_dataset()