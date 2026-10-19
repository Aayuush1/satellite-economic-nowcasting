"""
Applies the calibrated multi-sensor poverty model to generate high-resolution
poverty nowcasts across all 775 Local Government Areas (Admin-2) in Nigeria.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def generate_lga_nowcasts():
    data_dir = Path("data/processed")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load macro ground-truth state training data
    df_state = pd.read_csv(data_dir / "nigeria_modeling_dataset.csv")
    df_state['log_viirs'] = np.log1p(df_state['viirs_mean_radiance'])
    
    X_train = df_state[['log_viirs', 'modis_mean_ndvi']].values
    y_train = df_state['poverty_rate_pct'].values
    
    # 2. Fit the calibrated model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Model trained. Intercept: {model.intercept_:.2f}, Coefs: {model.coef_}")
    
    # 3. Load Admin-2 (LGA) features
    lga_file = data_dir / "nigeria_lga_satellite_features_2023.csv"
    if not lga_file.exists():
        raise FileNotFoundError(f"Missing {lga_file}. Run extract_lga_features.py first.")
    
    df_lga = pd.read_csv(lga_file)
    df_lga['log_viirs'] = np.log1p(df_lga['viirs_mean_radiance'])
    
    # 4. Predict poverty headcount rates at the LGA level
    X_lga = df_lga[['log_viirs', 'modis_mean_ndvi']].values
    df_lga['predicted_poverty_pct'] = model.predict(X_lga)
    
    # Clip predictions to physically realistic bounds [0%, 100%]
    df_lga['predicted_poverty_pct'] = df_lga['predicted_poverty_pct'].clip(lower=0.0, upper=100.0)
    
    # 5. Save predictions
    out_file = data_dir / "nigeria_lga_poverty_nowcast_2023.csv"
    df_lga.to_csv(out_file, index=False)
    print(f"✓ Saved 775 LGA nowcasts to {out_file}\n")
    
    # 6. Print extreme quantiles (wealthiest vs poorest hotspots)
    print("=== Top 5 Wealthiest LGAs (Lowest Predicted Poverty) ===")
    print(df_lga.sort_values('predicted_poverty_pct')[['state_name', 'lga_name', 'predicted_poverty_pct']].head(5).to_string(index=False))
    
    print("\n=== Top 5 Poorest LGAs (Highest Predicted Poverty) ===")
    print(df_lga.sort_values('predicted_poverty_pct', ascending=False)[['state_name', 'lga_name', 'predicted_poverty_pct']].head(5).to_string(index=False))

if __name__ == '__main__':
    generate_lga_nowcasts()