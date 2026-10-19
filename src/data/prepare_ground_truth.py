"""
Prepares the spatial target ground truth dataset.
Maps subnational multidimensional poverty headcount benchmarks to Nigerian states.
"""
from pathlib import Path
import pandas as pd

def create_state_ground_truth():
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the generated state index
    index_file = data_dir / "nigeria_states_index.csv"
    if not index_file.exists():
        raise FileNotFoundError(f"Missing {index_file}. Run notebooks/02_target_boundaries.ipynb first.")
    
    df_states = pd.read_csv(index_file)
    
    # 2. National Bureau of Statistics (NBS) / MPI state poverty headcount benchmarks (%)
    state_benchmarks = {
        'Abia': 29.8, 'Abuja': 12.3, 'Adamawa': 62.4, 'Akwa Ibom': 44.5,
        'Anambra': 22.1, 'Bauchi': 87.0, 'Bayelsa': 54.0, 'Benue': 56.7,
        'Borno': 78.2, 'Cross River': 48.9, 'Delta': 24.5, 'Ebonyi': 68.3,
        'Edo': 26.0, 'Ekiti': 28.4, 'Enugu': 43.1, 'Gombe': 78.4,
        'Imo': 32.1, 'Jigawa': 87.2, 'Kaduna': 67.5, 'Kano': 66.3,
        'Katsina': 82.5, 'Kebbi': 84.1, 'Kogi': 51.4, 'Kwara': 43.5,
        'Lagos': 8.5, 'Nasarawa': 60.2, 'Niger': 61.8, 'Ogun': 24.1,
        'Ondo': 27.2, 'Osun': 23.4, 'Oyo': 28.9, 'Plateau': 62.0,
        'Rivers': 23.8, 'Sokoto': 90.5, 'Taraba': 74.6, 'Yobe': 83.5,
        'Zamfara': 82.0
    }
    
    df_states['poverty_rate_pct'] = df_states['state_name'].map(state_benchmarks)
    
    # Impute any unmatched records with the national median
    median_val = df_states['poverty_rate_pct'].median()
    df_states['poverty_rate_pct'] = df_states['poverty_rate_pct'].fillna(median_val)
    
    out_path = data_dir / "nigeria_state_ground_truth.csv"
    df_states.to_csv(out_path, index=False)
    print(f"✓ Saved state ground truth targets ({len(df_states)} records) to {out_path}")
    print("\nSample records:")
    print(df_states.head())

if __name__ == '__main__':
    create_state_ground_truth()