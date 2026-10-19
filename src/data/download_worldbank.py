"""
Ingests macroeconomic ground truth indicators for Nigeria from the World Bank API.
"""
import wbgapi as wb
import pandas as pd
from pathlib import Path

# Core economic indicators:
# NY.GDP.PCAP.CD: GDP per capita (current US$)
# NY.GDP.PCAP.KD.ZG: GDP per capita growth (annual %)
# EG.ELC.ACCS.ZS: Access to electricity (% of population)
# SI.POV.NAHC: Poverty headcount ratio at national poverty lines (% of population)
INDICATORS = {
    'NY.GDP.PCAP.CD': 'gdp_per_capita_usd',
    'NY.GDP.PCAP.KD.ZG': 'gdp_pc_growth_annual_pct',
    'EG.ELC.ACCS.ZS': 'electricity_access_pct',
    'SI.POV.NAHC': 'national_poverty_headcount_pct'
}

def fetch_nigeria_macro_series(start_year: int = 2013, end_year: int = 2023) -> pd.DataFrame:
    print(f"Fetching World Bank indicators for Nigeria ({start_year}-{end_year})...")
    
    records = []
    for code, clean_name in INDICATORS.items():
        data = wb.data.DataFrame(
            series=code,
            economy='NGA',
            time=range(start_year, end_year + 1),
            labels=False
        )
        # Reshape wide to long
        data = data.reset_index().melt(id_vars=['economy'], var_name='year', value_name=clean_name)
        data['year'] = data['year'].str.extract(r'(\d+)').astype(int)
        records.append(data.set_index(['economy', 'year'])[clean_name])
    
    df_combined = pd.concat(records, axis=1).reset_index()
    return df_combined

if __name__ == '__main__':
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = fetch_nigeria_macro_series()
    out_path = out_dir / "nigeria_worldbank_macro.csv"
    df.to_csv(out_path, index=False)
    
    print(f"Data saved successfully to {out_path}")
    print(df.to_string(index=False))