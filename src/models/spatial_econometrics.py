"""
Implements Spatial Weight Matrices (Queen Contiguity), Global Moran's I,
and Spatial Autoregressive (SAR / Spatial Lag) models using FAO GAUL boundaries
extracted directly via Google Earth Engine, with robust polygon sanitization.
"""
from pathlib import Path
import ee
import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
from esda.moran import Moran
from spreg import GM_Lag
from shapely.geometry import shape, MultiPolygon, Polygon

PROJECT_ID = 'satellite-economic-nowcasting'

def clean_to_polygon(geom):
    """Ensures input geometry contains only Polygon or MultiPolygon entities."""
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        return geom
    elif geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        flat_polys = []
        for p in polys:
            if p.geom_type == 'MultiPolygon':
                flat_polys.extend(p.geoms)
            else:
                flat_polys.append(p)
        if flat_polys:
            return MultiPolygon(flat_polys)
    # Fallback to convex hull if no polygonal components exist
    return geom.convex_hull

def get_state_geometries():
    """Extracts GeoDataFrame of Nigerian states directly from Earth Engine FAO GAUL."""
    print(f"Initializing Earth Engine (Project: {PROJECT_ID})...")
    ee.Initialize(project=PROJECT_ID)
    
    states_fc = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
    )
    
    geojson = states_fc.getInfo()
    features = []
    for f in geojson['features']:
        raw_geom = shape(f['geometry'])
        clean_geom = clean_to_polygon(raw_geom)
        props = f['properties']
        features.append({
            'state_name': props.get('ADM1_NAME'),
            'geometry': clean_geom
        })
    
    gdf = gpd.GeoDataFrame(features, geometry='geometry', crs="EPSG:4326")
    return gdf

def run_spatial_econometrics():
    data_dir = Path("data/processed")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load multi-sensor modeling dataset
    df = pd.read_csv(data_dir / "nigeria_modeling_dataset.csv")
    df['log_viirs'] = np.log1p(df['viirs_mean_radiance'])

    # 2. Extract state boundaries directly from GEE
    gdf = get_state_geometries()
    
    # Harmonize naming and merge
    gdf['state_name'] = gdf['state_name'].str.strip()
    df['state_name'] = df['state_name'].str.strip()
    
    merged_gdf = gdf.merge(df, on='state_name', how='inner')
    print(f"Matched {len(merged_gdf)} state spatial polygons.")

    # 3. Construct Queen Contiguity Spatial Weight Matrix (W)
    w = libpysal.weights.Queen.from_dataframe(merged_gdf, use_index=False)
    w.transform = 'R'  # Row-standardized
    print(f"Constructed Row-Standardized Spatial Weight Matrix W with {w.n} entities.")

    # 4. Global Moran's I Test for Spatial Autocorrelation
    y = merged_gdf['poverty_rate_pct'].values
    moran = Moran(y, w)
    print("\n=== Global Moran's I Spatial Autocorrelation Test ===")
    print(f"Moran's I Statistic : {moran.I:.4f}")
    print(f"Expected I          : {moran.EI:.4f}")
    print(f"P-value (Normality) : {moran.p_norm:.5f}")
    if moran.p_norm < 0.05:
        print("✓ Reject null hypothesis: Subnational poverty exhibits strong, positive spatial clustering.")

    # 5. Spatial Lag Model (SAR) via Generalized Method of Moments
    X = merged_gdf[['log_viirs', 'modis_mean_ndvi']].values
    sar_model = GM_Lag(
        y=y.reshape(-1, 1),
        x=X,
        w=w,
        name_y='poverty_rate_pct',
        name_x=['log_viirs', 'modis_mean_ndvi'],
        name_w='Queen_Contiguity_W'
    )

    print("\n=== Spatial Lag Model (SAR) Output ===")
    print(sar_model.summary)

if __name__ == '__main__':
    run_spatial_econometrics()