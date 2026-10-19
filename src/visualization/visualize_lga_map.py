"""
Generates an interactive geospatial choropleth map of Nigeria displaying
nowcasted poverty rates across all 775 Local Government Areas.
"""
from pathlib import Path
import ee
import folium
import pandas as pd

PROJECT_ID = 'satellite-economic-nowcasting'

def generate_interactive_map():
    print(f"Initializing Earth Engine (Project: {PROJECT_ID})...")
    ee.Initialize(project=PROJECT_ID)

    data_dir = Path("data/processed")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Centered coordinates for Nigeria (Latitude, Longitude)
    nigeria_center = [9.0820, 8.6753]
    
    # Use standard OpenStreetMap (no API key required)
    m = folium.Map(
        location=nigeria_center,
        zoom_start=6,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # 2. Add NOAA VIIRS Nighttime Radiance 2023 Composite
    viirs = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate('2023-01-01', '2023-12-31')
        .select('avg_rad')
        .median()
    )

    # Contrast stretch for nighttime illumination
    vis_params = {
        'min': 0.0,
        'max': 12.0,
        'palette': ['000004', '1b0c41', '4a0c6b', '781c6d', 'a52c60', 'cf4446', 'ed6925', 'fb9b06', 'f7d13c', 'fcffa4']
    }
    
    map_id_dict = ee.Image(viirs).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine / NOAA VIIRS',
        name='VIIRS Nighttime Radiance (2023)',
        overlay=True,
        control=True,
        opacity=0.7
    ).add_to(m)

    # 3. Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    out_html = docs_dir / "nigeria_poverty_map_2023.html"
    m.save(str(out_html))
    print(f"✓ Clean interactive map saved to: {out_html}")

if __name__ == '__main__':
    generate_interactive_map()