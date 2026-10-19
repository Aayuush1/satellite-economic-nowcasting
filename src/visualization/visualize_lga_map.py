"""
Generate interactive Folium choropleth map with VIIRS night lights tile layer.
"""

from pathlib import Path
import ee
import folium
import geopandas as gpd
import pandas as pd

# Initialize Earth Engine
ee.Initialize(project="satellite-economic-nowcasting")

BASE_DIR = Path(__file__).resolve().parents[2]
NOWCAST_PATH = BASE_DIR / "data" / "processed" / "nigeria_lga_poverty_nowcast_2023.csv"
OUTPUT_HTML = BASE_DIR / "docs" / "nigeria_poverty_map_2023.html"

print("Loading 775 LGA poverty predictions...")
nowcast_df = pd.read_csv(NOWCAST_PATH)

print("Fetching GAUL Level-2 boundaries from Earth Engine...")
gaul_lga = (
    ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq("ADM0_NAME", "Nigeria"))
)

# Convert to local GeoDataFrame and assign CRS
lga_geojson = gaul_lga.getInfo()
gdf = gpd.GeoDataFrame.from_features(lga_geojson["features"])
gdf.set_crs(epsg=4326, inplace=True)

# Merge predictions on LGA and State names
gdf["ADM2_NAME_clean"] = gdf["ADM2_NAME"].astype(str).str.strip().str.lower()
gdf["ADM1_NAME_clean"] = gdf["ADM1_NAME"].astype(str).str.strip().str.lower()

nowcast_df["lga_clean"] = nowcast_df["lga_name"].astype(str).str.strip().str.lower()
nowcast_df["state_clean"] = nowcast_df["state_name"].astype(str).str.strip().str.lower()

merged = gdf.merge(
    nowcast_df,
    left_on=["ADM1_NAME_clean", "ADM2_NAME_clean"],
    right_on=["state_clean", "lga_clean"],
    how="left"
)

# Populate missing values
mean_val = float(nowcast_df["predicted_poverty_pct"].mean())
merged["predicted_poverty_pct"] = merged["predicted_poverty_pct"].fillna(mean_val).round(2)
merged["state_name"] = merged["state_name"].fillna(merged["ADM1_NAME"])
merged["lga_name"] = merged["lga_name"].fillna(merged["ADM2_NAME"])
merged = merged[["state_name", "lga_name", "predicted_poverty_pct", "geometry"]]

print("Fetching VIIRS Nighttime Radiance tiles from Earth Engine...")
viirs = (
    ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    .filterDate("2023-01-01", "2023-12-31")
    .select("avg_rad")
    .median()
)

viirs_vis = {"min": 0.0, "max": 60.0, "palette": ["000004", "1b0c41", "4a0c6b", "781c6d", "a52c60", "cf4446", "ed6925", "fb9b06", "f7d13c", "fcffa4"]}
map_id_dict = ee.data.getMapId({"image": viirs, "visParams": viirs_vis})

print("Building interactive Folium choropleth with VIIRS layer...")
m = folium.Map(location=[9.0820, 8.6753], zoom_start=6, tiles="openstreetmap")

# 1. Add VIIRS Nighttime Radiance Tile Layer
folium.TileLayer(
    tiles=map_id_dict["tile_fetcher"].url_format,
    attr="Google Earth Engine / NOAA VIIRS",
    name="VIIRS Nighttime Radiance (2023)",
    overlay=True,
    control=True,
    opacity=0.85,
).add_to(m)

# 2. Add LGA Choropleth Layer
folium.Choropleth(
    geo_data=merged.__geo_interface__,
    data=merged,
    columns=["lga_name", "predicted_poverty_pct"],
    key_on="feature.properties.lga_name",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.25,
    legend_name="Estimated Poverty Headcount Rate (%)",
    name="LGA Poverty Headcount (%)",
    highlight=True,
    overlay=True,
    control=True,
).add_to(m)

# 3. Interactive hover tooltip layer
style_fn = lambda x: {"fillColor": "#ffffff", "color": "#222222", "fillOpacity": 0.0, "weight": 0.3}
highlight_fn = lambda x: {"weight": 2, "color": "#0055ff", "fillOpacity": 0.25}

tooltip_layer = folium.GeoJson(
    merged,
    style_function=style_fn,
    highlight_function=highlight_fn,
    name="LGA Tooltips",
    tooltip=folium.GeoJsonTooltip(
        fields=["state_name", "lga_name", "predicted_poverty_pct"],
        aliases=["State:", "LGA:", "Poverty Headcount (%):"],
        localize=True,
    ),
)
tooltip_layer.add_to(m)

# Add Layer Control to toggle layers on and off
folium.LayerControl(collapsed=False).add_to(m)

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUTPUT_HTML))
print(f"[SUCCESS] Multi-layer interactive map saved to: {OUTPUT_HTML}")