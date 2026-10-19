"""
Unified CLI orchestrator for Nigeria Remote Sensing Economic Nowcasting Pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> None:
    print(f"\n[INFO] Starting: {description}")
    print(f"[CMD]  {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Failed at stage: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"[SUCCESS] Completed: {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Nigeria Subnational Poverty Nowcasting Master Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--extract-macro",
        action="store_true",
        help="Extract Admin-1 VIIRS nighttime lights and MODIS NDVI via Earth Engine",
    )
    parser.add_argument(
        "--extract-s2",
        action="store_true",
        help="Extract Admin-1 Sentinel-2 multi-spectral features (NDBI, MNDWI, BSI)",
    )
    parser.add_argument(
        "--extract-lga",
        action="store_true",
        help="Extract Admin-2 (775 LGAs) VIIRS and MODIS satellite features",
    )
    parser.add_argument(
        "--extract-panel",
        action="store_true",
        help="Extract 2018-2023 balanced multi-year panel features across 37 States",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Merge raw ground truth and extracted satellite features into modeling matrices",
    )
    parser.add_argument(
        "--spatial-test",
        action="store_true",
        help="Compute Global Moran's I and run Spatial Lag Model (SAR) regressions",
    )
    parser.add_argument(
        "--nowcast-lga",
        action="store_true",
        help="Generate downscaled 775-LGA poverty headcount estimates using linear baseline",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate interactive HTML Folium choropleth map for LGA poverty",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Execute the full pipeline sequentially from ingestion to map generation",
    )

    args = parser.parse_args()

    # If no flags passed, display help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    py_exec = sys.executable

    # Pipeline stages
    if args.run_all or args.extract_macro:
        run_command([py_exec, "src/data/extract_satellite_features.py"], "Extract Admin-1 Macro Satellite Features")

    if args.run_all or args.extract_s2:
        run_command([py_exec, "src/data/extract_sentinel2_features.py"], "Extract Sentinel-2 Spectral Indices")

    if args.run_all or args.extract_lga:
        run_command([py_exec, "src/data/extract_lga_features.py"], "Extract Admin-2 LGA Features")

    if args.run_all or args.extract_panel:
        run_command([py_exec, "src/data/extract_panel_features.py"], "Extract 2018-2023 Multi-Year Panel Features")

    if args.run_all or args.build_dataset:
        run_command([py_exec, "src/data/build_dataset.py"], "Build Unified Modeling Datasets")

    if args.run_all or args.spatial_test:
        run_command([py_exec, "src/models/spatial_econometrics.py"], "Spatial Autocorrelation and SAR Testing")

    if args.run_all or args.nowcast_lga:
        run_command([py_exec, "src/models/nowcast_lga.py"], "Generate 775-LGA Poverty Nowcast")

    if args.run_all or args.visualize:
        run_command([py_exec, "src/visualization/visualize_lga_map.py"], "Render Interactive Web GIS Folium Map")


if __name__ == "__main__":
    main()