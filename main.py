"""
Master CLI Orchestrator for Remote Sensing Economic Nowcasting Pipeline.
Supports modular execution of extraction, training, nowcasting, and econometric testing.
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Remote Sensing Economic Nowcasting Pipeline (Nigeria Admin-1/Admin-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Examples:
  python main.py --help
  python main.py --spatial-test
  python main.py --nowcast-lga
  python main.py --visualize
  python main.py --run-all
        """
    )

    # Individual task flags
    parser.add_argument(
        "--extract-macro",
        action="store_true",
        help="Step 1: Extract Admin-1 State Level VIIRS radiance & MODIS NDVI (2023)"
    )
    parser.add_argument(
        "--extract-s2",
        action="store_true",
        help="Step 2: Extract Admin-1 Sentinel-2 L2A optical indices (NDBI, BSI, MNDWI)"
    )
    parser.add_argument(
        "--extract-lga",
        action="store_true",
        help="Step 3: Extract Admin-2 (775 LGAs) VIIRS & MODIS features via Earth Engine"
    )
    parser.add_argument(
        "--nowcast-lga",
        action="store_true",
        help="Step 4: Train calibrated linear baseline and generate granular poverty nowcasts for 775 LGAs"
    )
    parser.add_argument(
        "--spatial-test",
        action="store_true",
        help="Step 5: Run Global Moran's I spatial autocorrelation and Queen-contiguity SAR spatial lag model"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Step 6: Render interactive Folium geospatial web map (docs/nigeria_poverty_map_2023.html)"
    )
    
    # Master batch flag
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run the complete end-to-end pipeline sequentially from extraction to interactive map"
    )

    args = parser.parse_args()

    # If the user runs `python main.py` without arguments, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(" SATELLITE ECONOMIC NOWCASTING PIPELINE (NIGERIA)")
    print("=" * 70)

    # Step 1: Macro State Feature Extraction (VIIRS + MODIS)
    if args.run_all or args.extract_macro:
        print("\n>>> [1/6] Extracting State Macro Satellite Features...")
        from src.data.extract_satellite_features import extract_all_states
        df_states = extract_all_states(2023)
        print("✓ Completed Admin-1 macro satellite extraction.")

    # Step 2: Sentinel-2 Surface Reflectance Indices
    if args.run_all or args.extract_s2:
        print("\n>>> [2/6] Extracting Sentinel-2 Optical Indices (NDBI, MNDWI, BSI)...")
        from src.data.extract_sentinel2_features import extract_sentinel2_features
        df_s2 = extract_sentinel2_features(2023)
        out_s2 = Path("data/processed/nigeria_sentinel2_features_2023.csv")
        df_s2.to_csv(out_s2, index=False)
        print(f"✓ Saved Sentinel-2 indices to: {out_s2}")

    # Step 3: Admin-2 (775 LGAs) Feature Extraction
    if args.run_all or args.extract_lga:
        print("\n>>> [3/6] Extracting 775 LGA Features via Earth Engine...")
        from src.data.extract_lga_features import extract_lga_features
        df_lga = extract_lga_features(2023)
        out_lga = Path("data/processed/nigeria_lga_satellite_features_2023.csv")
        df_lga.to_csv(out_lga, index=False)
        print(f"✓ Saved 775 LGA feature matrix to: {out_lga}")

    # Step 4: Disaggregated Poverty Inference (775 LGAs)
    if args.run_all or args.nowcast_lga:
        print("\n>>> [4/6] Executing Granular LGA Poverty Nowcast...")
        from src.models.nowcast_lga import generate_lga_nowcasts
        generate_lga_nowcasts()
        print("✓ Completed LGA poverty nowcasting.")

    # Step 5: Spatial Econometrics & Autocorrelation Testing
    if args.run_all or args.spatial_test:
        print("\n>>> [5/6] Testing Spatial Autocorrelation & SAR Lag Model...")
        from src.models.spatial_econometrics import run_spatial_econometrics
        run_spatial_econometrics()
        print("✓ Completed spatial econometric tests.")

    # Step 6: Interactive Geospatial Web Delivery
    if args.run_all or args.visualize:
        print("\n>>> [6/6] Compiling Interactive Folium Geospatial Map...")
        from src.visualization.visualize_lga_map import generate_interactive_map
        generate_interactive_map()
        print("✓ Rendered interactive map to docs/nigeria_poverty_map_2023.html")

    print("\n" + "=" * 70)
    print("✓ Pipeline execution complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()