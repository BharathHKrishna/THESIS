#!/usr/bin/env python3
"""
Simple runner for the energy pipeline
Usage: python run_energy_pipeline.py
"""

import argparse
from pipeline import EnergyFeatureExtractor

def run_small_test():
    """Run a small test first"""
    print("Running small test area...")
    extractor = EnergyFeatureExtractor(
        bbox=(8.410, 49.010, 8.415, 49.015),  # Small area
        cache_dir="./cache_test",
        epsg=25832
    )
    return extractor.run_complete_pipeline(tile_size=50)

def run_full_campus():
    """Run full KIT campus"""
    print("Running full KIT North Campus...")
    extractor = EnergyFeatureExtractor(
        bbox=(8.4080, 49.009, 8.430, 49.020),
        cache_dir="./cache",
        epsg=25832
    )
    return extractor.run_complete_pipeline(tile_size=100)

def run_custom_area(bbox, tile_size=100):
    """Run custom area"""
    print(f"Running custom area: {bbox}")
    extractor = EnergyFeatureExtractor(
        bbox=bbox,
        cache_dir="./cache_custom",
        epsg=25832
    )
    return extractor.run_complete_pipeline(tile_size=tile_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Energy Feature Extraction Pipeline")
    parser.add_argument("--mode", choices=["test", "full", "custom"], default="full",
                       help="Run mode: test (small area), full (KIT campus), or custom")
    parser.add_argument("--bbox", type=float, nargs=4,
                       help="Custom bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--tile-size", type=int, default=100,
                       help="Tile size in meters")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "test":
            results = run_small_test()
        elif args.mode == "custom" and args.bbox:
            results = run_custom_area(tuple(args.bbox), args.tile_size)
        else:
            results = run_full_campus()
        
        print("\n✅ Pipeline completed successfully!")
        print("📊 Check ./output/ folder for all results")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check internet connection for OSM data")
        print("2. Install missing packages: pip install osmnx geopandas")
        print("3. Try smaller area first: python run_energy_pipeline.py --mode test")