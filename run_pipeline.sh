#!/bin/bash
# Simple runner script for the energy pipeline

echo "=========================================="
echo "Energy Feature Extraction Pipeline Runner"
echo "=========================================="

# Set up environment
echo "Setting up environment..."
bash setup_environment.sh

# Run the pipeline
echo "Running pipeline..."
echo "This may take a few minutes to download OSM data..."
echo ""

# Run with different options
if [ "$1" == "quick" ]; then
    echo "Running quick test with small area..."
    python3 pipeline.py --bbox 8.410 49.010 8.415 49.015 --tile-size 50
elif [ "$1" == "full" ]; then
    echo "Running full pipeline..."
    python3 pipeline.py
elif [ "$1" == "test" ]; then
    echo "Running test mode..."
    python3 -c "
from pipeline import EnergyFeatureExtractor
extractor = EnergyFeatureExtractor(bbox=(8.410, 49.010, 8.415, 49.015))
print('✓ Extractor initialized successfully')
    "
else
    echo "Running standard pipeline for KIT North Campus..."
    python3 pipeline.py
fi

echo ""
echo "=========================================="
echo "Pipeline execution complete!"
echo "=========================================="
echo ""
echo "Output files are in ./output/"
echo "To view results:"
echo "  - Open energy_features_map.png for visualization"
echo "  - Check feature_matrix.csv for numerical results"
echo "  - View tile_map.geojson in QGIS or online GeoJSON viewer"