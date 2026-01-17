#!/bin/bash
# Setup script for Energy Feature Extraction Pipeline

echo "=========================================="
echo "Setting up Energy Feature Extraction Environment"
echo "=========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/osm
mkdir -p data/copernicus
mkdir -p data/viirs
mkdir -p cache
mkdir -p output
mkdir -p logs

echo "Directories created:"
echo "  data/           - For input datasets"
echo "  cache/          - For cached data"
echo "  output/         - For results"
echo "  logs/           - For log files"

# Check Python version
echo "Checking Python version..."
python3 --version

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Check if osmnx is installed (critical dependency)
echo "Checking critical dependencies..."
python3 -c "import osmnx; print('✓ OSMnx installed successfully')" || {
    echo "Installing OSMnx..."
    pip install osmnx
}

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the pipeline: python3 pipeline.py"
echo "2. Or use the runner: bash run_pipeline.sh"
echo ""
echo "Note: The pipeline will extract OSM data automatically."
echo "For Copernicus/VIIRS data, place files in data/copernicus/ and data/viirs/"