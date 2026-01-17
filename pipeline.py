"""
ENERGY FEATURE EXTRACTION PIPELINE:
1. Extracts ALL energy-relevant features from OSM for KIT North Campus
2. Calculates comprehensive energy parameters and statistics
3. Creates professional visualizations and reports
4. Saves all results in multiple formats for thesis use
================================================================================
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from shapely.geometry import box, Point, Polygon, LineString
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import pickle
from pathlib import Path
import math
import textwrap
from dataclasses import dataclass
import seaborn as sns
from scipy import stats

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class CampusConfig:
    """Configuration for KIT North Campus"""
    NAME: str = "KIT North Campus"
    BBOX: Tuple[float, float, float, float] = (8.4140, 49.0125, 8.4230, 49.0165)
    EPSG: int = 25832  # UTM Zone 32N for Germany
    EXPECTED_BUILDINGS: int = 80  # Expected for campus area
    TILE_SIZE: int = 50  # meters for spatial analysis
    PV_EFFICIENCY: float = 0.15  # kW per m² of solar panel

class KITEnergyExtractor:
    """
    Complete energy feature extractor for KIT North Campus
    Extracts ALL energy features and calculates comprehensive parameters
    """
    
    def __init__(self, config: CampusConfig = None):
        self.config = config or CampusConfig()
        self.setup_directories()
        self.initialize_geometries()
        logger.info(f"Initialized KIT Energy Extractor for {self.config.NAME}")
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['output', 'cache', 'output/plots', 'output/data', 'output/reports']
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def initialize_geometries(self):
        """Initialize campus geometry and calculate area"""
        self.bounding_box = box(*self.config.BBOX)
        self.campus_gdf = gpd.GeoDataFrame(
            {'name': [self.config.NAME], 'geometry': [self.bounding_box]},
            crs="EPSG:4326"
        ).to_crs(epsg=self.config.EPSG)
        
        # Calculate area metrics
        self.area_m2 = self.campus_gdf.geometry.area.iloc[0]
        self.area_km2 = self.area_m2 / 1_000_000
        self.area_ha = self.area_m2 / 10_000
        
        logger.info(f"Campus Area: {self.area_ha:.2f} ha ({self.area_km2:.4f} km²)")
    
    def extract_all_energy_features(self) -> Dict[str, Any]:
        """
        Extract ALL energy-relevant features from OpenStreetMap
        Returns comprehensive dictionary of all features
        """
        logger.info("="*70)
        logger.info("EXTRACTING ALL ENERGY FEATURES FROM OSM")
        logger.info("="*70)
        
        try:
            import osmnx as ox
            ox.settings.timeout = 90
            ox.settings.max_retries = 5
            ox.settings.log_console = False
            
            bbox_polygon = self.bounding_box
            all_features = {}
            
            # ==================== 1. POWER INFRASTRUCTURE ====================
            logger.info("\n1. POWER INFRASTRUCTURE EXTRACTION")
            power_features = {}
            
            # Core power infrastructure
            power_tags = {
                'power_plants': {'power': ['plant']},
                'generators': {'power': ['generator']},
                'substations': {'power': ['substation', 'transformer']},
                'switches': {'power': ['switch', 'terminal']},
                'power_lines': {'power': ['line', 'cable', 'minor_line']},
                'power_towers': {'power': ['tower', 'pole', 'portal']}
            }
            
            for feature_name, tags in power_tags.items():
                try:
                    gdf = ox.features.features_from_polygon(bbox_polygon, tags=tags)
                    if not gdf.empty:
                        gdf = gdf.to_crs(epsg=self.config.EPSG)
                        power_features[feature_name] = gdf
                        logger.info(f"   ✓ {feature_name}: {len(gdf)} features")
                        
                        # Add specific attributes
                        if feature_name == 'generators' and 'generator:method' in gdf.columns:
                            methods = gdf['generator:method'].value_counts()
                            for method, count in methods.items():
                                logger.info(f"     - {method}: {count}")
                except Exception as e:
                    logger.warning(f"   ✗ {feature_name}: {e}")
                    power_features[feature_name] = gpd.GeoDataFrame()
            
            all_features['power_infrastructure'] = power_features
            
            # ==================== 2. ENERGY GENERATION ====================
            logger.info("\n2. ENERGY GENERATION EXTRACTION")
            energy_gen = {}
            
            # Generator methods
            try:
                gen_methods = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'generator:method': True}
                )
                if not gen_methods.empty:
                    gen_methods = gen_methods.to_crs(epsg=self.config.EPSG)
                    energy_gen['methods'] = gen_methods
                    logger.info(f"   ✓ Generator methods: {len(gen_methods)}")
            except:
                energy_gen['methods'] = gpd.GeoDataFrame()
            
            # Generator sources
            try:
                gen_sources = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'generator:source': True}
                )
                if not gen_sources.empty:
                    gen_sources = gen_sources.to_crs(epsg=self.config.EPSG)
                    energy_gen['sources'] = gen_sources
                    logger.info(f"   ✓ Generator sources: {len(gen_sources)}")
            except:
                energy_gen['sources'] = gpd.GeoDataFrame()
            
            all_features['energy_generation'] = energy_gen
            
            # ==================== 3. BUILDINGS & ROOFS ====================
            logger.info("\n3. BUILDING EXTRACTION")
            building_features = {}
            
            # All buildings
            try:
                buildings = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'building': True}
                )
                if not buildings.empty:
                    buildings = buildings.to_crs(epsg=self.config.EPSG)
                    buildings['area_m2'] = buildings.geometry.area
                    building_features['all_buildings'] = buildings
                    logger.info(f"   ✓ Total buildings: {len(buildings)}")
                    
                    # Building type analysis
                    if 'building' in buildings.columns:
                        btypes = buildings['building'].value_counts()
                        logger.info(f"   Building types:")
                        for btype, count in btypes.head(8).items():
                            logger.info(f"     - {btype}: {count}")
            except Exception as e:
                logger.error(f"   ✗ Building extraction failed: {e}")
                building_features['all_buildings'] = gpd.GeoDataFrame()
            
            # Flat roofs (YOUR SPECIFIC FILTER)
            try:
                flat_roofs = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'building:roof:shape': 'flat'}
                )
                if not flat_roofs.empty:
                    flat_roofs = flat_roofs.to_crs(epsg=self.config.EPSG)
                    flat_roofs['roof_area_m2'] = flat_roofs.geometry.area
                    flat_roofs['pv_potential_kw'] = flat_roofs['roof_area_m2'] * self.config.PV_EFFICIENCY
                    building_features['flat_roofs'] = flat_roofs
                    logger.info(f"   ✓ Flat roofs: {len(flat_roofs)}")
                    logger.info(f"     Total PV potential: {flat_roofs['pv_potential_kw'].sum():.1f} kW")
                else:
                    logger.info("   ⓘ No flat roofs found")
                    building_features['flat_roofs'] = gpd.GeoDataFrame()
            except:
                building_features['flat_roofs'] = gpd.GeoDataFrame()
            
            # Other roof shapes for comparison
            try:
                other_roofs = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'building:roof:shape': ['gabled', 'hipped', 'pyramidal']}
                )
                if not other_roofs.empty:
                    other_roofs = other_roofs.to_crs(epsg=self.config.EPSG)
                    building_features['other_roofs'] = other_roofs
                    logger.info(f"   ✓ Other roof shapes: {len(other_roofs)}")
            except:
                building_features['other_roofs'] = gpd.GeoDataFrame()
            
            all_features['buildings'] = building_features
            
            # ==================== 4. ENERGY TRANSPORT ====================
            logger.info("\n4. ENERGY TRANSPORT EXTRACTION")
            transport_features = {}
            
            # Pipelines
            try:
                pipelines = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'man_made': 'pipeline'}
                )
                if not pipelines.empty:
                    pipelines = pipelines.to_crs(epsg=self.config.EPSG)
                    transport_features['pipelines'] = pipelines
                    
                    # Add substance info
                    if 'substance' in pipelines.columns:
                        substances = pipelines['substance'].value_counts()
                        logger.info(f"   ✓ Pipelines: {len(pipelines)}")
                        for substance, count in substances.items():
                            logger.info(f"     - {substance}: {count}")
            except:
                transport_features['pipelines'] = gpd.GeoDataFrame()
            
            all_features['energy_transport'] = transport_features
            
            # ==================== 5. ENERGY CONSUMPTION FEATURES ====================
            logger.info("\n5. ENERGY CONSUMPTION FEATURES EXTRACTION")
            consumption_features = {}
            
            # Charging stations
            try:
                charging = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'amenity': 'charging_station'}
                )
                if not charging.empty:
                    charging = charging.to_crs(epsg=self.config.EPSG)
                    consumption_features['charging_stations'] = charging
                    logger.info(f"   ✓ Charging stations: {len(charging)}")
                    
                    # Socket details if available
                    socket_cols = [c for c in charging.columns if 'socket' in c]
                    if socket_cols:
                        for col in socket_cols[:3]:
                            unique = charging[col].dropna().unique()
                            if len(unique) > 0:
                                logger.info(f"     - {col}: {unique[0]}")
            except:
                consumption_features['charging_stations'] = gpd.GeoDataFrame()
            
            # Solar panels
            try:
                solar = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'solar:panel': 'yes'}
                )
                if not solar.empty:
                    solar = solar.to_crs(epsg=self.config.EPSG)
                    consumption_features['solar_panels'] = solar
                    logger.info(f"   ✓ Solar panels: {len(solar)}")
            except:
                consumption_features['solar_panels'] = gpd.GeoDataFrame()
            
            all_features['energy_consumption'] = consumption_features
            
            # ==================== 6. LAND USE & OTHER ====================
            logger.info("\n6. LAND USE AND OTHER FEATURES")
            other_features = {}
            
            # Land use
            try:
                landuse = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'landuse': ['industrial', 'commercial', 'education', 'research']}
                )
                if not landuse.empty:
                    landuse = landuse.to_crs(epsg=self.config.EPSG)
                    other_features['landuse'] = landuse
                    logger.info(f"   ✓ Land use areas: {len(landuse)}")
            except:
                other_features['landuse'] = gpd.GeoDataFrame()
            
            # Communication lines (often co-located with power)
            try:
                comm_lines = ox.features.features_from_polygon(
                    bbox_polygon,
                    tags={'communication': 'line'}
                )
                if not comm_lines.empty:
                    comm_lines = comm_lines.to_crs(epsg=self.config.EPSG)
                    other_features['communication_lines'] = comm_lines
                    logger.info(f"   ✓ Communication lines: {len(comm_lines)}")
            except:
                other_features['communication_lines'] = gpd.GeoDataFrame()
            
            all_features['other_features'] = other_features
            
            logger.info("\n" + "="*70)
            logger.info("FEATURE EXTRACTION COMPLETE")
            logger.info("="*70)
            
            return all_features
            
        except ImportError:
            logger.error("OSMnx not installed! Run: pip install osmnx")
            raise
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def calculate_comprehensive_parameters(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive energy parameters from extracted features
        """
        logger.info("\n" + "="*70)
        logger.info("CALCULATING COMPREHENSIVE ENERGY PARAMETERS")
        logger.info("="*70)
        
        parameters = {
            'metadata': {
                'campus_name': self.config.NAME,
                'extraction_date': datetime.now().isoformat(),
                'area_ha': round(self.area_ha, 2),
                'area_km2': round(self.area_km2, 4),
                'tile_size_m': self.config.TILE_SIZE
            },
            'building_parameters': {},
            'energy_generation_parameters': {},
            'power_infrastructure_parameters': {},
            'renewable_potential_parameters': {},
            'spatial_distribution_parameters': {},
            'summary_statistics': {}
        }
        
        # ==================== BUILDING PARAMETERS ====================
        building_data = features.get('buildings', {})
        all_buildings = building_data.get('all_buildings', gpd.GeoDataFrame())
        flat_roofs = building_data.get('flat_roofs', gpd.GeoDataFrame())
        
        if not all_buildings.empty:
            # Basic counts
            total_buildings = len(all_buildings)
            parameters['building_parameters']['total_buildings'] = total_buildings
            
            # Area calculations
            if 'area_m2' in all_buildings.columns:
                total_building_area = all_buildings['area_m2'].sum()
                avg_building_area = all_buildings['area_m2'].mean()
                
                parameters['building_parameters']['total_building_area_m2'] = round(total_building_area, 2)
                parameters['building_parameters']['avg_building_area_m2'] = round(avg_building_area, 2)
                parameters['building_parameters']['building_coverage_percentage'] = round(
                    (total_building_area / self.area_m2) * 100, 2
                )
            
            # Building density
            building_density = total_buildings / self.area_ha
            parameters['building_parameters']['building_density_ha'] = round(building_density, 2)
            parameters['building_parameters']['building_density_km2'] = round(building_density * 100, 2)
            
            # Building type distribution
            if 'building' in all_buildings.columns:
                building_types = all_buildings['building'].value_counts()
                parameters['building_parameters']['unique_building_types'] = len(building_types)
                parameters['building_parameters']['dominant_building_type'] = (
                    building_types.index[0] if len(building_types) > 0 else "Unknown"
                )
        
        # ==================== FLAT ROOF PARAMETERS ====================
        if not flat_roofs.empty:
            flat_roof_count = len(flat_roofs)
            parameters['building_parameters']['flat_roof_count'] = flat_roof_count
            
            # Flat roof percentage
            if 'total_buildings' in parameters['building_parameters']:
                flat_roof_percentage = (
                    flat_roof_count / parameters['building_parameters']['total_buildings']
                ) * 100
                parameters['building_parameters']['flat_roof_percentage'] = round(flat_roof_percentage, 2)
            
            # PV potential calculations
            if 'pv_potential_kw' in flat_roofs.columns:
                total_pv_potential = flat_roofs['pv_potential_kw'].sum()
                avg_pv_per_roof = flat_roofs['pv_potential_kw'].mean()
                
                parameters['renewable_potential_parameters']['total_pv_potential_kw'] = round(total_pv_potential, 2)
                parameters['renewable_potential_parameters']['avg_pv_per_roof_kw'] = round(avg_pv_per_roof, 2)
                parameters['renewable_potential_parameters']['pv_potential_density_kw_ha'] = round(
                    total_pv_potential / self.area_ha, 2
                )
                
                # Estimated annual energy generation (kWh/year)
                # Assuming 850 full-load hours per year for Karlsruhe
                annual_energy_kwh = total_pv_potential * 850
                parameters['renewable_potential_parameters']['estimated_annual_energy_kwh'] = round(annual_energy_kwh, 2)
        
        # ==================== POWER INFRASTRUCTURE PARAMETERS ====================
        power_data = features.get('power_infrastructure', {})
        total_power_features = 0
        power_type_counts = {}
        
        for feature_type, gdf in power_data.items():
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                count = len(gdf)
                total_power_features += count
                power_type_counts[feature_type] = count
        
        parameters['power_infrastructure_parameters']['total_power_features'] = total_power_features
        parameters['power_infrastructure_parameters']['power_feature_density_ha'] = round(
            total_power_features / self.area_ha, 2
        )
        parameters['power_infrastructure_parameters']['feature_type_counts'] = power_type_counts
        
        # ==================== ENERGY GENERATION PARAMETERS ====================
        gen_data = features.get('energy_generation', {})
        
        # Generator methods
        gen_methods = gen_data.get('methods', gpd.GeoDataFrame())
        if not gen_methods.empty and 'generator:method' in gen_methods.columns:
            method_counts = gen_methods['generator:method'].value_counts().to_dict()
            parameters['energy_generation_parameters']['generator_methods'] = method_counts
            
            # Identify renewable methods
            renewable_methods = ['photovoltaic', 'wind_turbine', 'hydro', 'biomass', 'biogas']
            renewable_count = sum(
                count for method, count in method_counts.items() 
                if any(rm in str(method).lower() for rm in renewable_methods)
            )
            parameters['energy_generation_parameters']['renewable_generators'] = renewable_count
        
        # Generator sources
        gen_sources = gen_data.get('sources', gpd.GeoDataFrame())
        if not gen_sources.empty and 'generator:source' in gen_sources.columns:
            source_counts = gen_sources['generator:source'].value_counts().to_dict()
            parameters['energy_generation_parameters']['generator_sources'] = source_counts
        
        # ==================== ENERGY CONSUMPTION PARAMETERS ====================
        consumption_data = features.get('energy_consumption', {})
        
        # Charging stations
        charging = consumption_data.get('charging_stations', gpd.GeoDataFrame())
        if not charging.empty:
            parameters['energy_generation_parameters']['charging_stations_count'] = len(charging)
            parameters['energy_generation_parameters']['charging_station_density_ha'] = round(
                len(charging) / self.area_ha, 2
            )
        
        # Solar panels
        solar = consumption_data.get('solar_panels', gpd.GeoDataFrame())
        if not solar.empty:
            parameters['energy_generation_parameters']['solar_panel_installations'] = len(solar)
        
        # ==================== SPATIAL DISTRIBUTION PARAMETERS ====================
        # Calculate using tile-based analysis
        tiles = self.create_spatial_tiles(features)
        if tiles:
            parameters['spatial_distribution_parameters'] = tiles.get('tile_statistics', {})
            
            # Calculate spatial autocorrelation (Moran's I approximation)
            if 'tile_gdf' in tiles and 'building_count' in tiles['tile_gdf'].columns:
                building_counts = tiles['tile_gdf']['building_count'].values
                if len(building_counts) > 1:
                    mean_count = np.mean(building_counts)
                    variance = np.var(building_counts)
                    if variance > 0:
                        z_scores = (building_counts - mean_count) / np.sqrt(variance)
                        spatial_variation = np.std(z_scores)
                        parameters['spatial_distribution_parameters']['building_spatial_variation'] = round(
                            spatial_variation, 3
                        )
        
        # ==================== SUMMARY STATISTICS ====================
        # Calculate overall energy feature density
        total_energy_features = (
            parameters['building_parameters'].get('total_buildings', 0) +
            parameters['power_infrastructure_parameters'].get('total_power_features', 0) +
            parameters['energy_generation_parameters'].get('charging_stations_count', 0) +
            parameters['energy_generation_parameters'].get('solar_panel_installations', 0)
        )
        
        parameters['summary_statistics']['total_energy_features'] = total_energy_features
        parameters['summary_statistics']['energy_feature_density_ha'] = round(
            total_energy_features / self.area_ha, 2
        )
        
        # Energy sustainability index (simple calculation)
        renewable_score = (
            parameters['renewable_potential_parameters'].get('total_pv_potential_kw', 0) / 1000 +  # Convert to MW
            parameters['energy_generation_parameters'].get('renewable_generators', 0) * 10
        )
        parameters['summary_statistics']['renewable_energy_score'] = round(renewable_score, 2)
        
        logger.info("✓ Calculated comprehensive energy parameters")
        return parameters
    
    def create_spatial_tiles(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create spatial tiles for distribution analysis
        """
        building_data = features.get('buildings', {})
        all_buildings = building_data.get('all_buildings', gpd.GeoDataFrame())
        
        if all_buildings.empty:
            return {}
        
        # Get campus bounds
        bbox_proj = self.campus_gdf.geometry.iloc[0].bounds
        x_min, y_min, x_max, y_max = bbox_proj
        
        # Calculate grid
        tile_size = self.config.TILE_SIZE
        x_size = x_max - x_min
        y_size = y_max - y_min
        
        n_x_tiles = max(1, int(np.ceil(x_size / tile_size)))
        n_y_tiles = max(1, int(np.ceil(y_size / tile_size)))
        
        tiles = []
        
        for i in range(n_x_tiles):
            for j in range(n_y_tiles):
                # Create tile
                tile_x_min = x_min + i * tile_size
                tile_x_max = min(x_min + (i + 1) * tile_size, x_max)
                tile_y_min = y_min + j * tile_size
                tile_y_max = min(y_min + (j + 1) * tile_size, y_max)
                
                tile_geom = box(tile_x_min, tile_y_min, tile_x_max, tile_y_max)
                
                # Count features in tile
                try:
                    buildings_in_tile = all_buildings[all_buildings.intersects(tile_geom)]
                    building_count = len(buildings_in_tile)
                except:
                    building_count = 0
                
                tile_data = {
                    'tile_id': f"{i}_{j}",
                    'geometry': tile_geom,
                    'center_x': (tile_x_min + tile_x_max) / 2,
                    'center_y': (tile_y_min + tile_y_max) / 2,
                    'building_count': building_count,
                    'tile_area_m2': tile_geom.area,
                    'tile_area_ha': tile_geom.area / 10000
                }
                
                tiles.append(tile_data)
        
        # Create GeoDataFrame
        tile_gdf = gpd.GeoDataFrame(
            tiles,
            geometry='geometry',
            crs=f"EPSG:{self.config.EPSG}"
        )
        
        # Calculate tile statistics
        building_counts = [t['building_count'] for t in tiles]
        tile_stats = {
            'total_tiles': len(tiles),
            'tiles_with_buildings': sum(1 for t in tiles if t['building_count'] > 0),
            'max_buildings_per_tile': max(building_counts) if building_counts else 0,
            'avg_buildings_per_tile': np.mean(building_counts) if building_counts else 0,
            'building_count_variance': np.var(building_counts) if building_counts else 0,
            'tile_size_m': tile_size,
            'grid_dimensions': f"{n_x_tiles} × {n_y_tiles}",
            'spatial_distribution_gini': self.calculate_gini_coefficient(building_counts) if building_counts else 0
        }
        
        return {
            'tile_gdf': tile_gdf,
            'tile_features': tiles,
            'tile_statistics': tile_stats
        }
    
    def calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumulative = np.cumsum(sorted_values)
        
        # Gini coefficient formula
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        return round(gini, 3)
    
    def create_professional_visualizations(self, features: Dict[str, Any], parameters: Dict[str, Any]):
        """
        Create professional visualizations for thesis/publication
        """
        logger.info("\nCreating professional visualizations...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Colors for consistent theming
        campus_color = '#2E86AB'  # Blue for campus
        building_color = '#A23B72'  # Purple for buildings
        energy_color = '#F18F01'  # Orange for energy
        renewable_color = '#73AB84'  # Green for renewable
        
        # ==================== PLOT 1: Campus Overview ====================
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Plot campus boundary
        self.campus_gdf.boundary.plot(ax=ax1, color=campus_color, linewidth=3, label='Campus Boundary')
        
        # Plot buildings
        building_data = features.get('buildings', {})
        all_buildings = building_data.get('all_buildings', gpd.GeoDataFrame())
        flat_roofs = building_data.get('flat_roofs', gpd.GeoDataFrame())
        
        if not all_buildings.empty:
            all_buildings.plot(ax=ax1, color=building_color, alpha=0.6, label='All Buildings')
        
        if not flat_roofs.empty:
            flat_roofs.plot(ax=ax1, color=renewable_color, alpha=0.9, label='Flat Roofs (PV Potential)')
        
        # Plot power infrastructure
        power_data = features.get('power_infrastructure', {})
        for feature_type, gdf in power_data.items():
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                gdf.plot(ax=ax1, color=energy_color, markersize=40, alpha=0.8, 
                        label=f'Power: {feature_type.replace("_", " ").title()}')
        
        ax1.set_title(f'KIT North Campus: Energy Feature Distribution', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Easting (m)')
        ax1.set_ylabel('Northing (m)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f"Campus Area: {self.area_ha:.1f} ha\n"
        if 'total_buildings' in parameters['building_parameters']:
            stats_text += f"Buildings: {parameters['building_parameters']['total_buildings']}\n"
        if 'flat_roof_count' in parameters['building_parameters']:
            stats_text += f"Flat Roofs: {parameters['building_parameters']['flat_roof_count']}\n"
        if 'total_pv_potential_kw' in parameters['renewable_potential_parameters']:
            stats_text += f"PV Potential: {parameters['renewable_potential_parameters']['total_pv_potential_kw']:.0f} kW"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # ==================== PLOT 2: Building Distribution ====================
        ax2 = fig.add_subplot(gs[0, 2])
        
        if not all_buildings.empty and 'area_m2' in all_buildings.columns:
            # Histogram of building areas
            building_areas = all_buildings['area_m2']
            ax2.hist(building_areas, bins=20, color=building_color, alpha=0.7, edgecolor='black')
            ax2.set_title('Building Area Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Building Area (m²)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {building_areas.mean():.0f} m²\n"
            stats_text += f"Std: {building_areas.std():.0f} m²\n"
            stats_text += f"Max: {building_areas.max():.0f} m²"
            ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ==================== PLOT 3: Energy Infrastructure ====================
        ax3 = fig.add_subplot(gs[0, 3])
        
        power_counts = parameters['power_infrastructure_parameters'].get('feature_type_counts', {})
        if power_counts:
            labels = [k.replace('_', ' ').title() for k in power_counts.keys()]
            sizes = list(power_counts.values())
            
            # Create donut chart
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                              startangle=90, colors=sns.color_palette("Set2"))
            
            # Draw circle for donut
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax3.add_artist(centre_circle)
            
            ax3.set_title('Power Infrastructure Distribution', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No power infrastructure\nfound in campus area', 
                    ha='center', va='center', fontsize=11)
            ax3.set_title('Power Infrastructure', fontsize=12, fontweight='bold')
        
        # ==================== PLOT 4: PV Potential Analysis ====================
        ax4 = fig.add_subplot(gs[1, 2])
        
        if not flat_roofs.empty and 'pv_potential_kw' in flat_roofs.columns:
            pv_values = flat_roofs['pv_potential_kw']
            
            # Scatter plot of PV potential vs roof area
            if 'roof_area_m2' in flat_roofs.columns:
                ax4.scatter(flat_roofs['roof_area_m2'], pv_values, 
                           color=renewable_color, alpha=0.7, s=50)
                ax4.set_xlabel('Roof Area (m²)')
                ax4.set_ylabel('PV Potential (kW)')
                
                # Add regression line
                z = np.polyfit(flat_roofs['roof_area_m2'], pv_values, 1)
                p = np.poly1d(z)
                ax4.plot(flat_roofs['roof_area_m2'], p(flat_roofs['roof_area_m2']), 
                        color='red', alpha=0.8, linestyle='--', label=f'y = {z[0]:.3f}x + {z[1]:.2f}')
                ax4.legend()
            
            ax4.set_title('PV Potential Analysis', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # ==================== PLOT 5: Spatial Distribution ====================
        ax5 = fig.add_subplot(gs[1, 3])
        
        tiles = self.create_spatial_tiles(features)
        if tiles and 'tile_gdf' in tiles and 'building_count' in tiles['tile_gdf'].columns:
            tile_gdf = tiles['tile_gdf']
            
            # Create heatmap
            tile_gdf.plot(column='building_count', ax=ax5, cmap='YlOrRd', 
                         legend=True, legend_kwds={'label': 'Buildings per Tile'})
            
            # Overlay campus boundary
            self.campus_gdf.boundary.plot(ax=ax5, color='black', linewidth=2)
            
            ax5.set_title('Building Density Heatmap', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Easting (m)')
            ax5.set_ylabel('Northing (m)')
            
            # Add spatial statistics
            if 'tile_statistics' in tiles:
                stats = tiles['tile_statistics']
                stats_text = f"Tiles: {stats['total_tiles']}\n"
                stats_text += f"Avg/Tile: {stats['avg_buildings_per_tile']:.1f}\n"
                stats_text += f"Max/Tile: {stats['max_buildings_per_tile']}"
                ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, fontsize=9,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # ==================== PLOT 6: Parameter Summary ====================
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create parameter summary table
        summary_data = []
        
        # Add building parameters
        for key, value in parameters['building_parameters'].items():
            if isinstance(value, (int, float)):
                summary_data.append([
                    key.replace('_', ' ').title(),
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    "Building Metrics"
                ])
        
        # Add energy parameters
        for key, value in parameters['renewable_potential_parameters'].items():
            if isinstance(value, (int, float)):
                summary_data.append([
                    key.replace('_', ' ').title(),
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    "Renewable Energy"
                ])
        
        # Create table
        if summary_data:
            table = ax6.table(cellText=summary_data,
                             colLabels=['Parameter', 'Value', 'Category'],
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.4, 0.3, 0.3])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style table
            for i in range(len(summary_data) + 1):
                table[(i, 0)].set_facecolor('#F0F0F0')
                table[(i, 2)].set_facecolor('#F5F5F5')
        
        ax6.set_title('Energy Feature Parameter Summary', fontsize=14, fontweight='bold', pad=20)
        
        # ==================== PLOT 7: Feature Comparison ====================
        ax7 = fig.add_subplot(gs[3, :2])
        
        # Prepare comparison data
        categories = ['Buildings', 'Power Features', 'Charging Stations', 'Solar Panels']
        values = [
            parameters['building_parameters'].get('total_buildings', 0),
            parameters['power_infrastructure_parameters'].get('total_power_features', 0),
            parameters['energy_generation_parameters'].get('charging_stations_count', 0),
            parameters['energy_generation_parameters'].get('solar_panel_installations', 0)
        ]
        
        colors = [building_color, energy_color, '#FF6B6B', renewable_color]
        
        bars = ax7.bar(categories, values, color=colors, alpha=0.8)
        ax7.set_title('Energy Feature Count Comparison', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Count')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        
        # ==================== PLOT 8: Energy Sustainability Dashboard ====================
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        # Create sustainability dashboard
        dashboard_text = "ENERGY SUSTAINABILITY DASHBOARD\n"
        dashboard_text += "=" * 35 + "\n\n"
        
        # Campus metrics
        dashboard_text += f"Campus Area: {self.area_ha:.1f} ha\n"
        dashboard_text += f"Building Density: {parameters['building_parameters'].get('building_density_ha', 0):.1f}/ha\n\n"
        
        # Renewable energy metrics
        if 'total_pv_potential_kw' in parameters['renewable_potential_parameters']:
            pv_pot = parameters['renewable_potential_parameters']['total_pv_potential_kw']
            dashboard_text += f"PV Potential: {pv_pot:.0f} kW\n"
            
            # Estimated annual production
            annual_kwh = pv_pot * 850
            dashboard_text += f"Est. Annual: {annual_kwh:,.0f} kWh\n\n"
        
        # Energy infrastructure
        total_energy = parameters['summary_statistics'].get('total_energy_features', 0)
        energy_density = parameters['summary_statistics'].get('energy_feature_density_ha', 0)
        dashboard_text += f"Total Energy Features: {total_energy}\n"
        dashboard_text += f"Energy Density: {energy_density:.1f}/ha\n\n"
        
        # Sustainability score
        renew_score = parameters['summary_statistics'].get('renewable_energy_score', 0)
        dashboard_text += f"Renewable Energy Score: {renew_score:.1f}/100"
        
        ax8.text(0.1, 0.9, dashboard_text, fontsize=11, 
                verticalalignment='top', transform=ax8.transAxes,
                bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9, edgecolor=campus_color))
        
        ax8.set_title('Sustainability Assessment', fontsize=12, fontweight='bold', pad=10)
        
        # ==================== SAVE FIGURE ====================
        plt.suptitle('KIT NORTH CAMPUS: COMPREHENSIVE ENERGY ANALYSIS', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        output_path = Path('output/plots/comprehensive_energy_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        logger.info(f"✓ Saved comprehensive visualization to {output_path}")
        
        # Create additional individual plots
        self.create_individual_plots(features, parameters)
    
    def create_individual_plots(self, features: Dict[str, Any], parameters: Dict[str, Any]):
        """Create additional individual plots for detailed analysis"""
        
        # 1. Building Type Distribution
        building_data = features.get('buildings', {})
        all_buildings = building_data.get('all_buildings', gpd.GeoDataFrame())
        
        if not all_buildings.empty and 'building' in all_buildings.columns:
            plt.figure(figsize=(10, 6))
            building_types = all_buildings['building'].value_counts().head(10)
            building_types.plot(kind='bar', color='#A23B72', alpha=0.8)
            plt.title('Top 10 Building Types in KIT North Campus', fontsize=14, fontweight='bold')
            plt.xlabel('Building Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('output/plots/building_types.png', dpi=300)
            plt.close()
        
        # 2. Power Infrastructure Map
        plt.figure(figsize=(10, 8))
        
        # Plot campus
        self.campus_gdf.boundary.plot(color='#2E86AB', linewidth=2, label='Campus')
        
        # Plot power features
        power_data = features.get('power_infrastructure', {})
        for feature_type, gdf in power_data.items():
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                marker_size = {'power_plants': 100, 'generators': 80, 'substations': 60, 
                              'switches': 40, 'power_lines': 20, 'power_towers': 30}.get(feature_type, 50)
                gdf.plot(ax=plt.gca(), markersize=marker_size, alpha=0.8, 
                        label=feature_type.replace('_', ' ').title())
        
        plt.title('Power Infrastructure Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/plots/power_infrastructure.png', dpi=300)
        plt.close()
        
        # 3. Parameter Comparison Radar Chart
        self.create_radar_chart(parameters)
    
    def create_radar_chart(self, parameters: Dict[str, Any]):
        """Create radar chart for parameter comparison"""
        try:
            # Normalize parameters for radar chart
            categories = ['Building Density', 'PV Potential', 'Power Features', 
                         'Energy Diversity', 'Spatial Distribution']
            
            # Get normalized values (0-1 scale)
            building_density = min(parameters['building_parameters'].get('building_density_ha', 0) / 10, 1)
            
            pv_potential = min(parameters['renewable_potential_parameters'].get('total_pv_potential_kw', 0) / 500, 1)
            
            power_features = min(parameters['power_infrastructure_parameters'].get('total_power_features', 0) / 20, 1)
            
            # Energy diversity (based on different feature types)
            unique_features = sum(1 for v in parameters['power_infrastructure_parameters'].get('feature_type_counts', {}).values() if v > 0)
            energy_diversity = min(unique_features / 5, 1)
            
            # Spatial distribution (Gini coefficient inverted)
            spatial_dist = 1 - parameters['spatial_distribution_parameters'].get('building_spatial_variation', 0.5)
            
            values = [building_density, pv_potential, power_features, energy_diversity, spatial_dist]
            values += values[:1]  # Close the polygon
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
            ax.fill(angles, values, alpha=0.25, color='#2E86AB')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
            ax.grid(True)
            
            plt.title('Energy Feature Parameter Radar Chart', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('output/plots/parameter_radar.png', dpi=300)
            plt.close()
        except:
            pass  # Skip radar chart if data insufficient
    
    def save_all_results(self, features: Dict[str, Any], parameters: Dict[str, Any]):
        """
        Save all results in multiple formats
        """
        logger.info("\nSaving all results...")
        
        # ==================== 1. Save Parameters as JSON ====================
        params_file = Path('output/data/energy_parameters.json')
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=2, default=str)
        logger.info(f"✓ Parameters saved to {params_file}")
        
        # ==================== 2. Save Parameters as CSV ====================
        # Flatten parameters for CSV
        flat_params = {}
        for category, data in parameters.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for subkey, subvalue in value.items():
                            flat_params[f"{category}_{key}_{subkey}"] = subvalue
                    else:
                        flat_params[f"{category}_{key}"] = value
        
        params_df = pd.DataFrame([flat_params])
        params_csv = Path('output/data/energy_parameters.csv')
        params_df.to_csv(params_csv, index=False)
        logger.info(f"✓ Parameters CSV saved to {params_csv}")
        
        # ==================== 3. Save Feature Counts Summary ====================
        summary_data = {
            'Campus Area (ha)': [self.area_ha],
            'Total Buildings': [parameters['building_parameters'].get('total_buildings', 0)],
            'Flat Roofs': [parameters['building_parameters'].get('flat_roof_count', 0)],
            'Flat Roof %': [parameters['building_parameters'].get('flat_roof_percentage', 0)],
            'PV Potential (kW)': [parameters['renewable_potential_parameters'].get('total_pv_potential_kw', 0)],
            'Power Features': [parameters['power_infrastructure_parameters'].get('total_power_features', 0)],
            'Charging Stations': [parameters['energy_generation_parameters'].get('charging_stations_count', 0)],
            'Solar Panels': [parameters['energy_generation_parameters'].get('solar_panel_installations', 0)],
            'Building Density (/ha)': [parameters['building_parameters'].get('building_density_ha', 0)],
            'Energy Density (/ha)': [parameters['summary_statistics'].get('energy_feature_density_ha', 0)]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = Path('output/data/feature_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"✓ Feature summary saved to {summary_file}")
        
        # ==================== 4. Save Building Data ====================
        building_data = features.get('buildings', {})
        all_buildings = building_data.get('all_buildings', gpd.GeoDataFrame())
        if not all_buildings.empty:
            buildings_file = Path('output/data/buildings.geojson')
            all_buildings.to_file(buildings_file, driver='GeoJSON')
            logger.info(f"✓ Building data saved to {buildings_file}")
            
            # Save building statistics
            if 'area_m2' in all_buildings.columns:
                building_stats = all_buildings['area_m2'].describe()
                building_stats_file = Path('output/data/building_statistics.json')
                building_stats.to_json(building_stats_file)
        
        # ==================== 5. Save Flat Roof Data ====================
        flat_roofs = building_data.get('flat_roofs', gpd.GeoDataFrame())
        if not flat_roofs.empty:
            flat_roofs_file = Path('output/data/flat_roofs.geojson')
            flat_roofs.to_file(flat_roofs_file, driver='GeoJSON')
            logger.info(f"✓ Flat roof data saved to {flat_roofs_file}")
            
            # Save PV potential details
            if 'pv_potential_kw' in flat_roofs.columns:
                pv_summary = flat_roofs['pv_potential_kw'].describe()
                pv_file = Path('output/data/pv_potential_summary.json')
                pv_summary.to_json(pv_file)
        
        # ==================== 6. Save Power Infrastructure Data ====================
        power_data = features.get('power_infrastructure', {})
        for feature_type, gdf in power_data.items():
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                power_file = Path(f'output/data/power_{feature_type}.geojson')
                gdf.to_file(power_file, driver='GeoJSON')
        
        # ==================== 7. Save Complete Results as Pickle ====================
        complete_results = {
            'features': features,
            'parameters': parameters,
            'campus_info': {
                'name': self.config.NAME,
                'bbox': self.config.BBOX,
                'area_ha': self.area_ha,
                'area_km2': self.area_km2
            },
            'extraction_date': datetime.now().isoformat()
        }
        
        pickle_file = Path('output/data/complete_results.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(complete_results, f)
        logger.info(f"✓ Complete results saved to {pickle_file}")
        
        # ==================== 8. Generate Report ====================
        self.generate_report(parameters)
        
        logger.info("\n" + "="*70)
        logger.info("ALL RESULTS SAVED SUCCESSFULLY")
        logger.info("="*70)
    
    def generate_report(self, parameters: Dict[str, Any]):
        """Generate a text report summary"""
        report = [
            "="*70,
            "KIT NORTH CAMPUS - ENERGY FEATURE ANALYSIS REPORT",
            "="*70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Campus Area: {self.area_ha:.1f} hectares",
            "",
            "SUMMARY STATISTICS:",
            "-"*40
        ]
        
        # Add key statistics
        key_stats = [
            ("Total Buildings", parameters['building_parameters'].get('total_buildings', 0)),
            ("Building Density", f"{parameters['building_parameters'].get('building_density_ha', 0):.1f}/ha"),
            ("Flat Roofs", parameters['building_parameters'].get('flat_roof_count', 0)),
            ("Flat Roof %", f"{parameters['building_parameters'].get('flat_roof_percentage', 0):.1f}%"),
            ("PV Potential", f"{parameters['renewable_potential_parameters'].get('total_pv_potential_kw', 0):.0f} kW"),
            ("Annual Energy", f"{parameters['renewable_potential_parameters'].get('estimated_annual_energy_kwh', 0):,.0f} kWh"),
            ("Power Features", parameters['power_infrastructure_parameters'].get('total_power_features', 0)),
            ("Charging Stations", parameters['energy_generation_parameters'].get('charging_stations_count', 0)),
            ("Solar Panels", parameters['energy_generation_parameters'].get('solar_panel_installations', 0))
        ]
        
        for label, value in key_stats:
            report.append(f"{label:25}: {value}")
        
        report.extend([
            "",
            "SPATIAL ANALYSIS:",
            "-"*40,
            f"Tile Size: {self.config.TILE_SIZE}m",
            f"Total Tiles: {parameters['spatial_distribution_parameters'].get('total_tiles', 0)}",
            f"Tiles with Buildings: {parameters['spatial_distribution_parameters'].get('tiles_with_buildings', 0)}",
            f"Spatial Variation: {parameters['spatial_distribution_parameters'].get('building_spatial_variation', 0):.3f}",
            "",
            "ENERGY SUSTAINABILITY ASSESSMENT:",
            "-"*40,
            f"Total Energy Features: {parameters['summary_statistics'].get('total_energy_features', 0)}",
            f"Energy Feature Density: {parameters['summary_statistics'].get('energy_feature_density_ha', 0):.1f}/ha",
            f"Renewable Energy Score: {parameters['summary_statistics'].get('renewable_energy_score', 0):.1f}/100",
            "",
            "="*70,
            "END OF REPORT",
            "="*70
        ])
        
        # Save report
        report_file = Path('output/reports/energy_analysis_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        # Also print to console
        print('\n'.join(report))
        logger.info(f"✓ Report generated: {report_file}")
    
    def run_complete_pipeline(self):
        """
        Run the complete energy feature extraction pipeline
        """
        print("\n" + "="*80)
        print("KIT NORTH CAMPUS - COMPREHENSIVE ENERGY FEATURE EXTRACTION")
        print("="*80)
        print(f"Area: {self.config.BBOX}")
        print(f"Expected Campus Size: ~{self.area_ha:.1f} hectares")
        print("="*80)
        
        try:
            # Step 1: Extract all features
            print("\n🚀 STEP 1: Extracting all energy features from OpenStreetMap...")
            features = self.extract_all_energy_features()
            
            # Step 2: Calculate parameters
            print("\n📊 STEP 2: Calculating comprehensive energy parameters...")
            parameters = self.calculate_comprehensive_parameters(features)
            
            # Step 3: Create visualizations
            print("\n🎨 STEP 3: Creating professional visualizations...")
            self.create_professional_visualizations(features, parameters)
            
            # Step 4: Save all results
            print("\n💾 STEP 4: Saving all results...")
            self.save_all_results(features, parameters)
            
            # Final summary
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("\n📁 OUTPUT FILES CREATED:")
            print("  output/plots/comprehensive_energy_analysis.png - Main visualization")
            print("  output/plots/*.png - Additional detailed plots")
            print("  output/data/energy_parameters.json - All calculated parameters")
            print("  output/data/feature_summary.csv - Key statistics")
            print("  output/data/buildings.geojson - Building footprints")
            print("  output/data/flat_roofs.geojson - Flat roof data")
            print("  output/reports/energy_analysis_report.txt - Summary report")
            print("  output/data/complete_results.pkl - Complete results (pickle)")
            print("\n📈 KEY FINDINGS:")
            
            if 'total_buildings' in parameters['building_parameters']:
                print(f"  • Buildings: {parameters['building_parameters']['total_buildings']}")
            if 'flat_roof_count' in parameters['building_parameters']:
                print(f"  • Flat Roofs: {parameters['building_parameters']['flat_roof_count']}")
            if 'total_pv_potential_kw' in parameters['renewable_potential_parameters']:
                print(f"  • PV Potential: {parameters['renewable_potential_parameters']['total_pv_potential_kw']:.0f} kW")
            if 'total_power_features' in parameters['power_infrastructure_parameters']:
                print(f"  • Power Features: {parameters['power_infrastructure_parameters']['total_power_features']}")
            
            print("\n" + "="*80)
            
            return {
                'success': True,
                'features': features,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"\n❌ PIPELINE FAILED: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🏛️  KIT NORTH CAMPUS ENERGY ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Extract ALL energy-relevant features from OpenStreetMap")
    print("  2. Calculate comprehensive energy parameters and metrics")
    print("  3. Create professional visualizations for your thesis")
    print("  4. Save all results in multiple formats")
    print("\n" + "-"*80)
    
    # Configuration
    config = CampusConfig()
    
    print(f"\n📏 CAMPUS CONFIGURATION:")
    print(f"  Name: {config.NAME}")
    print(f"  Bounding Box: {config.BBOX}")
    print(f"  Area: ~{config.EXPECTED_BUILDINGS} expected buildings")
    print(f"  Tile Size: {config.TILE_SIZE}m for spatial analysis")
    
    input("\nPress Enter to start the analysis...")
    
    # Create extractor and run pipeline
    extractor = KITEnergyExtractor(config)
    results = extractor.run_complete_pipeline()
    
    if results['success']:
        print("\n🎉 Analysis complete! Check the 'output' folder for all results.")
        print("   Use these files directly in your thesis.")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()