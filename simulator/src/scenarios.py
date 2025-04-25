from src.filter import Filter
import datetime
from typing import List, Optional, Dict, Any, Tuple
from matplotlib.patches import Polygon
from datetime import datetime, timedelta
from src.query import Query
import random
import math
# ===== REGION DEFINITIONS =====

# --- RIVER AND FLOOD PRONE REGIONS ---
def create_river_polygons() -> Dict[str, List[Polygon]]:
    """Create hierarchical and overlapping river system polygons"""
    river_systems = {}
    
    # Major continental river basins (larger, parent regions)
    river_systems["mississippi_basin"] = [
        Polygon([(-98.0, 29.0), (-98.5, 29.0), (-97.0, 47.0), (-89.0, 47.0), (-88.0, 30.0), (-90.0, 29.0)])
    ]
    
    # Subregions within Mississippi basin (overlapping with parent)
    river_systems["mississippi_upper"] = [
        Polygon([(-97.0, 38.0), (-97.5, 38.0), (-97.0, 47.0), (-89.0, 47.0), (-90.0, 38.0)])
    ]
    
    river_systems["mississippi_lower"] = [
        Polygon([(-98.0, 29.0), (-98.5, 29.0), (-97.0, 38.0), (-90.0, 38.0), (-88.0, 30.0), (-90.0, 29.0)])
    ]
    
    # Major tributary systems (overlapping with main basin)
    river_systems["missouri_river"] = [
        Polygon([(-112.0, 45.0), (-112.0, 47.0), (-97.0, 47.0), (-97.0, 38.0), (-95.0, 38.7), (-105.0, 45.0)])
    ]
    
    river_systems["ohio_river"] = [
        Polygon([(-89.0, 37.0), (-89.0, 42.0), (-80.0, 42.0), (-80.0, 37.0), (-86.0, 36.5)])
    ]
    
    # Similar hierarchical structure for other global river systems
    river_systems["ganges_basin"] = [
        Polygon([(77.0, 23.0), (77.0, 31.0), (92.0, 31.0), (92.0, 21.5), (80.0, 21.5)])
    ]
    
    river_systems["ganges_delta"] = [
        Polygon([(88.0, 21.5), (88.0, 26.0), (92.0, 26.0), (92.0, 21.5)])
    ]
    
    river_systems["brahmaputra"] = [
        Polygon([(88.0, 25.0), (88.0, 31.0), (97.0, 31.0), (97.0, 27.0), (92.0, 25.0)])
    ]
    
    # More river systems with sub-regions and tributaries
    river_systems["amazon_basin"] = [
        Polygon([(-80.0, -10.0), (-80.0, 5.0), (-45.0, 5.0), (-45.0, -10.0)])
    ]
    
    river_systems["amazon_upper"] = [
        Polygon([(-80.0, -5.0), (-80.0, 5.0), (-65.0, 5.0), (-65.0, -5.0)])
    ]
    
    river_systems["amazon_mouth"] = [
        Polygon([(-55.0, -5.0), (-55.0, 2.0), (-45.0, 2.0), (-45.0, -5.0)])
    ]
    
    # Nile region with subregions
    river_systems["nile_basin"] = [
        Polygon([(24.0, 5.0), (24.0, 31.0), (36.0, 31.0), (36.0, 5.0)])
    ]
    
    river_systems["nile_delta"] = [
        Polygon([(30.0, 30.0), (30.0, 31.5), (32.0, 31.5), (32.0, 30.0)])
    ]
    
    river_systems["yangtze_basin"] = [
        Polygon([(90.0, 28.0), (90.0, 35.0), (122.0, 35.0), (122.0, 28.0)])
    ]
    
    river_systems["mekong_basin"] = [
        Polygon([(98.0, 8.0), (98.0, 33.0), (110.0, 33.0), (110.0, 8.0)])
    ]
    
    river_systems["mekong_delta"] = [
        Polygon([(104.5, 8.5), (104.5, 11.0), (107.0, 11.0), (107.0, 8.5)])
    ]
    
    # Combine into a flat list for simple access
    all_river_polygons = []
    for polygons in river_systems.values():
        all_river_polygons.extend(polygons)
    
    return all_river_polygons

# --- WILDFIRE PRONE REGIONS ---
def create_wildfire_polygons() -> List[Polygon]:
    """Create hierarchical and overlapping wildfire prone regions"""
    wildfire_regions = []
    
    # North American regions
    wildfire_regions.append(
        Polygon([(-125.0, 30.0), (-125.0, 55.0), (-100.0, 55.0), (-100.0, 30.0)])  # western_north_america
    )
    
    # Subregions with overlap
    wildfire_regions.append(
        Polygon([(-125.0, 32.0), (-125.0, 42.0), (-115.0, 42.0), (-115.0, 32.0)])  # california_region
    )
    
    wildfire_regions.append(
        Polygon([(-125.0, 42.0), (-125.0, 50.0), (-115.0, 50.0), (-115.0, 42.0)])  # pacific_northwest
    )
    
    wildfire_regions.append(
        Polygon([(-115.0, 35.0), (-115.0, 50.0), (-105.0, 50.0), (-105.0, 35.0)])  # rocky_mountains
    )
    
    wildfire_regions.append(
        Polygon([(-115.0, 31.0), (-115.0, 38.0), (-103.0, 38.0), (-103.0, 31.0)])  # southwest_us
    )
    
    # Mediterranean regions
    wildfire_regions.append(
        Polygon([(-10.0, 35.0), (-10.0, 45.0), (35.0, 45.0), (35.0, 35.0)])  # mediterranean_basin
    )
    
    # Subregions with overlap
    wildfire_regions.append(
        Polygon([(-10.0, 36.0), (-10.0, 44.0), (3.0, 44.0), (3.0, 36.0)])  # iberian_peninsula
    )
    
    wildfire_regions.append(
        Polygon([(7.0, 37.0), (7.0, 45.0), (18.0, 45.0), (18.0, 37.0)])  # italian_peninsula
    )
    
    wildfire_regions.append(
        Polygon([(19.0, 35.0), (19.0, 42.0), (28.0, 42.0), (28.0, 35.0)])  # greek_peninsula
    )
    
    # Australian regions
    wildfire_regions.append(
        Polygon([(115.0, -38.0), (115.0, -25.0), (153.0, -25.0), (153.0, -38.0)])  # australia_fire_prone
    )
    
    # Subregions with overlap
    wildfire_regions.append(
        Polygon([(142.0, -39.0), (142.0, -33.0), (153.0, -33.0), (153.0, -39.0)])  # southeast_australia
    )
    
    wildfire_regions.append(
        Polygon([(115.0, -35.0), (115.0, -28.0), (125.0, -28.0), (125.0, -35.0)])  # western_australia
    )
    
    return wildfire_regions

# --- EARTHQUAKE PRONE REGIONS ---
def create_earthquake_polygons() -> List[Polygon]:
    """Create earthquake prone regions"""
    earthquake_regions = []
    
    # Pacific Ring of Fire (major tectonic region)
    earthquake_regions.append(
        Polygon([(-130.0, 30.0), (-130.0, 60.0), (-110.0, 60.0), (-110.0, 30.0)])  # north_america_west
    )
    
    earthquake_regions.append(
        Polygon([(130.0, 30.0), (130.0, 45.0), (145.0, 45.0), (145.0, 30.0)])  # japan_region
    )
    
    earthquake_regions.append(
        Polygon([(95.0, -10.0), (95.0, 20.0), (130.0, 20.0), (130.0, -10.0)])  # indonesia_philippines
    )
    
    earthquake_regions.append(
        Polygon([(-80.0, -55.0), (-80.0, 0.0), (-65.0, 0.0), (-65.0, -55.0)])  # south_america_west
    )
    
    # Specific fault zones and urban areas (metropolitan scale)
    earthquake_regions.append(
        Polygon([(-123.0, 32.0), (-123.0, 40.0), (-118.0, 40.0), (-118.0, 32.0)])  # san_andreas_fault
    )
    
    earthquake_regions.append(
        Polygon([(-122.5, 37.4), (-122.5, 38.2), (-121.7, 38.2), (-121.7, 37.4)])  # san_francisco_bay
    )
    
    earthquake_regions.append(
        Polygon([(-119.0, 33.5), (-119.0, 34.5), (-117.0, 34.5), (-117.0, 33.5)])  # los_angeles_region
    )
    
    earthquake_regions.append(
        Polygon([(139.5, 35.4), (139.5, 36.0), (140.2, 36.0), (140.2, 35.4)])  # tokyo_region
    )
    
    earthquake_regions.append(
        Polygon([(85.0, 27.5), (85.0, 28.0), (85.5, 28.0), (85.5, 27.5)])  # kathmandu_valley
    )
    
    earthquake_regions.append(
        Polygon([(28.5, 40.8), (28.5, 41.2), (29.5, 41.2), (29.5, 40.8)])  # istanbul_region
    )
    
    return earthquake_regions

# --- MARITIME REGIONS ---
def create_maritime_polygons() -> List[Polygon]:
    """Create hierarchical and overlapping maritime regions"""
    maritime_regions = []
    
    # Strategic maritime regions (regional scale)
    maritime_regions.append(
        Polygon([(105.0, 0.0), (105.0, 25.0), (125.0, 25.0), (125.0, 0.0)])  # south_china_sea
    )
    
    maritime_regions.append(
        Polygon([(120.0, 25.0), (120.0, 35.0), (130.0, 35.0), (130.0, 25.0)])  # east_china_sea
    )
    
    maritime_regions.append(
        Polygon([(-5.0, 30.0), (-5.0, 45.0), (35.0, 45.0), (35.0, 30.0)])  # mediterranean_sea
    )
    
    maritime_regions.append(
        Polygon([(48.0, 24.0), (48.0, 30.0), (57.0, 30.0), (57.0, 24.0)])  # persian_gulf
    )
    
    maritime_regions.append(
        Polygon([(43.0, 10.0), (43.0, 15.0), (51.0, 15.0), (51.0, 10.0)])  # gulf_of_aden
    )
    
    # Strategic choke points (local scale)
    maritime_regions.append(
        Polygon([(98.0, 1.0), (98.0, 6.5), (104.0, 6.5), (104.0, 1.0)])  # strait_of_malacca
    )
    
    maritime_regions.append(
        Polygon([(-6.0, 35.5), (-6.0, 36.5), (-5.0, 36.5), (-5.0, 35.5)])  # strait_of_gibraltar
    )
    
    maritime_regions.append(
        Polygon([(55.0, 25.0), (55.0, 27.0), (57.5, 27.0), (57.5, 25.0)])  # strait_of_hormuz
    )
    
    return maritime_regions

# --- AIRSPACE REGIONS ---
def create_airspace_polygons() -> List[Polygon]:
    """Create airspace regions of interest"""
    airspace_regions = []
    
    # National airspace (country scale)
    airspace_regions.append(
        Polygon([(-125.0, 25.0), (-125.0, 49.0), (-66.0, 49.0), (-66.0, 25.0)])  # us_airspace
    )
    
    airspace_regions.append(
        Polygon([(75.0, 18.0), (75.0, 54.0), (135.0, 54.0), (135.0, 18.0)])  # china_airspace
    )
    
    # Regional airspace (regional scale)
    airspace_regions.append(
        Polygon([(-85.0, 30.0), (-85.0, 45.0), (-65.0, 45.0), (-65.0, 30.0)])  # us_east_coast
    )
    
    airspace_regions.append(
        Polygon([(-125.0, 32.0), (-125.0, 49.0), (-115.0, 49.0), (-115.0, 32.0)])  # us_west_coast
    )
    
    airspace_regions.append(
        Polygon([(110.0, 20.0), (110.0, 40.0), (125.0, 40.0), (125.0, 20.0)])  # eastern_china
    )
    
    # Strategic areas and conflict zones (local scale)
    airspace_regions.append(
        Polygon([(118.0, 22.0), (118.0, 25.0), (122.0, 25.0), (122.0, 22.0)])  # taiwan_strait
    )
    
    airspace_regions.append(
        Polygon([(109.0, 5.0), (109.0, 22.0), (120.0, 22.0), (120.0, 5.0)])  # south_china_sea_airspace
    )
    
    airspace_regions.append(
        Polygon([(124.0, 34.0), (124.0, 43.0), (131.0, 43.0), (131.0, 34.0)])  # korean_peninsula
    )
    
    return airspace_regions

# --- URBAN REGIONS ---
def create_urban_polygons() -> List[Polygon]:
    """Create urban regions based on major cities"""
    urban_regions = []
    
    # Global major urban centers
    major_cities = [
        ("tokyo", 35.6895, 139.6917),
        ("delhi", 28.7041, 77.1025),
        ("shanghai", 31.2304, 121.4737),
        ("new_york", 40.7128, -74.0060),
        ("los_angeles", 34.0522, -118.2437),
        ("london", 51.5074, -0.1278),
        ("paris", 48.8566, 2.3522),
        ("beijing", 39.9042, 116.4074),
        ("mumbai", 19.0760, 72.8777),
        ("istanbul", 41.0082, 28.9784)
    ]
    
    # Create multi-scale urban regions
    for city_name, lat, lon in major_cities:
        # Core urban area (city proper)
        urban_regions.append(
            Polygon([
                (lon - 0.15, lat - 0.15),
                (lon - 0.15, lat + 0.15),
                (lon + 0.15, lat + 0.15),
                (lon + 0.15, lat - 0.15)
            ])
        )
        
        # Metropolitan area (overlapping with core)
        urban_regions.append(
            Polygon([
                (lon - 0.5, lat - 0.5),
                (lon - 0.5, lat + 0.5),
                (lon + 0.5, lat + 0.5),
                (lon + 0.5, lat - 0.5)
            ])
        )
    
    # Create urban corridors (connecting multiple cities)
    urban_regions.append(
        Polygon([(-77.5, 37.0), (-77.5, 43.0), (-71.0, 43.0), (-71.0, 37.0)])  # US Northeast Corridor
    )
    
    urban_regions.append(
        Polygon([(117.0, 30.0), (117.0, 40.0), (122.0, 40.0), (122.0, 30.0)])  # Chinese Eastern Seaboard
    )
    
    return urban_regions

# Combined function to get all regions
def get_all_regions() -> List[Polygon]:
    """Get all regions from all categories"""
    all_regions = []
    all_regions.extend(create_river_polygons())
    all_regions.extend(create_wildfire_polygons())
    all_regions.extend(create_earthquake_polygons())
    all_regions.extend(create_maritime_polygons())
    all_regions.extend(create_airspace_polygons())
    all_regions.extend(create_urban_polygons())
    return all_regions

# ===== ENHANCED FILTER SET =====

flood_filters_tpu = [
    Filter("F1", "Water Extent Change Detection", 4.4300, {"pass": 0.05, "fail": 0.95}),  # 10*0.44 + 0.03
    Filter("F2", "Water Turbidity Analysis", 1.6300, {"pass": 0.06, "fail": 0.94}),     # 10*0.16 + 0.03
    Filter("F3", "Water Depth Estimation", 1.9300, {"pass": 0.04, "fail": 0.96}),       # 10*0.19 + 0.03
    Filter("F4", "Vegetation Submersion Detection", 1.5300, {"pass": 0.055, "fail": 0.945}), # 10*0.15 + 0.03
    Filter("F5", "Infrastructure Submersion Detection", 1.7300, {"pass": 0.045, "fail": 0.955}), # 10*0.17 + 0.03
    Filter("F6", "Historical Flooding Pattern Comparison", 2.2300, {"pass": 0.03, "fail": 0.97}), # 10*0.22 + 0.03
    Filter("F7", "Flash Flood Risk Assessment", 2.0300, {"pass": 0.04, "fail": 0.96}), # 10*0.20 + 0.03
]
# -- Jetson Cost --
flood_filters_jetson = [
    Filter("F1", "Water Extent Change Detection", 1.3700, {"pass": 0.05, "fail": 0.95}),  # 3*0.44 + 0.05
    Filter("F2", "Water Turbidity Analysis", 0.5300, {"pass": 0.06, "fail": 0.94}),     # 3*0.16 + 0.05
    Filter("F3", "Water Depth Estimation", 0.6200, {"pass": 0.04, "fail": 0.96}),       # 3*0.19 + 0.05
    Filter("F4", "Vegetation Submersion Detection", 0.5000, {"pass": 0.055, "fail": 0.945}), # 3*0.15 + 0.05
    Filter("F5", "Infrastructure Submersion Detection", 0.5600, {"pass": 0.045, "fail": 0.955}), # 3*0.17 + 0.05
    Filter("F6", "Historical Flooding Pattern Comparison", 0.7100, {"pass": 0.03, "fail": 0.97}), # 3*0.22 + 0.05
    Filter("F7", "Flash Flood Risk Assessment", 0.6500, {"pass": 0.04, "fail": 0.96}), # 3*0.20 + 0.05
]

# ==== FIRE EVENT DETECTION FILTERS ====
# -- Edge TPU Cost --
wildfire_filters_tpu = [
    Filter("W1", "Smoke Plume Detection", 1.6300, {"pass": 0.055, "fail": 0.945}), # 10*0.16 + 0.03
    Filter("W2", "Thermal Anomaly Detection", 1.5300, {"pass": 0.065, "fail": 0.935}), # 10*0.15 + 0.03
    Filter("W3", "Active Fire Front Mapping", 1.8300, {"pass": 0.05, "fail": 0.95}), # 10*0.18 + 0.03
    Filter("W4", "Burn Scar Mapping", 4.7300, {"pass": 0.07, "fail": 0.93}),     # 10*0.47 + 0.03
    Filter("W5", "Vegetation Mortality Assessment", 1.9300, {"pass": 0.06, "fail": 0.94}), # 10*0.19 + 0.03
    Filter("W6", "Fire Spread Direction Prediction", 2.2300, {"pass": 0.035, "fail": 0.965}), # 10*0.22 + 0.03
    Filter("W7", "Fire Intensity Estimation", 2.0300, {"pass": 0.04, "fail": 0.96}), # 10*0.20 + 0.03
]
# -- Jetson Cost --
wildfire_filters_jetson = [
    Filter("W1", "Smoke Plume Detection", 0.5300, {"pass": 0.055, "fail": 0.945}), # 3*0.16 + 0.05
    Filter("W2", "Thermal Anomaly Detection", 0.5000, {"pass": 0.065, "fail": 0.935}), # 3*0.15 + 0.05
    Filter("W3", "Active Fire Front Mapping", 0.5900, {"pass": 0.05, "fail": 0.95}), # 3*0.18 + 0.05
    Filter("W4", "Burn Scar Mapping", 1.4600, {"pass": 0.07, "fail": 0.93}),     # 3*0.47 + 0.05
    Filter("W5", "Vegetation Mortality Assessment", 0.6200, {"pass": 0.06, "fail": 0.94}), # 3*0.19 + 0.05
    Filter("W6", "Fire Spread Direction Prediction", 0.7100, {"pass": 0.035, "fail": 0.965}), # 3*0.22 + 0.05
    Filter("W7", "Fire Intensity Estimation", 0.6500, {"pass": 0.04, "fail": 0.96}), # 3*0.20 + 0.05
]

# ==== EARTHQUAKE DAMAGE FILTERS ====
# -- Edge TPU Cost --
earthquake_filters_tpu = [
    Filter("E1", "Building Collapse Detection", 3.9300, {"pass": 0.035, "fail": 0.965}), # 10*0.39 + 0.03
    Filter("E2", "Structural Damage Patterns", 2.1300, {"pass": 0.045, "fail": 0.955}), # 10*0.21 + 0.03
    Filter("E3", "Building Alignment Change", 3.8300, {"pass": 0.05, "fail": 0.95}), # 10*0.38 + 0.03
    Filter("E4", "Road Network Disruption", 2.0300, {"pass": 0.055, "fail": 0.945}), # 10*0.20 + 0.03
    Filter("E5", "Bridge Failure Assessment", 2.2300, {"pass": 0.03, "fail": 0.97}), # 10*0.22 + 0.03
    Filter("E6", "Surface Rupture Detection", 2.3300, {"pass": 0.02, "fail": 0.98}), # 10*0.23 + 0.03
    Filter("E7", "Landslide Detection Post-Earthquake", 2.1300, {"pass": 0.04, "fail": 0.96}), # 10*0.21 + 0.03
]
# -- Jetson Cost --
earthquake_filters_jetson = [
    Filter("E1", "Building Collapse Detection", 1.2200, {"pass": 0.035, "fail": 0.965}), # 3*0.39 + 0.05
    Filter("E2", "Structural Damage Patterns", 0.6800, {"pass": 0.045, "fail": 0.955}), # 3*0.21 + 0.05
    Filter("E3", "Building Alignment Change", 1.1900, {"pass": 0.05, "fail": 0.95}), # 3*0.38 + 0.05
    Filter("E4", "Road Network Disruption", 0.6500, {"pass": 0.055, "fail": 0.945}), # 3*0.20 + 0.05
    Filter("E5", "Bridge Failure Assessment", 0.7100, {"pass": 0.03, "fail": 0.97}), # 3*0.22 + 0.05
    Filter("E6", "Surface Rupture Detection", 0.7400, {"pass": 0.02, "fail": 0.98}), # 3*0.23 + 0.05
    Filter("E7", "Landslide Detection Post-Earthquake", 0.6800, {"pass": 0.04, "fail": 0.96}), # 3*0.21 + 0.05
]

# ==== MARITIME VESSEL DETECTION FILTERS ====
# -- Edge TPU Cost --
ship_filters_tpu = [
    Filter("S1", "Large Vessel Detection (>100m)", 6.4300, {"pass": 0.08, "fail": 0.92}), # 10*0.64 + 0.03
    Filter("S2", "Medium Vessel Detection (20-100m)", 5.6300, {"pass": 0.07, "fail": 0.93}), # 10*0.56 + 0.03
    Filter("S3", "Small Vessel Detection (<20m)", 3.0300, {"pass": 0.05, "fail": 0.95}), # 10*0.30 + 0.03
    Filter("S4", "Vessel Wake Pattern Analysis", 1.7300, {"pass": 0.06, "fail": 0.94}), # 10*0.17 + 0.03
    Filter("S5", "Vessel Superstructure Classification", 2.0300, {"pass": 0.05, "fail": 0.95}), # 10*0.20 + 0.03
    Filter("S6", "Vessel Formation Detection", 2.2300, {"pass": 0.04, "fail": 0.96}), # 10*0.22 + 0.03
    Filter("S7", "Ship-to-Ship Transfer Detection", 2.3300, {"pass": 0.03, "fail": 0.97}), # 10*0.23 + 0.03
]
# -- Jetson Cost --
ship_filters_jetson = [
    Filter("S1", "Large Vessel Detection (>100m)", 1.9700, {"pass": 0.08, "fail": 0.92}), # 3*0.64 + 0.05
    Filter("S2", "Medium Vessel Detection (20-100m)", 1.7300, {"pass": 0.07, "fail": 0.93}), # 3*0.56 + 0.05
    Filter("S3", "Small Vessel Detection (<20m)", 0.9500, {"pass": 0.05, "fail": 0.95}), # 3*0.30 + 0.05
    Filter("S4", "Vessel Wake Pattern Analysis", 0.5600, {"pass": 0.06, "fail": 0.94}), # 3*0.17 + 0.05
    Filter("S5", "Vessel Superstructure Classification", 0.6500, {"pass": 0.05, "fail": 0.95}), # 3*0.20 + 0.05
    Filter("S6", "Vessel Formation Detection", 0.7100, {"pass": 0.04, "fail": 0.96}), # 3*0.22 + 0.05
    Filter("S7", "Ship-to-Ship Transfer Detection", 0.7400, {"pass": 0.03, "fail": 0.97}), # 3*0.23 + 0.05
]

# ==== VESSEL CLASSIFICATION FILTERS ====
# -- Edge TPU Cost --
ship_classification_filters_tpu = [
    Filter("SC1", "Naval Vessel Classification", 1.9300, {"pass": 0.055, "fail": 0.945}), # 10*0.19 + 0.03
    Filter("SC2", "Commercial Cargo Vessel Classification", 1.7300, {"pass": 0.065, "fail": 0.935}), # 10*0.17 + 0.03
    Filter("SC3", "Tanker Vessel Classification", 1.6300, {"pass": 0.07, "fail": 0.93}), # 10*0.16 + 0.03
    Filter("SC4", "Passenger Vessel Classification", 1.8300, {"pass": 0.06, "fail": 0.94}), # 10*0.18 + 0.03
    Filter("SC5", "Fishing Vessel Classification", 2.0300, {"pass": 0.05, "fail": 0.95}), # 10*0.20 + 0.03
]
# -- Jetson Cost --
ship_classification_filters_jetson = [
    Filter("SC1", "Naval Vessel Classification", 0.6200, {"pass": 0.055, "fail": 0.945}), # 3*0.19 + 0.05
    Filter("SC2", "Commercial Cargo Vessel Classification", 0.5600, {"pass": 0.065, "fail": 0.935}), # 3*0.17 + 0.05
    Filter("SC3", "Tanker Vessel Classification", 0.5300, {"pass": 0.07, "fail": 0.93}), # 3*0.16 + 0.05
    Filter("SC4", "Passenger Vessel Classification", 0.5900, {"pass": 0.06, "fail": 0.94}), # 3*0.18 + 0.05
    Filter("SC5", "Fishing Vessel Classification", 0.6500, {"pass": 0.05, "fail": 0.95}), # 3*0.20 + 0.05
]

# ==== AIRCRAFT DETECTION FILTERS ====
# -- Edge TPU Cost --
aircraft_filters_tpu = [
    Filter("A1", "Large Aircraft Detection", 3.0300, {"pass": 0.07, "fail": 0.93}),     # 10*0.30 + 0.03
    Filter("A2", "Small Aircraft Detection", 4.8300, {"pass": 0.055, "fail": 0.945}), # 10*0.48 + 0.03
    Filter("A3", "Helicopter Detection", 3.7300, {"pass": 0.06, "fail": 0.94}),     # 10*0.37 + 0.03
    Filter("A4", "Aircraft Contrail Analysis", 1.6300, {"pass": 0.065, "fail": 0.935}), # 10*0.16 + 0.03
    Filter("A5", "Aircraft Formation Detection", 2.1300, {"pass": 0.045, "fail": 0.955}), # 10*0.21 + 0.03
    Filter("A6", "Low-Altitude Flight Detection", 1.9300, {"pass": 0.05, "fail": 0.95}), # 10*0.19 + 0.03
    Filter("A7", "Military Aircraft Classification", 2.2300, {"pass": 0.04, "fail": 0.96}), # 10*0.22 + 0.03
    Filter("A8", "Commercial Aircraft Classification", 2.0300, {"pass": 0.05, "fail": 0.95}), # 10*0.20 + 0.03
]
# -- Jetson Cost --
aircraft_filters_jetson = [
    Filter("A1", "Large Aircraft Detection", 0.9500, {"pass": 0.07, "fail": 0.93}),     # 3*0.30 + 0.05
    Filter("A2", "Small Aircraft Detection", 1.4900, {"pass": 0.055, "fail": 0.945}), # 3*0.48 + 0.05
    Filter("A3", "Helicopter Detection", 1.1600, {"pass": 0.06, "fail": 0.94}),     # 3*0.37 + 0.05
    Filter("A4", "Aircraft Contrail Analysis", 0.5300, {"pass": 0.065, "fail": 0.935}), # 3*0.16 + 0.05
    Filter("A5", "Aircraft Formation Detection", 0.6800, {"pass": 0.045, "fail": 0.955}), # 3*0.21 + 0.05
    Filter("A6", "Low-Altitude Flight Detection", 0.6200, {"pass": 0.05, "fail": 0.95}), # 3*0.19 + 0.05
    Filter("A7", "Military Aircraft Classification", 0.7100, {"pass": 0.04, "fail": 0.96}), # 3*0.22 + 0.05
    Filter("A8", "Commercial Aircraft Classification", 0.6500, {"pass": 0.05, "fail": 0.95}), # 3*0.20 + 0.05
]

# ==== GENERAL-PURPOSE ENVIRONMENTAL FILTERS ====
# -- Edge TPU Cost --
general_environmental_filters_tpu = [
    Filter("G1", "Cloud Cover Assessment", 1.2300, {"pass": 0.30, "fail": 0.70}),     # 10*0.12 + 0.03
    Filter("G2", "Atmospheric Haze Detection", 1.4300, {"pass": 0.20, "fail": 0.80}), # 10*0.14 + 0.03
    Filter("G3", "Precipitation Pattern Analysis", 1.6300, {"pass": 0.15, "fail": 0.85}), # 10*0.16 + 0.03
    Filter("G4", "Urban Area Detection", 1.5300, {"pass": 0.25, "fail": 0.75}),     # 10*0.15 + 0.03
    Filter("G5", "Agricultural Field Detection", 1.4300, {"pass": 0.25, "fail": 0.75}), # 10*0.14 + 0.03
    Filter("G6", "Forest Cover Assessment", 1.6300, {"pass": 0.20, "fail": 0.80}),   # 10*0.16 + 0.03
    Filter("G7", "Deforestation Detection", 1.8300, {"pass": 0.08, "fail": 0.92}),   # 10*0.18 + 0.03
]
# -- Jetson Cost --
general_environmental_filters_jetson = [
    Filter("G1", "Cloud Cover Assessment", 0.4100, {"pass": 0.30, "fail": 0.70}),     # 3*0.12 + 0.05
    Filter("G2", "Atmospheric Haze Detection", 0.4700, {"pass": 0.20, "fail": 0.80}), # 3*0.14 + 0.05
    Filter("G3", "Precipitation Pattern Analysis", 0.5300, {"pass": 0.15, "fail": 0.85}), # 3*0.16 + 0.05
    Filter("G4", "Urban Area Detection", 0.5000, {"pass": 0.25, "fail": 0.75}),     # 3*0.15 + 0.05
    Filter("G5", "Agricultural Field Detection", 0.4700, {"pass": 0.25, "fail": 0.75}), # 3*0.14 + 0.05
    Filter("G6", "Forest Cover Assessment", 0.5300, {"pass": 0.20, "fail": 0.80}),   # 3*0.16 + 0.05
    Filter("G7", "Deforestation Detection", 0.5900, {"pass": 0.08, "fail": 0.92}),   # 3*0.18 + 0.05
]

# ==== INFRASTRUCTURE MONITORING FILTERS ====
# -- Edge TPU Cost --
infrastructure_filters_tpu = [
    Filter("I1", "Power Plant Activity Assessment", 3.8300, {"pass": 0.07, "fail": 0.93}), # 10*0.38 + 0.03
    Filter("I2", "Power Transmission Line Monitoring", 2.0300, {"pass": 0.06, "fail": 0.94}), # 10*0.20 + 0.03
    Filter("I3", "Pipeline Integrity Assessment", 5.2300, {"pass": 0.05, "fail": 0.95}), # 10*0.52 + 0.03
    Filter("I4", "Road Network Condition Analysis", 1.7300, {"pass": 0.09, "fail": 0.91}), # 10*0.17 + 0.03
    Filter("I5", "Railway Activity Monitoring", 1.6300, {"pass": 0.095, "fail": 0.905}), # 10*0.16 + 0.03
    Filter("I6", "Bridge Structural Assessment", 4.1300, {"pass": 0.055, "fail": 0.945}), # 10*0.41 + 0.03
    Filter("I7", "Airport Operations Analysis", 1.9300, {"pass": 0.075, "fail": 0.925}), # 10*0.19 + 0.03
]
# -- Jetson Cost --
infrastructure_filters_jetson = [
    Filter("I1", "Power Plant Activity Assessment", 1.1900, {"pass": 0.07, "fail": 0.93}), # 3*0.38 + 0.05
    Filter("I2", "Power Transmission Line Monitoring", 0.6500, {"pass": 0.06, "fail": 0.94}), # 3*0.20 + 0.05
    Filter("I3", "Pipeline Integrity Assessment", 1.6100, {"pass": 0.05, "fail": 0.95}), # 3*0.52 + 0.05
    Filter("I4", "Road Network Condition Analysis", 0.5600, {"pass": 0.09, "fail": 0.91}), # 3*0.17 + 0.05
    Filter("I5", "Railway Activity Monitoring", 0.5300, {"pass": 0.095, "fail": 0.905}), # 3*0.16 + 0.05
    Filter("I6", "Bridge Structural Assessment", 1.2800, {"pass": 0.055, "fail": 0.945}), # 3*0.41 + 0.05
    Filter("I7", "Airport Operations Analysis", 0.6200, {"pass": 0.075, "fail": 0.925}), # 3*0.19 + 0.05
]

# ==== BORDER AND SECURITY MONITORING FILTERS ====
# -- Edge TPU Cost --
security_filters_tpu = [
    Filter("B1", "Border Crossing Detection", 2.1300, {"pass": 0.05, "fail": 0.95}), # 10*0.21 + 0.03
    Filter("B2", "Border Infrastructure Assessment", 1.9300, {"pass": 0.06, "fail": 0.94}), # 10*0.19 + 0.03
    Filter("B3", "Unauthorized Path Detection", 2.2300, {"pass": 0.04, "fail": 0.96}), # 10*0.22 + 0.03
    Filter("B4", "Crowd Gathering Detection", 1.8300, {"pass": 0.08, "fail": 0.92}), # 10*0.18 + 0.03
    Filter("B5", "Vehicle Congregation Detection", 1.7300, {"pass": 0.085, "fail": 0.915}), # 10*0.17 + 0.03
    Filter("B6", "Security Checkpoint Activity", 1.6300, {"pass": 0.09, "fail": 0.91}), # 10*0.16 + 0.03
    Filter("B7", "Military Base Activity Assessment", 2.3300, {"pass": 0.035, "fail": 0.965}), # 10*0.23 + 0.03
    Filter("B8", "Sensitive Facility Monitoring", 2.2300, {"pass": 0.04, "fail": 0.96}), # 10*0.22 + 0.03
]
# -- Jetson Cost --
security_filters_jetson = [
    Filter("B1", "Border Crossing Detection", 0.6800, {"pass": 0.05, "fail": 0.95}), # 3*0.21 + 0.05
    Filter("B2", "Border Infrastructure Assessment", 0.6200, {"pass": 0.06, "fail": 0.94}), # 3*0.19 + 0.05
    Filter("B3", "Unauthorized Path Detection", 0.7100, {"pass": 0.04, "fail": 0.96}), # 3*0.22 + 0.05
    Filter("B4", "Crowd Gathering Detection", 0.5900, {"pass": 0.08, "fail": 0.92}), # 3*0.18 + 0.05
    Filter("B5", "Vehicle Congregation Detection", 0.5600, {"pass": 0.085, "fail": 0.915}), # 3*0.17 + 0.05
    Filter("B6", "Security Checkpoint Activity", 0.5300, {"pass": 0.09, "fail": 0.91}), # 3*0.16 + 0.05
    Filter("B7", "Military Base Activity Assessment", 0.7400, {"pass": 0.035, "fail": 0.965}), # 3*0.23 + 0.05
    Filter("B8", "Sensitive Facility Monitoring", 0.7100, {"pass": 0.04, "fail": 0.96}), # 3*0.22 + 0.05
]

# ==== WATER EVENT DETECTION FILTERS ====
flood_filters = [
    # Primary flood detection filters
    Filter("F1", "Water Extent Change Detection", 0.44, {"pass": 0.05, "fail": 0.95}),
    Filter("F2", "Water Turbidity Analysis", 0.16, {"pass": 0.06, "fail": 0.94}),
    Filter("F3", "Water Depth Estimation", 0.19, {"pass": 0.04, "fail": 0.96}),
    
    # Secondary flood indicators
    Filter("F4", "Vegetation Submersion Detection", 0.15, {"pass": 0.055, "fail": 0.945}),
    Filter("F5", "Infrastructure Submersion Detection", 0.17, {"pass": 0.045, "fail": 0.955}),
    
    # Specialized flood analysis
    Filter("F6", "Historical Flooding Pattern Comparison", 0.22, {"pass": 0.03, "fail": 0.97}),
    Filter("F7", "Flash Flood Risk Assessment", 0.20, {"pass": 0.04, "fail": 0.96}),
]

# ==== FIRE EVENT DETECTION FILTERS ====
wildfire_filters = [
    # Primary fire detection
    Filter("W1", "Smoke Plume Detection", 0.16, {"pass": 0.055, "fail": 0.945}),
    Filter("W2", "Thermal Anomaly Detection", 0.15, {"pass": 0.065, "fail": 0.935}),
    Filter("W3", "Active Fire Front Mapping", 0.18, {"pass": 0.05, "fail": 0.95}),
    
    # Fire impact assessment
    Filter("W4", "Burn Scar Mapping", 0.47, {"pass": 0.07, "fail": 0.93}),
    Filter("W5", "Vegetation Mortality Assessment", 0.19, {"pass": 0.06, "fail": 0.94}),
    
    # Fire behavior analysis
    Filter("W6", "Fire Spread Direction Prediction", 0.22, {"pass": 0.035, "fail": 0.965}),
    Filter("W7", "Fire Intensity Estimation", 0.20, {"pass": 0.04, "fail": 0.96}),
]

# ==== EARTHQUAKE DAMAGE FILTERS ====
earthquake_filters = [
    # Building damage assessment
    Filter("E1", "Building Collapse Detection", 0.39, {"pass": 0.035, "fail": 0.965}),
    Filter("E2", "Structural Damage Patterns", 0.21, {"pass": 0.045, "fail": 0.955}),
    Filter("E3", "Building Alignment Change", 0.38, {"pass": 0.05, "fail": 0.95}),
    
    # Infrastructure damage
    Filter("E4", "Road Network Disruption", 0.20, {"pass": 0.055, "fail": 0.945}),
    Filter("E5", "Bridge Failure Assessment", 0.22, {"pass": 0.03, "fail": 0.97}),
    
    # Geological effects
    Filter("E6", "Surface Rupture Detection", 0.23, {"pass": 0.02, "fail": 0.98}),
    Filter("E7", "Landslide Detection Post-Earthquake", 0.21, {"pass": 0.04, "fail": 0.96}),
]

# ==== MARITIME VESSEL DETECTION FILTERS ====
ship_filters = [
    # Basic vessel detection
    Filter("S1", "Large Vessel Detection (>100m)", 0.64, {"pass": 0.08, "fail": 0.92}),
    Filter("S2", "Medium Vessel Detection (20-100m)", 0.56, {"pass": 0.07, "fail": 0.93}),
    Filter("S3", "Small Vessel Detection (<20m)", 0.3, {"pass": 0.05, "fail": 0.95}),
    
    # Vessel characteristics
    Filter("S4", "Vessel Wake Pattern Analysis", 0.17, {"pass": 0.06, "fail": 0.94}),
    Filter("S5", "Vessel Superstructure Classification", 0.20, {"pass": 0.05, "fail": 0.95}),
    
    # Vessel behavior and context
    Filter("S6", "Vessel Formation Detection", 0.22, {"pass": 0.04, "fail": 0.96}),
    Filter("S7", "Ship-to-Ship Transfer Detection", 0.23, {"pass": 0.03, "fail": 0.97}),
]

# ==== VESSEL CLASSIFICATION FILTERS ====
ship_classification_filters = [
    # Vessel type classification
    Filter("SC1", "Naval Vessel Classification", 0.19, {"pass": 0.055, "fail": 0.945}),
    Filter("SC2", "Commercial Cargo Vessel Classification", 0.17, {"pass": 0.065, "fail": 0.935}),
    Filter("SC3", "Tanker Vessel Classification", 0.16, {"pass": 0.07, "fail": 0.93}),
    Filter("SC4", "Passenger Vessel Classification", 0.18, {"pass": 0.06, "fail": 0.94}),
    Filter("SC5", "Fishing Vessel Classification", 0.20, {"pass": 0.05, "fail": 0.95}),
]

# ==== AIRCRAFT DETECTION FILTERS ====
aircraft_filters = [
    # Basic aircraft detection
    Filter("A1", "Large Aircraft Detection", 0.30, {"pass": 0.07, "fail": 0.93}),
    Filter("A2", "Small Aircraft Detection", 0.48, {"pass": 0.055, "fail": 0.945}),
    Filter("A3", "Helicopter Detection", 0.37, {"pass": 0.06, "fail": 0.94}),
    
    # Aircraft characteristics and behavior
    Filter("A4", "Aircraft Contrail Analysis", 0.16, {"pass": 0.065, "fail": 0.935}),
    Filter("A5", "Aircraft Formation Detection", 0.21, {"pass": 0.045, "fail": 0.955}),
    Filter("A6", "Low-Altitude Flight Detection", 0.19, {"pass": 0.05, "fail": 0.95}),
    
    # Aircraft type classification
    Filter("A7", "Military Aircraft Classification", 0.22, {"pass": 0.04, "fail": 0.96}),
    Filter("A8", "Commercial Aircraft Classification", 0.20, {"pass": 0.05, "fail": 0.95}),
]

# ==== GENERAL-PURPOSE ENVIRONMENTAL FILTERS ====
general_environmental_filters = [
    # Atmospheric conditions
    Filter("G1", "Cloud Cover Assessment", 0.12, {"pass": 0.30, "fail": 0.70}),
    Filter("G2", "Atmospheric Haze Detection", 0.14, {"pass": 0.20, "fail": 0.80}),
    Filter("G3", "Precipitation Pattern Analysis", 0.16, {"pass": 0.15, "fail": 0.85}),
    
    # Land cover and change
    Filter("G4", "Urban Area Detection", 0.15, {"pass": 0.25, "fail": 0.75}),
    Filter("G5", "Agricultural Field Detection", 0.14, {"pass": 0.25, "fail": 0.75}),
    Filter("G6", "Forest Cover Assessment", 0.16, {"pass": 0.20, "fail": 0.80}),
    Filter("G7", "Deforestation Detection", 0.18, {"pass": 0.08, "fail": 0.92}),
]

# ==== INFRASTRUCTURE MONITORING FILTERS ====
infrastructure_filters = [
    # Critical infrastructure
    Filter("I1", "Power Plant Activity Assessment", 0.38, {"pass": 0.07, "fail": 0.93}),
    Filter("I2", "Power Transmission Line Monitoring", 0.20, {"pass": 0.06, "fail": 0.94}),
    Filter("I3", "Pipeline Integrity Assessment", 0.52, {"pass": 0.05, "fail": 0.95}),
    
    # Transportation infrastructure
    Filter("I4", "Road Network Condition Analysis", 0.17, {"pass": 0.09, "fail": 0.91}),
    Filter("I5", "Railway Activity Monitoring", 0.16, {"pass": 0.095, "fail": 0.905}),
    Filter("I6", "Bridge Structural Assessment", 0.41, {"pass": 0.055, "fail": 0.945}),
    Filter("I7", "Airport Operations Analysis", 0.19, {"pass": 0.075, "fail": 0.925}),
]

# ==== BORDER AND SECURITY MONITORING FILTERS ====
security_filters = [
    # Border monitoring
    Filter("B1", "Border Crossing Detection", 0.21, {"pass": 0.05, "fail": 0.95}),
    Filter("B2", "Border Infrastructure Assessment", 0.19, {"pass": 0.06, "fail": 0.94}),
    Filter("B3", "Unauthorized Path Detection", 0.22, {"pass": 0.04, "fail": 0.96}),
    
    # Security-relevant activities
    Filter("B4", "Crowd Gathering Detection", 0.18, {"pass": 0.08, "fail": 0.92}),
    Filter("B5", "Vehicle Congregation Detection", 0.17, {"pass": 0.085, "fail": 0.915}),
    Filter("B6", "Security Checkpoint Activity", 0.16, {"pass": 0.09, "fail": 0.91}),
    
    # Critical site monitoring
    Filter("B7", "Military Base Activity Assessment", 0.23, {"pass": 0.035, "fail": 0.965}),
    Filter("B8", "Sensitive Facility Monitoring", 0.22, {"pass": 0.04, "fail": 0.96}),
]

# Combine all filters into a comprehensive list
all_filters = (
    flood_filters + wildfire_filters + earthquake_filters + 
    ship_filters + ship_classification_filters + aircraft_filters + 
    general_environmental_filters + infrastructure_filters + security_filters
)

tpu_filters = (
    flood_filters_tpu + wildfire_filters_tpu + earthquake_filters_tpu +
    ship_filters_tpu + ship_classification_filters_tpu + aircraft_filters_tpu +
    general_environmental_filters_tpu + infrastructure_filters_tpu + security_filters_tpu
)

jetson_filters = (
    flood_filters_jetson + wildfire_filters_jetson + earthquake_filters_jetson +
    ship_filters_jetson + ship_classification_filters_jetson + aircraft_filters_jetson +
    general_environmental_filters_jetson + infrastructure_filters_jetson + security_filters_jetson)

def get_all_filters(hardware) -> List[Filter]:
    """Return all filters"""
    if hardware == "tpu":
        return tpu_filters
    if hardware == "gpu":
        return jetson_filters
    return all_filters

# Create filter lookup dictionary
filter_lookup = {f.filter_id: f for f in all_filters}

# ===== QUERY CREATION FUNCTIONS =====

# 1.1 ADVANCED FLOOD DETECTION
def create_flood_detection_query() -> Query:
    """Create an advanced flood detection query with complex filter combinations"""
    river_aois = create_river_polygons()
    
    # DNF formula with multiple complex conditions:
    # (F1 AND F2) OR (F1 AND F4 AND F7) OR (F1 AND F3 AND G1)
    filter_dnf = [
        ["F1", "F2"],               # Water extent + turbidity (basic detection)
        ["F1", "F4", "F7"],         # Water extent + vegetation submersion + flash flood risk
        ["F1", "F3", "G1"]          # Water extent + water depth + cloud cover (for partial visibility)
    ]
    
    query = Query(
        AOI=river_aois,
        priority_tier=9,            # High priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None                  # No specific time, run continuously
    )
    
    return query

# 1.2 ADVANCED WILDFIRE DETECTION
def create_wildfire_detection_query() -> Query:
    """Create an advanced wildfire detection query with complex filter combinations"""
    wildfire_aois = create_wildfire_polygons()
    
    # DNF formula with multiple complex conditions:
    # (W1 AND W2) OR (W2 AND W3 AND W7) OR (W4 AND W5 AND G6)
    filter_dnf = [
        ["W1", "W2"],               # Smoke plume + thermal anomaly (active fire)
        ["W2", "W3", "W7"],         # Thermal anomaly + active fire front + intensity estimation
        ["W4", "W5", "G6"]          # Burn scar + vegetation mortality + forest assessment (post-fire)
    ]
    
    query = Query(
        AOI=wildfire_aois,
        priority_tier=10,            # Highest priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None                  # No specific time, run continuously
    )
    
    return query

# 1.3 ADVANCED EARTHQUAKE DAMAGE DETECTION
def create_earthquake_detection_query() -> Query:
    """Create an advanced earthquake damage detection query with complex filter combinations"""
    earthquake_aois = create_earthquake_polygons()
    
    # DNF formula with multiple complex conditions:
    # (E1 AND E2) OR (E3 AND E4 AND E5) OR (E6 AND E7 AND I6)
    filter_dnf = [
        ["E1", "E2"],               # Building collapse + structural damage patterns
        ["E3", "E4", "E5"],         # Building alignment change + road disruption + bridge failure
        ["E6", "E7", "I6"]          # Surface rupture + landslide + bridge assessment
    ]
    
    # Set high priority for post-earthquake damage detection
    # Use one-time query with 24-hour horizon from current time
    next_24_hours = datetime.now() + timedelta(hours=24)
    
    query = Query(
        AOI=earthquake_aois,
        priority_tier=9,            # High priority
        type="one-time",
        filter_categories=filter_dnf,
        time=next_24_hours
    )
    
    return query

# 1.4 URBAN DISASTER MONITORING
def create_urban_disaster_query() -> Query:
    """Create an urban disaster monitoring query with complex filter combinations"""
    urban_aois = create_urban_polygons()
    
    # DNF formula combining multiple disaster types in urban settings:
    # (F1 AND F5 AND G4) OR (E1 AND E2 AND G4) OR (W1 AND W2 AND G4)
    filter_dnf = [
        ["F1", "F5", "G4"],         # Urban flooding: Water extent + infrastructure submersion + urban area
        ["E1", "E2", "G4"],         # Urban earthquake damage: Building collapse + structural damage + urban area
        ["W1", "W2", "G4"]          # Urban fire: Smoke plume + thermal anomaly + urban area
    ]
    
    query = Query(
        AOI=urban_aois,
        priority_tier=10,            # Highest priority for urban disasters
        type="recurring",
        filter_categories=filter_dnf,
        time=None
    )
    
    return query

# 2.1 ADVANCED MARITIME MONITORING
def create_maritime_monitoring_query() -> Query:
    """Create an advanced maritime vessel monitoring query with complex filter combinations"""
    maritime_aois = create_maritime_polygons()
    
    # DNF formula with multiple complex conditions:
    # (S1 AND S4 AND S5) OR (S2 AND S5 AND SC2) OR (S1 AND S6 AND SC1)
    filter_dnf = [
        ["S1", "S4", "S5"],         # Large vessel + wake pattern + superstructure (vessel characterization)
        ["S2", "S5", "SC2"],        # Medium vessel + superstructure + commercial classification
        ["S1", "S6", "SC1"]         # Large vessel + formation + naval classification (military vessels)
    ]
    
    query = Query(
        AOI=maritime_aois,
        priority_tier=7,            # Medium priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None
    )
    
    return query

# 2.2 ADVANCED AIRSPACE MONITORING
def create_airspace_monitoring_query() -> Query:
    """Create an advanced airspace monitoring query with complex filter combinations"""
    airspace_aois = create_airspace_polygons()
    
    # DNF formula with multiple complex conditions:
    # (A1 AND A4 AND A8) OR (A2 AND A6 AND A7) OR (A3 AND A5 AND A7)
    filter_dnf = [
        ["A1", "A4", "A8"],         # Large aircraft + contrail + commercial (commercial traffic)
        ["A2", "A6", "A7"],         # Small aircraft + low altitude + military (military operations)
        ["A3", "A5", "A7"]          # Helicopter + formation + military (tactical operations)
    ]
    
    query = Query(
        AOI=airspace_aois,
        priority_tier=8,            # High priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None
    )
    
    return query

# 2.3 CRITICAL INFRASTRUCTURE MONITORING
def create_infrastructure_monitoring_query() -> Query:
    """Create a critical infrastructure monitoring query with complex filter combinations"""
    # Combine areas with critical infrastructure (urban + earthquake zones)
    infra_aois = create_urban_polygons() + create_earthquake_polygons()[:3]  # Limit to important earthquake zones
    
    # DNF formula with multiple complex conditions:
    # (I1 AND I2 AND G4) OR (I3 AND I4 AND B8) OR (I6 AND I7 AND B2)
    filter_dnf = [
        ["I1", "I2", "G4"],         # Power plant + transmission lines + urban area
        ["I3", "I4", "B8"],         # Pipeline + road network + sensitive facility
        ["I6", "I7", "B2"]          # Bridge assessment + airport operations + border infrastructure
    ]
    
    query = Query(
        AOI=infra_aois,
        priority_tier=8,            # Medium priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None
    )
    
    return query

# 2.4 SECURITY AND BORDER MONITORING
def create_security_monitoring_query() -> Query:
    """Create a security and border monitoring query with complex filter combinations"""
    # Use select maritime regions and airspace that are security-sensitive
    security_aois = create_airspace_polygons()[-3:] + create_maritime_polygons()[-3:]  # Last items are more strategic/sensitive
    
    # DNF formula with multiple complex conditions:
    # (B1 AND B3 AND B5) OR (B4 AND B7 AND S3) OR (B2 AND B6 AND B8)
    filter_dnf = [
        ["B1", "B3", "B5"],         # Border crossing + unauthorized path + vehicle congregation
        ["B4", "B7", "S3"],         # Crowd gathering + military base + small vessel (coastal security)
        ["B2", "B6", "B8"]          # Border infrastructure + checkpoint + sensitive facility
    ]
    
    query = Query(
        AOI=security_aois,
        priority_tier=9,            # High priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None
    )
    
    return query

# 3.1 MULTI-DOMAIN COMBINED MONITORING
def create_combined_monitoring_query() -> Query:
    """Create a combined multi-domain query that integrates multiple monitoring types"""
    # Use a diverse set of regions to capture multi-domain events
    combined_aois = (create_river_polygons()[:3] + create_maritime_polygons()[:2] + 
                    create_urban_polygons()[:5] + create_earthquake_polygons()[:2])
    
    # Complex DNF formula that cuts across domains:
    # (F1 AND I4 AND G4) OR (S1 AND SC1 AND A7) OR (E1 AND B4 AND I6) OR (W1 AND G7 AND B7)
    filter_dnf = [
        ["F1", "I4", "G4"],         # Flooding impact on infrastructure: Water extent + roads + urban area
        ["S1", "SC1", "A7"],        # Combined naval-air operations: Large vessel + naval classification + military aircraft
        ["E1", "B4", "I6"],         # Disaster response: Building collapse + crowd gathering + bridge assessment
        ["W1", "G7", "B7"]          # Wildfire near sensitive sites: Smoke plume + deforestation + military base
    ]
    
    query = Query(
        AOI=combined_aois,
        priority_tier=10,            # Highest priority
        type="recurring",
        filter_categories=filter_dnf,
        time=None
    )
    
    return query

# 3.2 GLOBAL MONITORING SYSTEM
def create_global_monitoring_system() -> List[Query]:
    """Create a comprehensive set of diverse, complex queries for a global monitoring system"""
    # Create all individual queries
    global_system = [
        create_flood_detection_query(),
        create_wildfire_detection_query(),
        create_earthquake_detection_query(),
        create_urban_disaster_query(),
        create_maritime_monitoring_query(),
        create_airspace_monitoring_query(),
        create_infrastructure_monitoring_query(),
        create_security_monitoring_query(),
        create_combined_monitoring_query()
    ]
    
    # Add additional time-sensitive queries
    
    # Seasonal wildfire monitoring
    seasonal_aois = create_wildfire_polygons()[2:6]  # Select a subset of wildfire regions
    seasonal_filter_dnf = [
        ["W2", "W5", "W6"],         # Thermal anomaly + vegetation mortality + spread prediction
        ["W4", "W7", "G6"],         # Burn scar + intensity + forest assessment
        ["W1", "W3", "G2"]          # Smoke plume + active fire + atmospheric haze
    ]
    
    seasonal_query = Query(
        AOI=seasonal_aois,
        priority_tier=9,            # High priority
        type="one-time",            # Seasonal assessment
        filter_categories=seasonal_filter_dnf,
        time=datetime.now() + timedelta(days=90)  # 3 months in the future
    )
    
    # Crisis response simulation
    crisis_aois = create_urban_polygons()[4:6]  # Select a few urban areas
    crisis_filter_dnf = [
        ["B4", "B5", "I4"],         # Crowd + vehicles + road network (civil unrest)
        ["F1", "F5", "I6"],         # Flood + infrastructure submersion + bridges (flood crisis)
        ["E1", "E4", "B4"]          # Building collapse + road disruption + crowd (disaster response)
    ]
    
    crisis_query = Query(
        AOI=crisis_aois,
        priority_tier=10,            # Highest priority
        type="one-time",            # Immediate response
        filter_categories=crisis_filter_dnf,
        time=datetime.now() + timedelta(hours=48)  # 2 days in the future
    )
    
    # Add these to the global system
    global_system.append(seasonal_query)
    global_system.append(crisis_query)
    
    return global_system

# Function to run all natural disaster scenario queries
def run_natural_disaster_scenario():
    flood_query = create_flood_detection_query()
    wildfire_query = create_wildfire_detection_query()
    earthquake_query = create_earthquake_detection_query()
    urban_disaster_query = create_urban_disaster_query()
    
    print("Natural Disaster Scenario Queries:")
    print("1. Flood Detection Query:", flood_query)
    print("2. Wildfire Detection Query:", wildfire_query)
    print("3. Earthquake Detection Query:", earthquake_query)
    print("4. Urban Disaster Query:", urban_disaster_query)
    
    return [flood_query, wildfire_query, earthquake_query, urban_disaster_query]

# Function to run all military and security scenario queries
def run_military_scenario():
    maritime_query = create_maritime_monitoring_query()
    airspace_query = create_airspace_monitoring_query()
    infrastructure_query = create_infrastructure_monitoring_query()
    security_query = create_security_monitoring_query()
    
    print("Military & Security Scenario Queries:")
    print("1. Maritime Monitoring Query:", maritime_query)
    print("2. Airspace Monitoring Query:", airspace_query)
    print("3. Infrastructure Monitoring Query:", infrastructure_query)
    print("4. Security Monitoring Query:", security_query)
    
    return [maritime_query, airspace_query, infrastructure_query, security_query]

# Function to run combined scenario with all complex queries
def run_combined_scenario():
    combined_query = create_combined_monitoring_query()
    global_system = create_global_monitoring_system()
    
    print("Combined Scenario Queries:")
    print("1. Multi-Domain Combined Query:", combined_query)
    print("2. Global Monitoring System Queries:", len(global_system), "queries")
    
    return [combined_query] + global_system

# Main function to run all scenarios
def run_all_scenarios():
    natural_disaster_queries = run_natural_disaster_scenario()
    military_queries = run_military_scenario()
    combined_queries = run_combined_scenario()
    
    all_queries = natural_disaster_queries + military_queries + combined_queries
    
    print("\nComplete Satellite Inference Benchmark:")
    print(f"Total Queries: {len(all_queries)}")
    
    # Count query priorities
    priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for query in all_queries:
        priority = query.priority_tier
        if priority in priority_counts:
            priority_counts[priority] += 1
    
    print(f"Priority Distribution: {priority_counts}")
    
    # Calculate average filter complexity
    total_conditions = 0
    total_groups = 0
    for query in all_queries:
        for group in query.filter_categories:
            total_groups += 1
            total_conditions += len(group)
    
    avg_conditions = total_conditions / total_groups if total_groups > 0 else 0
    print(f"Average Filter Complexity: {avg_conditions:.2f} conditions per group")
    
    # Analyze overlap in filter conditions
    filter_usage = {f.filter_id: 0 for f in all_filters}
    for query in all_queries:
        for group in query.filter_categories:
            for filter_id in group:
                filter_usage[filter_id] = filter_usage.get(filter_id, 0) + 1
    
    top_filters = sorted(filter_usage.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 Most Used Filters:")
    for filter_id, count in top_filters:
        print(f"  {filter_id}: {count} uses")
    
    return all_queries


def create_global_grid(coverage_percentage: float = 10.0) -> List[Polygon]:
    """
    Create a global grid of polygons that covers the specified percentage of Earth's surface.
    
    Args:
        coverage_percentage: Percentage of Earth's surface to cover (0-100)
    
    Returns:
        List of polygon regions representing the coverage
    """
    # Validate input
    if not 0 <= coverage_percentage <= 100:
        raise ValueError("Coverage percentage must be between 0 and 100")
    
    # Earth is approximately 510 million km², so we'll create a grid system
    # that covers the requested percentage
    
    # Calculate number of grid cells needed
    # We'll use 5° x 5° grid cells (each about 0.77% of Earth's surface)
    # This is approximate since Earth is not a perfect sphere
    
    cell_coverage_pct = 0.77  # Approximate percentage covered by one 5°x5° cell
    num_cells_needed = math.ceil((coverage_percentage / 100) * (100 / cell_coverage_pct))
    
    grid_polygons = []
    
    # Create coverage with a mix of strategic and random sampling
    
    # 1. STRATEGIC COVERAGE - Always include key areas regardless of percentage
    # Major urban centers and critical regions (about 20% of our allocation)
    strategic_locations = [
        # Format: (lon_min, lat_min, lon_max, lat_max, name)
        (-74.5, 40.5, -73.5, 41.0, "new_york"),       # New York
        (-118.5, 33.5, -117.5, 34.5, "los_angeles"),  # Los Angeles
        (115.5, 39.5, 117.0, 40.5, "beijing"),        # Beijing
        (139.5, 35.5, 140.5, 36.5, "tokyo"),          # Tokyo
        (77.0, 28.5, 78.0, 29.0, "delhi"),            # Delhi
        (30.0, 30.0, 32.0, 32.0, "nile_delta"),       # Nile Delta
        (90.0, 23.0, 92.0, 25.0, "ganges_delta"),     # Ganges Delta
        (-122.5, 37.0, -121.5, 38.0, "san_francisco"), # San Francisco (earthquake-prone)
        (106.5, 10.0, 107.5, 11.0, "mekong_delta"),   # Mekong Delta (flood-prone)
        (-120.0, 35.0, -118.0, 37.0, "california_fire") # California wildfire region
    ]
    
    # Adjust the number of strategic locations based on coverage percentage
    strategic_limit = min(len(strategic_locations), max(1, int(num_cells_needed * 0.2)))
    
    # Add strategic locations first
    for i in range(strategic_limit):
        lon_min, lat_min, lon_max, lat_max, name = strategic_locations[i]
        grid_polygons.append(
            Polygon([(lon_min, lat_min), (lon_min, lat_max), 
                     (lon_max, lat_max), (lon_max, lat_min)])
        )
    
    # 2. RANDOM GLOBAL COVERAGE - Distribute remaining cells
    remaining_cells = num_cells_needed - strategic_limit
    
    # Create a global grid to sample from
    # Longitude: -180 to 180 in 5° increments
    # Latitude: -90 to 90 in 5° increments
    all_grid_cells = []
    
    for lon in range(-180, 180, 5):
        for lat in range(-90, 90, 5):
            # Skip cells that are mostly ocean (very simple approach)
            # This is a simplified approach - a real implementation would use a land mask
            if random.random() < 0.3:  # 30% chance to include a cell to approximate land distribution
                all_grid_cells.append((lon, lat))
    
    # Shuffle the grid cells and take the required number
    random.shuffle(all_grid_cells)
    selected_cells = all_grid_cells[:remaining_cells]
    
    # Convert selected grid cells to polygons
    for lon, lat in selected_cells:
        grid_polygons.append(
            Polygon([(lon, lat), (lon, lat+5), 
                     (lon+5, lat+5), (lon+5, lat)])
        )
    
    return grid_polygons

def create_coverage_scaling_query(
    coverage_percentage: float = 10.0,
    priority_tier: int = 5,
    query_type: str = "recurring",
    focus_categories: List[str] = None
) -> Query:
    """
    Create a query that dynamically scales to cover the specified percentage of Earth's surface.
    
    Args:
        coverage_percentage: Percentage of Earth's surface to cover (0-100)
        priority_tier: Priority level for this query (1-10)
        query_type: Either "recurring" or "one-time"
        focus_categories: Optional list of focus categories to prioritize 
                         ["flood", "wildfire", "earthquake", "maritime", "airspace", "urban"]
    
    Returns:
        Query object configured for the specified coverage
    """
    # Generate polygons covering the requested percentage of Earth
    coverage_polygons = create_global_grid(coverage_percentage)
    
    # Define filter DNF based on requested focus categories
    if not focus_categories:
        focus_categories = ["general"]  # Default to general monitoring
    
    filter_dnf = []
    
    # Add filter combinations based on focus categories
    if "flood" in focus_categories or "general" in focus_categories:
        filter_dnf.append(["F1", "F2", "G1"])  # Water extent + turbidity + cloud cover
    
    if "wildfire" in focus_categories or "general" in focus_categories:
        filter_dnf.append(["W1", "W2", "G2"])  # Smoke plume + thermal anomaly + haze
    
    if "earthquake" in focus_categories or "general" in focus_categories:
        filter_dnf.append(["E1", "E4", "G4"])  # Building collapse + road disruption + urban area
    
    if "maritime" in focus_categories or "general" in focus_categories:
        filter_dnf.append(["S1", "S4", "G3"])  # Large vessel + wake pattern + precipitation
    
    if "airspace" in focus_categories or "general" in focus_categories:
        filter_dnf.append(["A1", "A4", "G1"])  # Large aircraft + contrail + cloud cover
    
    if "urban" in focus_categories or "general" in focus_categories:
        filter_dnf.append(["G4", "I1", "I4"])  # Urban area + power plant + road network
    
    # If no valid categories were specified, add a general monitoring set
    if not filter_dnf:
        filter_dnf = [
            ["G1", "G2", "G3"],      # Weather conditions
            ["G4", "G5", "G6"],      # Land cover types
            ["I4", "S1", "A1"]       # Infrastructure and mobility
        ]
    
    # Set up time for one-time queries
    query_time = None
    if query_type == "one-time":
        query_time = datetime.now() + timedelta(hours=24)
    
    # Create the query
    query = Query(
        AOI=coverage_polygons,
        priority_tier=priority_tier,
        type=query_type,
        filter_categories=filter_dnf,
        time=query_time
    )
    
    return query


def run_coverage_scaling_scenario(
    coverage_percentages: List[float] = [1.0, 5.0, 10.0, 25.0, 50.0],
    focus_categories: List[str] = None
) -> List[Query]:
    """
    Run the coverage scaling scenario with multiple coverage percentages.
    
    Args:
        coverage_percentages: List of coverage percentages to simulate
        focus_categories: Categories to focus on for monitoring
        
    Returns:
        List of Query objects with varying coverage
    """
    coverage_queries = []
    
    for percentage in coverage_percentages:
        # Adjust priority based on coverage (higher coverage = lower priority)
        priority = max(1, min(10, 10 - int(percentage / 10)))
        
        query = create_coverage_scaling_query(
            coverage_percentage=percentage,
            priority_tier=priority,
            focus_categories=focus_categories
        )
        
        coverage_queries.append(query)
        
        print(f"Created coverage query at {percentage}% of Earth's surface " +
              f"(priority: {priority}, polygons: {len(query.AOI)})")
    
    return coverage_queries
