from __future__ import annotations

import re
from decimal import Decimal

TOOL_NAME = "IGEL-custom-inventory-generator"
TOOL_VERSION = "1.0.0"

DATE_FORMAT = "%d-%m-%Y"
TIME_RESOLUTION_PATTERN = re.compile(r"^(?P<value>[1-9]\d*)(?P<unit>[dwm])$")

REQUIRED_ENGINE_COLUMNS = {"Species", "Absolute_Mass_Fraction"}
ENGINE_FILE_PATTERN = re.compile(r"^(?P<engine_name>.+)_primary_exhaust_indices\.csv$")
REQUIRED_LAUNCH_LIST_COLUMNS = {"Launch_Tag", "Launch_Date", "Multiplier"}
REQUIRED_PROFILE_TOTAL_COLUMN = "species_mass_Total"
ENGINE_PROFILE_COLUMN_PATTERN = re.compile(r"^species_mass_(?P<engine_name>.+)$")
SPECIES_MASS_COLUMN_PATTERN = re.compile(r"^species_mass_(?P<species>.+)$")
TIMESTEP_KEY_PATTERN = re.compile(r"^(?P<start>\d{4}-\d{2}-\d{2})__(?P<end>\d{4}-\d{2}-\d{2})$")

BASE_ALTITUDE_RESOLUTION_KM = Decimal("0.01")
BASE_LATITUDE_RESOLUTION_DEG = Decimal("0.01")
BASE_LONGITUDE_RESOLUTION_DEG = Decimal("0.01")

MIN_CREATED_ALTITUDE_RESOLUTION_KM = Decimal("0.02")
MIN_CREATED_LATITUDE_RESOLUTION_DEG = Decimal("0.02")
MIN_CREATED_LONGITUDE_RESOLUTION_DEG = Decimal("0.02")

AVOGADRO_CONSTANT = 6.02214076e23
EARTH_RADIUS_M = 6_371_000.0

SPECIES_DB = {
                'H':        {'MolarMass': 1.008,   'Hill': 'H'},
                'HCN':      {'MolarMass': 27.025,  'Hill': 'CHN'},
                'H2':       {'MolarMass': 2.016,   'Hill': 'H2'},
                'H2O':      {'MolarMass': 18.015,  'Hill': 'H2O'},
                'H2O2':     {'MolarMass': 34.014,  'Hill': 'H2O2'},
                'HCL':      {'MolarMass': 36.461,   'Hill': 'ClH'},
                'O':        {'MolarMass': 15.999,  'Hill': 'O'},
                'O2':       {'MolarMass': 31.9988, 'Hill': 'O2'},
                'OH':       {'MolarMass': 17.007,  'Hill': 'HO'},
                'CO':       {'MolarMass': 28.010,  'Hill': 'CO'},
                'CO2':      {'MolarMass': 44.001,  'Hill': 'CO2'},
                'COOH':     {'MolarMass': 45.017,  'Hill': 'CHO2'},
                'HCHO':     {'MolarMass': 30.026,  'Hill': 'CH2O'},
                'HCO':      {'MolarMass': 29.018,  'Hill': 'CHO'},
                'HCOOH':    {'MolarMass': 46.025,  'Hill': 'CH2O2'},
                'HO2':      {'MolarMass': 33.007,  'Hill': 'HO2'},
                'O3':       {'MolarMass': 47.998,  'Hill': 'O3'},
                'C2H4':     {'MolarMass': 28.053,  'Hill': 'C2H4'},
                'C2H6':     {'MolarMass': 30.069,  'Hill': 'C2H6'},
                'C3H8':     {'MolarMass': 44.096,  'Hill': 'C3H8'},
                'CH3CN':    {'MolarMass': 41.052,  'Hill': 'C2H3N'},
                'CH4':      {'MolarMass': 16.043,  'Hill': 'CH4'},
                'C(gr)':    {'MolarMass': 12.011,  'Hill': 'C'},
                'C':        {'MolarMass': 12.011,  'Hill': 'C'},
                'CL':       {'MolarMass': 35.453,  'Hill': 'Cl'},
                'CL2':      {'MolarMass': 70.906,  'Hill': 'Cl2'},
                'NH3':      {'MolarMass': 17.031,  'Hill': 'H3N'},
                'N2':       {'MolarMass': 28.0134, 'Hill': 'N2'},
                'NO':       {'MolarMass': 30.006,  'Hill': 'NO'},
                'FeCL3':    {'MolarMass': 162.204, 'Hill': 'Cl3Fe'},
                'ALH2O2':   {'MolarMass': 60.996,  'Hill': 'AlH2O2'},
                'ALCL':     {'MolarMass': 62.434,  'Hill': 'AlCl'},
                'ALCL2':    {'MolarMass': 97.887,  'Hill': 'AlCl2'},
                'ALCL3':    {'MolarMass': 133.340, 'Hill': 'AlCl3'},
                'ALHCL':    {'MolarMass': 79.442,  'Hill': 'AlClH'},
                'ALHCL2':   {'MolarMass': 98.895,  'Hill': 'AlCl2H'},
                'ALOCL':    {'MolarMass': 78.434,  'Hill': 'AlClO'},
                'ALOHCL2':  {'MolarMass': 114.895, 'Hill': 'AlCl2HO'},
                'ALO3':     {'MolarMass': 74.980,  'Hill': 'AlO3'},
                'AL2O3':    {'MolarMass': 101.961, 'Hill': 'Al2O3'},
                'ALOH':     {'MolarMass': 43.989,  'Hill': 'AlHO'},
                'ALOHCL':   {'MolarMass': 79.442,  'Hill': 'AlClHO'},
                'ALH3O3':   {'MolarMass': 78.004,  'Hill': 'AlH3O3'},
                'ALH2O2CL': {'MolarMass': 96.449,  'Hill': 'AlClH2O2'},
}

NETCDF_TIME_UNITS = "days since 1750-1-1 00:00:00"
NETCDF_TIME_CALENDAR = "proleptic_gregorian"
NETCDF_TIME_AXIS_NAME = "time"
NETCDF_VERTICAL_AXIS_NAME = "lev"
NETCDF_LAT_AXIS_NAME = "lat"
NETCDF_LON_AXIS_NAME = "lon"
NETCDF_TIME_BOUNDS_NAME = "time_bnds"
NETCDF_BOUNDS_VERTEX_DIM = "nv"
NETCDF_LAT_BOUNDS_NAME = "lat_bnds"
NETCDF_LON_BOUNDS_NAME = "lon_bnds"
NETCDF_VERTICAL_BOUNDS_NAME = "lev_bnds"

GLOBAL_LAT_MIN_DEG = -90.0
GLOBAL_LAT_MAX_DEG = 90.0
GLOBAL_LON_MIN_DEG = -180.0
GLOBAL_LON_MAX_DEG = 180.0

NETCDF_COMPRESSION_LEVEL = 4
NETCDF_COORD_ROUND_DECIMALS = 10

INVENTORY_SPATIAL_INDEX_COLUMNS = [
    "ALTITUDE_MIN",
    "ALTITUDE_MAX",
    "LATITUDE_MIN",
    "LATITUDE_MAX",
    "LONGITUDE_MIN",
    "LONGITUDE_MAX",
]

LAUNCH_DATE_INPUT_FORMAT = "%Y %b %d"

ALLOWED_INVENTORY_EMISSION_UNITS = {
    "mass",
    "molecules",
    "molecule_rate",
    "molecule_rate_per_volume",
}

ALLOWED_INVENTORY_VERTICAL_COORDINATES = {
    "pressure",
    "altitude",
}

ALLOWED_INVENTORY_NETCDF_MODES = {
    "single_file",
    "per_timestep_files",
}

ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS = {
    "discard",
    "include_at_top_layer",
}