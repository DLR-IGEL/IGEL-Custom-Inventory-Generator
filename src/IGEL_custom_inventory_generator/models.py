from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MetadataConfig:
    config_version: str
    inventory_name: str
    institution: str
    creator_name: str
    creator_email: str
    project: str

@dataclass(frozen=True)
class InputDataConfig:
    launch_list: Path
    propellant_use_profiles_dir: Path
    engine_data_dir: Path


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    run_name: str
    overwrite: bool
    export_individual_profiles: bool
    inventory_emission_unit: str
    inventory_vertical_coordinate: str
    inventory_netcdf_mode: str


@dataclass(frozen=True)
class DomainConfig:
    altitude_min_km: float
    altitude_max_km: float
    latitude_min_deg: float
    latitude_max_deg: float
    longitude_min_deg: float
    longitude_max_deg: float
    inclusive: bool
    emissions_above_inventory: str


@dataclass(frozen=True)
class GridConfig:
    altitude_resolution_km: float
    latitude_resolution_deg: float
    longitude_resolution_deg: float


@dataclass(frozen=True)
class ProcessingConfig:
    post_combustion: str | None
    time_resolution: str


@dataclass(frozen=True)
class TimeRangeConfig:
    start_date: datetime
    end_date: datetime


@dataclass(frozen=True)
class Config:
    metadata: MetadataConfig
    input_data: InputDataConfig
    output: OutputConfig
    domain: DomainConfig
    grid: GridConfig
    processing: ProcessingConfig
    time_range: TimeRangeConfig