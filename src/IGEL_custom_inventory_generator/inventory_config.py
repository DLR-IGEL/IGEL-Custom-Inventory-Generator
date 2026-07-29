from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from constants import (
    ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS,
    ALLOWED_INVENTORY_EMISSION_UNITS,
    ALLOWED_INVENTORY_NETCDF_MODES,
    ALLOWED_INVENTORY_VERTICAL_COORDINATES,
    DATE_FORMAT,
    TIME_RESOLUTION_PATTERN,
)
from errors import ConfigError
from models import (
    Config,
    DomainConfig,
    GridConfig,
    InputDataConfig,
    MetadataConfig,
    OutputConfig,
    ProcessingConfig,
    TimeRangeConfig,
)


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Top-level YAML structure must be a mapping/dictionary.")

    try:
        metadata_raw = raw["metadata"]
        input_data_raw = raw["input_data"]
        output_raw = raw["output"]
        domain_raw = raw["domain"]
        grid_raw = raw["grid"]
        processing_raw = raw["processing"]
        time_range_raw = raw["time_range"]
    except KeyError as exc:
        raise ConfigError(f"Missing required top-level section: {exc.args[0]}") from exc

    config = Config(
        metadata=_parse_metadata(metadata_raw),
        input_data=_parse_input_data(input_data_raw),
        output=_parse_output(output_raw),
        domain=_parse_domain(domain_raw),
        grid=_parse_grid(grid_raw),
        processing=_parse_processing(processing_raw),
        time_range=_parse_time_range(time_range_raw),
    )

    _validate_config(config)
    return config


def _parse_inventory_netcdf_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "output.inventory_netcdf_mode must be one of: "
            f"{sorted(ALLOWED_INVENTORY_NETCDF_MODES)}"
        )

    value = value.strip()
    if value not in ALLOWED_INVENTORY_NETCDF_MODES:
        raise ConfigError(
            "output.inventory_netcdf_mode must be one of: "
            f"{sorted(ALLOWED_INVENTORY_NETCDF_MODES)}"
        )

    return value


def _parse_metadata(data: dict[str, Any]) -> MetadataConfig:
    _ensure_mapping(data, "metadata")
    return MetadataConfig(
        config_version=str(_require_key(data, "config_version", "metadata")),
        inventory_name=str(_require_key(data, "inventory_name", "metadata")),
        institution=str(_require_key(data, "institution", "metadata")),
        creator_name=str(_require_key(data, "creator_name", "metadata")),
        creator_email=str(_require_key(data, "creator_email", "metadata")),
        project=str(_require_key(data, "project", "metadata")),
    )


def _parse_inventory_vertical_coordinate(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "output.inventory_vertical_coordinate must be one of: "
            f"{sorted(ALLOWED_INVENTORY_VERTICAL_COORDINATES)}"
        )

    value = value.strip()
    if value not in ALLOWED_INVENTORY_VERTICAL_COORDINATES:
        raise ConfigError(
            "output.inventory_vertical_coordinate must be one of: "
            f"{sorted(ALLOWED_INVENTORY_VERTICAL_COORDINATES)}"
        )

    return value


def _parse_inventory_emission_unit(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "output.inventory_emission_unit must be one of: "
            f"{sorted(ALLOWED_INVENTORY_EMISSION_UNITS)}"
        )

    value = value.strip()
    if value not in ALLOWED_INVENTORY_EMISSION_UNITS:
        raise ConfigError(
            "output.inventory_emission_unit must be one of: "
            f"{sorted(ALLOWED_INVENTORY_EMISSION_UNITS)}"
        )

    return value


def _parse_input_data(data: dict[str, Any]) -> InputDataConfig:
    _ensure_mapping(data, "input_data")
    return InputDataConfig(
        launch_list=Path(_require_key(data, "launch_list", "input_data")),
        propellant_use_profiles_dir=Path(
            _require_key(data, "propellant_use_profiles_dir", "input_data")
        ),
        engine_data_dir=Path(_require_key(data, "engine_data_dir", "input_data")),
    )


def _parse_output(data: dict[str, Any]) -> OutputConfig:
    _ensure_mapping(data, "output")
    return OutputConfig(
        directory=Path(_require_key(data, "directory", "output")),
        run_name=str(_require_key(data, "run_name", "output")),
        overwrite=bool(_require_key(data, "overwrite", "output")),
        export_individual_profiles=bool(
            _require_key(data, "export_individual_profiles", "output")
        ),
        inventory_emission_unit=_parse_inventory_emission_unit(
            _require_key(data, "inventory_emission_unit", "output")
        ),
        inventory_vertical_coordinate=_parse_inventory_vertical_coordinate(
            _require_key(data, "inventory_vertical_coordinate", "output")
        ),
        inventory_netcdf_mode=_parse_inventory_netcdf_mode(
            _require_key(data, "inventory_netcdf_mode", "output")
        ),
    )


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{field_name} must be a boolean (true or false).")


def _parse_domain(data: dict[str, Any]) -> DomainConfig:
    _ensure_mapping(data, "domain")
    return DomainConfig(
        altitude_min_km=float(_require_key(data, "altitude_min_km", "domain")),
        altitude_max_km=float(_require_key(data, "altitude_max_km", "domain")),
        latitude_min_deg=float(_require_key(data, "latitude_min_deg", "domain")),
        latitude_max_deg=float(_require_key(data, "latitude_max_deg", "domain")),
        longitude_min_deg=float(_require_key(data, "longitude_min_deg", "domain")),
        longitude_max_deg=float(_require_key(data, "longitude_max_deg", "domain")),
        inclusive=_parse_bool(_require_key(data, "inclusive", "domain"), "domain.inclusive"),
        emissions_above_inventory=_parse_emissions_above_inventory(
            _require_key(data, "emissions_above_inventory", "domain")
        ),
    )

def _parse_emissions_above_inventory(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "domain.emissions_above_inventory must be one of: "
            f"{sorted(ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS)}"
        )

    value = value.strip()
    if value not in ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS:
        raise ConfigError(
            "domain.emissions_above_inventory must be one of: "
            f"{sorted(ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS)}"
        )

    return value

def _parse_grid(data: dict[str, Any]) -> GridConfig:
    _ensure_mapping(data, "grid")
    return GridConfig(
        altitude_resolution_km=float(_require_key(data, "altitude_resolution_km", "grid")),
        latitude_resolution_deg=float(_require_key(data, "latitude_resolution_deg", "grid")),
        longitude_resolution_deg=float(_require_key(data, "longitude_resolution_deg", "grid")),
    )


def _parse_processing(data: dict[str, Any]) -> ProcessingConfig:
    _ensure_mapping(data, "processing")
    return ProcessingConfig(
        post_combustion=_parse_post_combustion(_require_key(data, "post_combustion", "processing")),
        time_resolution=str(_require_key(data, "time_resolution", "processing")),
    )


def _parse_post_combustion(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ConfigError("processing.post_combustion must be 'CSVEM' or null.")


def _parse_time_range(data: dict[str, Any]) -> TimeRangeConfig:
    _ensure_mapping(data, "time_range")
    start_date = datetime.strptime(
        str(_require_key(data, "start_date", "time_range")),
        DATE_FORMAT,
    )
    end_date = datetime.strptime(
        str(_require_key(data, "end_date", "time_range")),
        DATE_FORMAT,
    )
    return TimeRangeConfig(start_date=start_date, end_date=end_date)


def _validate_config(config: Config) -> None:
    _validate_multiple_of_0p02(config.grid.altitude_resolution_km, "grid.altitude_resolution_km")
    _validate_multiple_of_0p02(config.grid.latitude_resolution_deg, "grid.latitude_resolution_deg")
    _validate_multiple_of_0p02(config.grid.longitude_resolution_deg, "grid.longitude_resolution_deg")

    if config.domain.altitude_min_km < 0:
        raise ConfigError("domain.altitude_min_km must be >= 0.")
    if config.domain.altitude_max_km <= config.domain.altitude_min_km:
        raise ConfigError("domain.altitude_max_km must be greater than domain.altitude_min_km.")
    
    if (config.domain.altitude_max_km > 1000) and config.output.inventory_vertical_coordinate == "pressure" :
        raise ConfigError("domain.altitude_max_km must be lower or equal to 1000km if output.inventory_vertical_coordinate is set to pressure.")

    if not (-90 <= config.domain.latitude_min_deg <= 90):
        raise ConfigError("domain.latitude_min_deg must be between -90 and 90.")
    if not (-90 <= config.domain.latitude_max_deg <= 90):
        raise ConfigError("domain.latitude_max_deg must be between -90 and 90.")
    if config.domain.latitude_max_deg <= config.domain.latitude_min_deg:
        raise ConfigError("domain.latitude_max_deg must be greater than domain.latitude_min_deg.")

    if not (-180 <= config.domain.longitude_min_deg <= 180):
        raise ConfigError("domain.longitude_min_deg must be between -180 and 180.")
    if not (-180 <= config.domain.longitude_max_deg <= 180):
        raise ConfigError("domain.longitude_max_deg must be between -180 and 180.")
    if config.domain.longitude_max_deg <= config.domain.longitude_min_deg:
        raise ConfigError("domain.longitude_max_deg must be greater than domain.longitude_min_deg.")

    if config.processing.post_combustion not in ("CSVEM", None):
        raise ConfigError("processing.post_combustion must be either 'CSVEM' or null.")

    if not TIME_RESOLUTION_PATTERN.match(config.processing.time_resolution):
        raise ConfigError("processing.time_resolution must look like '1d', '2w', or '3m'.")

    if config.time_range.end_date < config.time_range.start_date:
        raise ConfigError("time_range.end_date must be on or after time_range.start_date.")

    if not config.output.run_name.strip():
        raise ConfigError("output.run_name must not be empty.")
    if not config.metadata.institution.strip():
        raise ConfigError("metadata.institution must not be empty.")
    if not config.metadata.creator_name.strip():
        raise ConfigError("metadata.creator_name must not be empty.")
    if not config.metadata.creator_email.strip():
        raise ConfigError("metadata.creator_email must not be empty.")
    if not config.metadata.project.strip():
        raise ConfigError("metadata.project must not be empty.")

    if not config.input_data.launch_list.exists():
        raise ConfigError(f"input_data.launch_list does not exist: {config.input_data.launch_list}")

    if not config.input_data.propellant_use_profiles_dir.exists():
        raise ConfigError(
            "input_data.propellant_use_profiles_dir does not exist: "
            f"{config.input_data.propellant_use_profiles_dir}"
        )

    if not config.input_data.engine_data_dir.exists():
        raise ConfigError(
            f"input_data.engine_data_dir does not exist: {config.input_data.engine_data_dir}"
        )

    if config.output.inventory_netcdf_mode not in ALLOWED_INVENTORY_NETCDF_MODES:
        raise ConfigError(
            "output.inventory_netcdf_mode must be one of: "
            f"{sorted(ALLOWED_INVENTORY_NETCDF_MODES)}"
        )
    if config.domain.emissions_above_inventory not in ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS:
        raise ConfigError(
            "domain.emissions_above_inventory must be one of: "
            f"{sorted(ALLOWED_EMISSIONS_ABOVE_INVENTORY_OPTIONS)}"
        )


def _validate_multiple_of_0p02(value: float, field_name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{field_name} must be > 0.")

    scaled = value / 0.02
    if not math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError(f"{field_name} must be a natural multiple of 0.01.")


def _require_key(data: dict[str, Any], key: str, section: str) -> Any:
    if key not in data:
        raise ConfigError(f"Missing required key '{key}' in section '{section}'.")
    return data[key]


def _ensure_mapping(data: Any, section: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{section}' must be a mapping/dictionary.")