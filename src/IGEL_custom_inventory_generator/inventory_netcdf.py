from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, UTC

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .constants import (
    GLOBAL_LAT_MAX_DEG,
    GLOBAL_LAT_MIN_DEG,
    GLOBAL_LON_MAX_DEG,
    GLOBAL_LON_MIN_DEG,
    NETCDF_BOUNDS_VERTEX_DIM,
    NETCDF_COMPRESSION_LEVEL,
    NETCDF_COORD_ROUND_DECIMALS,
    NETCDF_LAT_AXIS_NAME,
    NETCDF_LAT_BOUNDS_NAME,
    NETCDF_LON_AXIS_NAME,
    NETCDF_LON_BOUNDS_NAME,
    NETCDF_TIME_AXIS_NAME,
    NETCDF_TIME_BOUNDS_NAME,
    NETCDF_TIME_CALENDAR,
    NETCDF_TIME_UNITS,
    NETCDF_VERTICAL_AXIS_NAME,
    NETCDF_VERTICAL_BOUNDS_NAME,
    TOOL_NAME,
    TOOL_VERSION,
)
from .errors import ExportError
from .inventory_timesteps import parse_timestep_key
from .inventory_transform import _altitudes_km_to_pressures_pa
from .models import Config


def _round_coord_value(value: float) -> float:
    return round(float(value), NETCDF_COORD_ROUND_DECIMALS)


def _build_regular_midpoint_axis(
    min_value: float,
    max_value: float,
    resolution: float,
) -> np.ndarray:
    if resolution <= 0:
        raise ExportError("Axis resolution must be > 0.")

    count = int(round((max_value - min_value) / resolution)) + 1
    values = min_value + np.arange(count, dtype=float) * resolution
    return np.round(values, NETCDF_COORD_ROUND_DECIMALS)


def _build_bounds_from_regular_midpoints(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ExportError("Bounds can only be built from a one-dimensional midpoint axis.")
    if values.size < 2:
        raise ExportError("Cannot build bounds from fewer than two midpoint values.")

    diffs = np.diff(values)
    if not np.allclose(diffs, diffs[0], rtol=0.0, atol=1e-10):
        raise ExportError("Axis is not uniformly spaced; bounds cannot be inferred safely.")

    half = float(diffs[0]) / 2.0
    bounds = np.column_stack([values - half, values + half]).astype(np.float64)
    return bounds


def _get_inventory_species_columns_from_dict(
    timestep_inventory_dict: dict[str, pd.DataFrame],
) -> list[str]:
    columns: list[str] = []

    for df in timestep_inventory_dict.values():
        for col in df.columns:
            if col.startswith("species_") and col not in columns:
                columns.append(col)

    if not columns:
        raise ExportError(
            "No inventory species columns found in transformed timestep inventory data."
        )

    return columns


def _get_inventory_variable_units(inventory_emission_unit: str) -> str:
    if inventory_emission_unit == "mass":
        return "kg"
    if inventory_emission_unit == "molecules":
        return "molecules"
    if inventory_emission_unit == "molecule_rate":
        return "molecules s-1"
    if inventory_emission_unit == "molecule_rate_per_volume":
        return "molecules m-3 s-1"

    raise ExportError(
        f"Unsupported inventory_emission_unit for NetCDF metadata: {inventory_emission_unit}"
    )


def _build_vertical_axis_and_bounds_from_config(
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    altitude_midpoints_km = _build_regular_midpoint_axis(
        min_value=config.domain.altitude_min_km,
        max_value=config.domain.altitude_max_km,
        resolution=config.grid.altitude_resolution_km,
    )
    altitude_bounds_km = _build_bounds_from_regular_midpoints(altitude_midpoints_km)

    if config.output.inventory_vertical_coordinate == "altitude":
        lev_values = altitude_midpoints_km * 1000.0
        lev_bounds = altitude_bounds_km * 1000.0
        lev_attrs = {
            "standard_name": "altitude",
            "long_name": "altitude",
            "units": "m",
            "positive": "up",
            "axis": "Z",
        }
        return lev_values.astype(np.float64), lev_bounds.astype(np.float64), lev_attrs

    if config.output.inventory_vertical_coordinate == "pressure":
        lev_values = _altitudes_km_to_pressures_pa(altitude_midpoints_km)

        pressure_at_lower_alt = _altitudes_km_to_pressures_pa(altitude_bounds_km[:, 0])
        pressure_at_upper_alt = _altitudes_km_to_pressures_pa(altitude_bounds_km[:, 1])

        lev_bounds = np.column_stack(
            [pressure_at_lower_alt, pressure_at_upper_alt]
        ).astype(np.float64)

        lev_attrs = {
            "standard_name": "air_pressure",
            "long_name": "ambient_air_pressure",
            "units": "Pa",
            "positive": "down",
            "axis": "Z",
        }
        return np.asarray(lev_values, dtype=np.float64), lev_bounds, lev_attrs

    raise ExportError(
        "Unsupported inventory_vertical_coordinate for NetCDF export: "
        f"{config.output.inventory_vertical_coordinate}"
    )


def _to_cftime(ts: pd.Timestamp) -> cftime.datetime:
    ts = pd.Timestamp(ts).normalize()

    try:
        if NETCDF_TIME_CALENDAR == "proleptic_gregorian":
            return cftime.DatetimeProlepticGregorian(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR in {"standard", "gregorian"}:
            return cftime.DatetimeGregorian(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR == "julian":
            return cftime.DatetimeJulian(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR == "noleap":
            return cftime.DatetimeNoLeap(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR == "365_day":
            return cftime.DatetimeNoLeap(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR == "all_leap":
            return cftime.DatetimeAllLeap(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR == "366_day":
            return cftime.DatetimeAllLeap(ts.year, ts.month, ts.day, 0, 0, 0)
        if NETCDF_TIME_CALENDAR == "360_day":
            return cftime.Datetime360Day(ts.year, ts.month, ts.day, 0, 0, 0)

        raise ExportError(
            f"Unsupported NETCDF_TIME_CALENDAR '{NETCDF_TIME_CALENDAR}' for cftime conversion."
        )
    except ValueError as exc:
        raise ExportError(
            f"Could not represent date {ts.strftime('%Y-%m-%d')} "
            "in the configured NetCDF time calendar."
        ) from exc


def _build_time_values_and_bounds(
    timestep_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    time_values: list[float] = []
    time_bounds: list[list[float]] = []

    for timestep_key in timestep_keys:
        timestep_start, timestep_end = parse_timestep_key(timestep_key)
        timestep_end_exclusive = pd.Timestamp(timestep_end) + pd.Timedelta(days=1)

        start_cf = _to_cftime(pd.Timestamp(timestep_start))
        end_cf = _to_cftime(pd.Timestamp(timestep_end_exclusive))

        bound_start = float(
            cftime.date2num(
                start_cf,
                units=NETCDF_TIME_UNITS,
                calendar=NETCDF_TIME_CALENDAR,
            )
        )
        bound_end = float(
            cftime.date2num(
                end_cf,
                units=NETCDF_TIME_UNITS,
                calendar=NETCDF_TIME_CALENDAR,
            )
        )

        time_bounds.append([bound_start, bound_end])
        time_values.append((bound_start + bound_end) / 2.0)

    return (
        np.asarray(time_values, dtype=np.float64),
        np.asarray(time_bounds, dtype=np.float64),
    )


def _get_coordinate_column_names_for_inventory(
    inventory_vertical_coordinate: str,
) -> tuple[str, str, str]:
    if inventory_vertical_coordinate == "altitude":
        return "LATITUDE", "LONGITUDE", "ALTITUDE"
    if inventory_vertical_coordinate == "pressure":
        return "LATITUDE", "LONGITUDE", "PRESSURE"

    raise ExportError(
        f"Unsupported inventory_vertical_coordinate: {inventory_vertical_coordinate}"
    )


def _build_coordinate_index(values: np.ndarray) -> dict[float, int]:
    return {_round_coord_value(v): idx for idx, v in enumerate(values.tolist())}


def _build_inventory_dataset(
    timestep_inventory_dict: dict[str, pd.DataFrame],
    config: Config,
    species_columns: list[str],
) -> xr.Dataset:
    timestep_keys = sorted(timestep_inventory_dict.keys())

    if not timestep_keys:
        raise ExportError("No timestep inventories available for NetCDF export.")

    time_values, time_bounds = _build_time_values_and_bounds(timestep_keys)

    lat_values = _build_regular_midpoint_axis(
        min_value=GLOBAL_LAT_MIN_DEG,
        max_value=GLOBAL_LAT_MAX_DEG,
        resolution=config.grid.latitude_resolution_deg,
    ).astype(np.float64)
    lat_bounds = _build_bounds_from_regular_midpoints(lat_values)

    lon_values = _build_regular_midpoint_axis(
        min_value=GLOBAL_LON_MIN_DEG,
        max_value=GLOBAL_LON_MAX_DEG,
        resolution=config.grid.longitude_resolution_deg,
    ).astype(np.float64)
    lon_bounds = _build_bounds_from_regular_midpoints(lon_values)

    lev_values, lev_bounds, lev_attrs = _build_vertical_axis_and_bounds_from_config(config)

    lat_col, lon_col, lev_col = _get_coordinate_column_names_for_inventory(
        config.output.inventory_vertical_coordinate
    )

    lat_index = _build_coordinate_index(lat_values)
    lon_index = _build_coordinate_index(lon_values)
    lev_index = _build_coordinate_index(lev_values)

    shape = (
        len(timestep_keys),
        len(lev_values),
        len(lat_values),
        len(lon_values),
    )

    data_vars: dict[str, xr.DataArray] = {}
    variable_units = _get_inventory_variable_units(config.output.inventory_emission_unit)

    for species_column in tqdm(
        species_columns,
        desc="Building Inventory Dataset",
        unit="species",
    ):
        data = np.zeros(shape, dtype=np.float64)

        for time_idx, timestep_key in enumerate(timestep_keys):
            timestep_df = timestep_inventory_dict[timestep_key]

            if timestep_df.empty:
                continue

            required_columns = {lat_col, lon_col, lev_col, species_column}
            missing_columns = required_columns.difference(timestep_df.columns)
            if missing_columns:
                raise ExportError(
                    f"Timestep dataframe '{timestep_key}' is missing columns required for "
                    f"NetCDF export: {sorted(missing_columns)}"
                )

            export_df = timestep_df[[lat_col, lon_col, lev_col, species_column]].copy()

            for row in export_df.itertuples(index=False):
                lat_value = _round_coord_value(getattr(row, lat_col))
                lon_value = _round_coord_value(getattr(row, lon_col))
                emission_value = float(getattr(row, species_column))

                if lat_value not in lat_index:
                    raise ExportError(
                        f"Latitude value {lat_value} from timestep '{timestep_key}' "
                        "is not on the full global output grid."
                    )
                if lon_value not in lon_index:
                    raise ExportError(
                        f"Longitude value {lon_value} from timestep '{timestep_key}' "
                        "is not on the full global output grid."
                    )

                lev_idx = _find_nearest_coordinate_index(
                    value=float(getattr(row, lev_col)),
                    axis_values=lev_values,
                    tolerance=1.0 if config.output.inventory_vertical_coordinate == "pressure" else 1e-6,
                )

                data[
                    time_idx,
                    lev_idx,
                    lat_index[lat_value],
                    lon_index[lon_value],
                ] += emission_value

        data_vars[species_column] = xr.DataArray(
            data,
            dims=(
                NETCDF_TIME_AXIS_NAME,
                NETCDF_VERTICAL_AXIS_NAME,
                NETCDF_LAT_AXIS_NAME,
                NETCDF_LON_AXIS_NAME,
            ),
            attrs={
                "long_name": species_column,
                "units": variable_units,
            },
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            NETCDF_TIME_AXIS_NAME: xr.DataArray(
                time_values,
                dims=(NETCDF_TIME_AXIS_NAME,),
                attrs={
                    "standard_name": "time",
                    "long_name": "time",
                    "bounds": NETCDF_TIME_BOUNDS_NAME,
                    "axis": "T",
                    "units": NETCDF_TIME_UNITS,
                    "calendar": NETCDF_TIME_CALENDAR,
                },
            ),
            NETCDF_VERTICAL_AXIS_NAME: xr.DataArray(
                lev_values,
                dims=(NETCDF_VERTICAL_AXIS_NAME,),
                attrs={
                    **lev_attrs,
                    "bounds": NETCDF_VERTICAL_BOUNDS_NAME,
                },
            ),
            NETCDF_LAT_AXIS_NAME: xr.DataArray(
                lat_values,
                dims=(NETCDF_LAT_AXIS_NAME,),
                attrs={
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                    "bounds": NETCDF_LAT_BOUNDS_NAME,
                },
            ),
            NETCDF_LON_AXIS_NAME: xr.DataArray(
                lon_values,
                dims=(NETCDF_LON_AXIS_NAME,),
                attrs={
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                    "bounds": NETCDF_LON_BOUNDS_NAME,
                },
            ),
            NETCDF_BOUNDS_VERTEX_DIM: xr.DataArray(
                np.array([0, 1], dtype=np.int32),
                dims=(NETCDF_BOUNDS_VERTEX_DIM,),
                attrs={
                    "long_name": "bounds_vertex_index",
                    "units": "1",
                },
            ),
        },
        attrs=_build_global_netcdf_attributes(
            config=config,
            timestep_keys=timestep_keys,
        ),
    )

    ds[NETCDF_TIME_BOUNDS_NAME] = xr.DataArray(
        time_bounds,
        dims=(NETCDF_TIME_AXIS_NAME, NETCDF_BOUNDS_VERTEX_DIM),
        attrs={
            "long_name": "time bounds",
            "units": NETCDF_TIME_UNITS,
            "calendar": NETCDF_TIME_CALENDAR,
        },
    )

    ds[NETCDF_LAT_BOUNDS_NAME] = xr.DataArray(
        lat_bounds,
        dims=(NETCDF_LAT_AXIS_NAME, NETCDF_BOUNDS_VERTEX_DIM),
        attrs={
            "long_name": "latitude bounds",
            "units": "degrees_north",
        },
    )

    ds[NETCDF_LON_BOUNDS_NAME] = xr.DataArray(
        lon_bounds,
        dims=(NETCDF_LON_AXIS_NAME, NETCDF_BOUNDS_VERTEX_DIM),
        attrs={
            "long_name": "longitude bounds",
            "units": "degrees_east",
        },
    )

    ds[NETCDF_VERTICAL_BOUNDS_NAME] = xr.DataArray(
        lev_bounds,
        dims=(NETCDF_VERTICAL_AXIS_NAME, NETCDF_BOUNDS_VERTEX_DIM),
        attrs={
            "long_name": "vertical bounds",
            "units": lev_attrs["units"],
        },
    )

    return ds


def _build_netcdf_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}

    bounds_variables = {
        NETCDF_TIME_BOUNDS_NAME,
        NETCDF_LAT_BOUNDS_NAME,
        NETCDF_LON_BOUNDS_NAME,
        NETCDF_VERTICAL_BOUNDS_NAME,
    }

    for coord_name in ds.coords:
        if coord_name == NETCDF_BOUNDS_VERTEX_DIM:
            encoding[coord_name] = {
                "dtype": "int32",
                "_FillValue": None,
            }
        else:
            encoding[coord_name] = {
                "dtype": "float64",
                "_FillValue": None,
            }

    for bounds_name in bounds_variables:
        if bounds_name in ds.data_vars:
            encoding[bounds_name] = {
                "dtype": "float64",
                "_FillValue": None,
            }

    for data_var_name in ds.data_vars:
        if data_var_name in bounds_variables:
            continue

        encoding[data_var_name] = {
            "dtype": "float64",
            "zlib": True,
            "complevel": NETCDF_COMPRESSION_LEVEL,
            "shuffle": True,
        }

    return encoding


def _write_inventory_dataset_to_netcdf(
    ds: xr.Dataset,
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise ExportError(f"File already exists and overwrite is false: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_netcdf(
        path=output_path,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
        encoding=_build_netcdf_encoding(ds),
        unlimited_dims=[NETCDF_TIME_AXIS_NAME],
    )


def export_inventory_netcdfs(
    timestep_inventory_dict: dict[str, pd.DataFrame],
    config: Config,
) -> list[Path]:
    species_columns = _get_inventory_species_columns_from_dict(timestep_inventory_dict)
    export_dir = Path(config.output.directory) / "inventory_netcdf"
    export_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []

    if config.output.inventory_netcdf_mode == "single_file":
        ds = _build_inventory_dataset(
            timestep_inventory_dict=timestep_inventory_dict,
            config=config,
            species_columns=species_columns,
        )
        output_path = export_dir / f"{config.output.run_name}_inventory.nc"
        print(
            "Writing NetCDF file (this may take several minutes for large inventories "
            "with many timesteps)..."
        )

        _write_inventory_dataset_to_netcdf(
            ds=ds,
            output_path=output_path,
            overwrite=config.output.overwrite,
        )
        written_paths.append(output_path)
        print("Finished writing NetCDF file.")
        return written_paths

    if config.output.inventory_netcdf_mode == "per_timestep_files":
        for timestep_key in tqdm(
            sorted(timestep_inventory_dict.keys()),
            desc="Writing timestep NetCDFs",
            unit="file",
        ):
            ds = _build_inventory_dataset(
                timestep_inventory_dict={timestep_key: timestep_inventory_dict[timestep_key]},
                config=config,
                species_columns=species_columns,
            )
            safe_timestep_key = timestep_key.replace("__", "_to_")
            output_path = export_dir / f"{config.output.run_name}_{safe_timestep_key}_inventory.nc"
            _write_inventory_dataset_to_netcdf(
                ds=ds,
                output_path=output_path,
                overwrite=config.output.overwrite,
            )
            written_paths.append(output_path)

        return written_paths

    raise ExportError(
        f"Unsupported inventory_netcdf_mode: {config.output.inventory_netcdf_mode}"
    )


def _find_nearest_coordinate_index(
    value: float,
    axis_values: np.ndarray,
    tolerance: float,
) -> int:
    diffs = np.abs(axis_values - value)
    idx = int(np.argmin(diffs))

    if diffs[idx] > tolerance:
        raise ExportError(
            f"Coordinate value {value} is not on the expected output grid "
            f"(nearest grid value: {axis_values[idx]}, difference: {diffs[idx]})."
        )

    return idx


def _build_global_netcdf_attributes(
    config: Config,
    timestep_keys: list[str],
) -> dict[str, str]:
    created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    first_timestep = timestep_keys[0] if timestep_keys else ""
    last_timestep = timestep_keys[-1] if timestep_keys else ""

    attrs = {
        "Conventions": "CF-1.8",
        "title": config.metadata.inventory_name,
        "summary": (
            "IGEL Rocket launch emission inventory generated from launch-specific "
            "propellant use profiles, engine emission indices, optional "
            "post-combustion processing, timestep aggregation, and NetCDF export."
        ),
        "source": TOOL_NAME,
        "software_name": TOOL_NAME,
        "software_version": TOOL_VERSION,
        "history": f"{created_utc} - NetCDF inventory created by {TOOL_NAME} {TOOL_VERSION}",
        "date_created": created_utc,
        "inventory_name": config.metadata.inventory_name,
        "config_version": config.metadata.config_version,
        "run_name": config.output.run_name,
        "inventory_emission_unit": config.output.inventory_emission_unit,
        "inventory_vertical_coordinate": config.output.inventory_vertical_coordinate,
        "inventory_netcdf_mode": config.output.inventory_netcdf_mode,
        "post_combustion": str(config.processing.post_combustion),
        "time_resolution": config.processing.time_resolution,
        "time_coverage_start": config.time_range.start_date.strftime("%Y-%m-%d"),
        "time_coverage_end": config.time_range.end_date.strftime("%Y-%m-%d"),
        "timestep_first": first_timestep,
        "timestep_last": last_timestep,
        "domain_altitude_min_km": str(config.domain.altitude_min_km),
        "domain_altitude_max_km": str(config.domain.altitude_max_km),
        "domain_latitude_min_deg": str(config.domain.latitude_min_deg),
        "domain_latitude_max_deg": str(config.domain.latitude_max_deg),
        "domain_longitude_min_deg": str(config.domain.longitude_min_deg),
        "domain_longitude_max_deg": str(config.domain.longitude_max_deg),
        "domain_inclusive": str(config.domain.inclusive).lower(),
        "grid_altitude_resolution_km": str(config.grid.altitude_resolution_km),
        "grid_latitude_resolution_deg": str(config.grid.latitude_resolution_deg),
        "grid_longitude_resolution_deg": str(config.grid.longitude_resolution_deg),
        "input_launch_list": str(config.input_data.launch_list),
        "input_propellant_use_profiles_dir": str(config.input_data.propellant_use_profiles_dir),
        "input_engine_data_dir": str(config.input_data.engine_data_dir),
        "institution": config.metadata.institution,
        "creator_name": config.metadata.creator_name,
        "creator_email": config.metadata.creator_email,
        "project": config.metadata.project,
    }

    return attrs
