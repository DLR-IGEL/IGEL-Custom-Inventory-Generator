from __future__ import annotations

import math
from pathlib import Path

import cftime
import numpy as np
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset

from constants import (
    NETCDF_LAT_AXIS_NAME,
    NETCDF_LON_AXIS_NAME,
    NETCDF_TIME_AXIS_NAME,
    NETCDF_TIME_BOUNDS_NAME,
    NETCDF_VERTICAL_AXIS_NAME,
    SPECIES_DB,
)
from errors import VerificationError


REQUIRED_DIMS = {
    NETCDF_TIME_AXIS_NAME,
    NETCDF_VERTICAL_AXIS_NAME,
    NETCDF_LAT_AXIS_NAME,
    NETCDF_LON_AXIS_NAME,
}
REQUIRED_COORDS = REQUIRED_DIMS | {NETCDF_TIME_BOUNDS_NAME}
REQUIRED_GLOBAL_ATTRS = {
    "inventory_emission_unit",
    "inventory_vertical_coordinate",
    "run_name",
    "time_resolution",
    "time_coverage_start",
    "time_coverage_end",
    "grid_altitude_resolution_km",
    "grid_latitude_resolution_deg",
    "grid_longitude_resolution_deg",
    "domain_altitude_min_km",
    "domain_altitude_max_km",
    "domain_latitude_min_deg",
    "domain_latitude_max_deg",
    "domain_longitude_min_deg",
    "domain_longitude_max_deg",
}
ALLOWED_UNITS = {"mass", "molecules", "molecule_rate", "molecule_rate_per_volume"}
ALLOWED_VERTICAL = {"altitude", "pressure"}


class InventoryResource:
    def __init__(self, source: str | Path):
        self.source = str(source)
        self.local_path = Path(source)

        if not self.local_path.exists():
            raise VerificationError(f"NetCDF file does not exist: {self.local_path}")
        if not self.local_path.is_file():
            raise VerificationError(f"NetCDF path is not a file: {self.local_path}")

    def cleanup(self) -> None:
        return None

    @property
    def label(self) -> str:
        return str(self.local_path)

    def verification_log_path(self) -> Path:
        return self.local_path.with_name(f"{self.local_path.stem}_verification_log.txt")


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.12g}"
    return str(value)


def _coord_array(ds: xr.Dataset, name: str) -> np.ndarray:
    return np.asarray(ds[name].values, dtype=float)


def _species_variables(ds: xr.Dataset) -> list[str]:
    variables = [
        name for name in ds.data_vars
        if isinstance(name, str) and name.startswith("species_")
    ]
    if not variables:
        raise VerificationError("No species variables found in NetCDF dataset.")
    return sorted(variables)


def _species_name_from_var(var_name: str) -> str:
    if not isinstance(var_name, str) or "_" not in var_name:
        raise VerificationError(f"Invalid species variable name: {var_name!r}")
    return var_name.split("_")[-1]


def _molar_mass_kg_per_molecule(species_name: str) -> float:
    if species_name == "Total":
        raise VerificationError("Cannot convert aggregate 'Total' variable to mass directly.")
    if species_name not in SPECIES_DB:
        raise VerificationError(f"Species '{species_name}' is missing from SPECIES_DB.")
    molar_mass_g_per_mol = float(SPECIES_DB[species_name]["MolarMass"])
    return (molar_mass_g_per_mol / 1000.0) / 6.02214076e23


def _get_time_units_and_calendar(
    ds_raw: xr.Dataset,
    netcdf_path: str | Path | None = None,
) -> tuple[str, str]:
    for var_name in (NETCDF_TIME_BOUNDS_NAME, NETCDF_TIME_AXIS_NAME):
        if var_name not in ds_raw:
            continue
        var = ds_raw[var_name]
        units = var.encoding.get("units") or var.attrs.get("units")
        calendar = var.encoding.get("calendar") or var.attrs.get("calendar") or "standard"
        if units:
            return str(units), str(calendar)

    if netcdf_path is not None:
        with NetCDFDataset(str(netcdf_path), mode="r") as nc:
            for var_name in (NETCDF_TIME_BOUNDS_NAME, NETCDF_TIME_AXIS_NAME):
                if var_name not in nc.variables:
                    continue
                var = nc.variables[var_name]
                units = getattr(var, "units", None)
                calendar = getattr(var, "calendar", None) or "standard"
                if units:
                    return str(units), str(calendar)

    raise VerificationError(
        "Could not determine NetCDF time units/calendar from 'time_bnds' or 'time'."
    )


def _time_bounds_seconds(ds_raw: xr.Dataset) -> np.ndarray:
    if NETCDF_TIME_BOUNDS_NAME not in ds_raw:
        raise VerificationError("Dataset is missing time bounds variable 'time_bnds'.")
    time_bnds = np.asarray(ds_raw[NETCDF_TIME_BOUNDS_NAME].values, dtype=float)
    if time_bnds.ndim != 2 or time_bnds.shape[1] != 2:
        raise VerificationError("time_bnds must have shape (time, 2).")
    return (time_bnds[:, 1] - time_bnds[:, 0]) * 86400.0


def _decoded_time_bounds_strings(
    ds_raw: xr.Dataset,
    netcdf_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    if NETCDF_TIME_BOUNDS_NAME not in ds_raw:
        raise VerificationError("Dataset is missing time bounds variable 'time_bnds'.")

    units, calendar = _get_time_units_and_calendar(ds_raw, netcdf_path=netcdf_path)
    values = np.asarray(ds_raw[NETCDF_TIME_BOUNDS_NAME].values, dtype=float)
    decoded = cftime.num2date(values, units=units, calendar=calendar)

    out: list[tuple[str, str]] = []
    for start, end in decoded:
        out.append(
            (
                f"{start.year:04d}-{start.month:02d}-{start.day:02d}",
                f"{end.year:04d}-{end.month:02d}-{end.day:02d}",
            )
        )
    return out


def _uniform_spacing(values: np.ndarray) -> tuple[bool, float | None]:
    if values.size < 2:
        return True, None
    diffs = np.diff(values)
    first = float(diffs[0])
    return bool(np.allclose(diffs, first, rtol=0.0, atol=1e-8)), first


def _midpoint_edges_from_metadata(
    min_value: float,
    max_value: float,
    resolution: float,
    output_scale: float = 1.0,
) -> np.ndarray:
    mids = np.arange(min_value, max_value + resolution * 0.5, resolution, dtype=float)
    if mids.size == 0:
        raise VerificationError("Could not reconstruct midpoint axis from metadata.")
    edges = np.empty(mids.size + 1, dtype=float)
    edges[0] = mids[0] - resolution / 2.0
    edges[1:] = mids + resolution / 2.0
    return edges * output_scale


def _cell_volumes_from_metadata(ds: xr.Dataset) -> np.ndarray:
    attrs = ds.attrs
    lat_res = float(attrs["grid_latitude_resolution_deg"])
    lon_res = float(attrs["grid_longitude_resolution_deg"])
    alt_min_km = float(attrs["domain_altitude_min_km"])
    alt_max_km = float(attrs["domain_altitude_max_km"])
    alt_res_km = float(attrs["grid_altitude_resolution_km"])

    lat_vals = _coord_array(ds, NETCDF_LAT_AXIS_NAME)
    lon_vals = _coord_array(ds, NETCDF_LON_AXIS_NAME)
    lev_count = ds.sizes[NETCDF_VERTICAL_AXIS_NAME]

    lat_edges_deg = np.empty(lat_vals.size + 1, dtype=float)
    lat_edges_deg[0] = lat_vals[0] - lat_res / 2.0
    lat_edges_deg[1:] = lat_vals + lat_res / 2.0

    lon_edges_deg = np.empty(lon_vals.size + 1, dtype=float)
    lon_edges_deg[0] = lon_vals[0] - lon_res / 2.0
    lon_edges_deg[1:] = lon_vals + lon_res / 2.0

    alt_edges_m = _midpoint_edges_from_metadata(
        alt_min_km,
        alt_max_km,
        alt_res_km,
        output_scale=1000.0,
    )
    if alt_edges_m.size != lev_count + 1:
        raise VerificationError(
            "Metadata-derived vertical layer count does not match NetCDF lev dimension."
        )

    earth_radius_m = 6_371_000.0
    sin_lat_term = np.sin(np.deg2rad(lat_edges_deg[1:])) - np.sin(np.deg2rad(lat_edges_deg[:-1]))
    delta_lon = np.deg2rad(lon_edges_deg[1:] - lon_edges_deg[:-1])
    r_inner = earth_radius_m + alt_edges_m[:-1]
    r_outer = earth_radius_m + alt_edges_m[1:]
    layer_factor = (r_outer**3 - r_inner**3) / 3.0

    volumes = (
        layer_factor[:, None, None]
        * sin_lat_term[None, :, None]
        * delta_lon[None, None, :]
    )
    return volumes.astype(float)


def _convert_var_to_mass_kg(
    ds: xr.Dataset,
    ds_raw: xr.Dataset,
    var_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    unit_mode = str(ds.attrs.get("inventory_emission_unit", "")).strip()
    species_name = _species_name_from_var(var_name)
    data = np.asarray(ds[var_name].values, dtype=float)

    if data.ndim != 4:
        raise VerificationError(
            f"Variable '{var_name}' is expected to have 4 dimensions (time, lev, lat, lon)."
        )

    if species_name == "Total":
        raise VerificationError(
            f"Variable '{var_name}' is an aggregate total and cannot be converted to species mass."
        )

    if unit_mode == "mass":
        mass = data
    elif unit_mode == "molecules":
        mass = data * _molar_mass_kg_per_molecule(species_name)
    elif unit_mode == "molecule_rate":
        durations = _time_bounds_seconds(ds_raw)[:, None, None, None]
        mass = data * durations * _molar_mass_kg_per_molecule(species_name)
    elif unit_mode == "molecule_rate_per_volume":
        durations = _time_bounds_seconds(ds_raw)[:, None, None, None]
        volumes = _cell_volumes_from_metadata(ds)[None, :, :, :]
        mass = data * durations * volumes * _molar_mass_kg_per_molecule(species_name)
    else:
        raise VerificationError(f"Unsupported inventory_emission_unit '{unit_mode}'.")

    per_timestep = mass.sum(axis=(1, 2, 3))
    return mass, per_timestep


def _structural_validation(ds: xr.Dataset) -> list[str]:
    issues: list[str] = []

    missing_dims = REQUIRED_DIMS.difference(ds.sizes)
    if missing_dims:
        issues.append(f"Missing required dimensions: {sorted(missing_dims)}")

    missing_coords = {
        name
        for name in [
            NETCDF_TIME_AXIS_NAME,
            NETCDF_VERTICAL_AXIS_NAME,
            NETCDF_LAT_AXIS_NAME,
            NETCDF_LON_AXIS_NAME,
        ]
        if name not in ds.coords
    }
    if missing_coords:
        issues.append(f"Missing required coordinate variables: {sorted(missing_coords)}")

    if NETCDF_TIME_BOUNDS_NAME not in ds.variables:
        issues.append("Missing required variable 'time_bnds'.")

    missing_attrs = REQUIRED_GLOBAL_ATTRS.difference(ds.attrs)
    if missing_attrs:
        issues.append(f"Missing required global attributes: {sorted(missing_attrs)}")

    if NETCDF_TIME_AXIS_NAME in ds.variables:
        time_var = ds[NETCDF_TIME_AXIS_NAME]
        time_units = time_var.encoding.get("units") or time_var.attrs.get("units")
        time_calendar = time_var.encoding.get("calendar") or time_var.attrs.get("calendar")
        if not time_units:
            issues.append("Time coordinate 'time' is missing 'units' metadata.")
        if not time_calendar:
            issues.append("Time coordinate 'time' is missing 'calendar' metadata.")
    else:
        issues.append("Missing required time coordinate variable 'time'.")

    if NETCDF_TIME_BOUNDS_NAME in ds.variables:
        time_bnds_var = ds[NETCDF_TIME_BOUNDS_NAME]
        tb_units = (
            time_bnds_var.encoding.get("units")
            or time_bnds_var.attrs.get("units")
            or (
                ds[NETCDF_TIME_AXIS_NAME].encoding.get("units")
                if NETCDF_TIME_AXIS_NAME in ds.variables
                else None
            )
            or (
                ds[NETCDF_TIME_AXIS_NAME].attrs.get("units")
                if NETCDF_TIME_AXIS_NAME in ds.variables
                else None
            )
        )
        tb_calendar = (
            time_bnds_var.encoding.get("calendar")
            or time_bnds_var.attrs.get("calendar")
            or (
                ds[NETCDF_TIME_AXIS_NAME].encoding.get("calendar")
                if NETCDF_TIME_AXIS_NAME in ds.variables
                else None
            )
            or (
                ds[NETCDF_TIME_AXIS_NAME].attrs.get("calendar")
                if NETCDF_TIME_AXIS_NAME in ds.variables
                else None
            )
        )
        if not tb_units:
            issues.append("Time bounds variable 'time_bnds' is missing 'units' metadata.")
        if not tb_calendar:
            issues.append("Time bounds variable 'time_bnds' is missing 'calendar' metadata.")

    unit_mode = str(ds.attrs.get("inventory_emission_unit", "")).strip()
    vertical_mode = str(ds.attrs.get("inventory_vertical_coordinate", "")).strip()
    if unit_mode and unit_mode not in ALLOWED_UNITS:
        issues.append(f"Unsupported inventory_emission_unit metadata value: {unit_mode}")
    if vertical_mode and vertical_mode not in ALLOWED_VERTICAL:
        issues.append(
            f"Unsupported inventory_vertical_coordinate metadata value: {vertical_mode}"
        )

    for var_name in _species_variables(ds):
        da = ds[var_name]
        if tuple(da.dims) != (
            NETCDF_TIME_AXIS_NAME,
            NETCDF_VERTICAL_AXIS_NAME,
            NETCDF_LAT_AXIS_NAME,
            NETCDF_LON_AXIS_NAME,
        ):
            issues.append(f"Species variable '{var_name}' has unexpected dimensions: {da.dims}")
        if "units" not in da.attrs:
            issues.append(f"Species variable '{var_name}' is missing 'units' attribute.")
        if "long_name" not in da.attrs:
            issues.append(f"Species variable '{var_name}' is missing 'long_name' attribute.")

    return issues


def _data_sanity_checks(ds: xr.Dataset) -> list[str]:
    issues: list[str] = []

    for coord_name in [
        NETCDF_TIME_AXIS_NAME,
        NETCDF_VERTICAL_AXIS_NAME,
        NETCDF_LAT_AXIS_NAME,
        NETCDF_LON_AXIS_NAME,
    ]:
        values = np.asarray(ds[coord_name].values)
        if values.size > 1:
            if values.ndim != 1:
                issues.append(f"Coordinate '{coord_name}' is not one-dimensional.")
            if len(np.unique(values)) != len(values):
                issues.append(f"Coordinate '{coord_name}' contains duplicate values.")
            if coord_name in {
                NETCDF_LAT_AXIS_NAME,
                NETCDF_LON_AXIS_NAME,
                NETCDF_TIME_AXIS_NAME,
            }:
                diffs = np.diff(values.astype(float))
                if np.any(diffs <= 0):
                    issues.append(f"Coordinate '{coord_name}' is not strictly increasing.")

    if NETCDF_TIME_BOUNDS_NAME in ds.variables:
        time_bnds = np.asarray(ds[NETCDF_TIME_BOUNDS_NAME].values, dtype=float)
        if np.isnan(time_bnds).any():
            issues.append("time_bnds contains NaN values.")
        if np.isinf(time_bnds).any():
            issues.append("time_bnds contains infinite values.")
        if time_bnds.ndim != 2 or time_bnds.shape[1] != 2:
            issues.append("time_bnds does not have shape (time, 2).")
        elif np.any(time_bnds[:, 1] <= time_bnds[:, 0]):
            issues.append("time_bnds contains non-positive timestep durations.")

    for var_name in _species_variables(ds):
        data = np.asarray(ds[var_name].values, dtype=float)
        if np.isnan(data).any():
            issues.append(f"Species variable '{var_name}' contains NaN values.")
        if np.isinf(data).any():
            issues.append(f"Species variable '{var_name}' contains infinite values.")
        if (data < 0).any():
            issues.append(f"Species variable '{var_name}' contains negative values.")
        if np.allclose(data, 0.0, rtol=0.0, atol=0.0):
            issues.append(f"Species variable '{var_name}' contains only zeros.")

    return issues


def _metadata_cross_checks(
    ds: xr.Dataset,
    ds_raw: xr.Dataset,
    netcdf_path: str | Path | None = None,
) -> list[str]:
    issues: list[str] = []
    attrs = ds.attrs

    lat_vals = _coord_array(ds, NETCDF_LAT_AXIS_NAME)
    lon_vals = _coord_array(ds, NETCDF_LON_AXIS_NAME)
    lev_vals = _coord_array(ds, NETCDF_VERTICAL_AXIS_NAME)

    lat_uniform, lat_spacing = _uniform_spacing(lat_vals)
    lon_uniform, lon_spacing = _uniform_spacing(lon_vals)
    if not lat_uniform:
        issues.append("Latitude coordinate is not uniformly spaced.")
    if not lon_uniform:
        issues.append("Longitude coordinate is not uniformly spaced.")

    if lat_spacing is not None:
        expected = float(attrs["grid_latitude_resolution_deg"])
        if not math.isclose(abs(lat_spacing), expected, rel_tol=0.0, abs_tol=1e-8):
            issues.append(
                f"Latitude spacing {abs(lat_spacing)} does not match metadata "
                f"grid_latitude_resolution_deg={expected}."
            )
    if lon_spacing is not None:
        expected = float(attrs["grid_longitude_resolution_deg"])
        if not math.isclose(abs(lon_spacing), expected, rel_tol=0.0, abs_tol=1e-8):
            issues.append(
                f"Longitude spacing {abs(lon_spacing)} does not match metadata "
                f"grid_longitude_resolution_deg={expected}."
            )

    lat_min_expected = float(attrs["domain_latitude_min_deg"])
    lat_max_expected = float(attrs["domain_latitude_max_deg"])
    lon_min_expected = float(attrs["domain_longitude_min_deg"])
    lon_max_expected = float(attrs["domain_longitude_max_deg"])

    if lat_vals.size:
        if not math.isclose(float(lat_vals[0]), lat_min_expected, rel_tol=0.0, abs_tol=1e-8):
            issues.append(
                f"First latitude midpoint {lat_vals[0]} does not match metadata "
                f"domain_latitude_min_deg={lat_min_expected}."
            )
        if not math.isclose(float(lat_vals[-1]), lat_max_expected, rel_tol=0.0, abs_tol=1e-8):
            issues.append(
                f"Last latitude midpoint {lat_vals[-1]} does not match metadata "
                f"domain_latitude_max_deg={lat_max_expected}."
            )
    if lon_vals.size:
        if not math.isclose(float(lon_vals[0]), lon_min_expected, rel_tol=0.0, abs_tol=1e-8):
            issues.append(
                f"First longitude midpoint {lon_vals[0]} does not match metadata "
                f"domain_longitude_min_deg={lon_min_expected}."
            )
        if not math.isclose(float(lon_vals[-1]), lon_max_expected, rel_tol=0.0, abs_tol=1e-8):
            issues.append(
                f"Last longitude midpoint {lon_vals[-1]} does not match metadata "
                f"domain_longitude_max_deg={lon_max_expected}."
            )

    vertical_mode = str(attrs["inventory_vertical_coordinate"])
    if vertical_mode == "altitude":
        actual_units = str(ds[NETCDF_VERTICAL_AXIS_NAME].attrs.get("units", ""))
        if actual_units != "m":
            issues.append(f"Altitude lev units '{actual_units}' do not match expected 'm'.")
        alt_min_m = float(attrs["domain_altitude_min_km"]) * 1000.0
        alt_max_m = float(attrs["domain_altitude_max_km"]) * 1000.0
        alt_res_m = float(attrs["grid_altitude_resolution_km"]) * 1000.0
        if lev_vals.size > 1:
            lev_uniform, lev_spacing = _uniform_spacing(lev_vals)
            if not lev_uniform:
                issues.append("Altitude lev coordinate is not uniformly spaced.")
            elif lev_spacing is not None and not math.isclose(
                abs(lev_spacing),
                alt_res_m,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                issues.append(
                    f"Altitude lev spacing {abs(lev_spacing)} does not match metadata "
                    f"grid_altitude_resolution_km={attrs['grid_altitude_resolution_km']}."
                )
        if lev_vals.size:
            if not math.isclose(float(lev_vals[0]), alt_min_m, rel_tol=0.0, abs_tol=1e-6):
                issues.append(
                    f"First altitude midpoint {lev_vals[0]} does not match metadata "
                    f"lower altitude midpoint {alt_min_m}."
                )
            if not math.isclose(float(lev_vals[-1]), alt_max_m, rel_tol=0.0, abs_tol=1e-6):
                issues.append(
                    f"Last altitude midpoint {lev_vals[-1]} does not match metadata "
                    f"upper altitude midpoint {alt_max_m}."
                )
    else:
        actual_units = str(ds[NETCDF_VERTICAL_AXIS_NAME].attrs.get("units", ""))
        if actual_units != "Pa":
            issues.append(f"Pressure lev units '{actual_units}' do not match expected 'Pa'.")
        if lev_vals.size > 1 and not np.all(np.diff(lev_vals) < 0):
            issues.append("Pressure lev coordinate is not strictly decreasing with height.")

    if NETCDF_TIME_BOUNDS_NAME in ds_raw:
        try:
            bound_strings = _decoded_time_bounds_strings(ds_raw, netcdf_path=netcdf_path)
            if bound_strings:
                first_start = bound_strings[0][0]
                units, calendar = _get_time_units_and_calendar(ds_raw, netcdf_path=netcdf_path)
                raw_time_bnds = np.asarray(ds_raw[NETCDF_TIME_BOUNDS_NAME].values, dtype=float)
                last_inclusive_dt = cftime.num2date(
                    raw_time_bnds[-1, 1] - 1.0,
                    units=units,
                    calendar=calendar,
                )
                last_inclusive = (
                    f"{last_inclusive_dt.year:04d}-"
                    f"{last_inclusive_dt.month:02d}-"
                    f"{last_inclusive_dt.day:02d}"
                )
                if first_start != str(attrs["time_coverage_start"]):
                    issues.append(
                        f"Decoded first time bound {first_start} does not match metadata "
                        f"time_coverage_start={attrs['time_coverage_start']}."
                    )
                if last_inclusive != str(attrs["time_coverage_end"]):
                    issues.append(
                        f"Decoded last inclusive time bound {last_inclusive} does not match "
                        f"metadata time_coverage_end={attrs['time_coverage_end']}."
                    )
        except VerificationError as exc:
            issues.append(
                "Skipped time metadata cross-check because time units/calendar could not "
                f"be determined: {exc}"
            )

    return issues


def _summarize_metadata(ds: xr.Dataset) -> list[str]:
    lines = ["Metadata:"]
    for key in sorted(ds.attrs):
        lines.append(f"  {key}: {_format_scalar(ds.attrs[key])}")
    return lines


def _summarize_dimensions(ds: xr.Dataset) -> list[str]:
    return [
        "Dimensions:",
        f"  time: {ds.sizes.get(NETCDF_TIME_AXIS_NAME, 0)}",
        f"  lev:  {ds.sizes.get(NETCDF_VERTICAL_AXIS_NAME, 0)}",
        f"  lat:  {ds.sizes.get(NETCDF_LAT_AXIS_NAME, 0)}",
        f"  lon:  {ds.sizes.get(NETCDF_LON_AXIS_NAME, 0)}",
    ]


def _mass_summaries(
    ds: xr.Dataset,
    ds_raw: xr.Dataset,
) -> tuple[list[str], dict[str, float], dict[str, np.ndarray]]:
    lines = ["Total emissions by species (kg):"]
    totals: dict[str, float] = {}
    per_timestep: dict[str, np.ndarray] = {}

    grand_total_kg = 0.0
    for var_name in _species_variables(ds):
        species_name = _species_name_from_var(var_name)
        if species_name == "Total":
            continue
        _, timestep_totals = _convert_var_to_mass_kg(ds, ds_raw, var_name)
        total = float(np.sum(timestep_totals))
        totals[species_name] = total
        per_timestep[species_name] = timestep_totals
        grand_total_kg += total
        lines.append(f"  {species_name}: {total:.12g}")

    lines.append(f"  TOTAL_ALL_SPECIES: {grand_total_kg:.12g}")
    return lines, totals, per_timestep


def _per_timestep_lines(
    ds_raw: xr.Dataset,
    per_timestep: dict[str, np.ndarray],
    netcdf_path: str | Path | None = None,
) -> list[str]:
    lines = ["Per-timestep species totals (kg):"]
    species_names = sorted(per_timestep)

    try:
        bounds = _decoded_time_bounds_strings(ds_raw, netcdf_path=netcdf_path)
        for t_idx, (start, end_exclusive) in enumerate(bounds):
            lines.append(
                f"  Timestep {t_idx + 1}: {start} to {end_exclusive} (end exclusive)"
            )
            timestep_grand_total_kg = 0.0
            for species_name in species_names:
                value = float(per_timestep[species_name][t_idx])
                timestep_grand_total_kg += value
                lines.append(f"    {species_name}: {value:.12g}")
            lines.append(f"    TOTAL_ALL_SPECIES: {timestep_grand_total_kg:.12g}")
    except VerificationError as exc:
        lines.append(
            "  Time bounds could not be decoded, so timesteps are listed by index only: "
            f"{exc}"
        )
        timestep_count = len(next(iter(per_timestep.values()))) if per_timestep else 0
        for t_idx in range(timestep_count):
            lines.append(f"  Timestep {t_idx + 1}:")
            timestep_grand_total_kg = 0.0
            for species_name in species_names:
                value = float(per_timestep[species_name][t_idx])
                timestep_grand_total_kg += value
                lines.append(f"    {species_name}: {value:.12g}")
            lines.append(f"    TOTAL_ALL_SPECIES: {timestep_grand_total_kg:.12g}")

    all_timestep_total_kg = sum(
        sum(float(v) for v in timestep_totals)
        for timestep_totals in per_timestep.values()
    )
    lines.append(f"\nSum over all timestep totals (kg): {all_timestep_total_kg:.12g}")
    return lines


def _issues_block(title: str, issues: list[str]) -> list[str]:
    lines = [title]
    if issues:
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  No issues found")
    return lines


def build_verification_report(netcdf_source: str | Path) -> tuple[str, Path]:
    resource = InventoryResource(netcdf_source)

    try:
        with xr.open_dataset(
            resource.local_path,
            decode_times=False,
            mask_and_scale=True,
        ) as ds_raw, xr.open_dataset(
            resource.local_path,
            decode_times=False,
            mask_and_scale=True,
        ) as ds:
            report_lines: list[str] = []
            report_lines.append(f"Verification target: {resource.label}")
            report_lines.extend(_summarize_dimensions(ds))
            report_lines.append("")
            report_lines.extend(_summarize_metadata(ds))
            report_lines.append("")

            structural_issues = _structural_validation(ds)
            sanity_issues = _data_sanity_checks(ds)
            metadata_issues = _metadata_cross_checks(
                ds,
                ds_raw,
                netcdf_path=resource.local_path,
            )

            report_lines.extend(_issues_block("Structural validation:", structural_issues))
            report_lines.append("")
            report_lines.extend(_issues_block("Data sanity checks:", sanity_issues))
            report_lines.append("")
            report_lines.extend(_issues_block("Metadata vs data cross-check:", metadata_issues))
            report_lines.append("")

            mass_lines, _totals, per_timestep = _mass_summaries(ds, ds_raw)
            report_lines.extend(mass_lines)
            report_lines.append("")
            report_lines.extend(
                _per_timestep_lines(
                    ds_raw,
                    per_timestep,
                    netcdf_path=resource.local_path,
                )
            )

        report_text = "\n".join(report_lines) + "\n"
        log_path = resource.verification_log_path()
        log_path.write_text(report_text, encoding="utf-8")
        return report_text, log_path
    finally:
        resource.cleanup()


def verify_inventory_netcdf(netcdf_source: str | Path) -> Path:
    report_text, log_path = build_verification_report(netcdf_source=netcdf_source)
    print(report_text, end="")
    print(f"Verification log written to: {log_path}")
    return log_path