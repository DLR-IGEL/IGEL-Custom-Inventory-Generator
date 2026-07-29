from __future__ import annotations

import re

import numpy as np
import pandas as pd
import ussa1976

from constants import (
    ALLOWED_INVENTORY_EMISSION_UNITS,
    ALLOWED_INVENTORY_VERTICAL_COORDINATES,
    AVOGADRO_CONSTANT,
    EARTH_RADIUS_M,
    SPECIES_MASS_COLUMN_PATTERN,
)
from errors import InventoryTransformError
from inventory_timesteps import timestep_duration_seconds_from_key


def degrees_to_radians(deg: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    return np.deg2rad(deg)


def calculate_shell_volume_m3(
    lat_min_deg: pd.Series | np.ndarray,
    lat_max_deg: pd.Series | np.ndarray,
    lon_min_deg: pd.Series | np.ndarray,
    lon_max_deg: pd.Series | np.ndarray,
    alt_min_km: pd.Series | np.ndarray,
    alt_max_km: pd.Series | np.ndarray,
) -> np.ndarray:
    lat_min_rad = np.deg2rad(np.asarray(lat_min_deg, dtype=float))
    lat_max_rad = np.deg2rad(np.asarray(lat_max_deg, dtype=float))
    lon_min_rad = np.deg2rad(np.asarray(lon_min_deg, dtype=float))
    lon_max_rad = np.deg2rad(np.asarray(lon_max_deg, dtype=float))

    delta_lon = lon_max_rad - lon_min_rad
    sin_lat_term = np.sin(lat_max_rad) - np.sin(lat_min_rad)

    r_inner_m = EARTH_RADIUS_M + np.asarray(alt_min_km, dtype=float) * 1000.0
    r_outer_m = EARTH_RADIUS_M + np.asarray(alt_max_km, dtype=float) * 1000.0

    volume_m3 = delta_lon * sin_lat_term * (r_outer_m**3 - r_inner_m**3) / 3.0
    return volume_m3


def calculate_volume_for_row(row: pd.Series) -> float:
    return float(
        calculate_shell_volume_m3(
            lat_min_deg=[row["LATITUDE_MIN"]],
            lat_max_deg=[row["LATITUDE_MAX"]],
            lon_min_deg=[row["LONGITUDE_MIN"]],
            lon_max_deg=[row["LONGITUDE_MAX"]],
            alt_min_km=[row["ALTITUDE_MIN"]],
            alt_max_km=[row["ALTITUDE_MAX"]],
        )[0]
    )


def _get_species_mass_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    species_columns: list[tuple[str, str]] = []

    for col in df.columns:
        match = SPECIES_MASS_COLUMN_PATTERN.match(col)
        if not match:
            continue

        species_name = match.group("species").strip()
        if species_name == "Total":
            continue

        species_columns.append((col, species_name))

    if not species_columns:
        raise InventoryTransformError(
            "No species mass columns found matching 'species_mass_<species>'."
        )

    return species_columns


def _recompute_total_column(
    df: pd.DataFrame,
    total_column_name: str,
    species_value_columns: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df[total_column_name] = df[species_value_columns].sum(axis=1)
    return df


def _convert_species_mass_to_molecules(
    mass_kg: pd.Series,
    species_name: str,
    species_db: dict[str, dict[str, float]],
) -> pd.Series:
    if species_name not in species_db:
        raise InventoryTransformError(f"Species '{species_name}' not found in species_db.")

    if "MolarMass" not in species_db[species_name]:
        raise InventoryTransformError(
            f"Species '{species_name}' is missing 'MolarMass' in species_db."
        )

    molar_mass_g_per_mol = float(species_db[species_name]["MolarMass"])
    if molar_mass_g_per_mol <= 0:
        raise InventoryTransformError(
            f"Species '{species_name}' has invalid molar mass: {molar_mass_g_per_mol}"
        )

    return mass_kg * 1000.0 / molar_mass_g_per_mol * AVOGADRO_CONSTANT


def MoleculesCount(
    df: pd.DataFrame,
    species_db: dict[str, dict[str, float]],
) -> pd.DataFrame:
    df = df.copy()
    species_columns = _get_species_mass_columns(df)

    molecule_columns: list[str] = []

    for mass_col, species_name in species_columns:
        mass_kg = pd.to_numeric(df[mass_col], errors="coerce")
        if mass_kg.isna().any():
            raise InventoryTransformError(
                f"Column '{mass_col}' contains non-numeric values."
            )

        out_col = f"species_molecules_{species_name}"
        df[out_col] = _convert_species_mass_to_molecules(
            mass_kg=mass_kg,
            species_name=species_name,
            species_db=species_db,
        )
        molecule_columns.append(out_col)

    df = df.drop(
        columns=[col for col in df.columns if col.startswith("species_mass_")],
        errors="ignore",
    )
    df = _recompute_total_column(
        df=df,
        total_column_name="species_molecules_Total",
        species_value_columns=molecule_columns,
    )

    return df


def MoleculeVolumeRateCalculator(
    df: pd.DataFrame,
    timespan_s: float,
    species_db: dict[str, dict[str, float]],
) -> pd.DataFrame:
    if timespan_s <= 0:
        raise InventoryTransformError("timespan_s must be > 0.")

    if "CELL_VOLUME_M3" not in df.columns:
        raise InventoryTransformError(
            "CELL_VOLUME_M3 is required for molecule_rate_per_volume conversion."
        )

    result_df = MoleculesCount(df=df, species_db=species_db)

    species_molecule_columns = [
        col
        for col in result_df.columns
        if col.startswith("species_molecules_") and col != "species_molecules_Total"
    ]

    rate_columns: list[str] = []

    for col in species_molecule_columns:
        match = re.match(r"^species_molecules_(?P<species>.+)$", col)
        if not match:
            continue

        species_name = match.group("species")
        out_col = f"species_molecule_rate_per_volume_{species_name}"
        result_df[out_col] = result_df[col] / (result_df["CELL_VOLUME_M3"] * timespan_s)
        rate_columns.append(out_col)

    result_df = result_df.drop(
        columns=species_molecule_columns + ["species_molecules_Total"],
        errors="ignore",
    )

    result_df = _recompute_total_column(
        df=result_df,
        total_column_name="species_molecule_rate_per_volume_Total",
        species_value_columns=rate_columns,
    )

    return result_df


def convert_inventory_emission_units(
    df: pd.DataFrame,
    inventory_emission_unit: str,
    timestep_seconds: float,
    species_db: dict[str, dict[str, float]],
) -> pd.DataFrame:
    if inventory_emission_unit not in ALLOWED_INVENTORY_EMISSION_UNITS:
        raise InventoryTransformError(
            "Unsupported inventory_emission_unit: "
            f"{inventory_emission_unit}"
        )

    if inventory_emission_unit == "mass":
        result_df = df.copy()
        species_columns = [col for col, _ in _get_species_mass_columns(result_df)]
        result_df = _recompute_total_column(
            df=result_df.drop(columns=["species_mass_Total"], errors="ignore"),
            total_column_name="species_mass_Total",
            species_value_columns=species_columns,
        )
        return result_df

    if inventory_emission_unit == "molecules":
        return MoleculesCount(df=df, species_db=species_db)

    if inventory_emission_unit == "molecule_rate":
        if timestep_seconds <= 0:
            raise InventoryTransformError("timestep_seconds must be > 0.")

        molecules_df = MoleculesCount(df=df, species_db=species_db)
        species_molecule_columns = [
            col
            for col in molecules_df.columns
            if col.startswith("species_molecules_") and col != "species_molecules_Total"
        ]

        result_df = molecules_df.copy()
        rate_columns: list[str] = []

        for col in species_molecule_columns:
            match = re.match(r"^species_molecules_(?P<species>.+)$", col)
            if not match:
                continue

            species_name = match.group("species")
            out_col = f"species_molecule_rate_{species_name}"
            result_df[out_col] = result_df[col] / timestep_seconds
            rate_columns.append(out_col)

        result_df = result_df.drop(
            columns=species_molecule_columns + ["species_molecules_Total"],
            errors="ignore",
        )
        result_df = _recompute_total_column(
            df=result_df,
            total_column_name="species_molecule_rate_Total",
            species_value_columns=rate_columns,
        )
        return result_df

    if inventory_emission_unit == "molecule_rate_per_volume":
        return MoleculeVolumeRateCalculator(
            df=df,
            timespan_s=timestep_seconds,
            species_db=species_db,
        )

    raise InventoryTransformError(
        f"Unhandled inventory_emission_unit: {inventory_emission_unit}"
    )


def _altitudes_km_to_pressures_pa(altitudes_km: pd.Series | np.ndarray) -> np.ndarray:
    altitudes_m = np.asarray(altitudes_km, dtype=float) * 1000.0

    # For pressure-coordinate bounds, the lowest altitude bin may extend below 0 km
    # (e.g. -0.5 km to +0.5 km for a 0 km-centered layer). Pressure is therefore
    # evaluated at 0 m for any negative bound.
    if np.any(altitudes_m < 0):
        altitudes_m = np.maximum(altitudes_m, 0.0)

    if np.any(altitudes_m > 1_000_000.0):
        raise InventoryTransformError(
            "Altitude for pressure conversion exceeds 1000 km, which is outside "
            "the supported USSA1976 range."
        )

    unique_altitudes_m = np.unique(altitudes_m)
    ds = ussa1976.compute(z=unique_altitudes_m, variables=["p"])
    unique_pressures_pa = np.asarray(ds["p"].values, dtype=float)

    altitude_to_pressure = dict(zip(unique_altitudes_m.tolist(), unique_pressures_pa.tolist()))
    pressures_pa = np.array([altitude_to_pressure[val] for val in altitudes_m], dtype=float)

    return pressures_pa


def PressureAltitudeCalc(df: pd.DataFrame) -> pd.DataFrame:
    if "ALTITUDE" not in df.columns:
        raise InventoryTransformError("ALTITUDE column is required.")

    result_df = df.copy()
    altitude_km = pd.to_numeric(result_df["ALTITUDE"], errors="coerce")
    if altitude_km.isna().any():
        raise InventoryTransformError("ALTITUDE contains non-numeric values.")

    result_df["PRESSURE"] = _altitudes_km_to_pressures_pa(altitude_km)
    return result_df


def pressures_to_altitudes_km(pressures_pa: list[float] | np.ndarray) -> list[float]:
    pressure_input = np.asarray(pressures_pa, dtype=float)

    alt_grid_m = np.arange(0.0, 1_000_001.0, 100.0)
    ds = ussa1976.compute(z=alt_grid_m, variables=["p"])
    pressure_grid_pa = np.asarray(ds["p"].values, dtype=float)
    alt_grid_km = alt_grid_m / 1000.0

    order = np.argsort(pressure_grid_pa)
    xp = pressure_grid_pa[order]
    fp = alt_grid_km[order]

    altitudes_km = np.interp(pressure_input, xp, fp)
    return altitudes_km.tolist()


def convert_inventory_vertical_coordinate(
    df: pd.DataFrame,
    inventory_vertical_coordinate: str,
) -> pd.DataFrame:
    if inventory_vertical_coordinate not in ALLOWED_INVENTORY_VERTICAL_COORDINATES:
        raise InventoryTransformError(
            "Unsupported inventory_vertical_coordinate: "
            f"{inventory_vertical_coordinate}"
        )

    if "ALTITUDE" not in df.columns:
        raise InventoryTransformError(
            "ALTITUDE midpoint column is required for vertical coordinate conversion."
        )

    result_df = df.copy()

    altitude_km = pd.to_numeric(result_df["ALTITUDE"], errors="coerce")
    if altitude_km.isna().any():
        raise InventoryTransformError("ALTITUDE contains non-numeric values.")

    if inventory_vertical_coordinate == "altitude":
        result_df["ALTITUDE"] = altitude_km * 1000.0

        if "ALTITUDE_MIN" in result_df.columns and "ALTITUDE_MAX" in result_df.columns:
            alt_min = pd.to_numeric(result_df["ALTITUDE_MIN"], errors="coerce")
            alt_max = pd.to_numeric(result_df["ALTITUDE_MAX"], errors="coerce")
            if alt_min.isna().any() or alt_max.isna().any():
                raise InventoryTransformError("ALTITUDE_MIN/MAX contain non-numeric values.")
            result_df["ALTITUDE_MIN"] = alt_min * 1000.0
            result_df["ALTITUDE_MAX"] = alt_max * 1000.0

        return result_df

    result_df["PRESSURE"] = _altitudes_km_to_pressures_pa(altitude_km)

    if "ALTITUDE_MIN" in result_df.columns and "ALTITUDE_MAX" in result_df.columns:
        alt_min = pd.to_numeric(result_df["ALTITUDE_MIN"], errors="coerce")
        alt_max = pd.to_numeric(result_df["ALTITUDE_MAX"], errors="coerce")
        if alt_min.isna().any() or alt_max.isna().any():
            raise InventoryTransformError("ALTITUDE_MIN/MAX contain non-numeric values.")

        result_df["PRESSURE_MAX"] = _altitudes_km_to_pressures_pa(alt_min)
        result_df["PRESSURE_MIN"] = _altitudes_km_to_pressures_pa(alt_max)

    result_df = result_df.drop(columns=["ALTITUDE"], errors="ignore")

    return result_df


def collapse_spatial_bins_to_midpoints(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "ALTITUDE_MIN",
        "ALTITUDE_MAX",
        "LATITUDE_MIN",
        "LATITUDE_MAX",
        "LONGITUDE_MIN",
        "LONGITUDE_MAX",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise InventoryTransformError(
            "Cannot collapse spatial bins to midpoints. Missing columns: "
            f"{sorted(missing)}"
        )

    result_df = df.copy()

    alt_min = pd.to_numeric(result_df["ALTITUDE_MIN"], errors="coerce")
    alt_max = pd.to_numeric(result_df["ALTITUDE_MAX"], errors="coerce")
    lat_min = pd.to_numeric(result_df["LATITUDE_MIN"], errors="coerce")
    lat_max = pd.to_numeric(result_df["LATITUDE_MAX"], errors="coerce")
    lon_min = pd.to_numeric(result_df["LONGITUDE_MIN"], errors="coerce")
    lon_max = pd.to_numeric(result_df["LONGITUDE_MAX"], errors="coerce")

    if (
        alt_min.isna().any()
        or alt_max.isna().any()
        or lat_min.isna().any()
        or lat_max.isna().any()
        or lon_min.isna().any()
        or lon_max.isna().any()
    ):
        raise InventoryTransformError("Spatial boundary columns contain non-numeric values.")

    result_df["CELL_VOLUME_M3"] = calculate_shell_volume_m3(
        lat_min_deg=lat_min,
        lat_max_deg=lat_max,
        lon_min_deg=lon_min,
        lon_max_deg=lon_max,
        alt_min_km=alt_min,
        alt_max_km=alt_max,
    )

    result_df["ALTITUDE"] = (alt_min + alt_max) / 2.0
    result_df["LATITUDE"] = (lat_min + lat_max) / 2.0
    result_df["LONGITUDE"] = (lon_min + lon_max) / 2.0

    return result_df

    alt_min = pd.to_numeric(result_df["ALTITUDE_MIN"], errors="coerce")
    alt_max = pd.to_numeric(result_df["ALTITUDE_MAX"], errors="coerce")
    lat_min = pd.to_numeric(result_df["LATITUDE_MIN"], errors="coerce")
    lat_max = pd.to_numeric(result_df["LATITUDE_MAX"], errors="coerce")
    lon_min = pd.to_numeric(result_df["LONGITUDE_MIN"], errors="coerce")
    lon_max = pd.to_numeric(result_df["LONGITUDE_MAX"], errors="coerce")

    if (
        alt_min.isna().any()
        or alt_max.isna().any()
        or lat_min.isna().any()
        or lat_max.isna().any()
        or lon_min.isna().any()
        or lon_max.isna().any()
    ):
        raise InventoryTransformError("Spatial boundary columns contain non-numeric values.")

    result_df["CELL_VOLUME_M3"] = calculate_shell_volume_m3(
        lat_min_deg=lat_min,
        lat_max_deg=lat_max,
        lon_min_deg=lon_min,
        lon_max_deg=lon_max,
        alt_min_km=alt_min,
        alt_max_km=alt_max,
    )

    result_df["ALTITUDE"] = (alt_min + alt_max) / 2.0
    result_df["LATITUDE"] = (lat_min + lat_max) / 2.0
    result_df["LONGITUDE"] = (lon_min + lon_max) / 2.0

    result_df = result_df.drop(
        columns=[
            "ALTITUDE_MIN",
            "ALTITUDE_MAX",
            "LATITUDE_MIN",
            "LATITUDE_MAX",
            "LONGITUDE_MIN",
            "LONGITUDE_MAX",
        ],
        errors="ignore",
    )

    return result_df


def transform_timestep_inventory_dataframe(
    timestep_inventory_df: pd.DataFrame,
    timestep_key: str,
    inventory_emission_unit: str,
    inventory_vertical_coordinate: str,
    species_db: dict[str, dict[str, float]],
) -> pd.DataFrame:
    timestep_seconds = timestep_duration_seconds_from_key(timestep_key)

    result_df = collapse_spatial_bins_to_midpoints(
        df=timestep_inventory_df,
    )

    result_df = convert_inventory_emission_units(
        df=result_df,
        inventory_emission_unit=inventory_emission_unit,
        timestep_seconds=timestep_seconds,
        species_db=species_db,
    )

    result_df = convert_inventory_vertical_coordinate(
        df=result_df,
        inventory_vertical_coordinate=inventory_vertical_coordinate,
    )

    return result_df


def transform_timestep_inventory_dict(
    timestep_inventory_dict: dict[str, pd.DataFrame],
    inventory_emission_unit: str,
    inventory_vertical_coordinate: str,
    species_db: dict[str, dict[str, float]],
) -> dict[str, pd.DataFrame]:
    transformed_dict: dict[str, pd.DataFrame] = {}

    for timestep_key, timestep_df in timestep_inventory_dict.items():
        transformed_dict[timestep_key] = transform_timestep_inventory_dataframe(
            timestep_inventory_df=timestep_df,
            timestep_key=timestep_key,
            inventory_emission_unit=inventory_emission_unit,
            inventory_vertical_coordinate=inventory_vertical_coordinate,
            species_db=species_db,
        )

    return transformed_dict