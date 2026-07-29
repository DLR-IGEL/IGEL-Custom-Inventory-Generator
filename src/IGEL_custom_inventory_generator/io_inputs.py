from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .constants import (
    BASE_ALTITUDE_RESOLUTION_KM,
    BASE_LATITUDE_RESOLUTION_DEG,
    BASE_LONGITUDE_RESOLUTION_DEG,
    ENGINE_FILE_PATTERN,
    ENGINE_PROFILE_COLUMN_PATTERN,
    MIN_CREATED_ALTITUDE_RESOLUTION_KM,
    MIN_CREATED_LATITUDE_RESOLUTION_DEG,
    MIN_CREATED_LONGITUDE_RESOLUTION_DEG,
    REQUIRED_ENGINE_COLUMNS,
    REQUIRED_LAUNCH_LIST_COLUMNS,
    REQUIRED_PROFILE_TOTAL_COLUMN,
)
from .errors import EngineDataError, ExportError, LaunchListError, PropellantUseProfileError


def load_engine_data(engine_data_dir: str | Path) -> dict[str, pd.DataFrame]:
    engine_data_dir = Path(engine_data_dir)

    if not engine_data_dir.exists():
        raise EngineDataError(f"Engine data directory does not exist: {engine_data_dir}")
    if not engine_data_dir.is_dir():
        raise EngineDataError(f"Engine data path is not a directory: {engine_data_dir}")

    engine_dict: dict[str, pd.DataFrame] = {}
    matching_files = sorted(engine_data_dir.glob("*_primary_exhaust_indices.csv"))

    if not matching_files:
        raise EngineDataError(
            "No engine CSV files found matching '*_primary_exhaust_indices.csv' "
            f"in {engine_data_dir}"
        )

    for csv_path in matching_files:
        match = ENGINE_FILE_PATTERN.match(csv_path.name)
        if not match:
            continue

        engine_name = match.group("engine_name").strip()

        if not engine_name:
            raise EngineDataError(f"Could not extract engine name from file: {csv_path.name}")

        if engine_name in engine_dict:
            raise EngineDataError(
                f"Duplicate engine name '{engine_name}' derived from filename: {csv_path.name}"
            )

        df = pd.read_csv(csv_path)
        validate_engine_dataframe(df=df, file_name=csv_path.name)

        missing_columns = REQUIRED_ENGINE_COLUMNS.difference(df.columns)
        if missing_columns:
            raise EngineDataError(
                f"Engine file '{csv_path.name}' is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        engine_dict[engine_name] = df

    return engine_dict


def validate_engine_dataframe(df: pd.DataFrame, file_name: str) -> None:
    missing_columns = REQUIRED_ENGINE_COLUMNS.difference(df.columns)
    if missing_columns:
        raise EngineDataError(
            f"Engine file '{file_name}' is missing required columns: {sorted(missing_columns)}"
        )

    if df["Species"].isna().any():
        raise EngineDataError(f"Engine file '{file_name}' contains missing values in 'Species'.")

    numeric_mass = pd.to_numeric(df["Absolute_Mass_Fraction"], errors="coerce")
    if numeric_mass.isna().any():
        raise EngineDataError(
            f"Engine file '{file_name}' contains non-numeric values in 'Absolute_Mass_Fraction'."
        )


def load_launch_list(launch_list_path: str | Path) -> pd.DataFrame:
    launch_list_path = Path(launch_list_path)

    if not launch_list_path.exists():
        raise LaunchListError(f"Launch list file does not exist: {launch_list_path}")
    if not launch_list_path.is_file():
        raise LaunchListError(f"Launch list path is not a file: {launch_list_path}")

    df = pd.read_csv(launch_list_path)
    validate_launch_list_dataframe(df=df, file_name=launch_list_path.name)

    missing_columns = REQUIRED_LAUNCH_LIST_COLUMNS.difference(df.columns)
    if missing_columns:
        raise LaunchListError(
            f"Launch list file '{launch_list_path.name}' is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def validate_launch_list_dataframe(df: pd.DataFrame, file_name: str) -> None:
    missing_columns = REQUIRED_LAUNCH_LIST_COLUMNS.difference(df.columns)
    if missing_columns:
        raise LaunchListError(
            f"Launch list file '{file_name}' is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["Launch_Tag"].isna().any():
        raise LaunchListError(f"Launch list file '{file_name}' contains missing Launch_Tag values.")

    if df["Launch_JD"].isna().any():
        raise LaunchListError(f"Launch list file '{file_name}' contains missing Launch_JD values.")

    if df["Launch_Date"].isna().any():
        raise LaunchListError(f"Launch list file '{file_name}' contains missing Launch_Date values.")


def _to_decimal(value: float | str) -> Decimal:
    return Decimal(str(value))


def _normalize_profile_species_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename propellant-use profile species columns by removing a trailing '_sum'.

    Examples
    --------
    species_mass_Total_sum   -> species_mass_Total
    species_mass_EngineA_sum -> species_mass_EngineA
    """
    rename_map: dict[str, str] = {}

    for col in df.columns:
        if col.startswith("species_mass_") and col.endswith("_sum"):
            rename_map[col] = col[:-4]

    if not rename_map:
        return df

    return df.rename(columns=rename_map)


def _validate_created_resolution(
    target_resolution: float,
    base_resolution: Decimal,
    minimum_resolution: Decimal,
    field_name: str,
) -> int:
    target_dec = _to_decimal(target_resolution)

    if target_dec < minimum_resolution:
        raise PropellantUseProfileError(
            f"{field_name}={target_resolution} is smaller than the minimum allowed "
            f"created resolution of {minimum_resolution}."
        )

    factor = target_dec / base_resolution
    if factor != factor.to_integral_value():
        raise PropellantUseProfileError(
            f"{field_name}={target_resolution} is not an exact multiple of "
            f"the base resolution {base_resolution}."
        )

    factor_int = int(factor)
    if factor_int < 2:
        raise PropellantUseProfileError(
            f"{field_name} must be at least double the ground resolution."
        )

    return factor_int


def _validate_values_on_base_grid(
    values: pd.Series,
    base_resolution: Decimal,
    field_name: str,
) -> None:
    for value in values:
        value_dec = _to_decimal(value)
        scaled = value_dec / base_resolution
        if scaled != scaled.to_integral_value():
            raise PropellantUseProfileError(
                f"Column '{field_name}' contains value {value} that is not aligned with "
                f"the base grid {base_resolution}."
            )


def _validate_base_bin_widths(df: pd.DataFrame) -> None:
    checks = [
        ("ALTITUDE_MIN", "ALTITUDE_MAX", Decimal("0.01"), "altitude"),
        ("LATITUDE_MIN", "LATITUDE_MAX", Decimal("0.01"), "latitude"),
        ("LONGITUDE_MIN", "LONGITUDE_MAX", Decimal("0.01"), "longitude"),
    ]

    for min_col, max_col, expected_width, label in checks:
        for min_val, max_val in zip(df[min_col], df[max_col]):
            width = _to_decimal(max_val) - _to_decimal(min_val)
            if width != expected_width:
                raise PropellantUseProfileError(
                    f"Profile dataframe contains a {label} bin with width {width}, "
                    f"expected {expected_width}."
                )


def _compute_target_bounds_from_min(
    min_series: pd.Series,
    target_resolution: float,
    base_resolution: Decimal,
    minimum_resolution: Decimal,
    field_name: str,
) -> tuple[list[float], list[float]]:
    factor = _validate_created_resolution(
        target_resolution=target_resolution,
        base_resolution=base_resolution,
        minimum_resolution=minimum_resolution,
        field_name=field_name,
    )

    target_dec = _to_decimal(target_resolution)
    shifted_origin = -target_dec / Decimal("2")

    target_mins: list[float] = []
    target_maxs: list[float] = []

    for value in min_series:
        value_dec = _to_decimal(value)

        shifted_index = (value_dec - shifted_origin) / base_resolution
        if shifted_index != shifted_index.to_integral_value():
            raise PropellantUseProfileError(
                f"Value {value} in '{field_name}' is not aligned with the created-grid "
                f"boundary origin {shifted_origin}."
            )

        shifted_index_int = int(shifted_index)
        target_index = shifted_index_int // factor

        target_min = shifted_origin + Decimal(target_index * factor) * base_resolution
        target_max = target_min + target_dec

        target_mins.append(float(target_min))
        target_maxs.append(float(target_max))

    return target_mins, target_maxs


def _validate_propellant_use_profile_dataframe(
    df: pd.DataFrame,
    file_name: str,
    engine_dict: dict[str, pd.DataFrame],
) -> list[str]:
    required_columns = {
        "ALTITUDE_MIN",
        "ALTITUDE_MAX",
        "LATITUDE_MIN",
        "LATITUDE_MAX",
        "LONGITUDE_MIN",
        "LONGITUDE_MAX",
        "NUM",
        REQUIRED_PROFILE_TOTAL_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise PropellantUseProfileError(
            f"Profile file '{file_name}' is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    engine_columns: list[str] = []
    extracted_engine_names: list[str] = []

    for column in df.columns:
        if column == REQUIRED_PROFILE_TOTAL_COLUMN:
            continue

        match = ENGINE_PROFILE_COLUMN_PATTERN.match(column)
        if not match:
            continue

        engine_name = match.group("engine_name").strip()
        if engine_name == "Total":
            continue

        engine_columns.append(column)
        extracted_engine_names.append(engine_name)

    if not engine_columns:
        raise PropellantUseProfileError(
            f"Profile file '{file_name}' must contain at least one engine-specific column "
            "matching 'species_mass_<engine_name>'."
        )

    missing_engines = sorted(
        engine_name for engine_name in extracted_engine_names if engine_name not in engine_dict
    )
    if missing_engines:
        raise PropellantUseProfileError(
            f"Profile file '{file_name}' references engine(s) not found in engine data: "
            f"{missing_engines}"
        )

    return engine_columns


def _rebin_propellant_use_profile(
    df: pd.DataFrame,
    altitude_resolution_km: float,
    latitude_resolution_deg: float,
    longitude_resolution_deg: float,
) -> pd.DataFrame:
    df = df.copy()

    species_columns = [col for col in df.columns if col.startswith("species_mass_")]
    sum_columns = ["NUM"] + species_columns

    numeric_columns = [
        "ALTITUDE_MIN",
        "ALTITUDE_MAX",
        "LATITUDE_MIN",
        "LATITUDE_MAX",
        "LONGITUDE_MIN",
        "LONGITUDE_MAX",
        "NUM",
        *species_columns,
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    _validate_values_on_base_grid(df["ALTITUDE_MIN"], BASE_ALTITUDE_RESOLUTION_KM, "ALTITUDE_MIN")
    _validate_values_on_base_grid(df["ALTITUDE_MAX"], BASE_ALTITUDE_RESOLUTION_KM, "ALTITUDE_MAX")
    _validate_values_on_base_grid(df["LATITUDE_MIN"], BASE_LATITUDE_RESOLUTION_DEG, "LATITUDE_MIN")
    _validate_values_on_base_grid(df["LATITUDE_MAX"], BASE_LATITUDE_RESOLUTION_DEG, "LATITUDE_MAX")
    _validate_values_on_base_grid(df["LONGITUDE_MIN"], BASE_LONGITUDE_RESOLUTION_DEG, "LONGITUDE_MIN")
    _validate_values_on_base_grid(df["LONGITUDE_MAX"], BASE_LONGITUDE_RESOLUTION_DEG, "LONGITUDE_MAX")

    _validate_base_bin_widths(df)

    alt_mins, alt_maxs = _compute_target_bounds_from_min(
        min_series=df["ALTITUDE_MIN"],
        target_resolution=altitude_resolution_km,
        base_resolution=BASE_ALTITUDE_RESOLUTION_KM,
        minimum_resolution=MIN_CREATED_ALTITUDE_RESOLUTION_KM,
        field_name="altitude_resolution_km",
    )
    lat_mins, lat_maxs = _compute_target_bounds_from_min(
        min_series=df["LATITUDE_MIN"],
        target_resolution=latitude_resolution_deg,
        base_resolution=BASE_LATITUDE_RESOLUTION_DEG,
        minimum_resolution=MIN_CREATED_LATITUDE_RESOLUTION_DEG,
        field_name="latitude_resolution_deg",
    )
    lon_mins, lon_maxs = _compute_target_bounds_from_min(
        min_series=df["LONGITUDE_MIN"],
        target_resolution=longitude_resolution_deg,
        base_resolution=BASE_LONGITUDE_RESOLUTION_DEG,
        minimum_resolution=MIN_CREATED_LONGITUDE_RESOLUTION_DEG,
        field_name="longitude_resolution_deg",
    )

    df["ALTITUDE_MIN"] = alt_mins
    df["ALTITUDE_MAX"] = alt_maxs
    df["LATITUDE_MIN"] = lat_mins
    df["LATITUDE_MAX"] = lat_maxs
    df["LONGITUDE_MIN"] = lon_mins
    df["LONGITUDE_MAX"] = lon_maxs

    drop_columns = [col for col in ["BINS_ALT", "BINS_LAT", "BINS_LONG"] if col in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    grouped = (
        df.groupby(
            [
                "ALTITUDE_MIN",
                "ALTITUDE_MAX",
                "LATITUDE_MIN",
                "LATITUDE_MAX",
                "LONGITUDE_MIN",
                "LONGITUDE_MAX",
            ],
            as_index=False,
            dropna=False,
        )[sum_columns]
        .sum()
    )

    return grouped


def load_propellant_use_profiles(
    propellant_use_profiles_dir: str | Path,
    launch_list_df: pd.DataFrame,
    engine_dict: dict[str, pd.DataFrame],
    altitude_resolution_km: float,
    latitude_resolution_deg: float,
    longitude_resolution_deg: float,
) -> dict[str, pd.DataFrame]:
    propellant_use_profiles_dir = Path(propellant_use_profiles_dir)

    if not propellant_use_profiles_dir.exists():
        raise PropellantUseProfileError(
            f"Propellant use profiles directory does not exist: {propellant_use_profiles_dir}"
        )
    if not propellant_use_profiles_dir.is_dir():
        raise PropellantUseProfileError(
            f"Propellant use profiles path is not a directory: {propellant_use_profiles_dir}"
        )
    if "Launch_Tag" not in launch_list_df.columns:
        raise PropellantUseProfileError(
            "Launch list dataframe must contain the column 'Launch_Tag'."
        )

    profile_dict: dict[str, pd.DataFrame] = {}
    launch_tags = launch_list_df["Launch_Tag"].dropna().astype(str).unique()

    for launch_tag in tqdm(launch_tags, desc="Loading and resizing propellant profiles", unit="profile"):
        profile_path = propellant_use_profiles_dir / f"{launch_tag}_propellant_use_profile.csv"

        if not profile_path.exists():
            raise PropellantUseProfileError(
                f"Missing propellant use profile for launch tag '{launch_tag}': {profile_path.name}"
            )
        if not profile_path.is_file():
            raise PropellantUseProfileError(
                f"Expected a file for launch tag '{launch_tag}', but found something else: "
                f"{profile_path}"
            )

        df = pd.read_csv(profile_path)
        df = _normalize_profile_species_column_names(df)

        _validate_propellant_use_profile_dataframe(
            df=df,
            file_name=profile_path.name,
            engine_dict=engine_dict,
        )

        rebinned_df = _rebin_propellant_use_profile(
            df=df,
            altitude_resolution_km=altitude_resolution_km,
            latitude_resolution_deg=latitude_resolution_deg,
            longitude_resolution_deg=longitude_resolution_deg,
        )

        profile_dict[launch_tag] = rebinned_df

    return profile_dict


def export_primary_exhaust_profiles(
    primary_exhaust_dict: dict[str, pd.DataFrame],
    output_dir: str | Path,
    overwrite: bool,
) -> None:
    output_dir = Path(output_dir)
    export_dir = output_dir / "primary_exhaust_profiles"
    export_dir.mkdir(parents=True, exist_ok=True)

    for launch_tag, df in primary_exhaust_dict.items():
        output_path = export_dir / f"{launch_tag}_primary_exhaust_profile.csv"

        if output_path.exists() and not overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        df.to_csv(output_path, index=False)


def export_final_emission_profiles(
    final_emissions_dict: dict[str, pd.DataFrame],
    output_dir: str | Path,
    overwrite: bool,
) -> None:
    output_dir = Path(output_dir)
    export_dir = output_dir / "final_emission_profiles"
    export_dir.mkdir(parents=True, exist_ok=True)

    for launch_tag, df in final_emissions_dict.items():
        output_path = export_dir / f"{launch_tag}_final_emission_profile.csv"

        if output_path.exists() and not overwrite:
            raise ExportError(f"File already exists and overwrite is false: {output_path}")

        df.to_csv(output_path, index=False)
