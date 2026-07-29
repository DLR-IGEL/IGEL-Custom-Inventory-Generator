from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from constants import (
    INVENTORY_SPATIAL_INDEX_COLUMNS,
    LAUNCH_DATE_INPUT_FORMAT,
    TIME_RESOLUTION_PATTERN,
    TIMESTEP_KEY_PATTERN,
)
from errors import ConfigError, ExportError, LaunchListError, PostCombustionError
from models import DomainConfig

def _sum_species_mass_columns(df: pd.DataFrame) -> float:
    species_mass_columns = [
        col for col in df.columns
        if col.startswith("species_mass_") and col != "species_mass_Total"
    ]
    if not species_mass_columns:
        return 0.0
    return float(df[species_mass_columns].sum().sum())

def _normalize_longitude_bins_for_global_grid(
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shift longitude bins whose midpoint lies outside [-180, 180] back into the
    canonical global grid range, preserving bin width.

    Example:
      187.5 .. 188.5  ->  -172.5 .. -171.5
      180.5 .. 181.5  ->  -179.5 .. -178.5
    """
    required_columns = {"LONGITUDE_MIN", "LONGITUDE_MAX"}
    missing_columns = required_columns.difference(inventory_df.columns)
    if missing_columns:
        raise ExportError(
            "Inventory dataframe is missing longitude columns required for "
            f"longitude normalization: {sorted(missing_columns)}"
        )

    if inventory_df.empty:
        return inventory_df.copy()

    df = inventory_df.copy()

    lon_min = pd.to_numeric(df["LONGITUDE_MIN"], errors="coerce")
    lon_max = pd.to_numeric(df["LONGITUDE_MAX"], errors="coerce")
    if lon_min.isna().any() or lon_max.isna().any():
        raise ExportError("Longitude boundary columns contain non-numeric values.")

    lon_mid = (lon_min + lon_max) / 2.0

    shift_minus_360 = lon_mid > 180.0
    shift_plus_360 = lon_mid < -180.0

    if shift_minus_360.any():
        df.loc[shift_minus_360, "LONGITUDE_MIN"] = lon_min.loc[shift_minus_360] - 360.0
        df.loc[shift_minus_360, "LONGITUDE_MAX"] = lon_max.loc[shift_minus_360] - 360.0

    if shift_plus_360.any():
        df.loc[shift_plus_360, "LONGITUDE_MIN"] = lon_min.loc[shift_plus_360] + 360.0
        df.loc[shift_plus_360, "LONGITUDE_MAX"] = lon_max.loc[shift_plus_360] + 360.0

    return df



def split_time_range_into_timesteps(
    start_date: datetime,
    end_date: datetime,
    time_resolution: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    match = TIME_RESOLUTION_PATTERN.match(time_resolution)
    if not match:
        raise ConfigError(
            "processing.time_resolution must look like '1d', '2w', or '3m'."
        )

    value = int(match.group("value"))
    unit = match.group("unit")

    current_start = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    timesteps: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    while current_start <= end_ts:
        if unit == "d":
            next_start = current_start + pd.Timedelta(days=value)
        elif unit == "w":
            next_start = current_start + pd.Timedelta(weeks=value)
        elif unit == "m":
            next_start = current_start + pd.DateOffset(months=value)
        else:
            raise ConfigError(
                f"Unsupported time resolution unit '{unit}' in '{time_resolution}'."
            )

        current_end = min(next_start - pd.Timedelta(days=1), end_ts)
        timesteps.append((current_start, current_end))
        current_start = next_start

    return timesteps


def prepare_launch_list_for_inventory(launch_list_df: pd.DataFrame) -> pd.DataFrame:
    df = launch_list_df.copy()

    parsed_dates = pd.to_datetime(
        df["Launch_Date"].astype(str).str.strip(),
        format=LAUNCH_DATE_INPUT_FORMAT,
        errors="coerce",
    )

    if parsed_dates.isna().any():
        bad_values = (
            df.loc[parsed_dates.isna(), "Launch_Date"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise LaunchListError(
            "Could not parse one or more Launch_Date values with format "
            f"'{LAUNCH_DATE_INPUT_FORMAT}'. Bad values: {bad_values}"
        )

    multiplier_numeric = pd.to_numeric(df["Multiplier"], errors="coerce")
    if multiplier_numeric.isna().any():
        bad_tags = df.loc[multiplier_numeric.isna(), "Launch_Tag"].astype(str).tolist()
        raise LaunchListError(
            f"Non-numeric Multiplier values found for launch tags: {bad_tags}"
        )

    df["Launch_Date_Parsed"] = parsed_dates.dt.normalize()
    df["Multiplier"] = multiplier_numeric.astype(float)

    return df


def parse_timestep_key(timestep_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = TIMESTEP_KEY_PATTERN.match(str(timestep_key))
    if not match:
        raise ConfigError(
            f"Invalid timestep key '{timestep_key}'. Expected 'YYYY-MM-DD__YYYY-MM-DD'."
        )

    start_ts = pd.Timestamp(match.group("start"))
    end_ts = pd.Timestamp(match.group("end"))

    if end_ts < start_ts:
        raise ConfigError(f"Invalid timestep key '{timestep_key}': end before start.")

    return start_ts, end_ts


def timestep_duration_seconds_from_key(timestep_key: str) -> float:
    start_ts, end_ts = parse_timestep_key(timestep_key)
    duration = end_ts - start_ts + pd.Timedelta(days=1)
    return float(duration.total_seconds())


def _build_inventory_columns(
    final_emissions_dict: dict[str, pd.DataFrame],
) -> list[str]:
    if not final_emissions_dict:
        raise PostCombustionError(
            "final_emissions_dict is empty; cannot build timestep inventories."
        )

    ordered_columns: list[str] = INVENTORY_SPATIAL_INDEX_COLUMNS.copy()

    for df in final_emissions_dict.values():
        for col in df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

    return ordered_columns


def _empty_timestep_inventory_dataframe(
    inventory_columns: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(columns=inventory_columns)


def _scale_single_launch_profile(
    launch_profile_df: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    scaled_df = launch_profile_df.copy(deep=True)

    numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_scale = [
        col for col in numeric_columns if col not in INVENTORY_SPATIAL_INDEX_COLUMNS
    ]

    if columns_to_scale:
        scaled_df[columns_to_scale] = scaled_df[columns_to_scale] * multiplier

    return scaled_df


def combine_final_emission_profiles_for_timestep(
    launches_in_timestep_df: pd.DataFrame,
    final_emissions_dict: dict[str, pd.DataFrame],
    inventory_columns: list[str],
) -> pd.DataFrame:
    if launches_in_timestep_df.empty:
        return _empty_timestep_inventory_dataframe(inventory_columns)

    weighted_profiles: list[pd.DataFrame] = []

    launches_grouped = (
        launches_in_timestep_df.groupby("Launch_Tag", as_index=False)["Multiplier"].sum()
    )

    for _, launch_row in launches_grouped.iterrows():
        launch_tag = str(launch_row["Launch_Tag"])
        multiplier = float(launch_row["Multiplier"])

        if launch_tag not in final_emissions_dict:
            raise PostCombustionError(
                f"Launch tag '{launch_tag}' found in launch list but missing in final_emissions_dict."
            )

        scaled_profile = _scale_single_launch_profile(
            launch_profile_df=final_emissions_dict[launch_tag],
            multiplier=multiplier,
        )
        weighted_profiles.append(scaled_profile)

    combined_df = pd.concat(weighted_profiles, ignore_index=True, sort=False)

    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_sum = [
        col for col in numeric_columns if col not in INVENTORY_SPATIAL_INDEX_COLUMNS
    ]

    if not columns_to_sum:
        raise PostCombustionError(
            "No numeric emission columns found to combine for timestep inventory."
        )

    combined_df = (
        combined_df.groupby(
            INVENTORY_SPATIAL_INDEX_COLUMNS,
            as_index=False,
            dropna=False,
        )[columns_to_sum]
        .sum()
        .sort_values(INVENTORY_SPATIAL_INDEX_COLUMNS)
        .reset_index(drop=True)
    )

    combined_df = combined_df.reindex(columns=inventory_columns, fill_value=0)
    return combined_df


def enforce_domain_boundaries_on_inventory_df(
    inventory_df: pd.DataFrame,
    domain: DomainConfig,
) -> pd.DataFrame:
    """
    Apply horizontal/vertical domain filtering to a timestep inventory dataframe.

    Horizontal filtering is always midpoint-based.

    Vertical handling:
      - cut:
          bins with altitude midpoint above altitude_max_km are discarded
      - include_at_top_layer:
          bins with altitude midpoint above altitude_max_km are moved into the
          uppermost inventory layer, preserving latitude/longitude and summing
          with any existing top-layer emissions in the same horizontal cell
    """
    required_columns = set(INVENTORY_SPATIAL_INDEX_COLUMNS)
    missing_columns = required_columns.difference(inventory_df.columns)
    if missing_columns:
        raise ExportError(
            "Inventory dataframe is missing required spatial columns for domain filtering: "
            f"{sorted(missing_columns)}"
        )

    if inventory_df.empty:
        return inventory_df.copy()

    df = inventory_df.copy()
    df = _normalize_longitude_bins_for_global_grid(df)

    altitude_mid = (df["ALTITUDE_MIN"] + df["ALTITUDE_MAX"]) / 2.0
    latitude_mid = (df["LATITUDE_MIN"] + df["LATITUDE_MAX"]) / 2.0
    longitude_mid =  (df["LONGITUDE_MIN"] + df["LONGITUDE_MAX"]) / 2.0


    species_mass_columns = [
        col for col in df.columns
        if col.startswith("species_mass_") and col != "species_mass_Total"
    ]

    def _mass(mask) -> float:
        if not species_mass_columns:
            return 0.0
        if mask.sum() == 0:
            return 0.0
        return float(df.loc[mask, species_mass_columns].sum().sum())

    if domain.inclusive:
        latlon_mask = (
            (latitude_mid >= domain.latitude_min_deg)
            & (latitude_mid <= domain.latitude_max_deg)
            & (longitude_mid >= domain.longitude_min_deg)
            & (longitude_mid <= domain.longitude_max_deg)
        )
        lower_alt_mask = altitude_mid >= domain.altitude_min_km
        upper_alt_mask = altitude_mid <= domain.altitude_max_km
    else:
        latlon_mask = (
            (latitude_mid > domain.latitude_min_deg)
            & (latitude_mid < domain.latitude_max_deg)
            & (longitude_mid > domain.longitude_min_deg)
            & (longitude_mid < domain.longitude_max_deg)
        )
        lower_alt_mask = altitude_mid > domain.altitude_min_km
        upper_alt_mask = altitude_mid < domain.altitude_max_km

    df = df.loc[latlon_mask & lower_alt_mask].copy()
    if df.empty:
        return df.reset_index(drop=True)

    altitude_mid = (df["ALTITUDE_MIN"] + df["ALTITUDE_MAX"]) / 2.0

    if domain.emissions_above_inventory == "discard":
        df = df.loc[upper_alt_mask.loc[df.index]].copy()
        return df.reset_index(drop=True)

    if domain.emissions_above_inventory == "include_at_top_layer":
        above_mask = ~upper_alt_mask.loc[df.index]
    
        if above_mask.any():
            in_domain_df = df.loc[~above_mask].copy()
    
            if in_domain_df.empty:
                raise ExportError(
                    "Cannot use 'include_at_top_layer' because no in-domain vertical layer "
                    "exists below the configured top boundary."
                )
    
            in_domain_mid = (in_domain_df["ALTITUDE_MIN"] + in_domain_df["ALTITUDE_MAX"]) / 2.0
            top_mid = in_domain_mid.max()
    
            top_layer_rows = in_domain_df.loc[in_domain_mid == top_mid]
            top_alt_min = float(top_layer_rows.iloc[0]["ALTITUDE_MIN"])
            top_alt_max = float(top_layer_rows.iloc[0]["ALTITUDE_MAX"])
    
            # move all above-domain emissions into the actual existing top layer
            df.loc[above_mask, "ALTITUDE_MIN"] = top_alt_min
            df.loc[above_mask, "ALTITUDE_MAX"] = top_alt_max
    
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns_to_sum = [
                col for col in numeric_columns if col not in INVENTORY_SPATIAL_INDEX_COLUMNS
            ]
    
            df = (
                df.groupby(
                    INVENTORY_SPATIAL_INDEX_COLUMNS,
                    as_index=False,
                    dropna=False,
                )[columns_to_sum]
                .sum()
                .sort_values(INVENTORY_SPATIAL_INDEX_COLUMNS)
                .reset_index(drop=True)
            )
    return df.reset_index(drop=True)



def build_timestep_inventory_dataframes(
    launch_list_df: pd.DataFrame,
    final_emissions_dict: dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    time_resolution: str,
    domain: DomainConfig,
) -> dict[str, pd.DataFrame]:
    prepared_launch_list = prepare_launch_list_for_inventory(launch_list_df)
    timesteps = split_time_range_into_timesteps(
        start_date=start_date,
        end_date=end_date,
        time_resolution=time_resolution,
    )
    inventory_columns = _build_inventory_columns(final_emissions_dict)

    timestep_inventory_dict: dict[str, pd.DataFrame] = {}

    total_selected_launch_mass = 0.0
    total_combined_before_domain = 0.0
    total_after_domain = 0.0

    for timestep_start, timestep_end in timesteps:
        launches_in_timestep = prepared_launch_list[
            (prepared_launch_list["Launch_Date_Parsed"] >= timestep_start)
            & (prepared_launch_list["Launch_Date_Parsed"] <= timestep_end)
        ].copy()

        timestep_key = (
            f"{timestep_start.strftime('%Y-%m-%d')}"
            f"__{timestep_end.strftime('%Y-%m-%d')}"
        )

        # Sum the referenced launch profiles before any combining/filtering
        launches_grouped = (
            launches_in_timestep.groupby("Launch_Tag", as_index=False)["Multiplier"].sum()
            if not launches_in_timestep.empty
            else pd.DataFrame(columns=["Launch_Tag", "Multiplier"])
        )

        timestep_selected_mass = 0.0
        for _, launch_row in launches_grouped.iterrows():
            launch_tag = str(launch_row["Launch_Tag"])
            multiplier = float(launch_row["Multiplier"])
            if launch_tag not in final_emissions_dict:
                raise PostCombustionError(
                    f"Launch tag '{launch_tag}' found in launch list but missing in final_emissions_dict."
                )
            timestep_selected_mass += (
                _sum_species_mass_columns(final_emissions_dict[launch_tag]) * multiplier
            )

        timestep_inventory_df = combine_final_emission_profiles_for_timestep(
            launches_in_timestep_df=launches_in_timestep,
            final_emissions_dict=final_emissions_dict,
            inventory_columns=inventory_columns,
        )

        timestep_combined_mass = _sum_species_mass_columns(timestep_inventory_df)

        timestep_inventory_df = enforce_domain_boundaries_on_inventory_df(
            inventory_df=timestep_inventory_df,
            domain=domain,
        )

        timestep_after_domain_mass = _sum_species_mass_columns(timestep_inventory_df)

        timestep_inventory_df = timestep_inventory_df.reindex(
            columns=inventory_columns,
            fill_value=0,
        )

        timestep_inventory_dict[timestep_key] = timestep_inventory_df

        total_selected_launch_mass += timestep_selected_mass
        total_combined_before_domain += timestep_combined_mass
        total_after_domain += timestep_after_domain_mass

    return timestep_inventory_dict