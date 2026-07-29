from __future__ import annotations

import pandas as pd

from IGEL_custom_inventory_generator.constants import (
    INVENTORY_SPATIAL_INDEX_COLUMNS,
    REQUIRED_LAUNCH_LIST_COLUMNS,
)
from IGEL_custom_inventory_generator.inventory_timesteps import (
    combine_final_emission_profiles_for_timestep,
    prepare_launch_list_for_inventory,
)
from IGEL_custom_inventory_generator.io_inputs import validate_launch_list_dataframe


def _launch_profile(mass_kg: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ALTITUDE_MIN": [0.0],
            "ALTITUDE_MAX": [1.0],
            "LATITUDE_MIN": [0.0],
            "LATITUDE_MAX": [1.0],
            "LONGITUDE_MIN": [0.0],
            "LONGITUDE_MAX": [1.0],
            "species_mass_CO2": [mass_kg],
            "species_mass_Total": [mass_kg],
        }
    )


def test_launch_list_schema() -> None:
    assert REQUIRED_LAUNCH_LIST_COLUMNS == {"Launch_Tag", "Launch_Date", "Launch_JD"}


def test_each_launch_row_contributes_one_profile() -> None:
    launches = pd.DataFrame(
        {
            "Launch_Tag": ["ExampleLaunch", "ExampleLaunch"],
            "Launch_Date": ["2024 Jan 01", "2024 Jan 01"],
            "Launch_JD": [2460310.5, 2460310.5],
        }
    )
    validate_launch_list_dataframe(launches, "Launch_List.csv")
    prepared = prepare_launch_list_for_inventory(launches)

    inventory_columns = [
        *INVENTORY_SPATIAL_INDEX_COLUMNS,
        "species_mass_CO2",
        "species_mass_Total",
    ]
    profile = _launch_profile()

    one_launch = combine_final_emission_profiles_for_timestep(
        prepared.iloc[:1],
        {"ExampleLaunch": profile},
        inventory_columns,
    )
    two_launches = combine_final_emission_profiles_for_timestep(
        prepared,
        {"ExampleLaunch": profile},
        inventory_columns,
    )

    assert one_launch["species_mass_CO2"].sum() == 10.0
    assert two_launches["species_mass_CO2"].sum() == 20.0
