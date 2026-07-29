from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from IGEL_custom_inventory_generator.emissions_final import apply_csvem_postcombustion
from IGEL_custom_inventory_generator.emissions_primary import calculate_primary_exhaust
from IGEL_custom_inventory_generator.constants import SPECIES_DB


def test_primary_emissions_preserve_engine_mass_fractions() -> None:
    profiles = {
        "ExampleLaunch": pd.DataFrame(
            {
                "ALTITUDE_MIN": [0.0],
                "ALTITUDE_MAX": [0.01],
                "LATITUDE_MIN": [0.0],
                "LATITUDE_MAX": [0.01],
                "LONGITUDE_MIN": [0.0],
                "LONGITUDE_MAX": [0.01],
                "NUM": [1],
                "species_mass_Total": [10.0],
                "species_mass_ExampleEngine": [10.0],
            }
        )
    }
    engines = {
        "ExampleEngine": pd.DataFrame(
            {
                "Species": ["CO2", "H2O", "Total"],
                "Absolute_Mass_Fraction": [0.75, 0.25, 1.0],
            }
        )
    }

    result = calculate_primary_exhaust(profiles, engines)["ExampleLaunch"]

    assert result["species_mass_CO2"].iloc[0] == 7.5
    assert result["species_mass_H2O"].iloc[0] == 2.5
    assert result["species_mass_Total"].iloc[0] == 10.0


def test_csvem_regression_at_zero_kilometres() -> None:
    profile = pd.DataFrame(
        {
            "ALTITUDE_MIN": [-0.005],
            "ALTITUDE_MAX": [0.005],
            "LATITUDE_MIN": [0.0],
            "LATITUDE_MAX": [0.01],
            "LONGITUDE_MIN": [0.0],
            "LONGITUDE_MAX": [0.01],
            "species_mass_Total": [100.0],
            "species_mass_CO": [10.0],
            "species_mass_CO2": [20.0],
        }
    )

    result = apply_csvem_postcombustion(profile, SPECIES_DB)

    expected_co = 0.0025 * (10.0 + 20.0)
    expected_co2 = 20.0 + (
        SPECIES_DB["CO2"]["MolarMass"] / SPECIES_DB["CO"]["MolarMass"]
    ) * (10.0 - expected_co)
    expected_no = 100.0 * 0.033 * np.exp(0.0)

    assert result["species_mass_CO"].iloc[0] == pytest.approx(expected_co)
    assert result["species_mass_CO2"].iloc[0] == pytest.approx(expected_co2)
    assert result["species_mass_NO"].iloc[0] == pytest.approx(expected_no)
