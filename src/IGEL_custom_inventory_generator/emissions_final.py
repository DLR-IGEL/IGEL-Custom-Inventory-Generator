from __future__ import annotations

import numpy as np
import pandas as pd

from constants import SPECIES_DB
from errors import PostCombustionError


def apply_csvem_postcombustion(
    primary_exhaust_df: pd.DataFrame,
    species_db: dict[str, dict[str, float]],
) -> pd.DataFrame:
    final_emissions = primary_exhaust_df.copy(deep=True)

    required_altitude_columns = {"ALTITUDE_MIN", "ALTITUDE_MAX", "species_mass_Total"}
    missing = required_altitude_columns.difference(final_emissions.columns)
    if missing:
        raise PostCombustionError(
            f"Primary emission profile is missing required columns: {sorted(missing)}"
        )

    altitude = (final_emissions["ALTITUDE_MIN"] + final_emissions["ALTITUDE_MAX"]) / 2.0

    h2o = primary_exhaust_df.get("species_mass_H2O", 0)
    h = primary_exhaust_df.get("species_mass_H", 0)
    h2 = primary_exhaust_df.get("species_mass_H2", 0)
    oh = primary_exhaust_df.get("species_mass_OH", 0)

    if (
        (getattr(h2o, "sum", lambda: h2o)() != 0)
        or (getattr(h, "sum", lambda: h)() != 0)
        or (getattr(h2, "sum", lambda: h2)() != 0)
        or (getattr(oh, "sum", lambda: oh)() != 0)
    ):
        final_emissions['species_mass_H2O']=(
            h2o
            + (species_db["H2O"]['MolarMass'] / species_db["H2"]['MolarMass']) * (h2+h-(oh*species_db["H2"]['MolarMass'] / (species_db["OH"]['MolarMass']*2)))
            + (species_db["H2O"]['MolarMass'] / (species_db["OH"]['MolarMass'])) * oh
        )
        final_emissions.drop(
            ["species_mass_H", "species_mass_OH", "species_mass_H2"],
            axis=1,
            errors="ignore",
            inplace=True,
        )

    no = primary_exhaust_df.get("species_mass_NO", 0)
    total = primary_exhaust_df.get("species_mass_Total", 0)
    final_emissions["species_mass_NO"] = (
        no + total * 0.033 * np.exp(-0.26 * altitude)
    ).where(lambda x: x >= 1e-3, 0)

    co = primary_exhaust_df.get("species_mass_CO", 0)
    co2 = primary_exhaust_df.get("species_mass_CO2", 0)

    if (
        (getattr(co, "sum", lambda: co)() != 0)
        or (getattr(co2, "sum", lambda: co2)() != 0)
    ):
        final_emissions["species_mass_CO"] = np.minimum(
            co,
            0.0025 * np.exp(0.067 * altitude) * (co + co2),
        )
        final_emissions["species_mass_CO2"] = (
            co2
            + (species_db["CO2"]["MolarMass"] / species_db["CO"]["MolarMass"])
            * (co - final_emissions["species_mass_CO"])
        )

    c = primary_exhaust_df.get("species_mass_C", 0)

    if getattr(c, "sum", lambda: c)() != 0:
        final_emissions["species_mass_C"] = (
            c * np.maximum(0.04, np.minimum(1, 0.04 * np.exp(0.12 * (altitude - 15))))
        )

    cl = primary_exhaust_df.get("species_mass_CL", 0)
    cl2 = primary_exhaust_df.get("species_mass_CL2", 0)
    hcl = primary_exhaust_df.get("species_mass_HCL", 0)

    if (
        (getattr(cl, "sum", lambda: cl)() != 0)
        or (getattr(cl2, "sum", lambda: cl2)() != 0)
        or (getattr(hcl, "sum", lambda: hcl)() != 0)
    ):
        cl_total = cl + cl2 + hcl

        alt_table = np.array([0, 0.5, 6, 10, 15, 18, 20, 30, 40], dtype=float)
        cl_ratio_table = np.array(
            [
                0.056421991,
                0.056421991,
                0.058159109,
                0.102840691,
                0.145462362,
                0.091503587,
                0.04977766,
                0.155916954,
                0.34290904,
            ],
            dtype=float,
        )
        cl2_ratio_table = np.array(
            [
                0.943578009,
                0.943578009,
                0.941840891,
                0.897159309,
                0.854537638,
                0.908496413,
                0.95022234,
                0.844083046,
                0.65709096,
            ],
            dtype=float,
        )

        mask_above_40 = altitude > 40
        mask_below_eq_40 = ~mask_above_40

        final_emissions["species_mass_HCL"] = 0.0
        final_emissions["species_mass_CL"] = 0.0
        final_emissions["species_mass_CL2"] = 0.0

        hcl_fraction = 0.627 / (1 + np.exp(0.226 * (altitude - 20.9))) + 0.304
        cl_ratio = np.interp(altitude, alt_table, cl_ratio_table)
        cl2_ratio = np.interp(altitude, alt_table, cl2_ratio_table)

        final_emissions.loc[mask_below_eq_40, "species_mass_HCL"] = (
            cl_total[mask_below_eq_40] * hcl_fraction[mask_below_eq_40]
        )

        cl_rest = (
            cl_total[mask_below_eq_40]
            - final_emissions.loc[mask_below_eq_40, "species_mass_HCL"]
        )

        final_emissions.loc[mask_below_eq_40, "species_mass_CL"] = (
            cl_rest * cl_ratio[mask_below_eq_40]
        )
        final_emissions.loc[mask_below_eq_40, "species_mass_CL2"] = (
            cl_rest * cl2_ratio[mask_below_eq_40]
        )

        final_emissions.loc[mask_above_40, "species_mass_HCL"] = (
            cl_total[mask_above_40] * 0.31281
        )
        final_emissions.loc[mask_above_40, "species_mass_CL"] = (
            cl_total[mask_above_40] * 0.23564
        )
        final_emissions.loc[mask_above_40, "species_mass_CL2"] = (
            cl_total[mask_above_40] * 0.45154
        )

    final_emissions["species_mass_Total"] = (
        final_emissions
        .drop(["species_mass_Total"], axis=1, errors="ignore")
        .filter(like="species_mass_")
        .sum(axis=1)
    )

    return final_emissions


def calculate_final_emissions(
    primary_exhaust_dict: dict[str, pd.DataFrame],
    post_combustion_method: str | None,
) -> dict[str, pd.DataFrame]:
    final_emissions_dict: dict[str, pd.DataFrame] = {}

    for launch_tag, primary_df in primary_exhaust_dict.items():
        if post_combustion_method is None:
            final_df = primary_df.copy(deep=True)
        elif post_combustion_method == "CSVEM":
            final_df = apply_csvem_postcombustion(
                primary_exhaust_df=primary_df,
                species_db=SPECIES_DB,
            )
        else:
            raise PostCombustionError(
                f"Unsupported post-combustion method: {post_combustion_method}"
            )

        final_emissions_dict[launch_tag] = final_df

    return final_emissions_dict