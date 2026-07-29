from __future__ import annotations

import pandas as pd

from constants import ENGINE_PROFILE_COLUMN_PATTERN
from errors import PrimaryEmissionError


def _build_engine_species_factors(
    engine_dict: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    engine_species_factors: dict[str, dict[str, float]] = {}

    for engine_name, engine_df in engine_dict.items():
        required_columns = {"Species", "Absolute_Mass_Fraction"}
        missing = required_columns.difference(engine_df.columns)
        if missing:
            raise PrimaryEmissionError(
                f"Engine '{engine_name}' is missing required columns: {sorted(missing)}"
            )

        species_series = engine_df["Species"].astype(str).str.strip()
        mass_fraction_series = pd.to_numeric(
            engine_df["Absolute_Mass_Fraction"],
            errors="coerce",
        )

        if mass_fraction_series.isna().any():
            bad_rows = engine_df.loc[mass_fraction_series.isna(), "Absolute_Mass_Fraction"]
            raise PrimaryEmissionError(
                f"Engine '{engine_name}' contains non-numeric values in "
                f"'Absolute_Mass_Fraction': {bad_rows.tolist()}"
            )

        species_factors: dict[str, float] = {}
        for species_name, mass_fraction in zip(species_series, mass_fraction_series):
            if species_name == "Total":
                continue
            species_factors[species_name] = float(mass_fraction)

        if not species_factors:
            raise PrimaryEmissionError(
                f"Engine '{engine_name}' has no usable species rows after skipping 'Total'."
            )

        engine_species_factors[engine_name] = species_factors

    return engine_species_factors


def calculate_primary_exhaust(
    propellant_use_profile_dict: dict[str, pd.DataFrame],
    engine_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    engine_species_factors = _build_engine_species_factors(engine_dict)

    primary_exhaust_dict: dict[str, pd.DataFrame] = {}

    for launch_tag, profile_df in propellant_use_profile_dict.items():
        df = profile_df.copy()

        if "species_mass_Total" not in df.columns:
            raise PrimaryEmissionError(
                f"Launch tag '{launch_tag}' dataframe is missing 'species_mass_Total'."
            )

        engine_columns: dict[str, str] = {}
        for column in df.columns:
            if column == "species_mass_Total":
                continue

            match = ENGINE_PROFILE_COLUMN_PATTERN.match(column)
            if not match:
                continue

            engine_name = match.group("engine_name").strip()
            if engine_name == "Total":
                continue

            if engine_name not in engine_species_factors:
                raise PrimaryEmissionError(
                    f"Launch tag '{launch_tag}' references unknown engine '{engine_name}'."
                )

            engine_columns[column] = engine_name

        if not engine_columns:
            raise PrimaryEmissionError(
                f"Launch tag '{launch_tag}' contains no engine columns of the form "
                "'species_mass_<engine_name>'."
            )

        output_df = df.drop(columns=list(engine_columns.keys())).copy()

        for engine_column, engine_name in engine_columns.items():
            engine_mass = pd.to_numeric(df[engine_column], errors="coerce")
            if engine_mass.isna().any():
                raise PrimaryEmissionError(
                    f"Launch tag '{launch_tag}', column '{engine_column}' contains non-numeric values."
                )

            species_factors = engine_species_factors[engine_name]

            for species_name, mass_fraction in species_factors.items():
                species_column = f"species_mass_{species_name}"
                contribution = engine_mass * mass_fraction

                if species_column in output_df.columns:
                    output_df[species_column] = output_df[species_column] + contribution
                else:
                    output_df[species_column] = contribution

        primary_exhaust_dict[launch_tag] = output_df

    return primary_exhaust_dict