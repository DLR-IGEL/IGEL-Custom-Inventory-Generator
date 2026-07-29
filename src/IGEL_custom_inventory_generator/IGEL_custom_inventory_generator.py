from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constants import DATE_FORMAT, SPECIES_DB
from .emissions_final import calculate_final_emissions
from .emissions_primary import calculate_primary_exhaust
from .errors import ConfigError, VerificationError
from .inventory_config import load_config
from .inventory_netcdf import export_inventory_netcdfs
from .inventory_timesteps import build_timestep_inventory_dataframes
from .inventory_transform import transform_timestep_inventory_dict
from .inventory_verify import verify_inventory_netcdf
from .io_inputs import (
    export_final_emission_profiles,
    export_primary_exhaust_profiles,
    load_engine_data,
    load_launch_list,
    load_propellant_use_profiles,
)
from .models import Config



def run_inventory_generator(config: Config) -> None:
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting DLR-IGEL-Custom-Inventory-Generator")
    print("This tool was created by Moritz Herberhold as part of the DLR S3D-BETTER project")
    print("Configuration loaded successfully.")
    print(f"Inventory name:                 {config.metadata.inventory_name}")
    print(f"Config version:                 {config.metadata.config_version}")
    print(f"Institution:                    {config.metadata.institution}")
    print(f"Creator name:                   {config.metadata.creator_name}")
    print(f"Creator email:                  {config.metadata.creator_email}")
    print(f"Project:                        {config.metadata.project}")
    print(f"Launch list:                    {config.input_data.launch_list}")
    print(f"Launch profiles dir:            {config.input_data.propellant_use_profiles_dir}")
    print(f"Engine data dir:                {config.input_data.engine_data_dir}")
    print(f"Output dir:                     {output_dir}")
    print(f"Run name:                       {config.output.run_name}")
    print(f"Export individual profiles:     {config.output.export_individual_profiles}")
    print(f"Inventory emission unit:        {config.output.inventory_emission_unit}")
    print(f"Vertical coordinate:            {config.output.inventory_vertical_coordinate}")
    print(f"Inventory NetCDF mode:          {config.output.inventory_netcdf_mode}")
    print(
          f"Altitude domain:                {config.domain.altitude_min_km} "
        f"to {config.domain.altitude_max_km} km"
    )
    print(
          f"Latitude domain:                {config.domain.latitude_min_deg} "
        f"to {config.domain.latitude_max_deg} deg"
    )
    print(
          f"Longitude domain:               {config.domain.longitude_min_deg} "
        f"to {config.domain.longitude_max_deg} deg"
    )
    print(
          f"Include domain boundary:        {config.domain.inclusive}"
    )
    print(
          f"Emissions above inventory:      {config.domain.emissions_above_inventory}"
    )
    print(f"Altitude resolution:            {config.grid.altitude_resolution_km} km")
    print(f"Latitude resolution:            {config.grid.latitude_resolution_deg} deg")
    print(f"Longitude resolution:           {config.grid.longitude_resolution_deg} deg")
    print(f"Post combustion:                {config.processing.post_combustion}")
    print(f"Time resolution:                {config.processing.time_resolution}")
    print(f"Start date:                     {config.time_range.start_date.strftime(DATE_FORMAT)}")
    print(f"End date:                       {config.time_range.end_date.strftime(DATE_FORMAT)}")

    engine_dict = load_engine_data(config.input_data.engine_data_dir)
    print(f"Loaded {len(engine_dict)} engine data tables.")

    launch_list_df = load_launch_list(config.input_data.launch_list)
    print(f"Loaded launch list with {len(launch_list_df)} rows.")

    propellant_use_profile_dict = load_propellant_use_profiles(
        propellant_use_profiles_dir=config.input_data.propellant_use_profiles_dir,
        launch_list_df=launch_list_df,
        engine_dict=engine_dict,
        altitude_resolution_km=config.grid.altitude_resolution_km,
        latitude_resolution_deg=config.grid.latitude_resolution_deg,
        longitude_resolution_deg=config.grid.longitude_resolution_deg,
    )
    print(f"Loaded and resized {len(propellant_use_profile_dict)} propellant use profiles.")
    primary_exhaust_dict = calculate_primary_exhaust(
        propellant_use_profile_dict=propellant_use_profile_dict,
        engine_dict=engine_dict,
    )
    print(f"Calculated primary emissions for {len(primary_exhaust_dict)} launch tags.")

    if config.output.export_individual_profiles:
        export_primary_exhaust_profiles(
            primary_exhaust_dict=primary_exhaust_dict,
            output_dir=config.output.directory,
            overwrite=config.output.overwrite,
        )
        print(
            f"Exported {len(primary_exhaust_dict)} individual primary emission profiles "
            f"to {config.output.directory / 'primary_exhaust_profiles'}"
        )

    final_emissions_dict = calculate_final_emissions(
        primary_exhaust_dict=primary_exhaust_dict,
        post_combustion_method=config.processing.post_combustion,
    )
    print(f"Calculated final emissions for {len(final_emissions_dict)} launch tags.")

    if config.output.export_individual_profiles:
        export_final_emission_profiles(
            final_emissions_dict=final_emissions_dict,
            output_dir=config.output.directory,
            overwrite=config.output.overwrite,
        )
        print(
            f"Exported {len(final_emissions_dict)} individual final emission profiles "
            f"to {config.output.directory / 'final_emission_profiles'}"
        )

    timestep_inventory_dict = build_timestep_inventory_dataframes(
        launch_list_df=launch_list_df,
        final_emissions_dict=final_emissions_dict,
        start_date=config.time_range.start_date,
        end_date=config.time_range.end_date,
        time_resolution=config.processing.time_resolution,
        domain=config.domain,
    )
    print(
        f"Built combined timestep emission profiles for "
        f"{len(timestep_inventory_dict)} timesteps."
    )

    timestep_inventory_dict = transform_timestep_inventory_dict(
        timestep_inventory_dict=timestep_inventory_dict,
        inventory_emission_unit=config.output.inventory_emission_unit,
        inventory_vertical_coordinate=config.output.inventory_vertical_coordinate,
        species_db=SPECIES_DB,
    )
    print(
        f"Transformed timestep inventories to emission unit "
        f"'{config.output.inventory_emission_unit}' and vertical coordinate "
        f"'{config.output.inventory_vertical_coordinate}'."
    )

    inventory_netcdf_paths = export_inventory_netcdfs(
        timestep_inventory_dict=timestep_inventory_dict,
        config=config,
    )

    print(f"Exported {len(inventory_netcdf_paths)} NetCDF inventory file(s):")
    for inventory_path in inventory_netcdf_paths:
        print(f"  - {inventory_path}")

    print("\nCustom Inventory Generation finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify IGEL NetCDF emission inventories."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file for inventory generation.",
    )
    group.add_argument(
        "--verify",
        type=Path,
        help="Path to a local NetCDF inventory file to verify.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        raise SystemExit(0)

    args = parser.parse_args(argv)
    if not args.config and not args.verify:
        parser.print_help()
        raise SystemExit(0)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)

        if args.verify:
            print("NetCDF verification started...")
            verify_inventory_netcdf(args.verify)
            return 0

        config = load_config(args.config)
        run_inventory_generator(config)
        return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except VerificationError as exc:
        print(f"Verification error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
