from __future__ import annotations

from pathlib import Path

import pytest

import IGEL_custom_inventory_generator
from IGEL_custom_inventory_generator.IGEL_custom_inventory_generator import (
    build_arg_parser,
)
from IGEL_custom_inventory_generator.inventory_config import load_config
from IGEL_custom_inventory_generator.inventory_netcdf import (
    _build_regular_midpoint_axis,
)


def test_package_version_and_cli_parser_are_importable() -> None:
    assert IGEL_custom_inventory_generator.__version__ == "1.0.0"
    assert build_arg_parser().prog


def test_example_configuration_is_loadable() -> None:
    config_path = Path(__file__).parents[1] / "config" / "example_config.yaml"
    config = load_config(config_path)

    assert config.output.run_name == "minimal_example"
    assert config.processing.post_combustion is None


def test_legacy_global_grid_convention_is_regression_locked() -> None:
    lat = _build_regular_midpoint_axis(-90.0, 90.0, 1.0)
    lon = _build_regular_midpoint_axis(-180.0, 180.0, 1.0)

    assert len(lat) == 181
    assert (lat[0], lat[-1]) == pytest.approx((-90.0, 90.0))
    assert len(lon) == 361
    assert (lon[0], lon[-1]) == pytest.approx((-180.0, 180.0))
