from __future__ import annotations

from pathlib import Path

import xarray as xr
import yaml

from IGEL_custom_inventory_generator.IGEL_custom_inventory_generator import main


def test_minimal_inventory_creation(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[1]
    example_config_path = repository_root / "config" / "example_config.yaml"
    raw_config = yaml.safe_load(example_config_path.read_text(encoding="utf-8"))
    raw_config["output"]["directory"] = str(tmp_path / "output")
    raw_config["output"]["export_individual_profiles"] = False

    test_config_path = tmp_path / "config.yaml"
    test_config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False),
        encoding="utf-8",
    )

    assert main(["--config", str(test_config_path)]) == 0

    inventory_path = (
        tmp_path
        / "output"
        / "inventory_netcdf"
        / "minimal_example_inventory.nc"
    )
    assert inventory_path.is_file()

    with xr.open_dataset(inventory_path) as dataset:
        assert dataset.sizes["time"] == 1
        assert dataset.sizes["lat"] == 181
        assert dataset.sizes["lon"] == 361
        assert float(dataset["species_mass_CO2"].sum()) == 1.0
