from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr

from IGEL_custom_inventory_generator import inventory_verify


def test_production_grid_chunk_shape_is_memory_bounded() -> None:
    shape = (12, 81, 181, 361)
    chunk_shape = inventory_verify._bounded_chunk_shape(shape)

    assert chunk_shape[0] == 1
    assert (
        math.prod(chunk_shape) * np.dtype(np.float64).itemsize
        <= inventory_verify.VERIFICATION_MAX_CHUNK_BYTES
    )


def test_species_scan_reads_bounded_chunks_and_preserves_totals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inventory_path = tmp_path / "inventory.nc"
    values = np.arange(1, 121, dtype=np.float64).reshape(2, 3, 4, 5)
    dataset = xr.Dataset(
        data_vars={
            "species_mass_CO2": (
                ("time", "lev", "lat", "lon"),
                values,
                {"units": "kg", "long_name": "carbon dioxide"},
            ),
            "time_bnds": (
                ("time", "nv"),
                np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            ),
        },
        coords={
            "time": np.array([0.5, 1.5], dtype=np.float64),
            "lev": np.arange(3, dtype=np.float64),
            "lat": np.arange(4, dtype=np.float64),
            "lon": np.arange(5, dtype=np.float64),
        },
        attrs={"inventory_emission_unit": "mass"},
    )
    dataset.to_netcdf(
        inventory_path,
        encoding={"species_mass_CO2": {"chunksizes": (1, 3, 4, 5)}},
    )

    monkeypatch.setattr(inventory_verify, "VERIFICATION_MAX_CHUNK_BYTES", 64)
    observed_chunk_bytes: list[int] = []
    original_reader = inventory_verify._read_netcdf_chunk

    def recording_reader(variable, chunk_slices):
        chunk = original_reader(variable, chunk_slices)
        observed_chunk_bytes.append(chunk.nbytes)
        return chunk

    monkeypatch.setattr(inventory_verify, "_read_netcdf_chunk", recording_reader)

    with xr.open_dataset(
        inventory_path,
        decode_times=False,
        cache=False,
    ) as ds_raw, xr.open_dataset(
        inventory_path,
        decode_times=False,
        cache=False,
    ) as ds:
        issues, _lines, totals, per_timestep = inventory_verify._scan_species_data(
            ds,
            ds_raw,
            inventory_path,
        )

    assert issues == []
    assert observed_chunk_bytes
    assert max(observed_chunk_bytes) <= 64
    np.testing.assert_allclose(
        per_timestep["CO2"],
        values.sum(axis=(1, 2, 3)),
    )
    assert totals["CO2"] == float(values.sum())
