# IGEL Custom Inventory Generator

The IGEL Custom Inventory Generator creates gridded rocket-launch
emission inventories from launch-specific propellant-use profiles and engine
emission indices. It aggregates emissions in space and time and writes
CF-1.8-labelled NetCDF files for use in atmospheric research.

This repository contains the software used in the context of the DLR Inventory
of Global Emissions by Launchers (IGEL) 2024. The IGEL 2024 input dataset is a
separate research artifact; users must supply the launch list, propellant-use
profiles, and engine data described below. The accompanying IGEL 2024 dataset 
can be found under https://doi.org/10.5281/zenodo.21667787 and a visualization 
of the dataset is provided under https://dlr-igel.github.io/IGEL-2024/

## Features

- Primary exhaust calculation from engine-specific absolute mass fractions
- Optional Commercial Space Vehicle Emissions Modeling (CSVEM)
  post-combustion treatment
- Configurable daily, weekly, or monthly temporal aggregation
- Configurable horizontal and vertical grid resolution
- Mass, molecule-count, molecule-rate, and volumetric molecule-rate output
- Altitude or USSA1976 pressure vertical coordinates
- Single-file or per-timestep NetCDF export
- Optional export of intermediate launch profiles
- Local NetCDF verification with structural, sanity, metadata, and mass checks

## Requirements

- Python 3.11 or 3.12
- A platform supported by the scientific Python dependencies
- Sufficient memory for the selected global grid and number of timesteps

## Installation

### Conda environment

The pinned environment is the recommended route for reproducing a published
inventory:

```bash
conda env create -f environment.yml
conda activate IGEL-custom-inventory-generator
python -m pip install --no-deps -e .
```

### pip

Install the package and its declared runtime dependencies:

```bash
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
```

## Command-line usage

After installation:

```bash
IGEL_custom_inventory_generator --config config/example_config.yaml
```

The equivalent module invocation is:

```bash
python -m IGEL_custom_inventory_generator --config config/example_config.yaml
```

Display help:

```bash
IGEL_custom_inventory_generator --help
```

## Configuration

The commented [`config/example_config.yaml`](config/example_config.yaml)
contains every required setting. Paths are interpreted relative to the current
working directory, so run the example from the repository root or replace the
paths with absolute paths.

The main sections are:

- `metadata`: dataset and creator metadata written to NetCDF
- `input_data`: launch list, propellant-use profiles, and engine data
- `output`: output location, format, units, and overwrite behavior
- `processing`: post-combustion method and time resolution
- `domain`: spatial inclusion boundaries
- `grid`: target spatial resolution
- `time_range`: inclusive inventory date range

## Input data

### Launch list

The launch-list CSV requires:

| Column | Meaning |
|---|---|
| `Launch_Tag` | Identifier used to locate the corresponding propellant-use profile |
| `Launch_Date` | Launch date in `%Y %b %d` format, for example `2024 Jan 03` |

Each row represents one launch and contributes one copy of its referenced
profile.

### Propellant-use profiles

For each unique launch tag, provide:

```text
<Launch_Tag>_propellant_use_profile.csv
```

Required spatial and total columns:

- `ALTITUDE_MIN`, `ALTITUDE_MAX`
- `LATITUDE_MIN`, `LATITUDE_MAX`
- `LONGITUDE_MIN`, `LONGITUDE_MAX`

At least one engine-specific column is required:

```text
species_mass_<engine_name>
```

A trailing `_sum` is accepted and removed while loading.

### Engine data

For each referenced engine, provide:

```text
<engine_name>_primary_exhaust_indices.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `Species` | Species identifier used by the generator |
| `Absolute_Mass_Fraction` | Species mass per unit engine propellant mass |

Species names must be available in the internal species database when molecule
conversion is requested.

## Processing methodology

The generator performs these steps:

1. Validate the configuration and input schemas.
2. Re-bin propellant-use profiles to the requested grid.
3. Calculate primary species emissions from engine mass fractions.
4. Apply optional CSVEM post-combustion equations.
5. Select launches for each timestep and combine their profiles.
6. Apply configured domain handling.
7. Convert spatial bins to midpoint coordinates.
8. Convert emission and vertical-coordinate units.
9. Construct and write the NetCDF inventory.

The optional post-combustion implementation follows the altitude-dependent
formulations described by Barker et al. (2024), which in turn cite the National
Academies CSVEM report. These formulations carry substantial uncertainty,
particularly above 40 km. Users should account for that uncertainty when
interpreting post-combustion inventories.

References:

- Barker, C. R. et al. (2024), “Global 3D rocket launch and re-entry air
  pollutant and CO2 emissions at the onset of the megaconstellation era,”
  *Scientific Data*, 11, 1079.
  <https://doi.org/10.1038/s41597-024-03910-z>
- National Academies of Sciences, Engineering, and Medicine (2021),
  *Commercial Space Vehicle Emissions Modeling*.
  <https://doi.org/10.17226/26142>
- Herberhold, M., Wilken, J., Callsen, S., Nützel, M., Yamashita, H., Sippel,
  M., & Silvestri, S. (2026). *Inventory of Global Emissions by Launchers for
  2024 (IGEL 2024)* [Data set]. Zenodo. [10.5281/zenodo.21667787]

> Herberhold, M., Wilken, J., Callsen, S., Nützel, M., Yamashita, H., Sippel,
  M., & Silvestri, S. *Inventory of global emissions by launchers for 2024*.
  [Currently in review]

## Output

Generated files are written below:

```text
<output.directory>/
├── final_emission_profiles/       # optional
├── inventory_netcdf/
└── primary_exhaust_profiles/      # optional
```

NetCDF emission variables follow one of these patterns:

```text
species_mass_<species>
species_molecules_<species>
species_molecule_rate_<species>
species_molecule_rate_per_volume_<species>
```

The main dimensions are `(time, lev, lat, lon)`. Coordinate and time bounds,
configuration metadata, creator metadata, and processing choices are included
in the file.

## Verification

Verify a local inventory file:

```bash
IGEL_custom_inventory_generator --verify /path/to/inventory.nc
```

Verification prints a report and writes:

```text
<inventory_name>_verification_log.txt
```

next to the verified file. It checks structure, numeric sanity, metadata
consistency, and species totals converted back to kilograms. Species data is
scanned in bounded chunks so verification also works with production-scale
inventories on memory-constrained systems.


## Testing

Run the regression test suite:

```bash
pytest -q
```

The GitHub Actions workflow builds the package and runs the same suite on the
supported Python versions.

## Limitations

- Runtime and memory use increase with grid resolution, number of species, and
  number of timesteps.
- NetCDF writing is single-threaded.
- Pressure conversion is limited to the USSA1976 range up to 1000 km.
- CSVEM coefficients are empirical and uncertain.


## Citation

Please cite the software using [`CITATION.cff`](CITATION.cff) and cite the IGEL
2024 data publication separately when using the published dataset.

## License

Copyright © 2026 Moritz Herberhold.

The software is distributed under the [MIT License](LICENSE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and review guidance.
Bug reports and focused pull requests are welcome.
