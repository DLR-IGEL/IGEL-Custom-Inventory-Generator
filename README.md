# IGEL Custom Emission Inventory Generator

## Overview

This tool generates **global emission inventories from rocket launches** based on launch-specific propellant use profiles and engine emission indices. It processes launch data, computes primary and post-combustion emissions, aggregates them temporally and spatially, and exports the results as **CF-compliant NetCDF files**.

The workflow is designed for reproducibility and supports configurable:

* spatial resolution
* temporal resolution
* emission units
* vertical coordinate systems
* domain boundaries
* post combustion

\---

## Features

* Calculation of **primary emissions** from engine data
* Optional **post-combustion modeling (CSVEM)**
* Temporal aggregation into configurable timesteps
* Spatial aggregation on a configurable global grid
* Conversion to multiple emission formats:

  * mass (kg)
  * molecules
  * molecule rate (1/s)
  * molecule rate per volume/flux (1/(m³·s))
* Vertical coordinate options:

  * altitude
  * pressure (USSA1976)
* Export to:

  * single NetCDF file
  * one NetCDF per timestep
* CF-compliant metadata and dimensions
* Optional export of intermediate emission profiles

\---

## Installation

### Requirements

* Python ≥ 3.10
* Recommended: Conda environment

### Install dependencies

This project uses a fully specified Conda environment to ensure reproducibility.

### Create predefined environment (recommended)

```bash
conda env create -f environment.yml
conda activate IGEL-custom-inventory-generator
```

### Alternative: Create own environment

Using pip:

```bash
pip install numpy pandas xarray netCDF4 pyyaml tqdm ussa1976 scipy python-dateutil
```

Or using conda:

```bash
conda create -n IGEL-custom-inventory-generator python=3.12 numpy pandas xarray netcdf4 pyyaml tqdm scipy math python-dateutil
conda activate IGEL-custom-inventory-generator
pip install ussa1976
```

\---

## Usage

Run the generator using a configuration file, or run verification on an existing local NetCDF inventory:

```bash
python your/path/to/IGEL\_custom\_inventory\_generator.py --config your/path/to/config.yaml
```

\---

## Input Data

### 1\. Launch List (CSV)

Required columns:

* `Launch\_Tag`
* `Launch\_Date` (format: `YYYY Mon DD`, e.g. `2024 Jan 3`)

Each row represents one launch and is counted once.

\---

### 2\. Propellant Use Profiles

One CSV per launch\_tag:

```
<Launch\_Tag>\_propellant\_use\_profile.csv
```

Contains spatial bins and engine-specific emissions:

* `ALTITUDE\_MIN`, `ALTITUDE\_MAX`
* `LATITUDE\_MIN`, `LATITUDE\_MAX`
* `LONGITUDE\_MIN`, `LONGITUDE\_MAX`
* `species\_mass\_<engine>\_sum`

\---

### 3\. Engine Data

One CSV per engine:

```
<engine\_name>\_primary\_exhaust\_indices.csv
```

Required columns:

* `Species`
* `Absolute mass fractions`

\---

## Configuration

A commented config.yaml is provided with the distributed code. It explains each input parameter.
Example for IGEL 2024 Inventory `config.yaml`:

```yaml

metadata:
  config\_version: "1.0"                       
  inventory\_name: "custom\_inventory"          
  institution: "DLR-SRT"                      
  creator\_name: "Moritz Herberhold"          
  creator\_email: "moritz.herberhold@dlr.de"   
  project: "S3D-BETTER"                       

output:
  directory: ./output
  run\_name: example\_run
  overwrite: true
  export\_individual\_profiles: true
  inventory\_emission\_unit: molecule\_rate\_per\_volume
  inventory\_vertical\_coordinate: pressure
  inventory\_netcdf\_mode: single\_file

processing:
  post\_combustion: CSVEM
  time\_resolution: 1m

domain:
  altitude\_min\_km: 0
  altitude\_max\_km: 80
  latitude\_min\_deg: -90
  latitude\_max\_deg: 90
  longitude\_min\_deg: -180
  longitude\_max\_deg: 180
  inclusive: true
  emissions\_above\_inventory: discard
grid:
  altitude\_resolution\_km: 1
  latitude\_resolution\_deg: 1
  longitude\_resolution\_deg: 1

time\_range:
  start\_date: 01-01-2024
  end\_date: 31-12-2024
```

\---

## Workflow

1. Load input data
2. Compute primary emissions
3. Apply post-combustion (optional)
4. Aggregate emissions per timestep
5. Enforce domain boundaries
6. Collapse bins to midpoints
7. Convert emission units and vertical coordinates
8. Export NetCDF inventory

\---

## Output

### NetCDF Inventory

* Dimensions: `(time, lev, lat, lon)`
* Global coverage:

  * latitude: -90 to 90
  * longitude: -180 to 180
* Time axis:

  * CF-compliant
  * includes bounds
* Compression enabled

### Variables

Each species:

```
species\_<type>\_<species>

Example:

```

species\_molecule\_rate\_per\_volume\_CO2

```

### Metadata

Includes:

\* emission units
\* coordinate units
\* run name
\* inventory configuration

---

## Performance Notes

\* Runtime scales with:

  \* number of timesteps
  \* grid resolution
  \* number of launches
\* NetCDF writing can take \*\*several minutes\*\* for large datasets
\* Progress bars are shown during data processing

---

## Testing

Run tests using:

```bash
pytest -q
```

\---

## Reproducibility

To ensure reproducibility:

* use fixed environment versions
* store config files with outputs
* track input datasets

\---

## Limitations

* Memory usage increases with grid size
* NetCDF writing is currently single-threaded
* Vertical pressure conversion limited to USSA1976 range

\---

## Citation

If you use this tool, please cite:

```
\[Your publication / data descriptor here]
```

\---

## License

Specify your license here (e.g. MIT, BSD-3-Clause).

\---

## Contact

Maintainer: Moritz Herberhold DLR-SRT (moritz.herberhold@dlr.de)
Project context: Global Emission Inventory to determine impact of rocket launches on our environment created by DLR-SRT in the S3D-BETTER project



## Verification Mode

The CLI also supports verification of an existing NetCDF inventory:

```bash
python IGEL\_custom\_inventory\_generator.py --verify /path/to/inventory.nc

```

Verification mode:

* prints all global metadata
* performs structural validation
* performs data sanity checks
* performs metadata-vs-data cross-checks
* converts every species back to total mass in kg regardless of stored inventory unit
* prints total mass per species
* prints per-timestep mass totals per species
* optionally compares two inventory files
* writes a text log named:

```
<inventoryfilename>\_verification\_log.txt
```

For local files, the log is written next to the verified inventory file. For HTTP(S) URLs, the log is written to the current working directory.

## Help

Both of the following show the command line help:

```bash
python IGEL\_custom\_inventory\_generator.py
python IGEL\_custom\_inventory\_generator.py --help
```
