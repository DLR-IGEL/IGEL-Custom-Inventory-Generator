# Contributing

Thank you for contributing to the IGEL Custom Emission Inventory Generator.

## Development setup

```bash
conda env create -f environment.yml
conda activate IGEL-custom-inventory-generator
python -m pip install --no-deps -e ".[test]"
```

Alternatively, use a Python 3.11 or 3.12 virtual environment:

```bash
python -m pip install -e ".[test]"
```

## Before submitting a change

Run:

```bash
python -m build
pytest -q
IGEL_custom_inventory_generator --help
```

Changes to scientific calculations must include:

- a concise description of the scientific rationale;
- a reference or derivation for new equations or constants;
- regression tests showing the intended numerical change;
- a note in `CHANGELOG.md`;
- confirmation of whether existing IGEL inventories remain reproducible.

Changes that intentionally alter published numerical output should update the
software version and must not be mixed with documentation-only changes.

## Pull requests

Keep pull requests focused. Describe:

- what changed and why;
- whether inventory values or file structure change;
- the tests and reference data used;
- any compatibility or reproducibility implications.

Do not commit research input data unless its distribution and licensing have
been approved. Use small synthetic fixtures for tests.

## Reporting issues

Open a GitHub issue with the software version or commit, operating system,
Python version, configuration excerpt, and the smallest reproducible example
that does not disclose restricted research data.
