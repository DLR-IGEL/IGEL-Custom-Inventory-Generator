# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Publication metadata, citation information, contribution guidance, and CI.
- A minimal, non-scientific example input dataset and runnable configuration.
- Regression tests for launch counting, primary emissions, CSVEM calculations,
  the preserved grid convention, package imports, and end-to-end NetCDF export.

### Changed

- Package-internal imports now work from an installed wheel.
- Packaging metadata declares runtime and test dependencies.
- Documentation now matches the implemented input schemas and command-line
  interface.
- NetCDF verification scans species data in bounded chunks instead of loading
  and retaining complete four-dimensional species arrays.

### Removed

- Generated Python bytecode from version control.
- Unreachable duplicate code that could never affect inventory calculations.

## [1.0.0]

- Initial public version of the custom inventory generator.
- Launch-list entries are applied once each.

[Unreleased]: https://github.com/DLR-IGEL/IGEL-Custom-Inventory-Generator/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/DLR-IGEL/IGEL-Custom-Inventory-Generator/releases/tag/v1.0.0
