# Changelog

All notable changes to the Macrodata Refinement (MDR) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Future development items will be listed here

### Fixed
- Fixed documentation build warnings by resolving duplicate object descriptions
- Fixed reference to non-existent functions in documentation
- Removed unsupported theme options in documentation configuration

## [0.1.0] - 2025-03-25

### Added
- New visualization capabilities
- Enhanced data transformation features
- Core refinement module with data cleaning capabilities:
  - Detection and handling of outliers
  - Imputation of missing values
  - Data smoothing techniques
  - Configurable refinement pipeline
- Validation module for data quality assessment:
  - Range validation
  - Missing value detection
  - Outlier detection
  - Configurable validation checks
- Transformation module for data conversion:
  - Normalization (min-max, z-score, robust)
  - Scaling and offsetting
  - Logarithmic and power transformations
  - Rolling window functions
- I/O module for file operations:
  - Support for CSV, JSON, Excel, Parquet, and HDF5 formats
  - File format detection and conversion
  - Type inference and casting
- Utility module with helper functions:
  - Logging configuration
  - Data array validation
  - Seasonality detection
  - Dictionary flattening and unflattening
- API interfaces:
  - REST API for web access
  - Command-line interface for script usage
- Visualization module:
  - Time series plots
  - Histograms and box plots
  - Heatmaps and correlation matrices
  - Validation and refinement comparisons
- Full documentation in Sphinx format
- GitHub workflow configurations for CI/CD, releases, and documentation
- Issue and pull request templates
- Comprehensive documentation
- Examples and tutorials
- Development tools and scripts

### Changed
- Improved test reliability by implementing deterministic approaches
- Enhanced error handling for CLI commands

### Fixed
- Modified `plot_correlation_matrix` to accept pandas DataFrames as input, making it more compatible with common data science workflows
- Fixed datetime parsing warnings in formats.py by implementing a more robust approach with explicit format detection
- Corrected CLI argument parsing for positional arguments
- Fixed interpolation method in `helpers.py` to properly handle spline interpolation order parameter
- Improved outlier detection tests to use mathematically guaranteed outlier datasets
- Fixed validation test cases to properly verify outlier detection functionality

---

Note: This changelog will be updated with each release to document new features, bug fixes, and breaking changes.