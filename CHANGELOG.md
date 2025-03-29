# Changelog

All notable changes to the Macrodata Refinement (MDR) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core functionality
- Robust type checking and validation for floating-point values

## [0.1.0] - 2025-03-25

### Added
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
- Comprehensive documentation
- Examples and tutorials
- Development tools and scripts

### Fixed
- None (initial release)

---

Note: This changelog will be updated with each release to document new features, bug fixes, and breaking changes.