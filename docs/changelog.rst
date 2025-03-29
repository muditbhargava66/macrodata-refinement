.. _changelog:

Changelog
=========

All notable changes to the Macrodata Refinement (MDR) project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~

- Future development items will be listed here

Fixed
~~~~~

- Fixed documentation build warnings by resolving duplicate object descriptions
- Fixed references to non-existent functions in documentation
- Removed unsupported theme options in documentation configuration

[0.1.0] - 2025-03-25
--------------------

Initial release of Macrodata Refinement.

Added
~~~~~

- New visualization capabilities
- Enhanced data transformation features
- Documentation system with Sphinx and Read the Docs
- Core refinement functionality:
  - Outlier removal
  - Missing value imputation
  - Data smoothing
- Data validation tools
- Transformation capabilities
- Extended visualization functions:
  - Time series plots
  - Histograms and box plots
  - Heatmaps and correlation matrices
  - Validation and refinement comparisons
- I/O support for multiple file formats:
  - CSV
  - JSON
  - Excel
  - Parquet
  - HDF5
- Extended CLI capabilities
- REST API
- Full documentation
- Unit and integration tests
- GitHub workflow configurations
- Issue and pull request templates
- Examples and tutorials

Changed
~~~~~~~

- Improved test reliability with deterministic approaches
- Enhanced error handling in refinement functions
- Enhanced type checking
- Optimized performance for large datasets

Fixed
~~~~~

- Modified ``plot_correlation_matrix`` to accept pandas DataFrames for better compatibility with data science workflows
- Fixed datetime parsing warnings with robust format detection
- Bug in outlier detection for very small datasets
- Issues with CLI argument parsing
- Interpolation method in helpers module for spline interpolation
- Validation test cases for proper outlier detection
