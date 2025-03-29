.. _changelog:

Changelog
========

All notable changes to the Macrodata Refinement (MDR) project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
-----------

Added
~~~~~

- Documentation system with Sphinx and Read the Docs
- Additional visualization options
- Extended CLI capabilities

Changed
~~~~~~~

- Improved error handling in refinement functions
- Enhanced type checking
- Optimized performance for large datasets

Fixed
~~~~~

- Bug in outlier detection for very small datasets
- Issue with certain file formats in I/O operations

[0.1.0] - 2025-03-15
------------------

Initial release of Macrodata Refinement.

Added
~~~~~

- Core refinement functionality:
  - Outlier removal
  - Missing value imputation
  - Data smoothing
- Data validation tools
- Transformation capabilities
- Basic visualization functions
- I/O support for multiple file formats:
  - CSV
  - JSON
  - Excel
  - Parquet
  - HDF5
- Command-line interface
- REST API
- Basic documentation
- Unit and integration tests
