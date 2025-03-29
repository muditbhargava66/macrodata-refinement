.. _api_io_formats:

Data Formats
============

.. automodule:: mdr.io.formats
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``formats`` module provides utilities for working with different data formats,
including format detection, conversion, and validation. It supports the core
I/O functionality of the MDR package.

Core Functions
--------------

.. autofunction:: mdr.io.formats.detect_format
   :no-index:
.. autofunction:: mdr.io.formats.convert_format
   :no-index:
.. autofunction:: mdr.io.formats.validate_format
   :no-index:

Supported Formats
-----------------

The module supports the following data formats:

- **CSV**: Comma-separated values files
- **JSON**: JavaScript Object Notation files
- **Excel**: Microsoft Excel workbooks (.xlsx, .xls)
- **Parquet**: Apache Parquet columnar storage files
- **HDF5**: Hierarchical Data Format version 5 files



Usage Examples
--------------

Format detection:

.. code-block:: python

    from mdr.io.formats import detect_format
    
    # Detect the format of a file
    format_info = detect_format("path/to/data.csv")
    print(f"Format: {format_info['format']}")
    print(f"Structure: {format_info['structure']}")

Format conversion:

.. code-block:: python

    from mdr.io.formats import convert_format
    
    # Convert a file from CSV to Parquet
    convert_format(
        "path/to/input.csv",
        "path/to/output.parquet",
        source_format="csv",
        target_format="parquet"
    )
