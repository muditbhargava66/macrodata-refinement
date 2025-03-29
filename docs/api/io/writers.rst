.. _api_io_writers:

Data Writers
============

.. automodule:: mdr.io.writers
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``writers`` module provides functions for writing processed data to various file formats.
These functions handle data serialization, formatting, and output to persistent storage.

Supported File Formats
----------------------

The module supports writing data to the following formats:

- **CSV**: Comma-separated values files
- **JSON**: JavaScript Object Notation files
- **Excel**: Microsoft Excel workbooks (.xlsx)
- **Parquet**: Apache Parquet columnar storage files
- **HDF5**: Hierarchical Data Format version 5 files

Core Functions
--------------

.. autofunction:: mdr.io.writers.write_csv
   :no-index:
.. autofunction:: mdr.io.writers.write_json
   :no-index:
.. autofunction:: mdr.io.writers.write_excel
   :no-index:
.. autofunction:: mdr.io.writers.write_parquet
   :no-index:
.. autofunction:: mdr.io.writers.write_hdf5
   :no-index:

Usage Examples
--------------

Writing to a CSV file:

.. code-block:: python

    import numpy as np
    from mdr.io.writers import write_csv
    
    # Create a dictionary of data variables
    data_dict = {
        "time": np.array([0, 1, 2, 3, 4]),
        "temperature": np.array([20.5, 21.3, 22.1, 21.7, 23.0]),
        "pressure": np.array([101.3, 101.4, 101.5, 101.2, 101.1])
    }
    
    # Write data to a CSV file
    write_csv(data_dict, "path/to/output.csv")

Writing to multiple formats:

.. code-block:: python

    from mdr.io.writers import write_csv, write_json, write_excel
    
    # Write to multiple formats for different use cases
    write_csv(data_dict, "path/to/output.csv")  # For general use
    write_json(data_dict, "path/to/output.json")  # For web applications
    write_excel(data_dict, "path/to/output.xlsx")  # For spreadsheet analysis
