.. _api_io_readers:

Data Readers
===========

.. automodule:: mdr.io.readers
   :members:
   :undoc-members:
   :show-inheritance:

Overview
-------

The ``readers`` module provides functions for reading data from various file formats
into numpy arrays or dictionaries of arrays. These functions handle data loading,
parsing, and initial preprocessing to prepare data for the MDR refinement pipeline.

Supported File Formats
--------------------

The module supports reading data from the following formats:

- **CSV**: Comma-separated values files
- **JSON**: JavaScript Object Notation files
- **Excel**: Microsoft Excel workbooks (.xlsx, .xls)
- **Parquet**: Apache Parquet columnar storage files
- **HDF5**: Hierarchical Data Format version 5 files

Core Functions
------------

.. autofunction:: mdr.io.readers.read_csv
.. autofunction:: mdr.io.readers.read_json
.. autofunction:: mdr.io.readers.read_excel
.. autofunction:: mdr.io.readers.read_parquet
.. autofunction:: mdr.io.readers.read_hdf5

Usage Examples
------------

Reading from a CSV file:

.. code-block:: python

    from mdr.io.readers import read_csv
    
    # Read data from a CSV file
    data_dict = read_csv("path/to/data.csv")
    
    # Print the variable names and shapes
    for var_name, values in data_dict.items():
        print(f"{var_name}: {values.shape}")

Reading from an Excel file with multiple sheets:

.. code-block:: python

    from mdr.io.readers import read_excel
    
    # Read data from specific sheets
    data_dict = read_excel(
        "path/to/data.xlsx",
        sheets=["Temperature", "Pressure"],
        column_mapping={
            "Temperature": {"Temp (C)": "temperature"},
            "Pressure": {"Press (hPa)": "pressure"}
        }
    )
