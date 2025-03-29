.. _api_api_cli:

Command-Line Interface
====================

.. automodule:: mdr.api.cli
   :members:
   :undoc-members:
   :show-inheritance:

Overview
-------

The ``cli`` module provides a command-line interface (CLI) for common operations
in the MDR package. This interface allows users to perform data refinement,
validation, and conversion from the terminal without writing Python code.

Commands
-------

The MDR CLI provides the following commands:

- **refine**: Refine data by removing outliers, imputing missing values, and smoothing
- **validate**: Validate data quality using configurable checks
- **convert**: Convert data between different file formats
- **transform**: Apply transformations to data
- **info**: Display information about data files

CLI Usage
--------

Basic command structure:

.. code-block:: bash

    mdr <command> [options] <input_file> [output_file]

Refining Data
~~~~~~~~~~~

.. code-block:: bash

    # Refine a CSV file with default settings
    mdr refine input.csv output.csv
    
    # Refine with custom parameters
    mdr refine input.csv output.csv --smoothing-factor 0.3 --outlier-threshold 2.5 --imputation-method linear

Validating Data
~~~~~~~~~~~~~

.. code-block:: bash

    # Validate a CSV file
    mdr validate input.csv --output-file validation_results.json
    
    # Validate with specific checks
    mdr validate input.csv --checks range missing outliers --output-file validation_results.json

Converting Formats
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Convert between file formats
    mdr convert input.csv output.parquet
    
    # Specify format explicitly
    mdr convert input.data output.csv --input-format csv --output-format parquet

Transforming Data
~~~~~~~~~~~~~~

.. code-block:: bash

    # Apply transformations to data
    mdr transform input.csv output.csv --normalize minmax --scale 2.0 1.0

Getting Information
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Display information about a data file
    mdr info input.csv

Command Reference
---------------

.. autofunction:: mdr.api.cli.main
.. autofunction:: mdr.api.cli.refine_command
.. autofunction:: mdr.api.cli.validate_command
.. autofunction:: mdr.api.cli.convert_command
.. autofunction:: mdr.api.cli.transform_command
.. autofunction:: mdr.api.cli.info_command

Advanced Usage
------------

The CLI can also be used in shell scripts or batch processing:

.. code-block:: bash

    #!/bin/bash
    
    # Process multiple files
    for file in data/*.csv; do
        basename=$(basename "$file" .csv)
        echo "Processing $file..."
        
        # Validate
        mdr validate "$file" --output-file "results/${basename}_validation.json"
        
        # Refine
        mdr refine "$file" "refined/${basename}.csv" --smoothing-factor 0.2
        
        # Transform
        mdr transform "refined/${basename}.csv" "transformed/${basename}.parquet" --normalize minmax
    done
