.. _api_utils_helpers:

Helper Utilities
==============

.. automodule:: mdr.utils.helpers
   :members:
   :undoc-members:
   :show-inheritance:

Overview
-------

The ``helpers`` module provides utility functions that support various operations
across the MDR package. These helper functions include data type conversion,
validation utilities, statistical functions, and more.

Core Functions
------------

Data Type Helpers
~~~~~~~~~~~~~~~

.. autofunction:: mdr.utils.helpers.ensure_numpy_array
.. autofunction:: mdr.utils.helpers.convert_to_numeric
.. autofunction:: mdr.utils.helpers.is_numeric_dtype
.. autofunction:: mdr.utils.helpers.is_time_dtype

Validation Helpers
~~~~~~~~~~~~~~~~

.. autofunction:: mdr.utils.helpers.validate_config
.. autofunction:: mdr.utils.helpers.validate_input
.. autofunction:: mdr.utils.helpers.check_dependencies

Statistical Helpers
~~~~~~~~~~~~~~~~~

.. autofunction:: mdr.utils.helpers.calculate_statistics
.. autofunction:: mdr.utils.helpers.moving_average
.. autofunction:: mdr.utils.helpers.calculate_z_scores

File and Path Helpers
~~~~~~~~~~~~~~~~~~~

.. autofunction:: mdr.utils.helpers.get_file_extension
.. autofunction:: mdr.utils.helpers.ensure_directory
.. autofunction:: mdr.utils.helpers.generate_timestamp

Usage Examples
------------

Converting data to numpy arrays:

.. code-block:: python

    import pandas as pd
    from mdr.utils.helpers import ensure_numpy_array
    
    # Create a pandas Series
    series = pd.Series([1, 2, 3, 4, 5])
    
    # Convert to numpy array
    array = ensure_numpy_array(series)
    
    print(f"Type: {type(array)}")
    print(f"Values: {array}")

Calculating statistics:

.. code-block:: python

    import numpy as np
    from mdr.utils.helpers import calculate_statistics
    
    # Create sample data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Calculate statistics
    stats = calculate_statistics(data)
    
    print(f"Mean: {stats['mean']}")
    print(f"Median: {stats['median']}")
    print(f"Std Dev: {stats['std']}")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
