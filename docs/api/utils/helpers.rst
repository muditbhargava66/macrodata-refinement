.. _api_utils_helpers:

Helper Utilities
================

.. automodule:: mdr.utils.helpers
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``helpers`` module provides utility functions that support various operations
across the MDR package. These helper functions include data type conversion,
validation utilities, statistical functions, and more.

Core Functions
--------------

Helper Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: mdr.utils.helpers.moving_average
   :no-index:

Usage Examples
--------------

Moving average example:

.. code-block:: python

    import numpy as np
    from mdr.utils.helpers import moving_average
    
    # Create sample data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Calculate moving average with window size 3
    ma_data = moving_average(data, window_size=3)
    
    print(f"Original data: {data}")
    print(f"Moving average: {ma_data}")
