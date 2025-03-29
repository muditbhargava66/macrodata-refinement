.. _api_core_validation:

Validation
==========

.. automodule:: mdr.core.validation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``validation`` module provides functions and classes for validating data quality through
various checks and tests. It helps identify issues such as:

- Missing values
- Statistical outliers
- Values outside expected ranges
- Inconsistencies between related variables

Core Components
---------------

ValidationResult
~~~~~~~~~~~~~~~~

.. autoclass:: mdr.core.validation.ValidationResult
   :members:
   :no-index:

The ``ValidationResult`` class encapsulates the results of validation checks, including 
whether the data passed validation, any error messages, and relevant statistics.

Data Validation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mdr.core.validation.validate_data
   :no-index:
.. autofunction:: mdr.core.validation.check_outliers
   :no-index:

Usage Examples
--------------

Basic validation of data:

.. code-block:: python

    import numpy as np
    from mdr.core.validation import validate_data
    
    # Create a dictionary of data variables
    data_dict = {
        "temperature": np.array([20.5, 21.3, np.nan, 21.7, 45.0]),
        "pressure": np.array([101.3, 101.4, 80.0, np.nan, np.nan])
    }
    
    # Validate the data
    validation_results = validate_data(
        data_dict,
        checks=["range", "missing", "outliers"],
        params={
            "range": {
                "min_value": 0.0,
                "max_value": 100.0
            },
            "missing": {
                "threshold": 0.1  # Allow up to 10% missing values
            },
            "outliers": {
                "threshold": 2.5,  # Z-score threshold for outliers
                "method": "zscore"
            }
        }
    )
    
    # Print validation results
    for var_name, result in validation_results.items():
        print(f"{var_name} validation: {'Passed' if result.is_valid else 'Failed'}")
        if not result.is_valid:
            for msg in result.error_messages:
                print(f"  - {msg}")
