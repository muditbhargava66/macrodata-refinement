.. _api_core_refinement:

Refinement
==========

.. automodule:: mdr.core.refinement
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``refinement`` module provides functions and classes for refining macrodata through various
statistical and analytical methods. Key capabilities include:

- Removing outliers from data
- Imputing missing values
- Smoothing noisy data
- Applying a complete refinement pipeline

Core Components
---------------

RefinementConfig
~~~~~~~~~~~~~~~~

.. autoclass:: mdr.core.refinement.RefinementConfig
   :members:
   :special-members: __post_init__
   :no-index:

The ``RefinementConfig`` class is used to configure the refinement process, specifying parameters
such as smoothing factor, outlier threshold, imputation method, and normalization type.

Data Refinement Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mdr.core.refinement.smooth_data
   :no-index:
.. autofunction:: mdr.core.refinement.remove_outliers
   :no-index:
.. autofunction:: mdr.core.refinement.impute_missing_values
   :no-index:
.. autofunction:: mdr.core.refinement.refine_data
   :no-index:
.. autofunction:: mdr.core.refinement.apply_refinement_pipeline
   :no-index:

Usage Examples
--------------

Basic refinement of a single data array:

.. code-block:: python

    import numpy as np
    from mdr.core.refinement import RefinementConfig, refine_data
    
    # Create sample data with outliers and missing values
    data = np.array([1.0, 2.0, np.nan, 4.0, 100.0])
    
    # Configure refinement
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    
    # Refine the data
    refined_data = refine_data(data, config)
    
    print("Original data:", data)
    print("Refined data:", refined_data)

Refinement of multiple variables:

.. code-block:: python

    import numpy as np
    from mdr.core.refinement import RefinementConfig, apply_refinement_pipeline
    
    # Create a dictionary of data variables
    data_dict = {
        "temperature": np.array([20.5, 21.3, np.nan, 21.7, 45.0]),
        "pressure": np.array([101.3, 101.4, 80.0, np.nan, np.nan])
    }
    
    # Configure refinement
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    
    # Apply refinement to all variables
    refined_dict = apply_refinement_pipeline(data_dict, config)
