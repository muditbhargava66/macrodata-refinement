.. _api_core_transformation:

Transformation
==============

.. automodule:: mdr.core.transformation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``transformation`` module provides functions for transforming data through various
statistical and mathematical operations. These transformations help prepare data
for analysis or visualization by standardizing scales, normalizing distributions,
or applying mathematical transformations.

Core Functions
--------------

.. autofunction:: mdr.core.transformation.transform_data
   :no-index:

Available Transformations
-------------------------

The module supports the following transformation types:

- **normalize**: Scale data to a standard range (e.g., 0-1, -1 to 1)
- **scale**: Apply linear scaling with a factor and offset
- **log**: Apply logarithmic transformation
- **power**: Apply power transformation
- **boxcox**: Apply Box-Cox transformation for normalizing distributions
- **winsorize**: Limit extreme values to reduce the effect of outliers

Usage Examples
--------------

Basic transformation of data:

.. code-block:: python

    import numpy as np
    from mdr.core.transformation import transform_data
    
    # Create sample data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Define transformation pipeline
    transformations = [
        {"type": "normalize", "method": "minmax"},
        {"type": "scale", "factor": 2.0, "offset": 1.0}
    ]
    
    # Apply transformations
    transformed_data = transform_data(data, transformations)
    
    print("Original data:", data)
    print("Transformed data:", transformed_data)

Applying multiple transformations to multiple variables:

.. code-block:: python

    import numpy as np
    from mdr.core.transformation import transform_data
    
    # Create a dictionary of data variables
    data_dict = {
        "temperature": np.array([20.5, 21.3, 22.1, 21.7, 23.0]),
        "pressure": np.array([101.3, 101.4, 101.5, 101.2, 101.1])
    }
    
    # Define transformation pipeline
    transformations = [
        {"type": "normalize", "method": "zscore"},
        {"type": "scale", "factor": 1.0, "offset": 0.0}
    ]
    
    # Apply transformations to each variable
    transformed_dict = {}
    for var_name, values in data_dict.items():
        transformed_dict[var_name] = transform_data(values, transformations)
