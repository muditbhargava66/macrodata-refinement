.. _usage:

Usage Guide
==========

This guide demonstrates how to use the Macrodata Refinement (MDR) package for data processing, validation, and transformation.

Basic Workflow
------------

A typical MDR workflow involves the following steps:

1. Loading data
2. Validating data quality
3. Refining data (removing outliers, imputing missing values, smoothing)
4. Transforming data
5. Visualizing results
6. Saving processed data

Quick Example
-----------

Here's a quick example that demonstrates these steps:

.. code-block:: python

    import numpy as np
    from mdr.core.refinement import RefinementConfig, refine_data
    from mdr.core.validation import validate_data
    from mdr.visualization.plots import plot_refinement_comparison
    import matplotlib.pyplot as plt

    # 1. Create sample data with outliers and missing values
    data = np.array([1.0, 2.0, np.nan, 4.0, 100.0])

    # 2. Validate the data
    validation_result = validate_data(
        {"sample": data},
        checks=["missing", "outliers"],
        params={
            "missing": {"threshold": 0.1},
            "outliers": {"threshold": 2.5}
        }
    )

    # Print validation results
    for var_name, result in validation_result.items():
        if result.is_valid:
            print(f"{var_name} passed validation")
        else:
            print(f"{var_name} failed validation:")
            for msg in result.error_messages:
                print(f"  - {msg}")

    # 3. Configure refinement
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )

    # 4. Refine the data
    refined_data = refine_data(data, config)

    # 5. Visualize the results
    fig, axes = plot_refinement_comparison(data, refined_data)
    plt.tight_layout()
    plt.show()

    print("Original data:", data)
    print("Refined data:", refined_data)

Data Refinement
-------------

Data refinement is the core functionality of MDR. It includes outlier removal, missing value imputation, and data smoothing.

Creating a Refinement Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create a configuration object that specifies how the refinement should be performed:

.. code-block:: python

    from mdr.core.refinement import RefinementConfig

    config = RefinementConfig(
        smoothing_factor=0.2,      # Smoothing intensity (0-1)
        outlier_threshold=2.5,     # Z-score threshold for outliers
        imputation_method="linear", # Method for filling missing values
        normalization_type="minmax" # Type of normalization to apply
    )

Applying Refinement
~~~~~~~~~~~~~~~~~

You can refine a single data array:

.. code-block:: python

    from mdr.core.refinement import refine_data
    
    refined_data = refine_data(data, config)

Or refine multiple variables at once:

.. code-block:: python

    from mdr.core.refinement import apply_refinement_pipeline
    
    data_dict = {
        "temperature": np.array([20.5, 21.3, np.nan, 21.7, 45.0]),
        "pressure": np.array([101.3, 101.4, 80.0, np.nan, np.nan])
    }
    
    refined_dict = apply_refinement_pipeline(data_dict, config)

Data Validation
-------------

MDR provides tools to validate data quality before refinement.

Available Validation Checks
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Range**: Check if values are within expected ranges
- **Missing**: Check the percentage of missing values
- **Outliers**: Identify statistical outliers
- **Consistency**: Check for internal consistency between variables

Validation Example
~~~~~~~~~~~~~~~~

.. code-block:: python

    from mdr.core.validation import validate_data
    
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

Data Transformation
-----------------

After refining your data, you may need to transform it for further analysis.

Available Transformations
~~~~~~~~~~~~~~~~~~~~~~~

- **Normalize**: Scale data to a standard range
- **Scale**: Apply linear scaling
- **Log**: Apply logarithmic transformation
- **Power**: Apply power transformation

Transformation Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mdr.core.transformation import transform_data
    
    transformations = [
        {"type": "normalize", "method": "minmax"},
        {"type": "scale", "factor": 2.0, "offset": 1.0}
    ]
    
    transformed_data = transform_data(refined_data, transformations)

Visualization
-----------

MDR provides various visualization tools to help understand your data before and after processing.

Time Series Plots
~~~~~~~~~~~~~~~

.. code-block:: python

    from mdr.visualization.plots import plot_time_series
    
    fig, ax = plot_time_series(data_dict, time_values)
    plt.show()

Refinement Comparison
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mdr.visualization.plots import plot_refinement_comparison
    
    fig, axes = plot_refinement_comparison(original_data, refined_data)
    plt.show()

Validation Results
~~~~~~~~~~~~~~~~

.. code-block:: python

    from mdr.visualization.plots import plot_validation_results
    
    fig, axes = plot_validation_results(validation_results)
    plt.show()

Command-Line Interface
--------------------

MDR provides a command-line interface for common operations:

Refining Data
~~~~~~~~~~~~

.. code-block:: bash

    mdr refine input.csv output.csv --smoothing-factor 0.2 --outlier-threshold 3.0

Validating Data
~~~~~~~~~~~~~

.. code-block:: bash

    mdr validate input.csv --output-file validation_results.json

Converting File Formats
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    mdr convert input.csv output.parquet

Advanced Usage
------------

For more advanced usage examples, please refer to the :ref:`examples` section, which includes:

- Working with multiple data sources
- Custom validation strategies
- Integration with other analysis workflows
- API server deployment
