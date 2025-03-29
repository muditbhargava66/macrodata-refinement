.. _examples:

Examples
=======

This section provides examples of how to use the Macrodata Refinement (MDR) package
for common data refinement tasks. These examples demonstrate key features and
workflows to help you get started quickly.

Basic Examples
------------

Basic Data Refinement
~~~~~~~~~~~~~~~~~~~

A simple example of refining data by removing outliers, imputing missing values, and smoothing:

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

Data Validation
~~~~~~~~~~~~~

Validating data quality before refinement:

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

Data Visualization
~~~~~~~~~~~~~~~~

Visualizing the effects of refinement:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mdr.core.refinement import RefinementConfig, refine_data
    from mdr.visualization.plots import plot_refinement_comparison
    
    # Create sample data with outliers
    data = np.array([1.0, 2.0, 3.0, 20.0, 5.0])
    
    # Configure and apply refinement
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    refined_data = refine_data(data, config)
    
    # Create a comparison plot
    fig, axes = plot_refinement_comparison(data, refined_data)
    plt.tight_layout()
    plt.show()

Advanced Examples
--------------

Complete Workflow
~~~~~~~~~~~~~~~

A complete workflow from data loading to saving the refined and transformed data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mdr.core.refinement import RefinementConfig, apply_refinement_pipeline
    from mdr.core.validation import validate_data
    from mdr.core.transformation import transform_data
    from mdr.io.readers import read_csv
    from mdr.io.writers import write_csv
    from mdr.visualization.plots import plot_time_series, save_plot
    import matplotlib.pyplot as plt
    
    # Step 1: Load data
    data_dict = read_csv("path/to/data.csv")
    
    # Extract time array if present
    time = data_dict.pop("time") if "time" in data_dict else None
    
    # Step 2: Validate data
    validation_results = validate_data(
        data_dict,
        checks=["range", "missing", "outliers"],
        params={
            "range": {"min_value": -10.0, "max_value": 200.0},
            "missing": {"threshold": 0.1},
            "outliers": {"threshold": 2.5}
        }
    )
    
    # Step 3: Refine data
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    refined_dict = apply_refinement_pipeline(data_dict, config)
    
    # Step 4: Transform data
    transformations = [
        {"type": "normalize", "method": "minmax"},
        {"type": "scale", "factor": 2.0, "offset": 1.0}
    ]
    
    transformed_dict = {}
    for var_name, values in refined_dict.items():
        transformed_dict[var_name] = transform_data(values, transformations)
    
    # Step 5: Visualize results
    if time is not None:
        # Plot original data
        fig, ax = plot_time_series(data_dict, time)
        save_plot(fig, "original_data.png")
        plt.close(fig)
        
        # Plot refined data
        fig, ax = plot_time_series(refined_dict, time)
        save_plot(fig, "refined_data.png")
        plt.close(fig)
        
        # Plot transformed data
        fig, ax = plot_time_series(transformed_dict, time)
        save_plot(fig, "transformed_data.png")
        plt.close(fig)
    
    # Step 6: Save results
    if time is not None:
        # Add time back to the dictionaries
        data_dict["time"] = time
        refined_dict["time"] = time
        transformed_dict["time"] = time
    
    # Save refined data
    write_csv(refined_dict, "refined_data.csv")
    
    # Save transformed data
    write_csv(transformed_dict, "transformed_data.csv")

Working with Multiple Data Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combining and refining data from multiple sources:

.. code-block:: python

    from mdr.io.readers import read_csv, read_excel
    from mdr.core.refinement import RefinementConfig, apply_refinement_pipeline
    from mdr.io.writers import write_csv
    
    # Load data from multiple sources
    temperature_dict = read_csv("temperature.csv")
    pressure_dict = read_excel("pressure.xlsx", sheets=["Pressure"])
    
    # Combine the data
    combined_dict = {**temperature_dict, **pressure_dict}
    
    # Configure refinement
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    
    # Refine the combined data
    refined_dict = apply_refinement_pipeline(combined_dict, config)
    
    # Save the refined data
    write_csv(refined_dict, "refined_combined_data.csv")

Using the Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the MDR command-line interface for batch processing:

.. code-block:: bash

    #!/bin/bash
    
    # Process a directory of CSV files
    for file in data/*.csv; do
        filename=$(basename "$file")
        echo "Processing $filename..."
        
        # Validate the data
        mdr validate "$file" --output-file "validation/${filename%.csv}_validation.json"
        
        # Refine the data
        mdr refine "$file" "refined/${filename}" \
            --smoothing-factor 0.2 \
            --outlier-threshold 2.5 \
            --imputation-method linear
        
        # Convert to parquet format
        mdr convert "refined/${filename}" "final/${filename%.csv}.parquet"
    done

Using the REST API
~~~~~~~~~~~~~~~~

Using the MDR REST API from a Python client:

.. code-block:: python

    import requests
    import json
    import numpy as np
    
    # Define the API URL
    api_url = "http://localhost:8000"
    
    # Create sample data with outliers and missing values
    data = [1.0, 2.0, None, 4.0, 100.0]
    
    # Configure refinement
    config = {
        "smoothing_factor": 0.2,
        "outlier_threshold": 2.5,
        "imputation_method": "linear",
        "normalization_type": "minmax"
    }
    
    # Send a refinement request
    response = requests.post(
        f"{api_url}/refinement",
        json={"data": data, "config": config}
    )
    
    # Parse the response
    result = response.json()
    print("Refined data:", result["refined_data"])

Jupyter Notebook Examples
-----------------------

For interactive examples, see the Jupyter notebooks in the `examples/notebooks` directory:

- `quickstart.ipynb`: An interactive tutorial covering the basics of MDR
- `advanced_refinement.ipynb`: Advanced refinement techniques
- `visualization_examples.ipynb`: Examples of various visualization options
- `custom_pipeline.ipynb`: Building custom refinement pipelines
