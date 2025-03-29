.. _api_visualization_plots:

Data Visualization
==================

.. automodule:: mdr.visualization.plots
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``plots`` module provides functions for visualizing data, refinement results,
and validation outcomes. These visualizations help understand the data and the
effects of refinement operations.

Plot Types
----------

The module provides the following types of plots:

- **Time Series**: Visualize data variables over time
- **Refinement Comparison**: Compare original and refined data
- **Validation Results**: Visualize data quality assessment results
- **Distribution**: Show data distributions before and after processing
- **Correlation**: Display relationships between variables

Core Functions
--------------

.. autofunction:: mdr.visualization.plots.plot_time_series
   :no-index:
.. autofunction:: mdr.visualization.plots.plot_refinement_comparison
   :no-index:
.. autofunction:: mdr.visualization.plots.plot_validation_results
   :no-index:
.. autofunction:: mdr.visualization.plots.plot_correlation_matrix
   :no-index:
.. autofunction:: mdr.visualization.plots.save_plot
   :no-index:

Input Format Compatibility
--------------------------

Many of the visualization functions accept multiple input formats:

- **NumPy arrays**: For single variable visualizations
- **Dictionary of arrays**: For multi-variable visualizations with named variables
- **Pandas DataFrames**: For direct use of pandas data structures (supported by most functions)

Customization Options
---------------------

Most plotting functions accept the following customization parameters:

- **figsize**: Tuple specifying the figure dimensions
- **title**: Custom title for the plot
- **labels**: Dictionary mapping variable names to display labels
- **colors**: Custom color scheme for the plot
- **style**: Matplotlib style sheet to use

Usage Examples
--------------

Time series plot:

.. code-block:: python

    import numpy as np
    from mdr.visualization.plots import plot_time_series
    import matplotlib.pyplot as plt
    
    # Create sample data
    time = np.array([0, 1, 2, 3, 4])
    data_dict = {
        "temperature": np.array([20.5, 21.3, 22.1, 21.7, 23.0]),
        "pressure": np.array([101.3, 101.4, 101.5, 101.2, 101.1])
    }
    
    # Create a time series plot
    fig, ax = plot_time_series(
        data_dict,
        time,
        title="Sensor Readings",
        labels={"temperature": "Temperature (°C)", "pressure": "Pressure (hPa)"}
    )
    plt.show()

Refinement comparison:

.. code-block:: python

    import numpy as np
    from mdr.core.refinement import RefinementConfig, refine_data
    from mdr.visualization.plots import plot_refinement_comparison
    import matplotlib.pyplot as plt
    
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
    fig, axes = plot_refinement_comparison(
        data,
        refined_data,
        title="Data Refinement Results"
    )
    plt.tight_layout()
    plt.show()

Correlation matrix with pandas DataFrame:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mdr.visualization.plots import plot_correlation_matrix, PlotConfig
    import matplotlib.pyplot as plt
    
    # Create a sample DataFrame
    data = {
        "temperature": [20.5, 21.3, 22.1, 21.7, 23.0],
        "pressure": [101.3, 101.4, 101.5, 101.2, 101.1],
        "humidity": [45.0, 47.0, 48.5, 50.2, 49.8]
    }
    df = pd.DataFrame(data)
    
    # Create a correlation matrix plot directly from the DataFrame
    fig, ax = plot_correlation_matrix(
        df,
        method="pearson",
        cmap="coolwarm",
        config=PlotConfig(title="Correlation Matrix")
    )
    plt.show()
