.. _api_visualization_plots:

Data Visualization
================

.. automodule:: mdr.visualization.plots
   :members:
   :undoc-members:
   :show-inheritance:

Overview
-------

The ``plots`` module provides functions for visualizing data, refinement results,
and validation outcomes. These visualizations help understand the data and the
effects of refinement operations.

Plot Types
---------

The module provides the following types of plots:

- **Time Series**: Visualize data variables over time
- **Refinement Comparison**: Compare original and refined data
- **Validation Results**: Visualize data quality assessment results
- **Distribution**: Show data distributions before and after processing
- **Correlation**: Display relationships between variables

Core Functions
------------

.. autofunction:: mdr.visualization.plots.plot_time_series
.. autofunction:: mdr.visualization.plots.plot_refinement_comparison
.. autofunction:: mdr.visualization.plots.plot_validation_results
.. autofunction:: mdr.visualization.plots.plot_distribution
.. autofunction:: mdr.visualization.plots.plot_correlation
.. autofunction:: mdr.visualization.plots.save_plot

Customization Options
-------------------

Most plotting functions accept the following customization parameters:

- **figsize**: Tuple specifying the figure dimensions
- **title**: Custom title for the plot
- **labels**: Dictionary mapping variable names to display labels
- **colors**: Custom color scheme for the plot
- **style**: Matplotlib style sheet to use

Usage Examples
------------

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
