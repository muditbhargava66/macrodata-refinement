"""
Macrodata Refinement (MDR) package.

A comprehensive toolkit for refining, validating, and transforming macrodata
through various statistical and analytical methods.

Key features:
- Data refinement: outlier removal, missing value imputation, smoothing
- Data validation: range checks, missing value detection, outlier detection
- Data transformation: normalization, scaling, and other transformations
- I/O support: reading and writing data in various formats
- Visualization: plotting and visualizing data and refinement results
- REST API and CLI interfaces

All functions and methods include strict type checks and assertions to
guarantee robust type safety, especially for floating-point values.
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Mudit Bhargava"
__email__ = "muditbhargava666@gmail.com"
__license__ = "MIT"

# Import key components from submodules
from mdr.core.refinement import (
    RefinementConfig,
    smooth_data,
    remove_outliers,
    impute_missing_values,
    refine_data,
    apply_refinement_pipeline
)

from mdr.core.validation import (
    ValidationResult,
    check_data_range,
    check_missing_values,
    check_outliers,
    check_data_integrity,
    validate_data
)

from mdr.core.transformation import (
    NormalizationType,
    normalize_data,
    scale_data,
    apply_logarithmic_transform,
    apply_power_transform,
    transform_data
)

from mdr.io.readers import (
    read_csv,
    read_json,
    read_excel,
    read_parquet,
    read_hdf5
)

from mdr.io.writers import (
    write_csv,
    write_json,
    write_excel,
    write_parquet,
    write_hdf5
)

from mdr.io.formats import (
    FormatType,
    detect_format,
    convert_format,
    validate_format
)

from mdr.utils.logging import (
    setup_logger,
    get_logger,
    set_log_level,
    LogLevel,
    LogHandler
)

from mdr.utils.helpers import (
    validate_numeric_array,
    validate_range,
    moving_average,
    detect_seasonality,
    interpolate_missing
)

from mdr.visualization.plots import (
    plot_time_series,
    plot_histogram,
    plot_boxplot,
    plot_heatmap,
    plot_scatter,
    plot_correlation_matrix,
    plot_validation_results,
    plot_refinement_comparison,
    save_plot,
    PlotConfig
)

# Define top-level exports
__all__ = [
    # Refinement
    "RefinementConfig",
    "smooth_data",
    "remove_outliers",
    "impute_missing_values",
    "refine_data",
    "apply_refinement_pipeline",
    
    # Validation
    "ValidationResult",
    "check_data_range",
    "check_missing_values",
    "check_outliers",
    "check_data_integrity",
    "validate_data",
    
    # Transformation
    "NormalizationType",
    "normalize_data",
    "scale_data",
    "apply_logarithmic_transform",
    "apply_power_transform",
    "transform_data",
    
    # I/O Readers
    "read_csv",
    "read_json",
    "read_excel",
    "read_parquet",
    "read_hdf5",
    
    # I/O Writers
    "write_csv",
    "write_json",
    "write_excel",
    "write_parquet",
    "write_hdf5",
    
    # Formats
    "FormatType",
    "detect_format",
    "convert_format",
    "validate_format",
    
    # Logging
    "setup_logger",
    "get_logger",
    "set_log_level",
    "LogLevel",
    "LogHandler",
    
    # Helpers
    "validate_numeric_array",
    "validate_range",
    "moving_average",
    "detect_seasonality",
    "interpolate_missing",
    
    # Visualization
    "plot_time_series",
    "plot_histogram",
    "plot_boxplot",
    "plot_heatmap",
    "plot_scatter",
    "plot_correlation_matrix",
    "plot_validation_results",
    "plot_refinement_comparison",
    "save_plot",
    "PlotConfig"
]