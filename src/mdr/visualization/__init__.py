"""
Visualization module for Macrodata Refinement (MDR).

This module provides functions and classes for visualizing macrodata
and analysis results.
"""

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

__all__ = [
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