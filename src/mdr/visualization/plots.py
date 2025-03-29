"""
Plotting utilities for Macrodata Refinement (MDR).

This module provides functions for creating and saving data visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from dataclasses import dataclass, field
import os


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior."""
    
    title: Optional[str] = None
    figsize: Tuple[float, float] = (10.0, 6.0)
    dpi: int = 100
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    legend: bool = True
    grid: bool = True
    style: str = "seaborn-v0_8-whitegrid"
    palette: str = "viridis"
    font_family: str = "sans-serif"
    font_size: int = 12
    figure_bgcolor: str = "#ffffff"
    axis_bgcolor: str = "#f8f8f8"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.title is not None:
            assert isinstance(self.title, str), "title must be a string"
        
        assert isinstance(self.figsize, tuple), "figsize must be a tuple"
        assert len(self.figsize) == 2, "figsize must be a tuple of length 2"
        assert isinstance(self.figsize[0], float), "figsize[0] must be a floating-point number"
        assert isinstance(self.figsize[1], float), "figsize[1] must be a floating-point number"
        assert self.figsize[0] > 0.0, "figsize[0] must be positive"
        assert self.figsize[1] > 0.0, "figsize[1] must be positive"
        
        assert isinstance(self.dpi, int), "dpi must be an integer"
        assert self.dpi > 0, "dpi must be positive"
        
        if self.xlabel is not None:
            assert isinstance(self.xlabel, str), "xlabel must be a string"
        
        if self.ylabel is not None:
            assert isinstance(self.ylabel, str), "ylabel must be a string"
        
        if self.xlim is not None:
            assert isinstance(self.xlim, tuple), "xlim must be a tuple"
            assert len(self.xlim) == 2, "xlim must be a tuple of length 2"
            assert isinstance(self.xlim[0], float), "xlim[0] must be a floating-point number"
            assert isinstance(self.xlim[1], float), "xlim[1] must be a floating-point number"
            assert self.xlim[0] <= self.xlim[1], "xlim[0] must be less than or equal to xlim[1]"
        
        if self.ylim is not None:
            assert isinstance(self.ylim, tuple), "ylim must be a tuple"
            assert len(self.ylim) == 2, "ylim must be a tuple of length 2"
            assert isinstance(self.ylim[0], float), "ylim[0] must be a floating-point number"
            assert isinstance(self.ylim[1], float), "ylim[1] must be a floating-point number"
            assert self.ylim[0] <= self.ylim[1], "ylim[0] must be less than or equal to ylim[1]"
        
        assert isinstance(self.legend, bool), "legend must be a boolean"
        assert isinstance(self.grid, bool), "grid must be a boolean"
        assert isinstance(self.style, str), "style must be a string"
        assert isinstance(self.palette, str), "palette must be a string"
        assert isinstance(self.font_family, str), "font_family must be a string"
        assert isinstance(self.font_size, int), "font_size must be an integer"
        assert self.font_size > 0, "font_size must be positive"
        assert isinstance(self.figure_bgcolor, str), "figure_bgcolor must be a string"
        assert isinstance(self.axis_bgcolor, str), "axis_bgcolor must be a string"


def _apply_plot_config(
    fig: plt.Figure,
    ax: plt.Axes,
    config: PlotConfig
) -> None:
    """
    Apply plot configuration to a figure and axes.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        config: Plot configuration
    """
    assert isinstance(config, PlotConfig), "config must be a PlotConfig object"
    
    # Apply style
    plt.style.use(config.style)
    
    # Apply title
    if config.title is not None:
        ax.set_title(config.title, fontsize=config.font_size + 2, fontfamily=config.font_family)
    
    # Apply labels
    if config.xlabel is not None:
        ax.set_xlabel(config.xlabel, fontsize=config.font_size, fontfamily=config.font_family)
    
    if config.ylabel is not None:
        ax.set_ylabel(config.ylabel, fontsize=config.font_size, fontfamily=config.font_family)
    
    # Apply limits
    if config.xlim is not None:
        ax.set_xlim(config.xlim)
    
    if config.ylim is not None:
        ax.set_ylim(config.ylim)
    
    # Apply grid
    ax.grid(config.grid)
    
    # Apply background colors
    fig.patch.set_facecolor(config.figure_bgcolor)
    ax.set_facecolor(config.axis_bgcolor)
    
    # Apply font properties to tick labels
    ax.tick_params(labelsize=config.font_size, labelcolor="black")
    
    # Apply legend if there are any labeled elements
    if config.legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, fontsize=config.font_size, loc="best")


def plot_time_series(
    data: Dict[str, np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot time series data.
    
    Args:
        data: Dictionary mapping variable names to data arrays
        timestamps: Optional array of timestamps for the x-axis
        config: Plot configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    
    if timestamps is not None:
        assert isinstance(timestamps, np.ndarray), "timestamps must be a numpy ndarray"
        
        # Check that the timestamps array has the same length as the data arrays
        first_key = next(iter(data.keys()))
        assert len(timestamps) == len(data[first_key]), \
            "timestamps array must have the same length as data arrays"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Time Series Plot",
            xlabel="Time" if timestamps is None else None,
            ylabel="Value"
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot each time series
    for name, values in data.items():
        if timestamps is not None:
            ax.plot(timestamps, values, label=name)
        else:
            ax.plot(values, label=name)
    
    # Apply configuration
    _apply_plot_config(fig, ax, config)
    
    return fig, ax


def plot_histogram(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    bins: int = 30,
    density: bool = False,
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of data.
    
    Args:
        data: Array of data or dictionary mapping variable names to data arrays
        bins: Number of histogram bins
        density: Whether to normalize the histogram
        config: Plot configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    assert isinstance(bins, int), "bins must be an integer"
    assert bins > 0, "bins must be positive"
    assert isinstance(density, bool), "density must be a boolean"
    
    # Convert single array to dictionary if needed
    if isinstance(data, np.ndarray):
        data = {"Data": data}
    
    assert isinstance(data, dict), "data must be a dictionary or numpy ndarray"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Histogram",
            xlabel="Value",
            ylabel="Frequency" if not density else "Density"
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot each histogram
    for name, values in data.items():
        # Filter out NaN values
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            ax.hist(
                valid_values,
                bins=bins,
                density=density,
                label=name,
                alpha=0.7
            )
    
    # Apply configuration
    _apply_plot_config(fig, ax, config)
    
    return fig, ax


def plot_boxplot(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    vert: bool = True,
    showfliers: bool = True,
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a box plot of data.
    
    Args:
        data: Array of data or dictionary mapping variable names to data arrays
        vert: Whether to draw the boxes vertically
        showfliers: Whether to show outliers
        config: Plot configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    assert isinstance(vert, bool), "vert must be a boolean"
    assert isinstance(showfliers, bool), "showfliers must be a boolean"
    
    # Convert single array to dictionary if needed
    if isinstance(data, np.ndarray):
        data = {"Data": data}
    
    assert isinstance(data, dict), "data must be a dictionary or numpy ndarray"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Box Plot",
            xlabel="" if vert else "Value",
            ylabel="Value" if vert else ""
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Prepare data for plotting
    box_data = []
    labels = []
    
    for name, values in data.items():
        # Filter out NaN values
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            box_data.append(valid_values)
            labels.append(name)
    
    # Plot the box plot
    if box_data:
        ax.boxplot(
            box_data,
            labels=labels,
            vert=vert,
            showfliers=showfliers,
            patch_artist=True
        )
    
    # Apply configuration
    _apply_plot_config(fig, ax, config)
    
    return fig, ax


def plot_heatmap(
    data: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of 2D data.
    
    Args:
        data: 2D array of data
        row_labels: Labels for the rows
        col_labels: Labels for the columns
        cmap: Colormap name
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        config: Plot configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert len(data.shape) == 2, "data must be a 2D array"
    assert isinstance(cmap, str), "cmap must be a string"
    
    if row_labels is not None:
        assert isinstance(row_labels, list), "row_labels must be a list"
        assert all(isinstance(label, str) for label in row_labels), "All row labels must be strings"
        assert len(row_labels) == data.shape[0], "Number of row labels must match data shape"
    
    if col_labels is not None:
        assert isinstance(col_labels, list), "col_labels must be a list"
        assert all(isinstance(label, str) for label in col_labels), "All column labels must be strings"
        assert len(col_labels) == data.shape[1], "Number of column labels must match data shape"
    
    if vmin is not None:
        assert isinstance(vmin, float), "vmin must be a floating-point number"
    
    if vmax is not None:
        assert isinstance(vmax, float), "vmax must be a floating-point number"
    
    if vmin is not None and vmax is not None:
        assert vmin <= vmax, "vmin must be less than or equal to vmax"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Heatmap",
            figsize=(8.0, 6.0)
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot the heatmap
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    
    # Add row and column labels
    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    
    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    
    # Apply configuration
    _apply_plot_config(fig, ax, config)
    
    return fig, ax


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[np.ndarray] = None,
    sizes: Optional[np.ndarray] = None,
    alpha: float = 0.7,
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a scatter plot of data.
    
    Args:
        x: X-coordinates
        y: Y-coordinates
        labels: Labels or categories for the points
        sizes: Sizes for the points
        alpha: Transparency for the points
        config: Plot configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    assert isinstance(x, np.ndarray), "x must be a numpy ndarray"
    assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
    assert len(x) == len(y), "x and y must have the same length"
    assert isinstance(alpha, float), "alpha must be a floating-point number"
    assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
    
    if labels is not None:
        assert isinstance(labels, np.ndarray), "labels must be a numpy ndarray"
        assert len(labels) == len(x), "labels must have the same length as x and y"
    
    if sizes is not None:
        assert isinstance(sizes, np.ndarray), "sizes must be a numpy ndarray"
        assert len(sizes) == len(x), "sizes must have the same length as x and y"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Scatter Plot",
            xlabel="X",
            ylabel="Y"
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot scatter points
    if labels is not None:
        # Get unique labels
        unique_labels = np.unique(labels)
        
        # Create a colormap
        cmap = plt.get_cmap(config.palette)
        colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]
        
        # Plot each category with a different color
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            if sizes is not None:
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=colors[i],
                    s=sizes[mask],
                    alpha=alpha,
                    label=str(label)
                )
            else:
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=colors[i],
                    alpha=alpha,
                    label=str(label)
                )
    else:
        # Plot all points with the same color
        if sizes is not None:
            ax.scatter(x, y, s=sizes, alpha=alpha)
        else:
            ax.scatter(x, y, alpha=alpha)
    
    # Apply configuration
    _apply_plot_config(fig, ax, config)
    
    return fig, ax


def plot_correlation_matrix(
    data: Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame],
    method: str = "pearson",
    cmap: str = "coolwarm",
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a correlation matrix.
    
    Args:
        data: 2D array of data, dictionary mapping variable names to data arrays, or pandas DataFrame
        method: Correlation method ('pearson', 'kendall', 'spearman')
        cmap: Colormap name
        config: Plot configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    assert isinstance(method, str), "method must be a string"
    assert method in ["pearson", "kendall", "spearman"], \
        "method must be one of ['pearson', 'kendall', 'spearman']"
    assert isinstance(cmap, str), "cmap must be a string"
    
    # Convert dictionary to DataFrame if needed
    if isinstance(data, dict):
        assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
        assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
        
        # Check that all arrays have the same length
        first_length = len(next(iter(data.values())))
        assert all(len(v) == first_length for v in data.values()), \
            "All arrays in data must have the same length"
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
    
    elif isinstance(data, np.ndarray):
        assert len(data.shape) == 2, "data must be a 2D array or dictionary of arrays"
        
        # Convert to DataFrame with default column names
        df = pd.DataFrame(data)
    
    elif isinstance(data, pd.DataFrame):
        # Use the DataFrame directly
        df = data
    
    else:
        raise ValueError("data must be a numpy ndarray, dictionary of arrays, or pandas DataFrame")
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title=f"Correlation Matrix ({method.capitalize()})",
            figsize=(8.0, 6.0)
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Calculate correlation matrix
    corr = df.corr(method=method)
    
    # Plot heatmap
    im = sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5
    )
    
    # Apply configuration
    _apply_plot_config(fig, ax, config)
    
    return fig, ax


def plot_validation_results(
    results: Dict[str, Dict[str, Any]],
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot validation results.
    
    Args:
        results: Dictionary mapping variable names to validation results
        config: Plot configuration
        
    Returns:
        Tuple of (figure, list of axes)
    """
    assert isinstance(results, dict), "results must be a dictionary"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Validation Results",
            figsize=(12.0, 8.0)
        )
    
    # Create figure and axes for multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=config.figsize, dpi=config.dpi)
    axes = axes.flatten()
    
    # Get variables and their validation status
    variable_names = list(results.keys())
    valid_status = [results[var]["is_valid"] for var in variable_names]
    
    # 1. Bar chart of valid vs. invalid variables
    ax1 = axes[0]
    valid_count = sum(valid_status)
    invalid_count = len(variable_names) - valid_count
    
    ax1.bar(
        ["Valid", "Invalid"],
        [valid_count, invalid_count],
        color=["green", "red"]
    )
    
    ax1.set_title("Validation Status")
    ax1.set_ylabel("Count")
    
    # 2. Pie chart of valid vs. invalid variables
    ax2 = axes[1]
    ax2.pie(
        [valid_count, invalid_count],
        labels=["Valid", "Invalid"],
        colors=["green", "red"],
        autopct="%1.1f%%",
        startangle=90
    )
    
    ax2.set_title("Validation Status")
    
    # 3. Bar chart of error counts by variable
    ax3 = axes[2]
    error_counts = [len(results[var]["error_messages"]) for var in variable_names]
    
    # Sort by error count
    sorted_indices = np.argsort(error_counts)[::-1]
    sorted_names = [variable_names[i] for i in sorted_indices]
    sorted_counts = [error_counts[i] for i in sorted_indices]
    
    # Limit to top 10 if there are many variables
    if len(sorted_names) > 10:
        sorted_names = sorted_names[:10]
        sorted_counts = sorted_counts[:10]
    
    ax3.barh(
        sorted_names,
        sorted_counts,
        color="orange"
    )
    
    ax3.set_title("Error Counts by Variable")
    ax3.set_xlabel("Number of Errors")
    
    # 4. Statistics summary
    ax4 = axes[3]
    ax4.axis("off")
    
    # Calculate some overall statistics
    total_vars = len(variable_names)
    total_errors = sum(error_counts)
    avg_errors = total_errors / total_vars if total_vars > 0 else 0
    
    stats_text = (
        f"Total Variables: {total_vars}\n"
        f"Valid Variables: {valid_count} ({valid_count/total_vars*100:.1f}%)\n"
        f"Invalid Variables: {invalid_count} ({invalid_count/total_vars*100:.1f}%)\n"
        f"Total Errors: {total_errors}\n"
        f"Avg. Errors per Variable: {avg_errors:.2f}"
    )
    
    ax4.text(
        0.5,
        0.5,
        stats_text,
        ha="center",
        va="center",
        fontsize=config.font_size
    )
    
    ax4.set_title("Summary Statistics")
    
    # Set overall title
    fig.suptitle(config.title, fontsize=config.font_size + 4)
    
    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    return fig, axes


def plot_refinement_comparison(
    original_data: np.ndarray,
    refined_data: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    config: Optional[PlotConfig] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot a comparison of original and refined data.
    
    Args:
        original_data: Original data array
        refined_data: Refined data array
        timestamps: Optional array of timestamps for the x-axis
        config: Plot configuration
        
    Returns:
        Tuple of (figure, list of axes)
    """
    assert isinstance(original_data, np.ndarray), "original_data must be a numpy ndarray"
    assert isinstance(refined_data, np.ndarray), "refined_data must be a numpy ndarray"
    assert original_data.shape == refined_data.shape, "original_data and refined_data must have the same shape"
    
    if timestamps is not None:
        assert isinstance(timestamps, np.ndarray), "timestamps must be a numpy ndarray"
        assert len(timestamps) == len(original_data), "timestamps must have the same length as data arrays"
    
    # Use default config if not provided
    if config is None:
        config = PlotConfig(
            title="Data Refinement Comparison",
            figsize=(12.0, 8.0)
        )
    
    # Create figure and axes for multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=config.figsize, dpi=config.dpi)
    axes = axes.flatten()
    
    # 1. Time series plot of original and refined data
    ax1 = axes[0]
    
    if timestamps is not None:
        ax1.plot(timestamps, original_data, label="Original", alpha=0.7)
        ax1.plot(timestamps, refined_data, label="Refined", alpha=0.7)
    else:
        ax1.plot(original_data, label="Original", alpha=0.7)
        ax1.plot(refined_data, label="Refined", alpha=0.7)
    
    ax1.set_title("Time Series Comparison")
    ax1.set_xlabel("Time" if timestamps is None else "")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Histogram of original and refined data
    ax2 = axes[1]
    
    # Filter out NaN values
    original_valid = original_data[~np.isnan(original_data)]
    refined_valid = refined_data[~np.isnan(refined_data)]
    
    ax2.hist(
        original_valid,
        bins=30,
        alpha=0.5,
        label="Original"
    )
    
    ax2.hist(
        refined_valid,
        bins=30,
        alpha=0.5,
        label="Refined"
    )
    
    ax2.set_title("Distribution Comparison")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True)
    
    # 3. Scatter plot of original vs. refined data
    ax3 = axes[2]
    
    # Create a mask for non-NaN values in both arrays
    mask = ~np.isnan(original_data) & ~np.isnan(refined_data)
    
    if np.any(mask):
        ax3.scatter(
            original_data[mask],
            refined_data[mask],
            alpha=0.5,
            edgecolor="k",
            linewidth=0.5
        )
        
        # Add diagonal line for reference
        min_val = min(original_data[mask].min(), refined_data[mask].min())
        max_val = max(original_data[mask].max(), refined_data[mask].max())
        ax3.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7)
    
    ax3.set_title("Original vs. Refined Values")
    ax3.set_xlabel("Original Value")
    ax3.set_ylabel("Refined Value")
    ax3.grid(True)
    
    # 4. Residuals plot
    ax4 = axes[3]
    
    if np.any(mask):
        residuals = refined_data[mask] - original_data[mask]
        
        if timestamps is not None and len(timestamps) == len(original_data):
            ax4.plot(timestamps[mask], residuals, "o-", alpha=0.5, markersize=3)
        else:
            ax4.plot(np.arange(len(residuals)), residuals, "o-", alpha=0.5, markersize=3)
        
        # Add horizontal line at zero
        ax4.axhline(y=0, color="k", linestyle="--", alpha=0.7)
    
    ax4.set_title("Refinement Residuals")
    ax4.set_xlabel("Time" if timestamps is not None else "Index")
    ax4.set_ylabel("Refined - Original")
    ax4.grid(True)
    
    # Set overall title
    fig.suptitle(config.title, fontsize=config.font_size + 4)
    
    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    return fig, axes


def save_plot(
    fig: plt.Figure,
    filepath: str,
    dpi: Optional[int] = None,
    format: Optional[str] = None,
    transparent: bool = False
) -> None:
    """
    Save a figure to a file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to the output file
        dpi: Resolution in dots per inch
        format: File format (auto-detected from extension if None)
        transparent: Whether to use a transparent background
    """
    assert isinstance(filepath, str), "filepath must be a string"
    
    if dpi is not None:
        assert isinstance(dpi, int), "dpi must be an integer"
        assert dpi > 0, "dpi must be positive"
    
    if format is not None:
        assert isinstance(format, str), "format must be a string"
    
    assert isinstance(transparent, bool), "transparent must be a boolean"
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the figure
    fig.savefig(
        filepath,
        dpi=dpi,
        format=format,
        bbox_inches="tight",
        transparent=transparent
    )