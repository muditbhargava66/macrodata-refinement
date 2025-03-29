"""
Unit tests for the visualization module of Macrodata Refinement (MDR).
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mdr.visualization.plots import (
    plot_time_series,
    plot_histogram,
    plot_boxplot,
    plot_heatmap,
    plot_scatter,
    plot_correlation_matrix,
    plot_validation_results,
    plot_refinement_comparison,
    PlotConfig
)


class TestPlotCorrelationMatrix:
    """Tests for the plot_correlation_matrix function."""
    
    def test_with_numpy_array(self) -> None:
        """Test correlation matrix with numpy array input."""
        # Create sample data
        data = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 3.0, 5.0, 3.0, 1.0]
        ]).T  # Transpose to get 5 samples with 3 variables
        
        # Create plot
        fig, ax = plot_correlation_matrix(
            data,
            method="pearson",
            cmap="coolwarm"
        )
        
        # Check basic properties
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Close the figure to avoid memory leaks
        plt.close(fig)
    
    def test_with_dict(self) -> None:
        """Test correlation matrix with dictionary input."""
        # Create sample data
        data = {
            "var1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "var2": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            "var3": np.array([1.0, 3.0, 5.0, 3.0, 1.0])
        }
        
        # Create plot
        fig, ax = plot_correlation_matrix(
            data,
            method="pearson",
            cmap="coolwarm"
        )
        
        # Check basic properties
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Close the figure to avoid memory leaks
        plt.close(fig)
    
    def test_with_dataframe(self) -> None:
        """Test correlation matrix with pandas DataFrame input."""
        # Create sample data
        data = {
            "var1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "var2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "var3": [1.0, 3.0, 5.0, 3.0, 1.0]
        }
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plot_correlation_matrix(
            df,
            method="pearson",
            cmap="coolwarm"
        )
        
        # Check basic properties
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Close the figure to avoid memory leaks
        plt.close(fig)
    
    def test_with_config(self) -> None:
        """Test correlation matrix with custom configuration."""
        # Create sample data
        data = {
            "var1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "var2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "var3": [1.0, 3.0, 5.0, 3.0, 1.0]
        }
        df = pd.DataFrame(data)
        
        # Create custom config
        config = PlotConfig(
            title="Test Correlation Matrix",
            figsize=(8.0, 6.0),
            dpi=100,
            grid=False,
            font_size=14
        )
        
        # Create plot
        fig, ax = plot_correlation_matrix(
            df,
            method="spearman",  # Use a different correlation method
            cmap="viridis",     # Use a different colormap
            config=config
        )
        
        # Check basic properties
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check config was applied
        assert ax.get_title() == "Test Correlation Matrix"
        
        # Close the figure to avoid memory leaks
        plt.close(fig)
    
    def test_invalid_input(self) -> None:
        """Test correlation matrix with invalid input."""
        # Test with invalid input type
        with pytest.raises(ValueError):
            plot_correlation_matrix(
                "invalid_input",  # type: ignore
                method="pearson",
                cmap="coolwarm"
            )
        
        # Test with invalid method
        with pytest.raises(AssertionError):
            plot_correlation_matrix(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                method="invalid_method",
                cmap="coolwarm"
            )
