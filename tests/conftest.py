"""
Pytest configuration and fixtures for Macrodata Refinement (MDR) tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from typing import Dict, List, Tuple, Any, Generator, Optional

from mdr.core.refinement import RefinementConfig


@pytest.fixture
def sample_data() -> Dict[str, np.ndarray]:
    """
    Generate sample data for testing.
    
    Returns:
        Dictionary of variable names to sample data arrays
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create linear data with some noise
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + np.random.normal(0, 1, 100)
    
    # Create sinusoidal data with some noise
    t = np.linspace(0, 2 * np.pi, 100)
    sin_data = np.sin(t) + np.random.normal(0, 0.1, 100)
    
    # Create exponential data with some noise
    exp_data = np.exp(x / 10) + np.random.normal(0, 0.5, 100)
    
    return {
        "linear": y,
        "sinusoidal": sin_data,
        "exponential": exp_data,
        "time": x
    }


@pytest.fixture
def sample_data_with_outliers() -> Dict[str, np.ndarray]:
    """
    Generate sample data with outliers for testing.
    
    Returns:
        Dictionary of variable names to sample data arrays with outliers
    """
    data = sample_data()
    
    # Add outliers
    data["linear"][10] = 100.0
    data["sinusoidal"][30] = 5.0
    data["exponential"][50] = 0.0
    
    return data


@pytest.fixture
def sample_data_with_missing() -> Dict[str, np.ndarray]:
    """
    Generate sample data with missing values for testing.
    
    Returns:
        Dictionary of variable names to sample data arrays with missing values
    """
    data = sample_data()
    
    # Add missing values
    data["linear"][20] = np.nan
    data["sinusoidal"][40:45] = np.nan
    data["exponential"][70] = np.nan
    
    return data


@pytest.fixture
def sample_data_with_outliers_and_missing() -> Dict[str, np.ndarray]:
    """
    Generate sample data with both outliers and missing values for testing.
    
    Returns:
        Dictionary of variable names to sample data arrays with outliers and missing values
    """
    data = sample_data_with_outliers()
    
    # Add missing values
    data["linear"][25] = np.nan
    data["sinusoidal"][45:50] = np.nan
    data["exponential"][75] = np.nan
    
    return data


@pytest.fixture
def refinement_config() -> RefinementConfig:
    """
    Create a sample refinement configuration for testing.
    
    Returns:
        RefinementConfig object
    """
    return RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=3.0,
        imputation_method="mean",
        normalization_type="minmax"
    )


@pytest.fixture
def temp_csv_file() -> Generator[str, None, None]:
    """
    Create a temporary CSV file for testing.
    
    Yields:
        Path to the temporary CSV file
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Clean up the file after the test
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for testing.
    
    Yields:
        Path to the temporary directory
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Create a sample pandas DataFrame for testing.
    
    Returns:
        Sample DataFrame
    """
    # Create sample data
    data = {
        "id": list(range(1, 101)),
        "value": np.random.normal(0, 1, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range(start="2023-01-01", periods=100)
    }
    
    return pd.DataFrame(data)