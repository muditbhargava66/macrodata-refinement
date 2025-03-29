"""
Refinement module for Macrodata Refinement (MDR).

This module provides functions and classes for refining macrodata
through various statistical and analytical methods.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RefinementConfig:
    """Configuration for data refinement operations."""
    
    smoothing_factor: float
    outlier_threshold: float
    imputation_method: str
    normalization_type: str
    
    def __post_init__(self) -> None:
        """Validate the configuration parameters."""
        assert isinstance(self.smoothing_factor, float), "smoothing_factor must be a floating-point number"
        assert isinstance(self.outlier_threshold, float), "outlier_threshold must be a floating-point number"
        assert isinstance(self.imputation_method, str), "imputation_method must be a string"
        assert isinstance(self.normalization_type, str), "normalization_type must be a string"
        
        assert 0.0 < self.smoothing_factor <= 1.0, "smoothing_factor must be between 0 and 1"
        assert self.outlier_threshold > 0.0, "outlier_threshold must be greater than 0"


def smooth_data(data: np.ndarray, factor: float) -> np.ndarray:
    """
    Apply smoothing to the input data.
    
    Args:
        data: Input data array to smooth
        factor: Smoothing factor (0 < factor <= 1)
        
    Returns:
        Smoothed data array
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(factor, float), "factor must be a floating-point number"
    assert 0.0 < factor <= 1.0, "factor must be between 0 and 1"
    
    # Apply exponential moving average for smoothing
    window_size = max(2, int(1.0 / factor))
    weights = np.exp(np.linspace(-1., 0., window_size))
    weights /= weights.sum()
    
    smoothed = np.convolve(data, weights, mode='same')
    
    # Handle edge effects
    smoothed[0] = data[0]
    smoothed[-1] = data[-1]
    
    return smoothed


def remove_outliers(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Remove outliers from the data using the specified threshold.
    
    Args:
        data: Input data array
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Data array with outliers replaced by median values
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(threshold, float), "threshold must be a floating-point number"
    assert threshold > 0.0, "threshold must be greater than 0"
    
    # Calculate z-scores
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    
    if mad == 0:
        # Handle case where MAD is zero (all values are the same)
        return data
    
    z_scores = 0.6745 * (data - median) / mad  # Approximately equivalent to z-scores
    
    # Create a copy to avoid modifying the original data
    refined_data = data.copy()
    
    # Replace outliers with median values
    outlier_mask = np.abs(z_scores) > threshold
    refined_data[outlier_mask] = median
    
    return refined_data


def impute_missing_values(
    data: np.ndarray, 
    method: str = "mean",
    window_size: int = 3
) -> np.ndarray:
    """
    Impute missing values in the data.
    
    Args:
        data: Input data array with potential NaN values
        method: Imputation method ('mean', 'median', 'linear', 'forward')
        window_size: Size of the window for local imputation methods
        
    Returns:
        Data array with missing values imputed
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(method, str), "method must be a string"
    assert isinstance(window_size, int), "window_size must be an integer"
    assert window_size > 0, "window_size must be greater than 0"
    
    # Create a copy to avoid modifying the original data
    imputed_data = data.copy()
    
    # Find indices of missing values
    missing_indices = np.where(np.isnan(imputed_data))[0]
    
    if len(missing_indices) == 0:
        return imputed_data
    
    if method == "mean":
        # Replace missing values with the mean of non-missing values
        mean_value = np.nanmean(imputed_data)
        imputed_data[missing_indices] = mean_value
        
    elif method == "median":
        # Replace missing values with the median of non-missing values
        median_value = np.nanmedian(imputed_data)
        imputed_data[missing_indices] = median_value
        
    elif method == "linear":
        # Linear interpolation
        valid_indices = np.where(~np.isnan(imputed_data))[0]
        
        if len(valid_indices) < 2:
            # Not enough valid points for interpolation, use forward fill
            imputed_data = pd.Series(imputed_data).fillna(method='ffill').values
        else:
            imputed_data = pd.Series(imputed_data).interpolate(method='linear').values
            
    elif method == "forward":
        # Forward fill
        imputed_data = pd.Series(imputed_data).fillna(method='ffill').values
        
        # If we still have NaN values at the beginning, fill with the first valid value
        if np.isnan(imputed_data[0]):
            first_valid_idx = np.where(~np.isnan(imputed_data))[0]
            if len(first_valid_idx) > 0:
                imputed_data[0:first_valid_idx[0]] = imputed_data[first_valid_idx[0]]
            else:
                imputed_data[:] = 0.0  # If all values are NaN, set to 0
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    return imputed_data


def refine_data(
    data: np.ndarray,
    config: RefinementConfig
) -> np.ndarray:
    """
    Apply a complete refinement pipeline to the data.
    
    Args:
        data: Input data array
        config: Refinement configuration
        
    Returns:
        Refined data array
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(config, RefinementConfig), "config must be a RefinementConfig object"
    
    # First, impute missing values
    refined = impute_missing_values(data, method=config.imputation_method)
    
    # Then remove outliers
    refined = remove_outliers(refined, threshold=config.outlier_threshold)
    
    # Finally, smooth the data
    refined = smooth_data(refined, factor=config.smoothing_factor)
    
    return refined


def apply_refinement_pipeline(
    data_dict: Dict[str, np.ndarray],
    config: RefinementConfig
) -> Dict[str, np.ndarray]:
    """
    Apply refinement pipeline to a dictionary of data arrays.
    
    Args:
        data_dict: Dictionary mapping variable names to data arrays
        config: Refinement configuration
        
    Returns:
        Dictionary with refined data arrays
    """
    assert isinstance(data_dict, dict), "data_dict must be a dictionary"
    assert isinstance(config, RefinementConfig), "config must be a RefinementConfig object"
    
    refined_dict = {}
    
    for key, data in data_dict.items():
        assert isinstance(data, np.ndarray), f"Value for key '{key}' must be a numpy ndarray"
        refined_dict[key] = refine_data(data, config)
    
    return refined_dict