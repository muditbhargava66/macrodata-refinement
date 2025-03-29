"""
Helper utilities for Macrodata Refinement (MDR).

This module provides utility functions for data validation, transformation,
and management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import sys
import os
import gc
from collections import deque


def validate_numeric_array(
    arr: np.ndarray,
    allow_nan: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Validate that an array consists of numeric values.
    
    Args:
        arr: Input array to validate
        allow_nan: Whether to allow NaN values
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        
    Returns:
        True if the array is valid, False otherwise
    """
    assert isinstance(arr, np.ndarray), "arr must be a numpy ndarray"
    assert isinstance(allow_nan, bool), "allow_nan must be a boolean"
    
    if min_val is not None:
        assert isinstance(min_val, float), "min_val must be a floating-point number"
    
    if max_val is not None:
        assert isinstance(max_val, float), "max_val must be a floating-point number"
    
    if min_val is not None and max_val is not None:
        assert min_val <= max_val, "min_val must be less than or equal to max_val"
    
    # Check if the array is numeric
    if not np.issubdtype(arr.dtype, np.number):
        return False
    
    # Check for NaN values if not allowed
    if not allow_nan and np.isnan(arr).any():
        return False
    
    # Check minimum value if specified
    if min_val is not None:
        if not allow_nan:
            if np.any(arr < min_val):
                return False
        else:
            if np.any(np.logical_and(~np.isnan(arr), arr < min_val)):
                return False
    
    # Check maximum value if specified
    if max_val is not None:
        if not allow_nan:
            if np.any(arr > max_val):
                return False
        else:
            if np.any(np.logical_and(~np.isnan(arr), arr > max_val)):
                return False
    
    return True


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    inclusive: bool = True
) -> bool:
    """
    Check if a value is within a specified range.
    
    Args:
        value: The value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        inclusive: Whether the range bounds are inclusive
        
    Returns:
        True if the value is within the range, False otherwise
    """
    assert isinstance(value, float), "value must be a floating-point number"
    assert isinstance(min_val, float), "min_val must be a floating-point number"
    assert isinstance(max_val, float), "max_val must be a floating-point number"
    assert isinstance(inclusive, bool), "inclusive must be a boolean"
    assert min_val <= max_val, "min_val must be less than or equal to max_val"
    
    if inclusive:
        return min_val <= value <= max_val
    else:
        return min_val < value < max_val


def moving_average(
    data: np.ndarray,
    window_size: int,
    center: bool = False
) -> np.ndarray:
    """
    Calculate the moving average of a data array.
    
    Args:
        data: Input data array
        window_size: Size of the moving window
        center: Whether to center the window
        
    Returns:
        Array of moving averages
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(window_size, int), "window_size must be an integer"
    assert window_size > 0, "window_size must be positive"
    assert isinstance(center, bool), "center must be a boolean"
    
    # Convert to pandas Series for easy rolling calculation
    series = pd.Series(data)
    
    # Calculate rolling mean
    rolling_mean = series.rolling(window=window_size, center=center).mean()
    
    # Convert back to numpy array
    result = rolling_mean.to_numpy()
    
    return result


def detect_seasonality(
    data: np.ndarray,
    max_lag: int = 365,
    threshold: float = 0.3
) -> Tuple[bool, Optional[int]]:
    """
    Detect seasonality in a time series using autocorrelation.
    
    Args:
        data: Input time series data
        max_lag: Maximum lag to consider
        threshold: Correlation threshold for seasonality detection
        
    Returns:
        Tuple of (is_seasonal, period), where period is the detected
        seasonal period or None if no seasonality is detected
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(max_lag, int), "max_lag must be an integer"
    assert max_lag > 0, "max_lag must be positive"
    assert isinstance(threshold, float), "threshold must be a floating-point number"
    assert 0.0 <= threshold <= 1.0, "threshold must be between 0 and 1"
    
    # Ensure we have enough data
    if len(data) < 2 * max_lag:
        max_lag = len(data) // 2
    
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 2 * max_lag:
        return False, None
    
    # Compute autocorrelation
    autocorr = np.correlate(clean_data, clean_data, mode='full')
    autocorr = autocorr[len(clean_data)-1:len(clean_data)+max_lag]
    autocorr = autocorr / np.max(autocorr)  # Normalize
    
    # Find peaks
    peaks = []
    for i in range(1, len(autocorr)-1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > threshold:
            peaks.append((i, autocorr[i]))
    
    # Sort peaks by correlation value
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Return the period of the highest peak (excluding lag 0)
    if peaks and peaks[0][0] > 0:
        return True, peaks[0][0]
    else:
        return False, None


def interpolate_missing(
    data: np.ndarray,
    method: str = 'linear',
    max_gap: Optional[int] = None,
    order: Optional[int] = None
) -> np.ndarray:
    """
    Interpolate missing values in a data array.
    
    Args:
        data: Input data array with potential NaN values
        method: Interpolation method ('linear', 'nearest', 'cubic', 'spline')
        max_gap: Maximum gap size to interpolate (None for no limit)
        
    Returns:
        Data array with missing values interpolated
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(method, str), "method must be a string"
    assert method in ['linear', 'nearest', 'cubic', 'spline'], \
        "method must be one of ['linear', 'nearest', 'cubic', 'spline']"
    
    if max_gap is not None:
        assert isinstance(max_gap, int), "max_gap must be an integer"
        assert max_gap > 0, "max_gap must be positive"
    
    # Create a pandas Series for interpolation
    series = pd.Series(data)
    
    # Find missing value indices
    missing_mask = series.isna()
    
    if not missing_mask.any():
        # No missing values to interpolate
        return data
    
    if max_gap is not None:
        # Find runs of missing values
        missing_indices = np.where(missing_mask)[0]
        
        # Find gaps between consecutive missing indices
        gaps = np.diff(missing_indices)
        
        # Find runs of consecutive missing values
        run_starts = np.where(gaps > 1)[0] + 1
        run_starts = np.insert(run_starts, 0, 0)
        
        # Find runs that are too large
        for i in range(len(run_starts) - 1):
            start_idx = missing_indices[run_starts[i]]
            end_idx = missing_indices[run_starts[i+1] - 1]
            
            if end_idx - start_idx >= max_gap:
                # Don't interpolate this gap
                series.iloc[start_idx:end_idx+1] = np.nan
    
    # Interpolate using the specified method
    kwargs = {}
    if method in ['spline', 'polynomial'] and order is not None:
        kwargs['order'] = order
    interpolated = series.interpolate(method=method, **kwargs)
    
    # Return as numpy array
    return interpolated.to_numpy()


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '.'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Input dictionary to flatten
        parent_key: Prefix for flattened keys
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    assert isinstance(d, dict), "d must be a dictionary"
    assert isinstance(parent_key, str), "parent_key must be a string"
    assert isinstance(sep, str), "sep must be a string"
    
    items = []
    for k, v in d.items():
        assert isinstance(k, str), "All dictionary keys must be strings"
        
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(
    d: Dict[str, Any],
    sep: str = '.'
) -> Dict[str, Any]:
    """
    Convert a flattened dictionary back to a nested dictionary.
    
    Args:
        d: Flattened dictionary
        sep: Separator used in flattened keys
        
    Returns:
        Nested dictionary
    """
    assert isinstance(d, dict), "d must be a dictionary"
    assert isinstance(sep, str), "sep must be a string"
    assert all(isinstance(k, str) for k in d.keys()), "All dictionary keys must be strings"
    
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        
        # Navigate to the correct nested dictionary
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    return result


def get_memory_usage(
    obj: Any = None,
    unit: str = 'MB'
) -> float:
    """
    Get the memory usage of an object or the current process.
    
    Args:
        obj: Python object to measure (None for current process)
        unit: Unit for the result ('B', 'KB', 'MB', 'GB')
        
    Returns:
        Memory usage in the specified unit
    """
    assert unit in ['B', 'KB', 'MB', 'GB'], "unit must be one of ['B', 'KB', 'MB', 'GB']"
    
    if obj is None:
        # Get memory usage of the current process
        import psutil
        process = psutil.Process(os.getpid())
        memory_bytes = process.memory_info().rss
    else:
        # Get memory usage of the specified object
        import sys
        memory_bytes = sys.getsizeof(obj)
        
        # For containers, recursively add the size of their contents
        if isinstance(obj, (list, tuple, set, dict)):
            if isinstance(obj, dict):
                memory_bytes += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
            else:
                memory_bytes += sum(sys.getsizeof(x) for x in obj)
    
    # Convert to the requested unit
    if unit == 'KB':
        return float(memory_bytes / 1024)
    elif unit == 'MB':
        return float(memory_bytes / (1024 * 1024))
    elif unit == 'GB':
        return float(memory_bytes / (1024 * 1024 * 1024))
    else:
        return float(memory_bytes)