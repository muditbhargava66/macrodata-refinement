"""
Transformation module for Macrodata Refinement (MDR).

This module provides functions for transforming macrodata
through various statistical and mathematical operations.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
from enum import Enum, auto


class NormalizationType(Enum):
    """Types of normalization methods."""
    
    MINMAX = auto()
    ZSCORE = auto()
    ROBUST = auto()
    DECIMAL_SCALING = auto()


def normalize_data(
    data: np.ndarray,
    method: Union[str, NormalizationType] = "minmax",
    params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize data using the specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'robust', 'decimal_scaling')
        params: Additional parameters for normalization
        
    Returns:
        Tuple of (normalized data array, normalization parameters)
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    
    if isinstance(method, str):
        try:
            method = NormalizationType[method.upper()]
        except KeyError:
            raise ValueError(f"Unknown normalization method: {method}")
    
    assert isinstance(method, NormalizationType), "method must be a string or NormalizationType"
    
    if params is None:
        params = {}
    
    assert isinstance(params, dict), "params must be a dictionary"
    
    # Handle missing values
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        # All values are missing, return the original data and empty parameters
        return data.copy(), {}
    
    # Create output array
    normalized = np.full_like(data, np.nan)
    
    # Apply normalization based on the specified method
    if method == NormalizationType.MINMAX:
        # Min-max normalization
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        # If min and max are the same, return zeros
        if data_min == data_max:
            normalized[valid_mask] = 0.0
            normalization_params = {"min": float(data_min), "max": float(data_max)}
        else:
            # Apply min-max scaling
            normalized[valid_mask] = (valid_data - data_min) / (data_max - data_min)
            normalization_params = {"min": float(data_min), "max": float(data_max)}
    
    elif method == NormalizationType.ZSCORE:
        # Z-score normalization
        data_mean = np.mean(valid_data)
        data_std = np.std(valid_data)
        
        # If standard deviation is zero, return zeros
        if data_std == 0:
            normalized[valid_mask] = 0.0
            normalization_params = {"mean": float(data_mean), "std": float(data_std)}
        else:
            # Apply z-score normalization
            normalized[valid_mask] = (valid_data - data_mean) / data_std
            normalization_params = {"mean": float(data_mean), "std": float(data_std)}
    
    elif method == NormalizationType.ROBUST:
        # Robust normalization using median and IQR
        data_median = np.median(valid_data)
        q1, q3 = np.percentile(valid_data, [25, 75])
        iqr = q3 - q1
        
        # If IQR is zero, return zeros
        if iqr == 0:
            normalized[valid_mask] = 0.0
            normalization_params = {"median": float(data_median), "iqr": float(iqr)}
        else:
            # Apply robust normalization
            normalized[valid_mask] = (valid_data - data_median) / iqr
            normalization_params = {"median": float(data_median), "iqr": float(iqr)}
    
    elif method == NormalizationType.DECIMAL_SCALING:
        # Decimal scaling normalization
        max_abs = np.max(np.abs(valid_data))
        
        if max_abs == 0:
            normalized[valid_mask] = 0.0
            normalization_params = {"scale": 1.0}
        else:
            # Calculate the number of digits in the maximum absolute value
            scale = 10 ** np.ceil(np.log10(max_abs))
            
            # Apply decimal scaling
            normalized[valid_mask] = valid_data / scale
            normalization_params = {"scale": float(scale)}
    
    return normalized, normalization_params


def scale_data(
    data: np.ndarray,
    factor: float,
    offset: float = 0.0
) -> np.ndarray:
    """
    Scale data by a factor and add an offset.
    
    Args:
        data: Input data array
        factor: Scaling factor
        offset: Offset to add after scaling
        
    Returns:
        Scaled data array
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(factor, float), "factor must be a floating-point number"
    assert isinstance(offset, float), "offset must be a floating-point number"
    
    return data * factor + offset


def apply_logarithmic_transform(
    data: np.ndarray,
    base: float = 10.0,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Apply logarithmic transformation to the data.
    
    Args:
        data: Input data array
        base: Logarithm base
        epsilon: Small value to add to prevent log(0)
        
    Returns:
        Log-transformed data array
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(base, float), "base must be a floating-point number"
    assert base > 0.0, "base must be positive"
    assert isinstance(epsilon, float), "epsilon must be a floating-point number"
    assert epsilon > 0.0, "epsilon must be positive"
    
    # Create a copy to avoid modifying the original data
    transformed = data.copy()
    
    # Replace negative values with NaN
    transformed[transformed <= 0] = np.nan
    
    # Apply log transformation
    valid_mask = ~np.isnan(transformed)
    transformed[valid_mask] = np.log(transformed[valid_mask] + epsilon) / np.log(base)
    
    return transformed


def apply_power_transform(
    data: np.ndarray,
    power: float,
    preserve_sign: bool = True
) -> np.ndarray:
    """
    Apply power transformation to the data.
    
    Args:
        data: Input data array
        power: Power to raise the data to
        preserve_sign: Whether to preserve the sign of the original data
        
    Returns:
        Power-transformed data array
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(power, float), "power must be a floating-point number"
    assert isinstance(preserve_sign, bool), "preserve_sign must be a boolean"
    
    # Create a copy to avoid modifying the original data
    transformed = data.copy()
    
    if preserve_sign:
        # Preserve the sign of the original data
        signs = np.sign(transformed)
        
        # Transform the absolute values
        valid_mask = ~np.isnan(transformed)
        transformed[valid_mask] = np.abs(transformed[valid_mask]) ** power * signs[valid_mask]
    else:
        # Apply power transformation directly
        valid_mask = ~np.isnan(transformed)
        transformed[valid_mask] = transformed[valid_mask] ** power
    
    return transformed


def apply_rolling_window(
    data: np.ndarray,
    window_size: int,
    window_function: Callable[[np.ndarray], float],
    center: bool = True
) -> np.ndarray:
    """
    Apply a rolling window function to the data.
    
    Args:
        data: Input data array
        window_size: Size of the rolling window
        window_function: Function to apply to each window (e.g., np.mean, np.median)
        center: Whether to center the window
        
    Returns:
        Data array with the rolling window function applied
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(window_size, int), "window_size must be an integer"
    assert window_size > 0, "window_size must be positive"
    assert callable(window_function), "window_function must be callable"
    assert isinstance(center, bool), "center must be a boolean"
    
    # Create pandas Series for easy rolling window operations
    series = pd.Series(data)
    
    # Apply rolling window function
    rolled = series.rolling(window=window_size, center=center).apply(
        lambda x: window_function(x.values)
    )
    
    return rolled.values


def transform_data(
    data: np.ndarray,
    transformations: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Apply a sequence of transformations to the data.
    
    Args:
        data: Input data array
        transformations: List of transformation specifications
        
    Returns:
        Transformed data array
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(transformations, list), "transformations must be a list"
    
    # Create a copy to avoid modifying the original data
    transformed = data.copy()
    
    for transform_spec in transformations:
        assert isinstance(transform_spec, dict), "Each transformation specification must be a dictionary"
        assert "type" in transform_spec, "Each transformation specification must have a 'type' field"
        
        transform_type = transform_spec["type"]
        
        if transform_type == "normalize":
            method = transform_spec.get("method", "minmax")
            params = transform_spec.get("params", {})
            transformed, _ = normalize_data(transformed, method=method, params=params)
            
        elif transform_type == "scale":
            factor = float(transform_spec.get("factor", 1.0))
            offset = float(transform_spec.get("offset", 0.0))
            transformed = scale_data(transformed, factor=factor, offset=offset)
            
        elif transform_type == "log":
            base = float(transform_spec.get("base", 10.0))
            epsilon = float(transform_spec.get("epsilon", 1e-10))
            transformed = apply_logarithmic_transform(transformed, base=base, epsilon=epsilon)
            
        elif transform_type == "power":
            power = float(transform_spec.get("power", 2.0))
            preserve_sign = transform_spec.get("preserve_sign", True)
            transformed = apply_power_transform(transformed, power=power, preserve_sign=preserve_sign)
            
        elif transform_type == "rolling":
            window_size = int(transform_spec.get("window_size", 3))
            
            # Get window function
            func_name = transform_spec.get("function", "mean")
            if func_name == "mean":
                window_function = np.nanmean
            elif func_name == "median":
                window_function = np.nanmedian
            elif func_name == "sum":
                window_function = np.nansum
            elif func_name == "min":
                window_function = np.nanmin
            elif func_name == "max":
                window_function = np.nanmax
            else:
                raise ValueError(f"Unknown window function: {func_name}")
            
            center = transform_spec.get("center", True)
            transformed = apply_rolling_window(
                transformed, 
                window_size=window_size, 
                window_function=window_function, 
                center=center
            )
            
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    return transformed