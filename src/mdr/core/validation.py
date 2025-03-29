"""
Validation module for Macrodata Refinement (MDR).

This module provides functions and classes for validating macrodata
and ensuring data quality.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of a data validation operation."""
    
    is_valid: bool
    error_messages: List[str]
    invalid_indices: Optional[np.ndarray] = None
    statistics: Optional[Dict[str, float]] = None
    
    def __post_init__(self) -> None:
        """Validate the ValidationResult instance."""
        assert isinstance(self.is_valid, bool), "is_valid must be a boolean"
        assert isinstance(self.error_messages, list), "error_messages must be a list"
        
        if self.invalid_indices is not None:
            assert isinstance(self.invalid_indices, np.ndarray), "invalid_indices must be a numpy ndarray"
        
        if self.statistics is not None:
            assert isinstance(self.statistics, dict), "statistics must be a dictionary"
            for key, value in self.statistics.items():
                assert isinstance(key, str), "statistics keys must be strings"
                assert isinstance(value, float), f"statistics value for key '{key}' must be a floating-point number"


def check_data_range(
    data: np.ndarray,
    min_value: float,
    max_value: float
) -> ValidationResult:
    """
    Check if all values in the data are within the specified range.
    
    Args:
        data: Input data array
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        ValidationResult object containing validation results
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(min_value, float), "min_value must be a floating-point number"
    assert isinstance(max_value, float), "max_value must be a floating-point number"
    assert min_value <= max_value, "min_value must be less than or equal to max_value"
    
    invalid_indices = np.where((data < min_value) | (data > max_value))[0]
    is_valid = len(invalid_indices) == 0
    
    error_messages = []
    if not is_valid:
        error_messages.append(
            f"Data contains {len(invalid_indices)} values outside the range [{min_value}, {max_value}]"
        )
    
    # Calculate statistics
    statistics = {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data))
    }
    
    return ValidationResult(
        is_valid=is_valid,
        error_messages=error_messages,
        invalid_indices=invalid_indices if not is_valid else None,
        statistics=statistics
    )


def check_missing_values(
    data: np.ndarray,
    threshold: float = 0.1
) -> ValidationResult:
    """
    Check for missing values in the data.
    
    Args:
        data: Input data array
        threshold: Maximum allowed fraction of missing values
        
    Returns:
        ValidationResult object containing validation results
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(threshold, float), "threshold must be a floating-point number"
    assert 0.0 <= threshold <= 1.0, "threshold must be between 0 and 1"
    
    missing_indices = np.where(np.isnan(data))[0]
    missing_fraction = len(missing_indices) / len(data)
    
    is_valid = missing_fraction <= threshold
    
    error_messages = []
    if not is_valid:
        error_messages.append(
            f"Data contains {len(missing_indices)} missing values ({missing_fraction:.2%}), "
            f"which exceeds the threshold of {threshold:.2%}"
        )
    
    # Calculate statistics related to missing values
    statistics = {
        "missing_count": float(len(missing_indices)),
        "missing_fraction": float(missing_fraction),
        "threshold": float(threshold)
    }
    
    return ValidationResult(
        is_valid=is_valid,
        error_messages=error_messages,
        invalid_indices=missing_indices if not is_valid else None,
        statistics=statistics
    )


def check_outliers(
    data: np.ndarray,
    threshold: float = 3.0,
    method: str = "zscore"
) -> ValidationResult:
    """
    Check for outliers in the data.
    
    Args:
        data: Input data array
        threshold: Threshold for outlier detection
        method: Method for outlier detection ('zscore', 'iqr', 'mad')
        
    Returns:
        ValidationResult object containing validation results
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(threshold, float), "threshold must be a floating-point number"
    assert threshold > 0.0, "threshold must be greater than 0"
    assert isinstance(method, str), "method must be a string"
    
    # Remove NaN values for outlier detection
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return ValidationResult(
            is_valid=False,
            error_messages=["Cannot check outliers: all values are missing"],
            statistics={"outlier_count": 0.0, "outlier_fraction": 0.0}
        )
    
    # Detect outliers based on the specified method
    if method == "zscore":
        # Z-score method
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        
        if std == 0:
            # All values are the same, no outliers
            outlier_indices = np.array([])
        else:
            z_scores = np.abs((data - mean) / std)
            outlier_indices = np.where((~np.isnan(z_scores)) & (z_scores > threshold))[0]
            
    elif method == "iqr":
        # IQR method
        q1, q3 = np.percentile(valid_data, [25, 75])
        iqr = q3 - q1
        
        if iqr == 0:
            # IQR is zero, all values are the same or nearly the same
            outlier_indices = np.array([])
        else:
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_indices = np.where((~np.isnan(data)) & 
                                      ((data < lower_bound) | (data > upper_bound)))[0]
            
    elif method == "mad":
        # Median Absolute Deviation method
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        
        if mad == 0:
            # MAD is zero, all values are the same or nearly the same
            outlier_indices = np.array([])
        else:
            z_scores = 0.6745 * np.abs(data - median) / mad  # Approximately equivalent to z-scores
            outlier_indices = np.where((~np.isnan(z_scores)) & (z_scores > threshold))[0]
            
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    outlier_fraction = len(outlier_indices) / len(data)
    is_valid = outlier_fraction <= 0.1  # Consider data invalid if more than 10% are outliers
    
    error_messages = []
    if not is_valid:
        error_messages.append(
            f"Data contains {len(outlier_indices)} outliers ({outlier_fraction:.2%}), "
            f"which may indicate data quality issues"
        )
    
    # Calculate statistics related to outliers
    statistics = {
        "outlier_count": float(len(outlier_indices)),
        "outlier_fraction": float(outlier_fraction)
    }
    
    return ValidationResult(
        is_valid=is_valid,
        error_messages=error_messages,
        invalid_indices=outlier_indices if not is_valid else None,
        statistics=statistics
    )


def check_data_integrity(
    data: np.ndarray,
    checks: List[str] = ["range", "missing", "outliers"],
    params: Dict[str, Any] = None
) -> ValidationResult:
    """
    Perform a comprehensive data integrity check.
    
    Args:
        data: Input data array
        checks: List of checks to perform
        params: Parameters for each check
        
    Returns:
        ValidationResult object containing validation results
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    assert isinstance(checks, list), "checks must be a list"
    
    if params is None:
        params = {}
    
    assert isinstance(params, dict), "params must be a dictionary"
    
    # Default parameters
    default_params = {
        "range": {"min_value": -np.inf, "max_value": np.inf},
        "missing": {"threshold": 0.1},
        "outliers": {"threshold": 3.0, "method": "zscore"}
    }
    
    # Update default parameters with provided parameters
    for check, check_params in default_params.items():
        if check in params:
            check_params.update(params[check])
    
    results = []
    all_errors = []
    invalid_indices_sets = []
    all_statistics = {}
    
    # Perform each requested check
    for check in checks:
        if check == "range":
            result = check_data_range(
                data,
                min_value=float(default_params["range"]["min_value"]),
                max_value=float(default_params["range"]["max_value"])
            )
            
        elif check == "missing":
            result = check_missing_values(
                data,
                threshold=float(default_params["missing"]["threshold"])
            )
            
        elif check == "outliers":
            result = check_outliers(
                data,
                threshold=float(default_params["outliers"]["threshold"]),
                method=default_params["outliers"]["method"]
            )
            
        else:
            raise ValueError(f"Unknown check: {check}")
        
        results.append(result)
        all_errors.extend(result.error_messages)
        
        if result.invalid_indices is not None:
            invalid_indices_sets.append(result.invalid_indices)
            
        if result.statistics is not None:
            all_statistics.update({f"{check}_{k}": v for k, v in result.statistics.items()})
    
    # Combine invalid indices from all checks
    combined_invalid_indices = np.unique(np.concatenate(invalid_indices_sets)) if invalid_indices_sets else None
    
    # Data is valid if all checks pass
    is_valid = all(result.is_valid for result in results)
    
    return ValidationResult(
        is_valid=is_valid,
        error_messages=all_errors,
        invalid_indices=combined_invalid_indices,
        statistics=all_statistics
    )


def validate_data(
    data_dict: Dict[str, np.ndarray],
    checks: List[str] = ["range", "missing", "outliers"],
    params: Dict[str, Dict[str, Any]] = None
) -> Dict[str, ValidationResult]:
    """
    Validate multiple data arrays.
    
    Args:
        data_dict: Dictionary mapping variable names to data arrays
        checks: List of checks to perform
        params: Parameters for each check, per variable
        
    Returns:
        Dictionary mapping variable names to ValidationResult objects
    """
    assert isinstance(data_dict, dict), "data_dict must be a dictionary"
    assert isinstance(checks, list), "checks must be a list"
    
    if params is None:
        params = {}
    
    assert isinstance(params, dict), "params must be a dictionary"
    
    validation_results = {}
    
    for key, data in data_dict.items():
        assert isinstance(data, np.ndarray), f"Value for key '{key}' must be a numpy ndarray"
        
        # Get parameters for this variable, if specified
        var_params = params.get(key, {})
        
        # Validate this variable's data
        validation_results[key] = check_data_integrity(data, checks, var_params)
    
    return validation_results