"""
Unit tests for the utility modules of Macrodata Refinement (MDR).
"""

import pytest
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Any, Tuple, Optional

from mdr.utils.helpers import (
    validate_numeric_array,
    validate_range,
    moving_average,
    detect_seasonality,
    interpolate_missing,
    flatten_dict,
    unflatten_dict,
    get_memory_usage
)

from mdr.utils.logging import (
    setup_logger,
    get_logger,
    set_log_level,
    LogLevel,
    log_execution_time,
    LogHandler
)


# ---- Helpers Module Tests ----

class TestValidateNumericArray:
    """Tests for the validate_numeric_array function."""
    
    def test_valid_input(self) -> None:
        """Test validation with valid input."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = validate_numeric_array(arr)
        
        assert result == True
    
    def test_invalid_array_type(self) -> None:
        """Test that non-ndarray arrays are rejected."""
        with pytest.raises(AssertionError):
            validate_numeric_array([1.0, 2.0, 3.0])  # type: ignore
    
    def test_invalid_allow_nan_type(self) -> None:
        """Test that non-bool allow_nan is rejected."""
        with pytest.raises(AssertionError):
            validate_numeric_array(np.array([1.0, 2.0, 3.0]), allow_nan="True")  # type: ignore
    
    def test_invalid_min_val_type(self) -> None:
        """Test that non-float min_val is rejected."""
        with pytest.raises(AssertionError):
            validate_numeric_array(np.array([1.0, 2.0, 3.0]), min_val="0.0")  # type: ignore
    
    def test_invalid_max_val_type(self) -> None:
        """Test that non-float max_val is rejected."""
        with pytest.raises(AssertionError):
            validate_numeric_array(np.array([1.0, 2.0, 3.0]), max_val="10.0")  # type: ignore
    
    def test_invalid_range(self) -> None:
        """Test that invalid ranges are rejected."""
        with pytest.raises(AssertionError):
            validate_numeric_array(np.array([1.0, 2.0, 3.0]), min_val=10.0, max_val=0.0)
    
    def test_non_numeric_array(self) -> None:
        """Test detection of non-numeric arrays."""
        # String array
        arr = np.array(["a", "b", "c"])
        result = validate_numeric_array(arr)
        assert result == False
    
    def test_with_nans(self) -> None:
        """Test handling of NaN values."""
        # Array with NaNs
        arr = np.array([1.0, np.nan, 3.0])
        
        # With allow_nan=True (default)
        result1 = validate_numeric_array(arr)
        assert result1 == True
        
        # With allow_nan=False
        result2 = validate_numeric_array(arr, allow_nan=False)
        assert result2 == False
    
    def test_with_min_max(self) -> None:
        """Test validation with min and max values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # All values within range
        result1 = validate_numeric_array(arr, min_val=0.0, max_val=10.0)
        assert result1 == True
        
        # Some values below min
        result2 = validate_numeric_array(arr, min_val=2.0, max_val=10.0)
        assert result2 == False
        
        # Some values above max
        result3 = validate_numeric_array(arr, min_val=0.0, max_val=4.0)
        assert result3 == False


class TestValidateRange:
    """Tests for the validate_range function."""
    
    def test_valid_input(self) -> None:
        """Test range validation with valid input."""
        result = validate_range(5.0, min_val=0.0, max_val=10.0)
        
        assert result == True
    
    def test_invalid_value_type(self) -> None:
        """Test that non-float values are rejected."""
        with pytest.raises(AssertionError):
            validate_range("5.0", min_val=0.0, max_val=10.0)  # type: ignore
    
    def test_invalid_min_val_type(self) -> None:
        """Test that non-float min_val is rejected."""
        with pytest.raises(AssertionError):
            validate_range(5.0, min_val="0.0", max_val=10.0)  # type: ignore
    
    def test_invalid_max_val_type(self) -> None:
        """Test that non-float max_val is rejected."""
        with pytest.raises(AssertionError):
            validate_range(5.0, min_val=0.0, max_val="10.0")  # type: ignore
    
    def test_invalid_inclusive_type(self) -> None:
        """Test that non-bool inclusive is rejected."""
        with pytest.raises(AssertionError):
            validate_range(5.0, min_val=0.0, max_val=10.0, inclusive="True")  # type: ignore
    
    def test_invalid_range(self) -> None:
        """Test that invalid ranges are rejected."""
        with pytest.raises(AssertionError):
            validate_range(5.0, min_val=10.0, max_val=0.0)
    
    def test_inclusive(self) -> None:
        """Test validation with inclusive bounds."""
        # Value at min bound
        result1 = validate_range(0.0, min_val=0.0, max_val=10.0, inclusive=True)
        assert result1 == True
        
        # Value at max bound
        result2 = validate_range(10.0, min_val=0.0, max_val=10.0, inclusive=True)
        assert result2 == True
    
    def test_exclusive(self) -> None:
        """Test validation with exclusive bounds."""
        # Value at min bound
        result1 = validate_range(0.0, min_val=0.0, max_val=10.0, inclusive=False)
        assert result1 == False
        
        # Value at max bound
        result2 = validate_range(10.0, min_val=0.0, max_val=10.0, inclusive=False)
        assert result2 == False
        
        # Value within bounds
        result3 = validate_range(5.0, min_val=0.0, max_val=10.0, inclusive=False)
        assert result3 == True


class TestMovingAverage:
    """Tests for the moving_average function."""
    
    def test_valid_input(self) -> None:
        """Test moving average with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(data, window_size=3)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
    
    def test_invalid_data_type(self) -> None:
        """Test that non-ndarray data is rejected."""
        with pytest.raises(AssertionError):
            moving_average([1.0, 2.0, 3.0], window_size=3)  # type: ignore
    
    def test_invalid_window_size_type(self) -> None:
        """Test that non-int window_size is rejected."""
        with pytest.raises(AssertionError):
            moving_average(np.array([1.0, 2.0, 3.0]), window_size="3")  # type: ignore
    
    def test_invalid_window_size_range(self) -> None:
        """Test that non-positive window_size is rejected."""
        with pytest.raises(AssertionError):
            moving_average(np.array([1.0, 2.0, 3.0]), window_size=0)
    
    def test_invalid_center_type(self) -> None:
        """Test that non-bool center is rejected."""
        with pytest.raises(AssertionError):
            moving_average(np.array([1.0, 2.0, 3.0]), window_size=3, center="True")  # type: ignore
    
    def test_smoothing_effect(self) -> None:
        """Test that moving average smooths the data."""
        # Create noisy data
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.5, 100)
        
        # Apply moving average
        result = moving_average(data, window_size=10)
        
        # Check that variance is reduced
        assert np.var(result[~np.isnan(result)]) < np.var(data)


class TestDetectSeasonality:
    """Tests for the detect_seasonality function."""
    
    def test_valid_input(self) -> None:
        """Test seasonality detection with valid input."""
        # Create seasonal data
        t = np.linspace(0, 10, 100)
        data = np.sin(t) + 0.1 * np.random.normal(0, 1, 100)
        
        is_seasonal, period = detect_seasonality(data, max_lag=50, threshold=0.5)
        
        assert isinstance(is_seasonal, bool)
        if is_seasonal:
            assert isinstance(period, int)
    
    def test_invalid_data_type(self) -> None:
        """Test that non-ndarray data is rejected."""
        with pytest.raises(AssertionError):
            detect_seasonality([1.0, 2.0, 3.0])  # type: ignore
    
    def test_invalid_max_lag_type(self) -> None:
        """Test that non-int max_lag is rejected."""
        with pytest.raises(AssertionError):
            detect_seasonality(np.array([1.0, 2.0, 3.0]), max_lag="50")  # type: ignore
    
    def test_invalid_max_lag_range(self) -> None:
        """Test that non-positive max_lag is rejected."""
        with pytest.raises(AssertionError):
            detect_seasonality(np.array([1.0, 2.0, 3.0]), max_lag=0)
    
    def test_invalid_threshold_type(self) -> None:
        """Test that non-float threshold is rejected."""
        with pytest.raises(AssertionError):
            detect_seasonality(np.array([1.0, 2.0, 3.0]), threshold="0.5")  # type: ignore
    
    def test_invalid_threshold_range(self) -> None:
        """Test that out-of-range threshold is rejected."""
        # Too small
        with pytest.raises(AssertionError):
            detect_seasonality(np.array([1.0, 2.0, 3.0]), threshold=-0.1)
        
        # Too large
        with pytest.raises(AssertionError):
            detect_seasonality(np.array([1.0, 2.0, 3.0]), threshold=1.1)
    
    def test_seasonal_data(self) -> None:
        """Test detection of seasonal data."""
        # Create strongly seasonal data with period=10
        t = np.linspace(0, 10, 100)
        seasonal_data = np.tile(np.sin(np.linspace(0, 2*np.pi, 10)), 10)
        
        is_seasonal, period = detect_seasonality(seasonal_data, max_lag=20, threshold=0.5)
        
        assert is_seasonal == True
        assert period == 10


class TestInterpolateMissing:
    """Tests for the interpolate_missing function."""
    
    def test_valid_input(self) -> None:
        """Test interpolation with valid input."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = interpolate_missing(data, method="linear")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert not np.isnan(result).any()  # No missing values in result
    
    def test_invalid_data_type(self) -> None:
        """Test that non-ndarray data is rejected."""
        with pytest.raises(AssertionError):
            interpolate_missing([1.0, 2.0, np.nan])  # type: ignore
    
    def test_invalid_method_type(self) -> None:
        """Test that non-string method is rejected."""
        with pytest.raises(AssertionError):
            interpolate_missing(np.array([1.0, 2.0, np.nan]), method=123)  # type: ignore
    
    def test_invalid_method_value(self) -> None:
        """Test that invalid method values are rejected."""
        with pytest.raises(AssertionError):
            interpolate_missing(np.array([1.0, 2.0, np.nan]), method="invalid")
    
    def test_invalid_max_gap_type(self) -> None:
        """Test that non-int max_gap is rejected."""
        with pytest.raises(AssertionError):
            interpolate_missing(np.array([1.0, 2.0, np.nan]), max_gap="3")  # type: ignore
    
    def test_invalid_max_gap_range(self) -> None:
        """Test that non-positive max_gap is rejected."""
        with pytest.raises(AssertionError):
            interpolate_missing(np.array([1.0, 2.0, np.nan]), max_gap=0)
    
    def test_no_missing_values(self) -> None:
        """Test interpolation with no missing values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = interpolate_missing(data)
        
        # Should return the original data
        assert np.array_equal(result, data)
    
    def test_methods(self) -> None:
        """Test different interpolation methods."""
        data = np.array([1.0, 2.0, np.nan, np.nan, 5.0])
        
        # Linear interpolation
        result_linear = interpolate_missing(data, method="linear")
        assert not np.isnan(result_linear).any()
        
        # Nearest interpolation
        result_nearest = interpolate_missing(data, method="nearest")
        assert not np.isnan(result_nearest).any()
        
        # Create a longer dataset for cubic interpolation (needs more points)
        longer_data = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 6.0, 7.0, 8.0])
        
        # Cubic interpolation - requires at least 4 non-NaN points
        result_cubic = interpolate_missing(longer_data, method="cubic")
        assert not np.isnan(result_cubic).any()
        
        # Spline interpolation - also requires order parameter
        result_spline = interpolate_missing(longer_data, method="spline", order=3)
        assert not np.isnan(result_spline).any()


class TestFlattenDict:
    """Tests for the flatten_dict function."""
    
    def test_valid_input(self) -> None:
        """Test flattening with valid input."""
        nested_dict = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        result = flatten_dict(nested_dict)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b.c", "b.d.e"}
        assert result["a"] == 1
        assert result["b.c"] == 2
        assert result["b.d.e"] == 3
    
    def test_invalid_dict_type(self) -> None:
        """Test that non-dict input is rejected."""
        with pytest.raises(AssertionError):
            flatten_dict("not a dict")  # type: ignore
    
    def test_invalid_parent_key_type(self) -> None:
        """Test that non-string parent_key is rejected."""
        with pytest.raises(AssertionError):
            flatten_dict({"a": 1}, parent_key=123)  # type: ignore
    
    def test_invalid_sep_type(self) -> None:
        """Test that non-string sep is rejected."""
        with pytest.raises(AssertionError):
            flatten_dict({"a": 1}, sep=123)  # type: ignore
    
    def test_invalid_key_type(self) -> None:
        """Test that non-string keys are rejected."""
        with pytest.raises(AssertionError):
            flatten_dict({1: "value"})  # type: ignore
    
    def test_empty_dict(self) -> None:
        """Test flattening an empty dictionary."""
        result = flatten_dict({})
        assert result == {}
    
    def test_flat_dict(self) -> None:
        """Test flattening an already flat dictionary."""
        flat_dict = {"a": 1, "b": 2, "c": 3}
        result = flatten_dict(flat_dict)
        assert result == flat_dict


class TestUnflattenDict:
    """Tests for the unflatten_dict function."""
    
    def test_valid_input(self) -> None:
        """Test unflattening with valid input."""
        flat_dict = {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }
        
        result = unflatten_dict(flat_dict)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        assert result["a"] == 1
        assert result["b"]["c"] == 2
        assert result["b"]["d"]["e"] == 3
    
    def test_invalid_dict_type(self) -> None:
        """Test that non-dict input is rejected."""
        with pytest.raises(AssertionError):
            unflatten_dict("not a dict")  # type: ignore
    
    def test_invalid_sep_type(self) -> None:
        """Test that non-string sep is rejected."""
        with pytest.raises(AssertionError):
            unflatten_dict({"a": 1}, sep=123)  # type: ignore
    
    def test_invalid_key_type(self) -> None:
        """Test that non-string keys are rejected."""
        with pytest.raises(AssertionError):
            unflatten_dict({1: "value"})  # type: ignore
    
    def test_empty_dict(self) -> None:
        """Test unflattening an empty dictionary."""
        result = unflatten_dict({})
        assert result == {}
    
    def test_already_nested_dict(self) -> None:
        """Test unflattening with keys that don't contain the separator."""
        dict_without_sep = {"a": 1, "b": 2, "c": 3}
        result = unflatten_dict(dict_without_sep)
        assert result == dict_without_sep


# ---- Logging Module Tests ----

class TestLogLevel:
    """Tests for the LogLevel enum."""
    
    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        assert hasattr(LogLevel, "DEBUG")
        assert hasattr(LogLevel, "INFO")
        assert hasattr(LogLevel, "WARNING")
        assert hasattr(LogLevel, "ERROR")
        assert hasattr(LogLevel, "CRITICAL")


class TestLogHandler:
    """Tests for the LogHandler enum."""
    
    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        assert hasattr(LogHandler, "CONSOLE")
        assert hasattr(LogHandler, "FILE")
        assert hasattr(LogHandler, "JSON")


class TestSetupLogger:
    """Tests for the setup_logger function."""
    
    def test_valid_input(self, temp_dir: str) -> None:
        """Test logger setup with valid input."""
        logger = setup_logger(
            name="test_logger",
            level=LogLevel.INFO,
            handlers=[LogHandler.CONSOLE],
            log_dir=temp_dir
        )
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
    
    def test_invalid_name_type(self) -> None:
        """Test that non-string name is rejected."""
        with pytest.raises(AssertionError):
            setup_logger(name=123)  # type: ignore
    
    def test_invalid_level_type(self) -> None:
        """Test that invalid level types are rejected."""
        # Not a LogLevel, string, or int
        with pytest.raises(AssertionError):
            setup_logger(level={"level": "info"})  # type: ignore
    
    def test_invalid_level_string(self) -> None:
        """Test that invalid level strings are rejected."""
        with pytest.raises(ValueError):
            setup_logger(level="INVALID")
    
    def test_invalid_level_int(self) -> None:
        """Test that invalid level integers are rejected."""
        with pytest.raises(ValueError):
            setup_logger(level=123456)
    
    def test_invalid_handlers_type(self) -> None:
        """Test that non-list handlers is rejected."""
        with pytest.raises(AssertionError):
            setup_logger(handlers="CONSOLE")  # type: ignore
    
    def test_invalid_handler_type(self) -> None:
        """Test that non-LogHandler handlers are rejected."""
        with pytest.raises(AssertionError):
            setup_logger(handlers=["CONSOLE"])  # type: ignore
    
    def test_invalid_log_dir_type(self) -> None:
        """Test that non-string log_dir is rejected."""
        with pytest.raises(AssertionError):
            setup_logger(log_dir=123)  # type: ignore
    
    def test_invalid_log_format_type(self) -> None:
        """Test that non-string log_format is rejected."""
        with pytest.raises(AssertionError):
            setup_logger(log_format=123)  # type: ignore
    
    def test_invalid_date_format_type(self) -> None:
        """Test that non-string date_format is rejected."""
        with pytest.raises(AssertionError):
            setup_logger(date_format=123)  # type: ignore
    
    def test_file_handler_without_log_dir(self) -> None:
        """Test that file handler requires log_dir."""
        with pytest.raises(ValueError):
            setup_logger(handlers=[LogHandler.FILE])
    
    def test_json_handler_without_log_dir(self) -> None:
        """Test that JSON handler requires log_dir."""
        with pytest.raises(ValueError):
            setup_logger(handlers=[LogHandler.JSON])


class TestGetLogger:
    """Tests for the get_logger function."""
    
    def test_returns_logger(self) -> None:
        """Test that get_logger returns a logger."""
        logger = get_logger()
        
        assert isinstance(logger, logging.Logger)


class TestSetLogLevel:
    """Tests for the set_log_level function."""
    
    def test_valid_input(self) -> None:
        """Test setting log level with valid input."""
        # Setup a logger first
        setup_logger(name="test_logger", level=LogLevel.INFO)
        
        # Set the log level
        set_log_level(LogLevel.DEBUG)
        
        # Check that the level was set
        logger = get_logger()
        assert logger.level == logging.DEBUG
    
    def test_invalid_level_type(self) -> None:
        """Test that invalid level types are rejected."""
        # Not a LogLevel, string, or int
        with pytest.raises(AssertionError):
            set_log_level({"level": "info"})  # type: ignore
    
    def test_invalid_level_string(self) -> None:
        """Test that invalid level strings are rejected."""
        with pytest.raises(ValueError):
            set_log_level("INVALID")
    
    def test_invalid_level_int(self) -> None:
        """Test that invalid level integers are rejected."""
        with pytest.raises(ValueError):
            set_log_level(123456)


class TestLogExecutionTime:
    """Tests for the log_execution_time decorator."""
    
    def test_valid_input(self) -> None:
        """Test the decorator with a valid function."""
        # Setup a logger first
        setup_logger(name="test_logger", level=LogLevel.DEBUG)
        
        # Define a function with the decorator
        @log_execution_time
        def test_function(x: int, y: int) -> int:
            return x + y
        
        # Call the function
        result = test_function(1, 2)
        
        # Check that the function still works
        assert result == 3
    
    def test_function_with_error(self) -> None:
        """Test the decorator with a function that raises an error."""
        # Setup a logger first
        setup_logger(name="test_logger", level=LogLevel.DEBUG)
        
        # Define a function with the decorator that raises an error
        @log_execution_time
        def error_function() -> None:
            raise ValueError("Test error")
        
        # Call the function
        with pytest.raises(ValueError):
            error_function()