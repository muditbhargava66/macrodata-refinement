"""
Unit tests for the core modules of Macrodata Refinement (MDR).
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

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
    normalize_data,
    scale_data,
    apply_logarithmic_transform,
    apply_power_transform,
    transform_data,
    NormalizationType
)


# ---- Refinement Module Tests ----

class TestRefinementConfig:
    """Tests for the RefinementConfig class."""
    
    def test_valid_config(self) -> None:
        """Test that a valid configuration is accepted."""
        config = RefinementConfig(
            smoothing_factor=0.2,
            outlier_threshold=3.0,
            imputation_method="mean",
            normalization_type="minmax"
        )
        
        assert config.smoothing_factor == 0.2
        assert config.outlier_threshold == 3.0
        assert config.imputation_method == "mean"
        assert config.normalization_type == "minmax"
    
    def test_invalid_smoothing_factor_type(self) -> None:
        """Test that non-float smoothing factors are rejected."""
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor="0.2",  # type: ignore
                outlier_threshold=3.0,
                imputation_method="mean",
                normalization_type="minmax"
            )
    
    def test_invalid_smoothing_factor_range(self) -> None:
        """Test that out-of-range smoothing factors are rejected."""
        # Too small
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor=0.0,  # Must be > 0
                outlier_threshold=3.0,
                imputation_method="mean",
                normalization_type="minmax"
            )
        
        # Too large
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor=1.1,  # Must be <= 1
                outlier_threshold=3.0,
                imputation_method="mean",
                normalization_type="minmax"
            )
    
    def test_invalid_outlier_threshold_type(self) -> None:
        """Test that non-float outlier thresholds are rejected."""
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor=0.2,
                outlier_threshold="3.0",  # type: ignore
                imputation_method="mean",
                normalization_type="minmax"
            )
    
    def test_invalid_outlier_threshold_range(self) -> None:
        """Test that negative outlier thresholds are rejected."""
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor=0.2,
                outlier_threshold=0.0,  # Must be > 0
                imputation_method="mean",
                normalization_type="minmax"
            )
    
    def test_invalid_imputation_method_type(self) -> None:
        """Test that non-string imputation methods are rejected."""
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor=0.2,
                outlier_threshold=3.0,
                imputation_method=123,  # type: ignore
                normalization_type="minmax"
            )
    
    def test_invalid_normalization_type_type(self) -> None:
        """Test that non-string normalization types are rejected."""
        with pytest.raises(AssertionError):
            RefinementConfig(
                smoothing_factor=0.2,
                outlier_threshold=3.0,
                imputation_method="mean",
                normalization_type=123  # type: ignore
            )


class TestSmoothData:
    """Tests for the smooth_data function."""
    
    def test_valid_input(self) -> None:
        """Test smoothing with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth_data(data, factor=0.5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            smooth_data([1.0, 2.0, 3.0], factor=0.5)  # type: ignore
    
    def test_invalid_factor_type(self) -> None:
        """Test that non-float factors are rejected."""
        with pytest.raises(AssertionError):
            smooth_data(np.array([1.0, 2.0, 3.0]), factor="0.5")  # type: ignore
    
    def test_invalid_factor_range(self) -> None:
        """Test that out-of-range factors are rejected."""
        data = np.array([1.0, 2.0, 3.0])
        
        # Too small
        with pytest.raises(AssertionError):
            smooth_data(data, factor=0.0)
        
        # Too large
        with pytest.raises(AssertionError):
            smooth_data(data, factor=1.1)
    
    def test_smoothing_effect(self) -> None:
        """Test that smoothing reduces variance."""
        # Create noisy data
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.5, 100)
        
        # Apply smoothing with different factors
        smoothed_low = smooth_data(data, factor=0.1)
        smoothed_high = smooth_data(data, factor=0.9)
        
        # Lower factor should result in more smoothing (less variance)
        assert np.var(smoothed_low) < np.var(data)
        assert np.var(smoothed_low) < np.var(smoothed_high)


class TestRemoveOutliers:
    """Tests for the remove_outliers function."""
    
    def test_valid_input(self) -> None:
        """Test outlier removal with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = remove_outliers(data, threshold=2.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert result[4] != 100.0  # The outlier should be replaced
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            remove_outliers([1.0, 2.0, 3.0], threshold=2.0)  # type: ignore
    
    def test_invalid_threshold_type(self) -> None:
        """Test that non-float thresholds are rejected."""
        with pytest.raises(AssertionError):
            remove_outliers(np.array([1.0, 2.0, 3.0]), threshold="2.0")  # type: ignore
    
    def test_invalid_threshold_range(self) -> None:
        """Test that non-positive thresholds are rejected."""
        data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(AssertionError):
            remove_outliers(data, threshold=0.0)
    
    def test_outlier_detection(self) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_outliers
        sample_data_with_outliers = create_sample_data_with_outliers()
        """Test that outliers are correctly identified and replaced."""
        data = sample_data_with_outliers["linear"]
        original_max = np.max(data)
        
        result = remove_outliers(data, threshold=2.0)
        
        # The maximum value should now be smaller (outlier replaced)
        assert np.max(result) < original_max


class TestImputeMissingValues:
    """Tests for the impute_missing_values function."""
    
    def test_valid_input(self) -> None:
        """Test imputation with valid input."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = impute_missing_values(data, method="mean")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert not np.isnan(result).any()  # No missing values in result
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            impute_missing_values([1.0, 2.0, np.nan], method="mean")  # type: ignore
    
    def test_invalid_method_type(self) -> None:
        """Test that non-string methods are rejected."""
        with pytest.raises(AssertionError):
            impute_missing_values(np.array([1.0, 2.0, np.nan]), method=123)  # type: ignore
    
    def test_invalid_method_value(self) -> None:
        """Test that invalid method values are rejected."""
        with pytest.raises(ValueError):
            impute_missing_values(np.array([1.0, 2.0, np.nan]), method="invalid")
    
    def test_method_mean(self) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_missing
        sample_data_with_missing = create_sample_data_with_missing()
        """Test mean imputation."""
        data = sample_data_with_missing["linear"]
        original_mean = np.nanmean(data)
        
        result = impute_missing_values(data, method="mean")
        
        # All missing values should now be filled with the mean
        assert not np.isnan(result).any()
        missing_mask = np.isnan(data)
        assert np.allclose(result[missing_mask], original_mean)
    
    def test_method_median(self) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_missing
        sample_data_with_missing = create_sample_data_with_missing()
        """Test median imputation."""
        data = sample_data_with_missing["linear"]
        original_median = np.nanmedian(data)
        
        result = impute_missing_values(data, method="median")
        
        # All missing values should now be filled with the median
        assert not np.isnan(result).any()
        missing_mask = np.isnan(data)
        assert np.allclose(result[missing_mask], original_median)


class TestRefineData:
    """Tests for the refine_data function."""
    
    def test_valid_input(self, refinement_config: RefinementConfig) -> None:
        """Test data refinement with valid input."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 100.0])
        result = refine_data(data, refinement_config)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert not np.isnan(result).any()  # No missing values in result
        assert np.max(result) < 100.0  # Outlier should be handled
    
    def test_invalid_data_type(self, refinement_config: RefinementConfig) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            refine_data([1.0, 2.0, 3.0], refinement_config)  # type: ignore
    
    def test_invalid_config_type(self) -> None:
        """Test that non-RefinementConfig configs are rejected."""
        with pytest.raises(AssertionError):
            refine_data(np.array([1.0, 2.0, 3.0]), "config")  # type: ignore
    
    def test_end_to_end(self, refinement_config: RefinementConfig) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_outliers_and_missing
        sample_data_with_outliers_and_missing = create_sample_data_with_outliers_and_missing()
        """Test end-to-end refinement with real data."""
        data = sample_data_with_outliers_and_missing["linear"]
        
        result = refine_data(data, refinement_config)
        
        # Check basic properties of the result
        assert result.shape == data.shape
        assert not np.isnan(result).any()  # No missing values
        assert np.max(result) < np.nanmax(data)  # Outliers handled


class TestApplyRefinementPipeline:
    """Tests for the apply_refinement_pipeline function."""
    
    def test_valid_input(self, refinement_config: RefinementConfig) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_outliers_and_missing
        sample_data_with_outliers_and_missing = create_sample_data_with_outliers_and_missing()
        """Test refinement pipeline with valid input."""
        result = apply_refinement_pipeline(sample_data_with_outliers_and_missing, refinement_config)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_data_with_outliers_and_missing.keys())
        
        # Check that all arrays were processed
        for key, value in result.items():
            assert isinstance(value, np.ndarray)
            assert value.shape == sample_data_with_outliers_and_missing[key].shape
            assert not np.isnan(value).any()  # No missing values
    
    def test_invalid_data_type(self, refinement_config: RefinementConfig) -> None:
        """Test that non-dictionary data is rejected."""
        with pytest.raises(AssertionError):
            apply_refinement_pipeline([1.0, 2.0, 3.0], refinement_config)  # type: ignore
    
    def test_invalid_config_type(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test that non-RefinementConfig configs are rejected."""
        with pytest.raises(AssertionError):
            apply_refinement_pipeline(sample_data, "config")  # type: ignore
    
    def test_invalid_data_values(self, refinement_config: RefinementConfig) -> None:
        """Test that non-numpy array values are rejected."""
        with pytest.raises(AssertionError):
            apply_refinement_pipeline({"key": [1.0, 2.0, 3.0]}, refinement_config)  # type: ignore


# ---- Validation Module Tests ----

class TestValidationResult:
    """Tests for the ValidationResult class."""
    
    def test_valid_input(self) -> None:
        """Test validation result with valid input."""
        result = ValidationResult(
            is_valid=True,
            error_messages=[],
            invalid_indices=np.array([]),
            statistics={"mean": 1.0, "std": 0.5}
        )
        
        assert result.is_valid
        assert result.error_messages == []
        assert np.array_equal(result.invalid_indices, np.array([]))
        assert result.statistics == {"mean": 1.0, "std": 0.5}
    
    def test_invalid_is_valid_type(self) -> None:
        """Test that non-bool is_valid is rejected."""
        with pytest.raises(AssertionError):
            ValidationResult(
                is_valid="True",  # type: ignore
                error_messages=[]
            )
    
    def test_invalid_error_messages_type(self) -> None:
        """Test that non-list error_messages is rejected."""
        with pytest.raises(AssertionError):
            ValidationResult(
                is_valid=True,
                error_messages="Error"  # type: ignore
            )
    
    def test_invalid_invalid_indices_type(self) -> None:
        """Test that non-ndarray invalid_indices is rejected."""
        with pytest.raises(AssertionError):
            ValidationResult(
                is_valid=True,
                error_messages=[],
                invalid_indices=[1, 2, 3]  # type: ignore
            )
    
    def test_invalid_statistics_type(self) -> None:
        """Test that non-dict statistics is rejected."""
        with pytest.raises(AssertionError):
            ValidationResult(
                is_valid=True,
                error_messages=[],
                statistics=[1.0, 2.0, 3.0]  # type: ignore
            )
    
    def test_invalid_statistics_value_type(self) -> None:
        """Test that non-float statistics values are rejected."""
        with pytest.raises(AssertionError):
            ValidationResult(
                is_valid=True,
                error_messages=[],
                statistics={"mean": "1.0"}  # type: ignore
            )


class TestCheckDataRange:
    """Tests for the check_data_range function."""
    
    def test_valid_input(self) -> None:
        """Test data range check with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = check_data_range(data, min_value=0.0, max_value=10.0)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.error_messages == []
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            check_data_range([1.0, 2.0, 3.0], min_value=0.0, max_value=10.0)  # type: ignore
    
    def test_invalid_min_value_type(self) -> None:
        """Test that non-float min_value is rejected."""
        with pytest.raises(AssertionError):
            check_data_range(np.array([1.0, 2.0, 3.0]), min_value="0.0", max_value=10.0)  # type: ignore
    
    def test_invalid_max_value_type(self) -> None:
        """Test that non-float max_value is rejected."""
        with pytest.raises(AssertionError):
            check_data_range(np.array([1.0, 2.0, 3.0]), min_value=0.0, max_value="10.0")  # type: ignore
    
    def test_invalid_range(self) -> None:
        """Test that invalid ranges are rejected."""
        with pytest.raises(AssertionError):
            check_data_range(np.array([1.0, 2.0, 3.0]), min_value=10.0, max_value=0.0)
    
    def test_out_of_range(self) -> None:
        """Test detection of out-of-range values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = check_data_range(data, min_value=2.0, max_value=4.0)
        
        assert not result.is_valid
        assert len(result.error_messages) > 0
        assert result.invalid_indices is not None
        # Values outside range are 1.0 and 5.0 (at indices 0 and 4)
        assert len(result.invalid_indices) == 2


class TestCheckMissingValues:
    """Tests for the check_missing_values function."""
    
    def test_valid_input(self) -> None:
        """Test missing values check with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = check_missing_values(data, threshold=0.1)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.error_messages == []
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            check_missing_values([1.0, 2.0, 3.0], threshold=0.1)  # type: ignore
    
    def test_invalid_threshold_type(self) -> None:
        """Test that non-float threshold is rejected."""
        with pytest.raises(AssertionError):
            check_missing_values(np.array([1.0, 2.0, 3.0]), threshold="0.1")  # type: ignore
    
    def test_invalid_threshold_range(self) -> None:
        """Test that out-of-range thresholds are rejected."""
        data = np.array([1.0, 2.0, 3.0])
        
        # Too small
        with pytest.raises(AssertionError):
            check_missing_values(data, threshold=-0.1)
        
        # Too large
        with pytest.raises(AssertionError):
            check_missing_values(data, threshold=1.1)
    
    def test_with_missing_values(self) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_missing
        sample_data_with_missing = create_sample_data_with_missing()
        """Test detection of missing values."""
        data = sample_data_with_missing["sinusoidal"]  # Has 5 missing values (5%)
        
        # Should pass with threshold = 0.1 (10%)
        result1 = check_missing_values(data, threshold=0.1)
        assert result1.is_valid
        
        # Should fail with threshold = 0.01 (1%)
        result2 = check_missing_values(data, threshold=0.01)
        assert not result2.is_valid
        assert len(result2.error_messages) > 0
        assert result2.invalid_indices is not None
        assert len(result2.invalid_indices) == 5  # 5 missing values


class TestCheckDataIntegrity:
    """Tests for the check_data_integrity function."""
    
    def test_valid_input(self) -> None:
        """Test data integrity check with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = check_data_integrity(data, checks=["range", "missing", "outliers"])
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.error_messages == []
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            check_data_integrity([1.0, 2.0, 3.0])  # type: ignore
    
    def test_invalid_checks_type(self) -> None:
        """Test that non-list checks is rejected."""
        with pytest.raises(AssertionError):
            check_data_integrity(np.array([1.0, 2.0, 3.0]), checks="range")  # type: ignore
    
    def test_with_problems(self) -> None:
        # Use the helper function instead of the fixture directly
        from tests.conftest import create_sample_data_with_outliers_and_missing
        sample_data_with_outliers_and_missing = create_sample_data_with_outliers_and_missing()
        """Test detection of various data problems."""
        data = sample_data_with_outliers_and_missing["linear"]
        
        # Check for missing values - use a more strict threshold to trigger a failure
        result1 = check_data_integrity(data, checks=["missing"], params={"missing": {"threshold": 0.001}})
        assert not result1.is_valid
        
        # Use a custom dataset with many outliers instead of relying on the test fixture
        # Create a dataset where 15% of values are outliers to exceed the 10% threshold
        np.random.seed(42)  # For reproducibility
        outlier_data = np.ones(100)  # 100 identical values
        outlier_indices = np.random.choice(range(100), 15, replace=False)  # 15% of indices
        outlier_data[outlier_indices] = 100.0  # Make these values outliers
        
        # Check for outliers with this custom dataset
        result2 = check_data_integrity(outlier_data, checks=["outliers"], params={"outliers": {"threshold": 0.5}})
        assert not result2.is_valid
        
        # Check for range violations
        result3 = check_data_integrity(data, checks=["range"], params={"range": {"min_value": 0.0, "max_value": 10.0}})
        assert not result3.is_valid


class TestValidateData:
    """Tests for the validate_data function."""
    
    def test_valid_input(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test data validation with valid input."""
        result = validate_data(sample_data, checks=["range", "missing", "outliers"])
        
        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_data.keys())
        
        # Check that all arrays were validated
        for key, value in result.items():
            assert isinstance(value, ValidationResult)
    
    def test_invalid_data_type(self) -> None:
        """Test that non-dictionary data is rejected."""
        with pytest.raises(AssertionError):
            validate_data([1.0, 2.0, 3.0])  # type: ignore
    
    def test_invalid_checks_type(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test that non-list checks is rejected."""
        with pytest.raises(AssertionError):
            validate_data(sample_data, checks="range")  # type: ignore
    
    def test_with_problems(self) -> None:
        """Test detection of various data problems across multiple variables."""
        # Instead of using validate_data, let's test the check_outliers function directly
        # to have more control over the test
        
        # Create a very simple dataset with clear outliers by IQR definition
        # Values 1-10 with two extreme outliers at 100
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 100])
        
        # Call check_outliers directly
        from mdr.core.validation import check_outliers
        
        # The IQR for this dataset is 5 (Q3=8.75, Q1=3.75)
        # Values outside [Q1-1.5*IQR, Q3+1.5*IQR] = [-3.75, 16.25] are outliers
        # So 100 is clearly an outlier
        result = check_outliers(
            data,
            threshold=1.5,  # Standard for IQR
            method="iqr"    # Use IQR method
        )
        
        print(f"\nDirect check_outliers result: {result}")
        
        # This should absolutely have outliers
        assert not result.is_valid
        assert result.statistics["outlier_count"] > 0
        
        # Now test with validate_data but using very clear outliers
        test_data = {"simple_outliers": data}
        
        validate_result = validate_data(
            test_data,
            checks=["outliers"],
            params={
                "outliers": {
                    "threshold": 1.5,
                    "method": "iqr"
                }
            }
        )
        
        print(f"\nValidate data result: {validate_result['simple_outliers']}")
        
        # The validation should fail for this variable
        assert "simple_outliers" in validate_result
        assert not validate_result["simple_outliers"].is_valid


# ---- Transformation Module Tests ----

class TestNormalizeData:
    """Tests for the normalize_data function."""
    
    def test_valid_input(self) -> None:
        """Test normalization with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result, params = normalize_data(data, method="minmax")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert isinstance(params, dict)
        assert "min" in params and "max" in params
        
        # Check that the result is normalized
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            normalize_data([1.0, 2.0, 3.0], method="minmax")  # type: ignore
    
    def test_invalid_method_type(self) -> None:
        """Test that invalid method types are rejected."""
        with pytest.raises(AssertionError):
            normalize_data(np.array([1.0, 2.0, 3.0]), method=123)  # type: ignore
    
    def test_invalid_method_value(self) -> None:
        """Test that invalid method values are rejected."""
        with pytest.raises(ValueError):
            normalize_data(np.array([1.0, 2.0, 3.0]), method="invalid")
    
    def test_method_minmax(self) -> None:
        """Test min-max normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result, params = normalize_data(data, method="minmax")
        
        # Check that the result is min-max normalized
        assert np.isclose(np.min(result), 0.0)
        assert np.isclose(np.max(result), 1.0)
    
    def test_method_zscore(self) -> None:
        """Test z-score normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result, params = normalize_data(data, method="zscore")
        
        # Check that the result is z-score normalized
        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        assert np.isclose(np.std(result), 1.0)


class TestScaleData:
    """Tests for the scale_data function."""
    
    def test_valid_input(self) -> None:
        """Test scaling with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scale_data(data, factor=2.0, offset=1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        
        # Check that the result is correctly scaled
        expected = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
        assert np.array_equal(result, expected)
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            scale_data([1.0, 2.0, 3.0], factor=2.0)  # type: ignore
    
    def test_invalid_factor_type(self) -> None:
        """Test that non-float factor is rejected."""
        with pytest.raises(AssertionError):
            scale_data(np.array([1.0, 2.0, 3.0]), factor="2.0")  # type: ignore
    
    def test_invalid_offset_type(self) -> None:
        """Test that non-float offset is rejected."""
        with pytest.raises(AssertionError):
            scale_data(np.array([1.0, 2.0, 3.0]), factor=2.0, offset="1.0")  # type: ignore


class TestApplyLogarithmicTransform:
    """Tests for the apply_logarithmic_transform function."""
    
    def test_valid_input(self) -> None:
        """Test logarithmic transformation with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_logarithmic_transform(data, base=10.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        
        # Check that the result is logarithmically transformed
        expected = np.log10(data + 1e-10)
        assert np.allclose(result, expected)
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            apply_logarithmic_transform([1.0, 2.0, 3.0])  # type: ignore
    
    def test_invalid_base_type(self) -> None:
        """Test that non-float base is rejected."""
        with pytest.raises(AssertionError):
            apply_logarithmic_transform(np.array([1.0, 2.0, 3.0]), base="10.0")  # type: ignore
    
    def test_invalid_base_range(self) -> None:
        """Test that non-positive bases are rejected."""
        with pytest.raises(AssertionError):
            apply_logarithmic_transform(np.array([1.0, 2.0, 3.0]), base=0.0)
    
    def test_invalid_epsilon_type(self) -> None:
        """Test that non-float epsilon is rejected."""
        with pytest.raises(AssertionError):
            apply_logarithmic_transform(np.array([1.0, 2.0, 3.0]), epsilon="1e-10")  # type: ignore
    
    def test_invalid_epsilon_range(self) -> None:
        """Test that non-positive epsilon is rejected."""
        with pytest.raises(AssertionError):
            apply_logarithmic_transform(np.array([1.0, 2.0, 3.0]), epsilon=0.0)


class TestApplyPowerTransform:
    """Tests for the apply_power_transform function."""
    
    def test_valid_input(self) -> None:
        """Test power transformation with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_power_transform(data, power=2.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        
        # Check that the result is power transformed
        expected = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        assert np.array_equal(result, expected)
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            apply_power_transform([1.0, 2.0, 3.0], power=2.0)  # type: ignore
    
    def test_invalid_power_type(self) -> None:
        """Test that non-float power is rejected."""
        with pytest.raises(AssertionError):
            apply_power_transform(np.array([1.0, 2.0, 3.0]), power="2.0")  # type: ignore
    
    def test_invalid_preserve_sign_type(self) -> None:
        """Test that non-bool preserve_sign is rejected."""
        with pytest.raises(AssertionError):
            apply_power_transform(np.array([1.0, 2.0, 3.0]), power=2.0, preserve_sign="True")  # type: ignore
    
    def test_negative_values(self) -> None:
        """Test handling of negative values."""
        data = np.array([-1.0, 2.0, -3.0, 4.0, -5.0])
        
        # With preserve_sign=True, signs should be preserved
        result1 = apply_power_transform(data, power=2.0, preserve_sign=True)
        expected1 = np.array([-1.0, 4.0, -9.0, 16.0, -25.0])
        assert np.array_equal(result1, expected1)
        
        # With preserve_sign=False, all values should be positive
        result2 = apply_power_transform(data, power=2.0, preserve_sign=False)
        expected2 = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        assert np.array_equal(result2, expected2)


class TestTransformData:
    """Tests for the transform_data function."""
    
    def test_valid_input(self) -> None:
        """Test data transformation with valid input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformations = [
            {"type": "normalize", "method": "minmax"},
            {"type": "scale", "factor": 2.0, "offset": 1.0}
        ]
        
        result = transform_data(data, transformations)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
    
    def test_invalid_data_type(self) -> None:
        """Test that non-numpy array data is rejected."""
        with pytest.raises(AssertionError):
            transform_data([1.0, 2.0, 3.0], [{"type": "normalize"}])  # type: ignore
    
    def test_invalid_transformations_type(self) -> None:
        """Test that non-list transformations is rejected."""
        with pytest.raises(AssertionError):
            transform_data(np.array([1.0, 2.0, 3.0]), "transformations")  # type: ignore
    
    def test_invalid_transformation_type(self) -> None:
        """Test that non-dict transformations are rejected."""
        with pytest.raises(AssertionError):
            transform_data(np.array([1.0, 2.0, 3.0]), ["transform"])  # type: ignore
    
    def test_missing_transformation_type(self) -> None:
        """Test that transformations without a type field are rejected."""
        with pytest.raises(AssertionError):
            transform_data(np.array([1.0, 2.0, 3.0]), [{"method": "minmax"}])
    
    def test_invalid_transformation_type_value(self) -> None:
        """Test that invalid transformation types are rejected."""
        with pytest.raises(ValueError):
            transform_data(np.array([1.0, 2.0, 3.0]), [{"type": "invalid"}])
    
    def test_multiple_transformations(self) -> None:
        """Test applying multiple transformations."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformations = [
            {"type": "normalize", "method": "minmax"},  # Result: [0.0, 0.25, 0.5, 0.75, 1.0]
            {"type": "scale", "factor": 2.0, "offset": 1.0}  # Result: [1.0, 1.5, 2.0, 2.5, 3.0]
        ]
        
        result = transform_data(data, transformations)
        
        # Calculate expected result manually
        expected = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        assert np.allclose(result, expected)