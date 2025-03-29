"""
Core functionality for Macrodata Refinement.

This module contains the core components for refining, validating,
and transforming macrodata.
"""

from mdr.core.refinement import (
    refine_data,
    apply_refinement_pipeline,
    RefinementConfig
)
from mdr.core.validation import (
    validate_data,
    check_data_integrity,
    ValidationResult
)
from mdr.core.transformation import (
    transform_data,
    normalize_data,
    scale_data
)

__all__ = [
    "refine_data",
    "apply_refinement_pipeline",
    "RefinementConfig",
    "validate_data",
    "check_data_integrity",
    "ValidationResult",
    "transform_data",
    "normalize_data",
    "scale_data"
]