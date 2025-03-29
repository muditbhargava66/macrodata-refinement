"""
Input/Output module for Macrodata Refinement (MDR).

This module provides functions and classes for reading and writing
macrodata from and to various file formats.
"""

from mdr.io.readers import (
    read_csv,
    read_json,
    read_excel,
    read_parquet,
    read_hdf5,
    DataReader
)
from mdr.io.writers import (
    write_csv,
    write_json,
    write_excel,
    write_parquet,
    write_hdf5,
    DataWriter
)
from mdr.io.formats import (
    detect_format,
    convert_format,
    validate_format,
    FormatType
)

__all__ = [
    # Readers
    "read_csv",
    "read_json",
    "read_excel",
    "read_parquet",
    "read_hdf5",
    "DataReader",
    
    # Writers
    "write_csv",
    "write_json",
    "write_excel",
    "write_parquet",
    "write_hdf5",
    "DataWriter",
    
    # Format utilities
    "detect_format",
    "convert_format",
    "validate_format",
    "FormatType"
]