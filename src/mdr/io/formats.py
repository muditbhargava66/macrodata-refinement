"""
Format utilities for Macrodata Refinement (MDR).

This module provides functions for detecting, validating, and converting
between different data formats.
"""

import os
import mimetypes
from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np
import pandas as pd
from enum import Enum, auto
import json
import csv
import tempfile
from datetime import datetime
import warnings

# Initialize mimetypes
mimetypes.init()


class FormatType(Enum):
    """Supported data format types."""
    
    CSV = auto()
    JSON = auto()
    EXCEL = auto()
    PARQUET = auto()
    HDF5 = auto()
    UNKNOWN = auto()


def detect_format(filepath: str) -> FormatType:
    """
    Detect the format of a file based on its extension or content.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Detected format type
    """
    assert isinstance(filepath, str), "filepath must be a string"
    
    # Check if the file exists
    if not os.path.isfile(filepath):
        raise ValueError(f"File does not exist: {filepath}")
    
    # Get the file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    # Map extensions to format types
    if ext in ['.csv', '.tsv', '.txt']:
        return FormatType.CSV
    elif ext in ['.json']:
        return FormatType.JSON
    elif ext in ['.xls', '.xlsx', '.xlsm', '.xlsb']:
        return FormatType.EXCEL
    elif ext in ['.parquet']:
        return FormatType.PARQUET
    elif ext in ['.h5', '.hdf5', '.he5']:
        return FormatType.HDF5
    
    # If extension is not recognized, try to identify by content
    try:
        with open(filepath, 'rb') as f:
            content = f.read(4096)  # Read the first 4KB
            
            # Check for Excel file signatures
            if content.startswith(b'\x50\x4B\x03\x04') or content.startswith(b'\xD0\xCF\x11\xE0'):
                return FormatType.EXCEL
            
            # Check for Parquet file signature
            if content.startswith(b'PAR1'):
                return FormatType.PARQUET
            
            # Check for HDF5 file signature
            if content.startswith(b'\x89HDF\r\n\x1a\n'):
                return FormatType.HDF5
            
            # Try to decode as text
            try:
                text_content = content.decode('utf-8')
                
                # Check for JSON format
                if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                    try:
                        json.loads(text_content)
                        return FormatType.JSON
                    except json.JSONDecodeError:
                        pass
                
                # Check for CSV format by detecting commas or tabs
                if ',' in text_content or '\t' in text_content:
                    # Check if it has a consistent number of fields
                    lines = text_content.split('\n')
                    if lines:
                        first_line_fields = len(lines[0].split(','))
                        consistent = all(len(line.split(',')) == first_line_fields for line in lines[1:3] if line.strip())
                        if consistent:
                            return FormatType.CSV
            except UnicodeDecodeError:
                pass
    
    except IOError:
        pass
    
    # If we can't determine the format, return UNKNOWN
    return FormatType.UNKNOWN


def validate_format(
    filepath: str,
    expected_format: FormatType
) -> bool:
    """
    Validate if a file has the expected format.
    
    Args:
        filepath: Path to the file
        expected_format: Expected format type
        
    Returns:
        True if the file has the expected format, False otherwise
    """
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(expected_format, FormatType), "expected_format must be a FormatType enum"
    
    try:
        detected_format = detect_format(filepath)
        return detected_format == expected_format
    except Exception:
        return False


def convert_format(
    data: Dict[str, np.ndarray],
    source_format: FormatType,
    target_format: FormatType,
    **options
) -> bytes:
    """
    Convert data from one format to another.
    
    Args:
        data: Dictionary mapping column names to data arrays
        source_format: Source format type
        target_format: Target format type
        **options: Additional options for the conversion
        
    Returns:
        Converted data as bytes
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    assert isinstance(source_format, FormatType), "source_format must be a FormatType enum"
    assert isinstance(target_format, FormatType), "target_format must be a FormatType enum"
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Use a temporary file to hold the converted data
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Write to the target format
        if target_format == FormatType.CSV:
            delimiter = options.get('delimiter', ',')
            float_format = options.get('float_format', '%.6f')
            
            assert isinstance(delimiter, str), "delimiter must be a string"
            assert len(delimiter) == 1, "delimiter must be a single character"
            assert isinstance(float_format, str), "float_format must be a string"
            
            df.to_csv(temp_path, sep=delimiter, index=False, float_format=float_format)
            
        elif target_format == FormatType.JSON:
            orient = options.get('orient', 'columns')
            date_format = options.get('date_format', 'iso')
            indent = options.get('indent', 4)
            
            assert isinstance(orient, str), "orient must be a string"
            assert orient in ["columns", "records", "index", "split", "values"], \
                "orient must be one of ['columns', 'records', 'index', 'split', 'values']"
            assert isinstance(date_format, str), "date_format must be a string"
            
            if indent is not None:
                assert isinstance(indent, int), "indent must be an integer"
                assert indent >= 0, "indent must be a non-negative integer"
            
            df.to_json(temp_path, orient=orient, date_format=date_format, indent=indent)
            
        elif target_format == FormatType.EXCEL:
            sheet_name = options.get('sheet_name', 'Sheet1')
            assert isinstance(sheet_name, str), "sheet_name must be a string"
            
            df.to_excel(temp_path, sheet_name=sheet_name, index=False)
            
        elif target_format == FormatType.PARQUET:
            compression = options.get('compression', 'snappy')
            assert isinstance(compression, str), "compression must be a string"
            
            df.to_parquet(temp_path, compression=compression, index=False)
            
        elif target_format == FormatType.HDF5:
            key = options.get('key', 'data')
            complevel = options.get('complevel', 9)
            complib = options.get('complib', 'zlib')
            
            assert isinstance(key, str), "key must be a string"
            
            if complevel is not None:
                assert isinstance(complevel, int), "complevel must be an integer"
                assert 0 <= complevel <= 9, "complevel must be an integer between 0 and 9"
            
            if complib is not None:
                assert isinstance(complib, str), "complib must be a string"
            
            df.to_hdf(temp_path, key=key, complevel=complevel, complib=complib)
            
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        # Read the converted data
        with open(temp_path, 'rb') as f:
            converted_data = f.read()
        
        return converted_data
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def convert_file_format(
    source_filepath: str,
    target_filepath: str,
    **options
) -> None:
    """
    Convert a file from one format to another.
    
    Args:
        source_filepath: Path to the source file
        target_filepath: Path to the target file
        **options: Additional options for the conversion
    """
    assert isinstance(source_filepath, str), "source_filepath must be a string"
    assert isinstance(target_filepath, str), "target_filepath must be a string"
    
    # Detect source format
    source_format = detect_format(source_filepath)
    if source_format == FormatType.UNKNOWN:
        raise ValueError(f"Could not detect format of source file: {source_filepath}")
    
    # Detect target format based on extension
    _, ext = os.path.splitext(target_filepath)
    ext = ext.lower()
    
    if ext in ['.csv', '.tsv', '.txt']:
        target_format = FormatType.CSV
    elif ext in ['.json']:
        target_format = FormatType.JSON
    elif ext in ['.xls', '.xlsx']:
        target_format = FormatType.EXCEL
    elif ext in ['.parquet']:
        target_format = FormatType.PARQUET
    elif ext in ['.h5', '.hdf5']:
        target_format = FormatType.HDF5
    else:
        raise ValueError(f"Unsupported target file extension: {ext}")
    
    # Read the source file
    if source_format == FormatType.CSV:
        delimiter = options.get('source_delimiter', ',')
        assert isinstance(delimiter, str), "source_delimiter must be a string"
        assert len(delimiter) == 1, "source_delimiter must be a single character"
        
        df = pd.read_csv(source_filepath, sep=delimiter)
    
    elif source_format == FormatType.JSON:
        orient = options.get('source_orient', 'columns')
        assert isinstance(orient, str), "source_orient must be a string"
        assert orient in ["columns", "records", "index", "split", "values"], \
            "source_orient must be one of ['columns', 'records', 'index', 'split', 'values']"
        
        df = pd.read_json(source_filepath, orient=orient)
    
    elif source_format == FormatType.EXCEL:
        sheet_name = options.get('source_sheet_name', 0)
        df = pd.read_excel(source_filepath, sheet_name=sheet_name)
    
    elif source_format == FormatType.PARQUET:
        df = pd.read_parquet(source_filepath)
    
    elif source_format == FormatType.HDF5:
        key = options.get('source_key', None)
        assert key is not None, "source_key must be provided for HDF5 files"
        assert isinstance(key, str), "source_key must be a string"
        
        df = pd.read_hdf(source_filepath, key=key)
    
    # Convert DataFrame to dictionary of numpy arrays
    data = {}
    for column in df.columns:
        data[str(column)] = df[column].to_numpy()
    
    # Write to the target file
    if target_format == FormatType.CSV:
        delimiter = options.get('target_delimiter', ',')
        float_format = options.get('float_format', '%.6f')
        
        assert isinstance(delimiter, str), "target_delimiter must be a string"
        assert len(delimiter) == 1, "target_delimiter must be a single character"
        assert isinstance(float_format, str), "float_format must be a string"
        
        df.to_csv(target_filepath, sep=delimiter, index=False, float_format=float_format)
    
    elif target_format == FormatType.JSON:
        orient = options.get('target_orient', 'columns')
        date_format = options.get('date_format', 'iso')
        indent = options.get('indent', 4)
        
        assert isinstance(orient, str), "target_orient must be a string"
        assert orient in ["columns", "records", "index", "split", "values"], \
            "target_orient must be one of ['columns', 'records', 'index', 'split', 'values']"
        assert isinstance(date_format, str), "date_format must be a string"
        
        if indent is not None:
            assert isinstance(indent, int), "indent must be an integer"
            assert indent >= 0, "indent must be a non-negative integer"
        
        df.to_json(target_filepath, orient=orient, date_format=date_format, indent=indent)
    
    elif target_format == FormatType.EXCEL:
        sheet_name = options.get('target_sheet_name', 'Sheet1')
        assert isinstance(sheet_name, str), "target_sheet_name must be a string"
        
        df.to_excel(target_filepath, sheet_name=sheet_name, index=False)
    
    elif target_format == FormatType.PARQUET:
        compression = options.get('compression', 'snappy')
        assert isinstance(compression, str), "compression must be a string"
        
        df.to_parquet(target_filepath, compression=compression, index=False)
    
    elif target_format == FormatType.HDF5:
        key = options.get('target_key', 'data')
        complevel = options.get('complevel', 9)
        complib = options.get('complib', 'zlib')
        
        assert isinstance(key, str), "target_key must be a string"
        
        if complevel is not None:
            assert isinstance(complevel, int), "complevel must be an integer"
            assert 0 <= complevel <= 9, "complevel must be an integer between 0 and 9"
        
        if complib is not None:
            assert isinstance(complib, str), "complib must be a string"
        
        df.to_hdf(target_filepath, key=key, complevel=complevel, complib=complib)


def infer_column_types(
    data: pd.DataFrame
) -> Dict[str, str]:
    """
    Infer the data types of columns in a DataFrame.
    
    Args:
        data: The DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to inferred types
    """
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    
    type_map = {}
    
    for column in data.columns:
        col_data = data[column]
        
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(col_data):
            # Check if it's integer-like
            if pd.api.types.is_integer_dtype(col_data) or col_data.dropna().apply(lambda x: x.is_integer()).all():
                type_map[str(column)] = "integer"
            else:
                type_map[str(column)] = "float"
                
        # Check if the column is datetime-like
        elif pd.api.types.is_datetime64_dtype(col_data) or (
            # Try to detect datetime format instead of relying on dateutil
            # First check if it's a string or object column
            (pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data)) and 
            # Then try common datetime formats
            try_common_datetime_formats(col_data)
        ):
            type_map[str(column)] = "datetime"
            
        # Check if the column is boolean
        elif pd.api.types.is_bool_dtype(col_data) or set(col_data.dropna().unique()).issubset({True, False, "True", "False", 1, 0}):
            type_map[str(column)] = "boolean"
            
        # Default to string
        else:
            type_map[str(column)] = "string"
    
    return type_map


def cast_column_types(
    data: pd.DataFrame,
    type_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Cast columns in a DataFrame to specified types.
    
    Args:
        data: The DataFrame to modify
        type_map: Dictionary mapping column names to target types
        
    Returns:
        DataFrame with columns cast to specified types
    """
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert isinstance(type_map, dict), "type_map must be a dictionary"
    assert all(isinstance(k, str) for k in type_map.keys()), "All keys in type_map must be strings"
    assert all(isinstance(v, str) for v in type_map.values()), "All values in type_map must be strings"
    
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    for column, target_type in type_map.items():
        if column not in result.columns:
            continue
        
        if target_type == "integer":
            # Convert to integer, replacing NaN with a sentinel value
            result[column] = pd.to_numeric(result[column], errors='coerce').fillna(0).astype(int)
            
        elif target_type == "float":
            # Convert to float
            result[column] = pd.to_numeric(result[column], errors='coerce').astype(float)
            
        elif target_type == "datetime":
            # Convert to datetime with warning suppressed
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                result[column] = pd.to_datetime(result[column], errors='coerce')
            
        elif target_type == "boolean":
            # Convert to boolean
            result[column] = result[column].map({'True': True, 'False': False, '1': True, '0': False, 1: True, 0: False})
            result[column] = result[column].astype(bool)
            
        elif target_type == "string":
            # Convert to string
            result[column] = result[column].astype(str)
    
    return result


def is_numeric_column(data: np.ndarray) -> bool:
    """
    Check if a numpy array contains numeric data.
    
    Args:
        data: The array to check
        
    Returns:
        True if the array contains numeric data, False otherwise
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    
    # Check if the array's dtype is numeric
    if np.issubdtype(data.dtype, np.number):
        return True
    
    # For object dtypes, try to convert to float and check for success
    if data.dtype == np.dtype('O'):
        try:
            numeric_data = pd.to_numeric(data, errors='coerce')
            # If we have more than 80% non-NaN values after conversion, consider it numeric
            return numeric_data.notna().mean() > 0.8
        except Exception:
            return False
    
    return False


def try_common_datetime_formats(col_data: pd.Series) -> bool:
    """
    Try to parse a column with common datetime formats.
    
    Args:
        col_data: The pandas Series to check
        
    Returns:
        True if the column contains datetime data, False otherwise
    """
    # Get a sample of the column (up to 100 non-null values) to check formats
    sample = col_data.dropna().head(100)
    if len(sample) == 0:
        return False
    
    # Common datetime formats to try
    formats = [
        '%Y-%m-%d',                  # 2023-01-31
        '%Y/%m/%d',                  # 2023/01/31
        '%d-%m-%Y',                  # 31-01-2023
        '%d/%m/%Y',                  # 31/01/2023
        '%m-%d-%Y',                  # 01-31-2023
        '%m/%d/%Y',                  # 01/31/2023
        '%Y-%m-%d %H:%M:%S',         # 2023-01-31 14:30:45
        '%Y-%m-%d %H:%M',            # 2023-01-31 14:30
        '%Y/%m/%d %H:%M:%S',         # 2023/01/31 14:30:45
        '%d-%m-%Y %H:%M:%S',         # 31-01-2023 14:30:45
        '%d/%m/%Y %H:%M:%S',         # 31/01/2023 14:30:45
        '%m-%d-%Y %H:%M:%S',         # 01-31-2023 14:30:45
        '%m/%d/%Y %H:%M:%S',         # 01/31/2023 14:30:45
        '%Y-%m-%dT%H:%M:%S',         # 2023-01-31T14:30:45 (ISO format)
        '%Y-%m-%dT%H:%M:%S.%f',      # 2023-01-31T14:30:45.123456
        '%Y%m%d',                    # 20230131
        '%Y%m%d%H%M%S',              # 20230131143045
    ]
    
    # Try each format
    for fmt in formats:
        try:
            # Try to parse all sample values with this format
            success_count = 0
            for val in sample:
                try:
                    if isinstance(val, str):
                        datetime.strptime(val, fmt)
                        success_count += 1
                except ValueError:
                    pass
            
            # If more than 90% of values match this format, consider it a datetime column
            if success_count / len(sample) > 0.9:
                return True
        except Exception:
            continue
    
    # If no format was good enough, fall back to pandas with warning suppressed
    try:
        # Suppress the specific warning about format inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
            datetime_data = pd.to_datetime(sample, errors='coerce')
        
        # If we have more than 80% non-NaN values after conversion, consider it datetime
        return datetime_data.notna().mean() > 0.8
    except Exception:
        return False


def is_datetime_column(data: np.ndarray) -> bool:
    """
    Check if a numpy array contains datetime data.
    
    Args:
        data: The array to check
        
    Returns:
        True if the array contains datetime data, False otherwise
    """
    assert isinstance(data, np.ndarray), "data must be a numpy ndarray"
    
    # Check if the array's dtype is datetime
    if np.issubdtype(data.dtype, np.datetime64):
        return True
    
    # For object dtypes, try common datetime formats
    if data.dtype == np.dtype('O'):
        try:
            # Convert to pandas Series for easier handling
            series = pd.Series(data)
            return try_common_datetime_formats(series)
        except Exception:
            return False
    
    return False