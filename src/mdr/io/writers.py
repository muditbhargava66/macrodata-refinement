"""
Data writers for Macrodata Refinement (MDR).

This module provides functions and classes for writing macrodata
to various file formats.
"""

import os
from typing import Dict, List, Union, Optional, Any, Tuple, Type, BinaryIO
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from enum import Enum, auto
import json
import csv


class DataDestination(Enum):
    """Types of data destinations."""
    
    FILE = auto()
    DATABASE = auto()
    API = auto()
    MEMORY = auto()


class DataWriter(ABC):
    """Abstract base class for data writers."""
    
    def __init__(self, dest_type: DataDestination = DataDestination.FILE):
        """
        Initialize the data writer.
        
        Args:
            dest_type: Type of data destination
        """
        assert isinstance(dest_type, DataDestination), "dest_type must be a DataDestination enum"
        self.dest_type = dest_type
    
    @abstractmethod
    def write(self, data: Dict[str, np.ndarray], destination: str, **options) -> None:
        """
        Write data to the destination.
        
        Args:
            data: Dictionary mapping variable names to data arrays
            destination: Destination identifier (file path, table name, etc.)
            **options: Additional writing options
        """
        pass
    
    @abstractmethod
    def validate_destination(self, destination: str) -> bool:
        """
        Validate if the destination can be written to.
        
        Args:
            destination: Destination identifier
            
        Returns:
            True if the destination is valid, False otherwise
        """
        pass


class FileWriter(DataWriter):
    """Base class for file-based data writers."""
    
    def __init__(self, encoding: str = "utf-8", overwrite: bool = False):
        """
        Initialize the file writer.
        
        Args:
            encoding: File encoding
            overwrite: Whether to overwrite existing files
        """
        super().__init__(DataDestination.FILE)
        assert isinstance(encoding, str), "encoding must be a string"
        assert isinstance(overwrite, bool), "overwrite must be a boolean"
        
        self.encoding = encoding
        self.overwrite = overwrite
    
    def validate_destination(self, destination: str) -> bool:
        """
        Validate if the file can be written to.
        
        Args:
            destination: File path
            
        Returns:
            True if the file is valid, False otherwise
        """
        assert isinstance(destination, str), "destination must be a string"
        
        # Check if the directory exists
        dir_path = os.path.dirname(destination)
        if dir_path and not os.path.isdir(dir_path):
            return False
        
        # Check if the file exists and if overwriting is allowed
        if os.path.exists(destination):
            if not self.overwrite:
                return False
            if not os.access(destination, os.W_OK):
                return False
        
        return True


class CSVWriter(FileWriter):
    """Writer for CSV files."""
    
    def __init__(
        self,
        delimiter: str = ",",
        quotechar: str = '"',
        encoding: str = "utf-8",
        overwrite: bool = False
    ):
        """
        Initialize the CSV writer.
        
        Args:
            delimiter: Field delimiter
            quotechar: Character for quoting fields
            encoding: File encoding
            overwrite: Whether to overwrite existing files
        """
        super().__init__(encoding=encoding, overwrite=overwrite)
        assert isinstance(delimiter, str), "delimiter must be a string"
        assert isinstance(quotechar, str), "quotechar must be a string"
        assert len(delimiter) == 1, "delimiter must be a single character"
        assert len(quotechar) == 1, "quotechar must be a single character"
        
        self.delimiter = delimiter
        self.quotechar = quotechar
    
    def write(
        self,
        data: Dict[str, np.ndarray],
        destination: str,
        index: bool = False,
        float_format: Optional[str] = "%.6f",
        date_format: Optional[str] = None,
        **options
    ) -> None:
        """
        Write data to a CSV file.
        
        Args:
            data: Dictionary mapping column names to data arrays
            destination: File path
            index: Whether to write row indices
            float_format: Format string for float values
            date_format: Format string for date values
            **options: Additional pandas.to_csv options
        """
        assert isinstance(data, dict), "data must be a dictionary"
        assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
        assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
        assert isinstance(destination, str), "destination must be a string"
        assert isinstance(index, bool), "index must be a boolean"
        
        if float_format is not None:
            assert isinstance(float_format, str), "float_format must be a string"
        
        if date_format is not None:
            assert isinstance(date_format, str), "date_format must be a string"
        
        if not self.validate_destination(destination):
            raise ValueError(f"Invalid or inaccessible destination: {destination}")
        
        # Convert dictionary of arrays to DataFrame
        df = pd.DataFrame(data)
        
        # Write to CSV
        df.to_csv(
            destination,
            sep=self.delimiter,
            index=index,
            quotechar=self.quotechar,
            encoding=self.encoding,
            float_format=float_format,
            date_format=date_format,
            **options
        )


class JSONWriter(FileWriter):
    """Writer for JSON files."""
    
    def write(
        self,
        data: Dict[str, np.ndarray],
        destination: str,
        orient: str = "columns",
        date_format: str = "iso",
        indent: Optional[int] = 4,
        **options
    ) -> None:
        """
        Write data to a JSON file.
        
        Args:
            data: Dictionary mapping column names to data arrays
            destination: File path
            orient: JSON format, one of ['columns', 'records', 'index', 'split', 'values']
            date_format: Format for date values
            indent: Number of spaces for indentation (None for no indentation)
            **options: Additional pandas.to_json options
        """
        assert isinstance(data, dict), "data must be a dictionary"
        assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
        assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
        assert isinstance(destination, str), "destination must be a string"
        assert isinstance(orient, str), "orient must be a string"
        assert orient in ["columns", "records", "index", "split", "values"], \
            "orient must be one of ['columns', 'records', 'index', 'split', 'values']"
        assert isinstance(date_format, str), "date_format must be a string"
        
        if indent is not None:
            assert isinstance(indent, int), "indent must be an integer"
            assert indent >= 0, "indent must be a non-negative integer"
        
        if not self.validate_destination(destination):
            raise ValueError(f"Invalid or inaccessible destination: {destination}")
        
        # Convert dictionary of arrays to DataFrame
        df = pd.DataFrame(data)
        
        # Write to JSON
        df.to_json(
            destination,
            orient=orient,
            date_format=date_format,
            indent=indent,
            **options
        )


class ExcelWriter(FileWriter):
    """Writer for Excel files."""
    
    def write(
        self,
        data: Dict[str, np.ndarray],
        destination: str,
        sheet_name: str = "Sheet1",
        float_format: Optional[str] = "%.6f",
        freeze_panes: Optional[Tuple[int, int]] = None,
        **options
    ) -> None:
        """
        Write data to an Excel file.
        
        Args:
            data: Dictionary mapping column names to data arrays
            destination: File path
            sheet_name: Name of the sheet
            float_format: Format string for float values
            freeze_panes: Tuple of (rows, cols) to freeze
            **options: Additional pandas.to_excel options
        """
        assert isinstance(data, dict), "data must be a dictionary"
        assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
        assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
        assert isinstance(destination, str), "destination must be a string"
        assert isinstance(sheet_name, str), "sheet_name must be a string"
        
        if float_format is not None:
            assert isinstance(float_format, str), "float_format must be a string"
        
        if freeze_panes is not None:
            assert isinstance(freeze_panes, tuple), "freeze_panes must be a tuple"
            assert len(freeze_panes) == 2, "freeze_panes must be a tuple of length 2"
            assert isinstance(freeze_panes[0], int), "freeze_panes[0] must be an integer"
            assert isinstance(freeze_panes[1], int), "freeze_panes[1] must be an integer"
            assert freeze_panes[0] >= 0, "freeze_panes[0] must be a non-negative integer"
            assert freeze_panes[1] >= 0, "freeze_panes[1] must be a non-negative integer"
        
        if not self.validate_destination(destination):
            raise ValueError(f"Invalid or inaccessible destination: {destination}")
        
        # Convert dictionary of arrays to DataFrame
        df = pd.DataFrame(data)
        
        # Write to Excel
        df.to_excel(
            destination,
            sheet_name=sheet_name,
            float_format=float_format,
            freeze_panes=freeze_panes,
            **options
        )


class ParquetWriter(FileWriter):
    """Writer for Parquet files."""
    
    def write(
        self,
        data: Dict[str, np.ndarray],
        destination: str,
        compression: str = "snappy",
        index: bool = False,
        **options
    ) -> None:
        """
        Write data to a Parquet file.
        
        Args:
            data: Dictionary mapping column names to data arrays
            destination: File path
            compression: Compression method
            index: Whether to include row indices
            **options: Additional pandas.to_parquet options
        """
        assert isinstance(data, dict), "data must be a dictionary"
        assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
        assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
        assert isinstance(destination, str), "destination must be a string"
        assert isinstance(compression, str), "compression must be a string"
        assert isinstance(index, bool), "index must be a boolean"
        
        if not self.validate_destination(destination):
            raise ValueError(f"Invalid or inaccessible destination: {destination}")
        
        try:
            # Convert dictionary of arrays to DataFrame
            df = pd.DataFrame(data)
            
            # Write to Parquet
            df.to_parquet(
                destination,
                compression=compression,
                index=index,
                **options
            )
            
        except ImportError:
            raise ImportError("pyarrow or fastparquet is required for writing Parquet files")


class HDF5Writer(FileWriter):
    """Writer for HDF5 files."""
    
    def write(
        self,
        data: Dict[str, np.ndarray],
        destination: str,
        key: str,
        mode: str = "a",
        complevel: Optional[int] = 9,
        complib: Optional[str] = "zlib",
        **options
    ) -> None:
        """
        Write data to an HDF5 file.
        
        Args:
            data: Dictionary mapping column names to data arrays
            destination: File path
            key: Group identifier in the HDF5 file
            mode: File open mode ('a' for append, 'w' for write)
            complevel: Compression level (0-9, 0 for no compression)
            complib: Compression library
            **options: Additional pandas.to_hdf options
        """
        assert isinstance(data, dict), "data must be a dictionary"
        assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
        assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
        assert isinstance(destination, str), "destination must be a string"
        assert isinstance(key, str), "key must be a string"
        assert isinstance(mode, str), "mode must be a string"
        assert mode in ["a", "w"], "mode must be 'a' (append) or 'w' (write)"
        
        if complevel is not None:
            assert isinstance(complevel, int), "complevel must be an integer"
            assert 0 <= complevel <= 9, "complevel must be an integer between 0 and 9"
        
        if complib is not None:
            assert isinstance(complib, str), "complib must be a string"
        
        if not self.validate_destination(destination):
            raise ValueError(f"Invalid or inaccessible destination: {destination}")
        
        try:
            # Convert dictionary of arrays to DataFrame
            df = pd.DataFrame(data)
            
            # Write to HDF5
            df.to_hdf(
                destination,
                key=key,
                mode=mode,
                complevel=complevel,
                complib=complib,
                **options
            )
            
        except ImportError:
            raise ImportError("tables is required for writing HDF5 files")


# Factory function to create writers for different file types
def get_writer(file_type: str, **options) -> DataWriter:
    """
    Get a writer for the specified file type.
    
    Args:
        file_type: Type of file ('csv', 'json', 'excel', 'parquet', 'hdf5')
        **options: Additional options for the writer
        
    Returns:
        Appropriate DataWriter instance
    """
    assert isinstance(file_type, str), "file_type must be a string"
    file_type = file_type.lower()
    
    if file_type == 'csv':
        return CSVWriter(**options)
    elif file_type == 'json':
        return JSONWriter(**options)
    elif file_type == 'excel' or file_type == 'xlsx' or file_type == 'xls':
        return ExcelWriter(**options)
    elif file_type == 'parquet':
        return ParquetWriter(**options)
    elif file_type == 'hdf5' or file_type == 'h5':
        return HDF5Writer(**options)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# Convenience functions for writing data to different file formats
def write_csv(
    data: Dict[str, np.ndarray],
    filepath: str,
    delimiter: str = ",",
    float_format: str = "%.6f",
    **options
) -> None:
    """
    Write data to a CSV file.
    
    Args:
        data: Dictionary mapping column names to data arrays
        filepath: Path to the CSV file
        delimiter: Field delimiter
        float_format: Format string for float values
        **options: Additional writing options
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(delimiter, str), "delimiter must be a string"
    assert len(delimiter) == 1, "delimiter must be a single character"
    assert isinstance(float_format, str), "float_format must be a string"
    
    writer = CSVWriter(delimiter=delimiter, overwrite=True)
    writer.write(data, filepath, float_format=float_format, **options)


def write_json(
    data: Dict[str, np.ndarray],
    filepath: str,
    orient: str = "columns",
    **options
) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Dictionary mapping column names to data arrays
        filepath: Path to the JSON file
        orient: JSON format
        **options: Additional writing options
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(orient, str), "orient must be a string"
    assert orient in ["columns", "records", "index", "split", "values"], \
        "orient must be one of ['columns', 'records', 'index', 'split', 'values']"
    
    writer = JSONWriter(overwrite=True)
    writer.write(data, filepath, orient=orient, **options)


def write_excel(
    data: Dict[str, np.ndarray],
    filepath: str,
    sheet_name: str = "Sheet1",
    **options
) -> None:
    """
    Write data to an Excel file.
    
    Args:
        data: Dictionary mapping column names to data arrays
        filepath: Path to the Excel file
        sheet_name: Name of the sheet
        **options: Additional writing options
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(sheet_name, str), "sheet_name must be a string"
    
    writer = ExcelWriter(overwrite=True)
    writer.write(data, filepath, sheet_name=sheet_name, **options)


def write_parquet(
    data: Dict[str, np.ndarray],
    filepath: str,
    compression: str = "snappy",
    **options
) -> None:
    """
    Write data to a Parquet file.
    
    Args:
        data: Dictionary mapping column names to data arrays
        filepath: Path to the Parquet file
        compression: Compression method
        **options: Additional writing options
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(compression, str), "compression must be a string"
    
    writer = ParquetWriter(overwrite=True)
    writer.write(data, filepath, compression=compression, **options)


def write_hdf5(
    data: Dict[str, np.ndarray],
    filepath: str,
    key: str,
    **options
) -> None:
    """
    Write data to an HDF5 file.
    
    Args:
        data: Dictionary mapping column names to data arrays
        filepath: Path to the HDF5 file
        key: Group identifier in the HDF5 file
        **options: Additional writing options
    """
    assert isinstance(data, dict), "data must be a dictionary"
    assert all(isinstance(k, str) for k in data.keys()), "All keys in data must be strings"
    assert all(isinstance(v, np.ndarray) for v in data.values()), "All values in data must be numpy arrays"
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(key, str), "key must be a string"
    
    writer = HDF5Writer(overwrite=True)
    writer.write(data, filepath, key=key, **options)