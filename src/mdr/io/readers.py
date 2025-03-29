"""
Data readers for Macrodata Refinement (MDR).

This module provides functions and classes for reading macrodata
from various file formats.
"""

import os
from typing import Dict, List, Union, Optional, Any, Tuple, Type
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from enum import Enum, auto
import json
import csv


class DataSource(Enum):
    """Types of data sources."""
    
    FILE = auto()
    DATABASE = auto()
    API = auto()
    MEMORY = auto()


class DataReader(ABC):
    """Abstract base class for data readers."""
    
    def __init__(self, source_type: DataSource = DataSource.FILE):
        """
        Initialize the data reader.
        
        Args:
            source_type: Type of data source
        """
        assert isinstance(source_type, DataSource), "source_type must be a DataSource enum"
        self.source_type = source_type
    
    @abstractmethod
    def read(self, source: str, **options) -> Dict[str, np.ndarray]:
        """
        Read data from the source.
        
        Args:
            source: Source identifier (file path, table name, etc.)
            **options: Additional reading options
            
        Returns:
            Dictionary mapping variable names to data arrays
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate if the source exists and is readable.
        
        Args:
            source: Source identifier
            
        Returns:
            True if the source is valid, False otherwise
        """
        pass


class FileReader(DataReader):
    """Base class for file-based data readers."""
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the file reader.
        
        Args:
            encoding: File encoding
        """
        super().__init__(DataSource.FILE)
        assert isinstance(encoding, str), "encoding must be a string"
        self.encoding = encoding
    
    def validate_source(self, source: str) -> bool:
        """
        Validate if the file exists and is readable.
        
        Args:
            source: File path
            
        Returns:
            True if the file is valid, False otherwise
        """
        assert isinstance(source, str), "source must be a string"
        return os.path.isfile(source) and os.access(source, os.R_OK)


class CSVReader(FileReader):
    """Reader for CSV files."""
    
    def __init__(
        self,
        delimiter: str = ",",
        quotechar: str = '"',
        encoding: str = "utf-8"
    ):
        """
        Initialize the CSV reader.
        
        Args:
            delimiter: Field delimiter
            quotechar: Character for quoting fields
            encoding: File encoding
        """
        super().__init__(encoding=encoding)
        assert isinstance(delimiter, str), "delimiter must be a string"
        assert isinstance(quotechar, str), "quotechar must be a string"
        assert len(delimiter) == 1, "delimiter must be a single character"
        assert len(quotechar) == 1, "quotechar must be a single character"
        
        self.delimiter = delimiter
        self.quotechar = quotechar
    
    def read(
        self,
        source: str,
        header: bool = True,
        index_col: Optional[Union[int, str]] = None,
        na_values: List[str] = None,
        parse_dates: bool = False,
        **options
    ) -> Dict[str, np.ndarray]:
        """
        Read data from a CSV file.
        
        Args:
            source: File path
            header: Whether to use the first row as column names
            index_col: Column to use as the index
            na_values: List of strings to interpret as NA/NaN
            parse_dates: Whether to parse date columns
            **options: Additional pandas.read_csv options
            
        Returns:
            Dictionary mapping column names to data arrays
        """
        assert isinstance(source, str), "source must be a string"
        assert isinstance(header, bool), "header must be a boolean"
        if index_col is not None:
            assert isinstance(index_col, (int, str)), "index_col must be an integer or string"
        if na_values is not None:
            assert isinstance(na_values, list), "na_values must be a list"
        assert isinstance(parse_dates, bool), "parse_dates must be a boolean"
        
        if not self.validate_source(source):
            raise ValueError(f"Invalid or inaccessible file: {source}")
        
        # Read CSV using pandas
        df = pd.read_csv(
            source,
            delimiter=self.delimiter,
            header=0 if header else None,
            index_col=index_col,
            na_values=na_values,
            parse_dates=parse_dates,
            quotechar=self.quotechar,
            encoding=self.encoding,
            **options
        )
        
        # Convert DataFrame to dictionary of numpy arrays
        data_dict = {}
        for column in df.columns:
            data_dict[str(column)] = df[column].to_numpy()
        
        return data_dict


class JSONReader(FileReader):
    """Reader for JSON files."""
    
    def read(
        self,
        source: str,
        orient: str = "columns",
        convert_dates: bool = True,
        **options
    ) -> Dict[str, np.ndarray]:
        """
        Read data from a JSON file.
        
        Args:
            source: File path
            orient: Expected JSON dict format, one of
                   ['columns', 'records', 'index', 'split', 'values']
            convert_dates: Whether to convert date strings to datetime objects
            **options: Additional pandas.read_json options
            
        Returns:
            Dictionary mapping column names to data arrays
        """
        assert isinstance(source, str), "source must be a string"
        assert isinstance(orient, str), "orient must be a string"
        assert orient in ["columns", "records", "index", "split", "values"], \
            "orient must be one of ['columns', 'records', 'index', 'split', 'values']"
        assert isinstance(convert_dates, bool), "convert_dates must be a boolean"
        
        if not self.validate_source(source):
            raise ValueError(f"Invalid or inaccessible file: {source}")
        
        # Read JSON using pandas
        df = pd.read_json(
            source,
            orient=orient,
            convert_dates=convert_dates,
            **options
        )
        
        # Convert DataFrame to dictionary of numpy arrays
        data_dict = {}
        for column in df.columns:
            data_dict[str(column)] = df[column].to_numpy()
        
        return data_dict


class ExcelReader(FileReader):
    """Reader for Excel files."""
    
    def read(
        self,
        source: str,
        sheet_name: Union[str, int, List, None] = 0,
        header: int = 0,
        na_values: List[str] = None,
        **options
    ) -> Dict[str, np.ndarray]:
        """
        Read data from an Excel file.
        
        Args:
            source: File path
            sheet_name: Name, index, or list of sheets to read
            header: Row to use for column names (0-indexed)
            na_values: List of strings to interpret as NA/NaN
            **options: Additional pandas.read_excel options
            
        Returns:
            Dictionary mapping column names to data arrays
        """
        assert isinstance(source, str), "source must be a string"
        assert isinstance(header, int), "header must be an integer"
        assert header >= 0, "header must be a non-negative integer"
        if na_values is not None:
            assert isinstance(na_values, list), "na_values must be a list"
        
        if not self.validate_source(source):
            raise ValueError(f"Invalid or inaccessible file: {source}")
        
        # Read Excel using pandas
        df = pd.read_excel(
            source,
            sheet_name=sheet_name,
            header=header,
            na_values=na_values,
            **options
        )
        
        # Handle multiple sheets
        if isinstance(df, dict):
            # Return the first sheet if multiple sheets are found
            sheet_name = next(iter(df))
            df = df[sheet_name]
        
        # Convert DataFrame to dictionary of numpy arrays
        data_dict = {}
        for column in df.columns:
            data_dict[str(column)] = df[column].to_numpy()
        
        return data_dict


class ParquetReader(FileReader):
    """Reader for Parquet files."""
    
    def read(
        self,
        source: str,
        columns: Optional[List[str]] = None,
        **options
    ) -> Dict[str, np.ndarray]:
        """
        Read data from a Parquet file.
        
        Args:
            source: File path
            columns: List of columns to read (None for all)
            **options: Additional pandas.read_parquet options
            
        Returns:
            Dictionary mapping column names to data arrays
        """
        assert isinstance(source, str), "source must be a string"
        if columns is not None:
            assert isinstance(columns, list), "columns must be a list"
            for col in columns:
                assert isinstance(col, str), "Each column name must be a string"
        
        if not self.validate_source(source):
            raise ValueError(f"Invalid or inaccessible file: {source}")
        
        try:
            # Try to read Parquet using pandas
            df = pd.read_parquet(
                source,
                columns=columns,
                **options
            )
            
            # Convert DataFrame to dictionary of numpy arrays
            data_dict = {}
            for column in df.columns:
                data_dict[str(column)] = df[column].to_numpy()
            
            return data_dict
            
        except ImportError:
            raise ImportError("pyarrow or fastparquet is required for reading Parquet files")


class HDF5Reader(FileReader):
    """Reader for HDF5 files."""
    
    def read(
        self,
        source: str,
        key: str,
        **options
    ) -> Dict[str, np.ndarray]:
        """
        Read data from an HDF5 file.
        
        Args:
            source: File path
            key: Group identifier in the HDF5 file
            **options: Additional pandas.read_hdf options
            
        Returns:
            Dictionary mapping column names to data arrays
        """
        assert isinstance(source, str), "source must be a string"
        assert isinstance(key, str), "key must be a string"
        
        if not self.validate_source(source):
            raise ValueError(f"Invalid or inaccessible file: {source}")
        
        try:
            # Try to read HDF5 using pandas
            df = pd.read_hdf(
                source,
                key=key,
                **options
            )
            
            # Convert DataFrame to dictionary of numpy arrays
            data_dict = {}
            for column in df.columns:
                data_dict[str(column)] = df[column].to_numpy()
            
            return data_dict
            
        except ImportError:
            raise ImportError("tables is required for reading HDF5 files")


# Factory function to create readers for different file types
def get_reader(file_type: str, **options) -> DataReader:
    """
    Get a reader for the specified file type.
    
    Args:
        file_type: Type of file ('csv', 'json', 'excel', 'parquet', 'hdf5')
        **options: Additional options for the reader
        
    Returns:
        Appropriate DataReader instance
    """
    assert isinstance(file_type, str), "file_type must be a string"
    file_type = file_type.lower()
    
    if file_type == 'csv':
        return CSVReader(**options)
    elif file_type == 'json':
        return JSONReader(**options)
    elif file_type == 'excel' or file_type == 'xlsx' or file_type == 'xls':
        return ExcelReader(**options)
    elif file_type == 'parquet':
        return ParquetReader(**options)
    elif file_type == 'hdf5' or file_type == 'h5':
        return HDF5Reader(**options)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# Convenience functions for reading different file types
def read_csv(
    filepath: str,
    delimiter: str = ",",
    header: bool = True,
    **options
) -> Dict[str, np.ndarray]:
    """
    Read data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        delimiter: Field delimiter
        header: Whether to use the first row as column names
        **options: Additional reading options
        
    Returns:
        Dictionary mapping column names to data arrays
    """
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(delimiter, str), "delimiter must be a string"
    assert len(delimiter) == 1, "delimiter must be a single character"
    assert isinstance(header, bool), "header must be a boolean"
    
    reader = CSVReader(delimiter=delimiter)
    return reader.read(filepath, header=header, **options)


def read_json(
    filepath: str,
    orient: str = "columns",
    **options
) -> Dict[str, np.ndarray]:
    """
    Read data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        orient: Expected JSON dict format
        **options: Additional reading options
        
    Returns:
        Dictionary mapping column names to data arrays
    """
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(orient, str), "orient must be a string"
    assert orient in ["columns", "records", "index", "split", "values"], \
        "orient must be one of ['columns', 'records', 'index', 'split', 'values']"
    
    reader = JSONReader()
    return reader.read(filepath, orient=orient, **options)


def read_excel(
    filepath: str,
    sheet_name: Union[str, int, List, None] = 0,
    **options
) -> Dict[str, np.ndarray]:
    """
    Read data from an Excel file.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Name, index, or list of sheets to read
        **options: Additional reading options
        
    Returns:
        Dictionary mapping column names to data arrays
    """
    assert isinstance(filepath, str), "filepath must be a string"
    
    reader = ExcelReader()
    return reader.read(filepath, sheet_name=sheet_name, **options)


def read_parquet(
    filepath: str,
    columns: Optional[List[str]] = None,
    **options
) -> Dict[str, np.ndarray]:
    """
    Read data from a Parquet file.
    
    Args:
        filepath: Path to the Parquet file
        columns: List of columns to read (None for all)
        **options: Additional reading options
        
    Returns:
        Dictionary mapping column names to data arrays
    """
    assert isinstance(filepath, str), "filepath must be a string"
    if columns is not None:
        assert isinstance(columns, list), "columns must be a list"
        for col in columns:
            assert isinstance(col, str), f"Each column name must be a string, got {type(col)}"
    
    reader = ParquetReader()
    return reader.read(filepath, columns=columns, **options)


def read_hdf5(
    filepath: str,
    key: str,
    **options
) -> Dict[str, np.ndarray]:
    """
    Read data from an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        key: Group identifier in the HDF5 file
        **options: Additional reading options
        
    Returns:
        Dictionary mapping column names to data arrays
    """
    assert isinstance(filepath, str), "filepath must be a string"
    assert isinstance(key, str), "key must be a string"
    
    reader = HDF5Reader()
    return reader.read(filepath, key=key, **options)