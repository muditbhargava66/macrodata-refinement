"""
Unit tests for the I/O modules of Macrodata Refinement (MDR).
"""

import pytest
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Any, Tuple, Generator

from mdr.io.readers import (
    CSVReader,
    JSONReader,
    ExcelReader,
    DataReader,
    read_csv,
    read_json
)
from mdr.io.writers import (
    CSVWriter,
    JSONWriter,
    DataWriter,
    write_csv,
    write_json
)
from mdr.io.formats import (
    detect_format,
    convert_format,
    validate_format,
    FormatType,
    infer_column_types,
    cast_column_types
)


# ---- Reader Module Tests ----

class TestCSVReader:
    """Tests for the CSVReader class."""
    
    def test_initialization(self) -> None:
        """Test CSVReader initialization with valid input."""
        reader = CSVReader(delimiter=",", quotechar='"', encoding="utf-8")
        
        assert reader.delimiter == ","
        assert reader.quotechar == '"'
        assert reader.encoding == "utf-8"
    
    def test_invalid_delimiter_type(self) -> None:
        """Test that non-string delimiters are rejected."""
        with pytest.raises(AssertionError):
            CSVReader(delimiter=1)  # type: ignore
    
    def test_invalid_delimiter_length(self) -> None:
        """Test that multi-char delimiters are rejected."""
        with pytest.raises(AssertionError):
            CSVReader(delimiter=",,")
    
    def test_invalid_quotechar_type(self) -> None:
        """Test that non-string quotechars are rejected."""
        with pytest.raises(AssertionError):
            CSVReader(quotechar=1)  # type: ignore
    
    def test_invalid_quotechar_length(self) -> None:
        """Test that multi-char quotechars are rejected."""
        with pytest.raises(AssertionError):
            CSVReader(quotechar='""')
    
    def test_invalid_encoding_type(self) -> None:
        """Test that non-string encodings are rejected."""
        with pytest.raises(AssertionError):
            CSVReader(encoding=1)  # type: ignore
    
    def test_read_valid_file(self, sample_dataframe: pd.DataFrame, temp_csv_file: str) -> None:
        """Test reading a valid CSV file."""
        # Write sample data to CSV
        sample_dataframe.to_csv(temp_csv_file, index=False)
        
        # Read the CSV file
        reader = CSVReader()
        data_dict = reader.read(temp_csv_file)
        
        # Check the result
        assert isinstance(data_dict, dict)
        assert set(data_dict.keys()) == set(sample_dataframe.columns)
        
        # Check that each column was read correctly
        for col in sample_dataframe.columns:
            assert col in data_dict
            assert isinstance(data_dict[col], np.ndarray)
            
            # Check values if the column is numeric
            if pd.api.types.is_numeric_dtype(sample_dataframe[col]):
                assert np.allclose(data_dict[col], sample_dataframe[col].values, equal_nan=True)
    
    def test_invalid_source_type(self) -> None:
        """Test that non-string sources are rejected."""
        reader = CSVReader()
        with pytest.raises(AssertionError):
            reader.read(123)  # type: ignore
    
    def test_nonexistent_file(self) -> None:
        """Test that nonexistent files are rejected."""
        reader = CSVReader()
        with pytest.raises(ValueError):
            reader.read("nonexistent_file.csv")


class TestJSONReader:
    """Tests for the JSONReader class."""
    
    def test_initialization(self) -> None:
        """Test JSONReader initialization with valid input."""
        reader = JSONReader(encoding="utf-8")
        
        assert reader.encoding == "utf-8"
    
    def test_invalid_encoding_type(self) -> None:
        """Test that non-string encodings are rejected."""
        with pytest.raises(AssertionError):
            JSONReader(encoding=1)  # type: ignore
    
    def test_read_valid_file(self, sample_dataframe: pd.DataFrame, temp_dir: str) -> None:
        """Test reading a valid JSON file."""
        # Create a JSON file
        json_path = os.path.join(temp_dir, "test.json")
        sample_dataframe.to_json(json_path, orient="columns")
        
        # Read the JSON file
        reader = JSONReader()
        data_dict = reader.read(json_path, orient="columns")
        
        # Check the result
        assert isinstance(data_dict, dict)
        assert set(data_dict.keys()) == set(sample_dataframe.columns)
        
        # Check that each column was read correctly
        for col in sample_dataframe.columns:
            assert col in data_dict
            assert isinstance(data_dict[col], np.ndarray)
    
    def test_invalid_source_type(self) -> None:
        """Test that non-string sources are rejected."""
        reader = JSONReader()
        with pytest.raises(AssertionError):
            reader.read(123)  # type: ignore
    
    def test_invalid_orient_type(self) -> None:
        """Test that non-string orient is rejected."""
        reader = JSONReader()
        with pytest.raises(AssertionError):
            reader.read("test.json", orient=123)  # type: ignore
    
    def test_invalid_orient_value(self) -> None:
        """Test that invalid orient values are rejected."""
        reader = JSONReader()
        with pytest.raises(AssertionError):
            reader.read("test.json", orient="invalid")


# Helper function to create read_csv convenience function
class TestReadCSV:
    """Tests for the read_csv convenience function."""
    
    def test_valid_input(self, sample_dataframe: pd.DataFrame, temp_csv_file: str) -> None:
        """Test reading a valid CSV file with the convenience function."""
        # Write sample data to CSV
        sample_dataframe.to_csv(temp_csv_file, index=False)
        
        # Read the CSV file
        data_dict = read_csv(temp_csv_file)
        
        # Check the result
        assert isinstance(data_dict, dict)
        assert set(data_dict.keys()) == set(sample_dataframe.columns)
    
    def test_invalid_filepath_type(self) -> None:
        """Test that non-string filepaths are rejected."""
        with pytest.raises(AssertionError):
            read_csv(123)  # type: ignore
    
    def test_invalid_delimiter_type(self) -> None:
        """Test that non-string delimiters are rejected."""
        with pytest.raises(AssertionError):
            read_csv("test.csv", delimiter=1)  # type: ignore
    
    def test_invalid_delimiter_length(self) -> None:
        """Test that multi-char delimiters are rejected."""
        with pytest.raises(AssertionError):
            read_csv("test.csv", delimiter=",,")
    
    def test_invalid_header_type(self) -> None:
        """Test that non-bool header is rejected."""
        with pytest.raises(AssertionError):
            read_csv("test.csv", header="True")  # type: ignore


# ---- Writer Module Tests ----

class TestCSVWriter:
    """Tests for the CSVWriter class."""
    
    def test_initialization(self) -> None:
        """Test CSVWriter initialization with valid input."""
        writer = CSVWriter(delimiter=",", quotechar='"', encoding="utf-8", overwrite=True)
        
        assert writer.delimiter == ","
        assert writer.quotechar == '"'
        assert writer.encoding == "utf-8"
        assert writer.overwrite == True
    
    def test_invalid_delimiter_type(self) -> None:
        """Test that non-string delimiters are rejected."""
        with pytest.raises(AssertionError):
            CSVWriter(delimiter=1)  # type: ignore
    
    def test_invalid_delimiter_length(self) -> None:
        """Test that multi-char delimiters are rejected."""
        with pytest.raises(AssertionError):
            CSVWriter(delimiter=",,")
    
    def test_invalid_quotechar_type(self) -> None:
        """Test that non-string quotechars are rejected."""
        with pytest.raises(AssertionError):
            CSVWriter(quotechar=1)  # type: ignore
    
    def test_invalid_quotechar_length(self) -> None:
        """Test that multi-char quotechars are rejected."""
        with pytest.raises(AssertionError):
            CSVWriter(quotechar='""')
    
    def test_invalid_encoding_type(self) -> None:
        """Test that non-string encodings are rejected."""
        with pytest.raises(AssertionError):
            CSVWriter(encoding=1)  # type: ignore
    
    def test_invalid_overwrite_type(self) -> None:
        """Test that non-bool overwrite is rejected."""
        with pytest.raises(AssertionError):
            CSVWriter(overwrite="True")  # type: ignore
    
    def test_write_valid_data(self, sample_data: Dict[str, np.ndarray], temp_csv_file: str) -> None:
        """Test writing valid data to a CSV file."""
        # Write the data
        writer = CSVWriter(overwrite=True)
        writer.write(sample_data, temp_csv_file)
        
        # Check that the file was created
        assert os.path.exists(temp_csv_file)
        
        # Read the file to verify content
        df = pd.read_csv(temp_csv_file)
        
        # Check that all variables were written
        assert set(df.columns) == set(sample_data.keys())
    
    def test_invalid_data_type(self, temp_csv_file: str) -> None:
        """Test that non-dict data is rejected."""
        writer = CSVWriter()
        with pytest.raises(AssertionError):
            writer.write("data", temp_csv_file)  # type: ignore
    
    def test_invalid_data_key_type(self, temp_csv_file: str) -> None:
        """Test that non-string data keys are rejected."""
        writer = CSVWriter()
        with pytest.raises(AssertionError):
            writer.write({1: np.array([1, 2, 3])}, temp_csv_file)  # type: ignore
    
    def test_invalid_data_value_type(self, temp_csv_file: str) -> None:
        """Test that non-ndarray data values are rejected."""
        writer = CSVWriter()
        with pytest.raises(AssertionError):
            writer.write({"key": [1, 2, 3]}, temp_csv_file)  # type: ignore
    
    def test_invalid_destination_type(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test that non-string destinations are rejected."""
        writer = CSVWriter()
        with pytest.raises(AssertionError):
            writer.write(sample_data, 123)  # type: ignore


class TestJSONWriter:
    """Tests for the JSONWriter class."""
    
    def test_initialization(self) -> None:
        """Test JSONWriter initialization with valid input."""
        writer = JSONWriter(encoding="utf-8", overwrite=True)
        
        assert writer.encoding == "utf-8"
        assert writer.overwrite == True
    
    def test_invalid_encoding_type(self) -> None:
        """Test that non-string encodings are rejected."""
        with pytest.raises(AssertionError):
            JSONWriter(encoding=1)  # type: ignore
    
    def test_invalid_overwrite_type(self) -> None:
        """Test that non-bool overwrite is rejected."""
        with pytest.raises(AssertionError):
            JSONWriter(overwrite="True")  # type: ignore
    
    def test_write_valid_data(
        self, sample_data: Dict[str, np.ndarray], temp_dir: str
    ) -> None:
        """Test writing valid data to a JSON file."""
        # Create a file path
        json_path = os.path.join(temp_dir, "test.json")
        
        # Write the data
        writer = JSONWriter(overwrite=True)
        writer.write(sample_data, json_path, orient="columns")
        
        # Check that the file was created
        assert os.path.exists(json_path)
        
        # Read the file to verify content
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        # Check that all variables were written
        assert set(json_data.keys()) == set(sample_data.keys())
    
    def test_invalid_data_type(self, temp_dir: str) -> None:
        """Test that non-dict data is rejected."""
        json_path = os.path.join(temp_dir, "test.json")
        writer = JSONWriter()
        with pytest.raises(AssertionError):
            writer.write("data", json_path)  # type: ignore
    
    def test_invalid_orient_type(
        self, sample_data: Dict[str, np.ndarray], temp_dir: str
    ) -> None:
        """Test that non-string orient is rejected."""
        json_path = os.path.join(temp_dir, "test.json")
        writer = JSONWriter()
        with pytest.raises(AssertionError):
            writer.write(sample_data, json_path, orient=123)  # type: ignore
    
    def test_invalid_orient_value(
        self, sample_data: Dict[str, np.ndarray], temp_dir: str
    ) -> None:
        """Test that invalid orient values are rejected."""
        json_path = os.path.join(temp_dir, "test.json")
        writer = JSONWriter()
        with pytest.raises(AssertionError):
            writer.write(sample_data, json_path, orient="invalid")


# Helper function to test write_csv convenience function
class TestWriteCSV:
    """Tests for the write_csv convenience function."""
    
    def test_valid_input(self, sample_data: Dict[str, np.ndarray], temp_csv_file: str) -> None:
        """Test writing valid data to a CSV file with the convenience function."""
        # Write the data
        write_csv(sample_data, temp_csv_file)
        
        # Check that the file was created
        assert os.path.exists(temp_csv_file)
        
        # Read the file to verify content
        df = pd.read_csv(temp_csv_file)
        
        # Check that all variables were written
        assert set(df.columns) == set(sample_data.keys())
    
    def test_invalid_data_type(self, temp_csv_file: str) -> None:
        """Test that non-dict data is rejected."""
        with pytest.raises(AssertionError):
            write_csv("data", temp_csv_file)  # type: ignore
    
    def test_invalid_filepath_type(self, sample_data: Dict[str, np.ndarray]) -> None:
        """Test that non-string filepaths are rejected."""
        with pytest.raises(AssertionError):
            write_csv(sample_data, 123)  # type: ignore
    
    def test_invalid_delimiter_type(
        self, sample_data: Dict[str, np.ndarray], temp_csv_file: str
    ) -> None:
        """Test that non-string delimiters are rejected."""
        with pytest.raises(AssertionError):
            write_csv(sample_data, temp_csv_file, delimiter=1)  # type: ignore
    
    def test_invalid_delimiter_length(
        self, sample_data: Dict[str, np.ndarray], temp_csv_file: str
    ) -> None:
        """Test that multi-char delimiters are rejected."""
        with pytest.raises(AssertionError):
            write_csv(sample_data, temp_csv_file, delimiter=",,")
    
    def test_invalid_float_format_type(
        self, sample_data: Dict[str, np.ndarray], temp_csv_file: str
    ) -> None:
        """Test that non-string float_format is rejected."""
        with pytest.raises(AssertionError):
            write_csv(sample_data, temp_csv_file, float_format=123)  # type: ignore


# ---- Formats Module Tests ----

class TestFormatType:
    """Tests for the FormatType enum."""
    
    def test_enum_values(self) -> None:
        """Test that the enum has the expected values."""
        assert hasattr(FormatType, "CSV")
        assert hasattr(FormatType, "JSON")
        assert hasattr(FormatType, "EXCEL")
        assert hasattr(FormatType, "PARQUET")
        assert hasattr(FormatType, "HDF5")
        assert hasattr(FormatType, "UNKNOWN")


class TestDetectFormat:
    """Tests for the detect_format function."""
    
    def test_detect_csv(self, sample_dataframe: pd.DataFrame, temp_dir: str) -> None:
        """Test detecting a CSV file."""
        # Create a CSV file
        csv_path = os.path.join(temp_dir, "test.csv")
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Detect the format
        format_type = detect_format(csv_path)
        
        assert format_type == FormatType.CSV
    
    def test_detect_json(self, sample_dataframe: pd.DataFrame, temp_dir: str) -> None:
        """Test detecting a JSON file."""
        # Create a JSON file
        json_path = os.path.join(temp_dir, "test.json")
        sample_dataframe.to_json(json_path, orient="columns")
        
        # Detect the format
        format_type = detect_format(json_path)
        
        assert format_type == FormatType.JSON
    
    def test_invalid_filepath_type(self) -> None:
        """Test that non-string filepaths are rejected."""
        with pytest.raises(AssertionError):
            detect_format(123)  # type: ignore
    
    def test_nonexistent_file(self) -> None:
        """Test that nonexistent files are rejected."""
        with pytest.raises(ValueError):
            detect_format("nonexistent_file.csv")


class TestValidateFormat:
    """Tests for the validate_format function."""
    
    def test_valid_csv(self, sample_dataframe: pd.DataFrame, temp_dir: str) -> None:
        """Test validating a CSV file."""
        # Create a CSV file
        csv_path = os.path.join(temp_dir, "test.csv")
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Validate the format
        is_valid = validate_format(csv_path, FormatType.CSV)
        
        assert is_valid == True
    
    def test_invalid_format(self, sample_dataframe: pd.DataFrame, temp_dir: str) -> None:
        """Test validating a file with the wrong format."""
        # Create a CSV file
        csv_path = os.path.join(temp_dir, "test.csv")
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Validate the format
        is_valid = validate_format(csv_path, FormatType.JSON)
        
        assert is_valid == False
    
    def test_invalid_filepath_type(self) -> None:
        """Test that non-string filepaths are rejected."""
        with pytest.raises(AssertionError):
            validate_format(123, FormatType.CSV)  # type: ignore
    
    def test_invalid_expected_format_type(self) -> None:
        """Test that non-FormatType expected formats are rejected."""
        with pytest.raises(AssertionError):
            validate_format("test.csv", "CSV")  # type: ignore


class TestInferColumnTypes:
    """Tests for the infer_column_types function."""
    
    def test_valid_input(self, sample_dataframe: pd.DataFrame) -> None:
        """Test inferring column types from a valid DataFrame."""
        type_map = infer_column_types(sample_dataframe)
        
        assert isinstance(type_map, dict)
        assert set(type_map.keys()) == set(sample_dataframe.columns)
        
        # Check that each type was inferred correctly
        for col, col_type in type_map.items():
            assert isinstance(col_type, str)
    
    def test_invalid_data_type(self) -> None:
        """Test that non-DataFrame data is rejected."""
        with pytest.raises(AssertionError):
            infer_column_types("data")  # type: ignore


class TestCastColumnTypes:
    """Tests for the cast_column_types function."""
    
    def test_valid_input(self, sample_dataframe: pd.DataFrame) -> None:
        """Test casting column types in a valid DataFrame."""
        type_map = {"id": "integer", "value": "float"}
        result = cast_column_types(sample_dataframe, type_map)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_dataframe.shape
        
        # Check that the types were cast correctly
        assert pd.api.types.is_integer_dtype(result["id"])
        assert pd.api.types.is_float_dtype(result["value"])
    
    def test_invalid_data_type(self) -> None:
        """Test that non-DataFrame data is rejected."""
        with pytest.raises(AssertionError):
            cast_column_types("data", {"id": "integer"})  # type: ignore
    
    def test_invalid_type_map_type(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that non-dict type_map is rejected."""
        with pytest.raises(AssertionError):
            cast_column_types(sample_dataframe, "type_map")  # type: ignore
    
    def test_invalid_type_map_key_type(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that non-string type_map keys are rejected."""
        with pytest.raises(AssertionError):
            cast_column_types(sample_dataframe, {1: "integer"})  # type: ignore
    
    def test_invalid_type_map_value_type(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that non-string type_map values are rejected."""
        with pytest.raises(AssertionError):
            cast_column_types(sample_dataframe, {"id": 1})  # type: ignore