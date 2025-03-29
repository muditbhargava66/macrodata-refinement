"""
Integration tests for the API interfaces of Macrodata Refinement (MDR).

These tests verify that the REST API and CLI interfaces work correctly
and integrate with the core functionality.
"""

import os
import json
import tempfile
import subprocess
import time
import threading
import pytest
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Any, Tuple, Generator, Optional, Callable

from mdr.api.rest import (
    start_server,
    stop_server,
    APIConfig,
    APIResponse,
    ResponseFormat,
    HTTPMethod
)
from mdr.api.cli import (
    parse_args,
    run_command,
    CommandRegistry,
    CLICommand
)
from mdr.core.refinement import RefinementConfig
from mdr.utils.logging import setup_logger, LogLevel


@pytest.fixture
def sample_data_path() -> str:
    """
    Get the path to the sample data JSON file.
    
    Returns:
        Path to the sample data file
    """
    # Get the path to the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the sample data file
    sample_path = os.path.join(current_dir, "..", "fixtures", "sample_data.json")
    
    return sample_path


@pytest.fixture
def sample_datasets(sample_data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load sample datasets from the sample data file.
    
    Args:
        sample_data_path: Path to the sample data file
        
    Returns:
        Dictionary mapping dataset names to dataset information
    """
    # Load the sample data
    with open(sample_data_path, 'r') as f:
        sample_data = json.load(f)
    
    return sample_data["datasets"]


@pytest.fixture
def convert_dataset_to_arrays(
    sample_datasets: Dict[str, Dict[str, Any]], dataset_name: str
) -> Dict[str, np.ndarray]:
    """
    Convert a named dataset from the JSON format to a dictionary of numpy arrays.
    
    Args:
        sample_datasets: Sample datasets
        dataset_name: Name of the dataset to convert
        
    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    assert dataset_name in sample_datasets, f"Unknown dataset: {dataset_name}"
    dataset = sample_datasets[dataset_name]
    
    # Convert each variable to a numpy array
    result = {}
    for var_name, values in dataset["variables"].items():
        # Convert None to np.nan
        float_values = [np.nan if v is None else float(v) for v in values]
        result[var_name] = np.array(float_values)
    
    return result


@pytest.fixture
def api_server() -> Generator[Tuple[str, int], None, None]:
    """
    Start a REST API server for testing and yield its URL.
    
    Yields:
        Tuple of (host, port)
    """
    # Configure the server
    config = APIConfig(
        host="localhost",
        port=8765,  # Use a port that's unlikely to be in use
        debug=True,
        cors_enabled=True
    )
    
    # Set up logging
    setup_logger(level=LogLevel.DEBUG)
    
    # Start the server
    start_server(config)
    
    try:
        # Wait for the server to start
        time.sleep(0.5)
        
        # Yield the server URL
        yield (config.host, config.port)
    finally:
        # Stop the server
        stop_server()


# Helper function to get the server URL
def get_server_url(host: str, port: int) -> str:
    """
    Get the URL for the API server.
    
    Args:
        host: Server host
        port: Server port
        
    Returns:
        Server URL
    """
    return f"http://{host}:{port}"


class TestRestAPI:
    """Integration tests for the REST API."""
    
    def test_refine_endpoint(self, api_server: Tuple[str, int]) -> None:
        """
        Test the /api/refine endpoint.
        
        Args:
            api_server: API server fixture
        """
        host, port = api_server
        server_url = get_server_url(host, port)
        
        # Create sample data with outliers
        data = [1.0, 2.0, 3.0, 100.0, 5.0]
        
        # Create request payload
        payload = {
            "data": data,
            "smoothing_factor": 0.2,
            "outlier_threshold": 2.0,
            "imputation_method": "mean",
            "normalization_type": "minmax"
        }
        
        # Send request to the refine endpoint
        response = requests.post(
            f"{server_url}/api/refine",
            json=payload
        )
        
        # Check that the request was successful
        assert response.status_code == 200
        
        # Parse the response
        response_data = response.json()
        
        # Check response structure
        assert "success" in response_data
        assert response_data["success"] == True
        assert "data" in response_data
        assert "message" in response_data
        
        # Check that refinement was performed
        refined_data = response_data["data"]
        assert isinstance(refined_data, list)
        assert len(refined_data) == len(data)
        
        # Check that the outlier was handled
        assert max(refined_data) < 100.0
    
    def test_validate_endpoint(self, api_server: Tuple[str, int]) -> None:
        """
        Test the /api/validate endpoint.
        
        Args:
            api_server: API server fixture
        """
        host, port = api_server
        server_url = get_server_url(host, port)
        
        # Create sample data with outliers and missing values
        data = {
            "temperature": [20.5, 21.3, None, 21.7, 45.0],
            "pressure": [101.3, 101.4, 80.0, None, None]
        }
        
        # Create request payload
        payload = {
            "data": data,
            "checks": ["range", "missing", "outliers"],
            "params": {
                "range": {
                    "min_value": 0.0,
                    "max_value": 100.0
                },
                "missing": {
                    "threshold": 0.1
                },
                "outliers": {
                    "threshold": 2.0
                }
            }
        }
        
        # Send request to the validate endpoint
        response = requests.post(
            f"{server_url}/api/validate",
            json=payload
        )
        
        # Check that the request was successful
        assert response.status_code == 200
        
        # Parse the response
        response_data = response.json()
        
        # Check response structure
        assert "success" in response_data
        assert response_data["success"] == True
        assert "data" in response_data
        assert "message" in response_data
        
        # Check validation results
        validation_results = response_data["data"]
        assert isinstance(validation_results, dict)
        assert set(validation_results.keys()) == set(data.keys())
        
        # Check that the issues were detected
        for var_name, result in validation_results.items():
            assert "is_valid" in result
            assert "error_messages" in result
            assert "statistics" in result
            
            # The data has issues, so validation should fail
            assert result["is_valid"] == False
            assert len(result["error_messages"]) > 0
    
    def test_invalid_request(self, api_server: Tuple[str, int]) -> None:
        """
        Test handling of invalid requests.
        
        Args:
            api_server: API server fixture
        """
        host, port = api_server
        server_url = get_server_url(host, port)
        
        # Create an invalid request payload (missing required fields)
        payload = {
            "smoothing_factor": 0.2
        }
        
        # Send request to the refine endpoint
        response = requests.post(
            f"{server_url}/api/refine",
            json=payload
        )
        
        # Check that the request was rejected
        assert response.status_code == 400
        
        # Parse the response
        response_data = response.json()
        
        # Check response structure
        assert "success" in response_data
        assert response_data["success"] == False
        assert "message" in response_data
        
        # Check error message
        assert "missing" in response_data["message"].lower()
    
    def test_nonexistent_endpoint(self, api_server: Tuple[str, int]) -> None:
        """
        Test handling of requests to nonexistent endpoints.
        
        Args:
            api_server: API server fixture
        """
        host, port = api_server
        server_url = get_server_url(host, port)
        
        # Send request to a nonexistent endpoint
        response = requests.get(f"{server_url}/api/nonexistent")
        
        # Check that the request was rejected
        assert response.status_code == 404
        
        # Parse the response
        response_data = response.json()
        
        # Check response structure
        assert "success" in response_data
        assert response_data["success"] == False
        assert "message" in response_data
        
        # Check error message
        assert "no handler" in response_data["message"].lower()


class TestCLI:
    """Integration tests for the command-line interface."""
    
    def test_refine_command(
        self,
        sample_datasets: Dict[str, Dict[str, Any]],
        temp_dir: str
    ) -> None:
        """
        Test the refine command.
        
        Args:
            sample_datasets: Sample datasets
            temp_dir: Temporary directory for test outputs
        """
        # Get sample dataset
        dataset_info = sample_datasets["data_with_outliers"]
        
        # Create input file
        input_path = os.path.join(temp_dir, "cli_input.csv")
        
        # Convert dataset to DataFrame
        data = {}
        for var_name, values in dataset_info["variables"].items():
            # Convert None to np.nan
            float_values = [np.nan if v is None else float(v) for v in values]
            data[var_name] = float_values
        
        df = pd.DataFrame(data)
        df.to_csv(input_path, index=False)
        
        # Create output path
        output_path = os.path.join(temp_dir, "cli_output.csv")
        
        # Build command-line arguments
        args = [
            "--log-level", "debug",
            "refine",
            input_path,
            output_path,
            "--smoothing-factor", "0.2",
            "--outlier-threshold", "2.0",
            "--imputation-method", "mean",
            "--normalization-type", "minmax"
        ]
        
        # Parse arguments
        parsed_args = parse_args(args)
        
        # Run command
        result = run_command(parsed_args)
        
        # Check that the command was successful
        assert result == 0
        
        # Check that the output file was created
        assert os.path.exists(output_path)
        
        # Read the output file
        output_df = pd.read_csv(output_path)
        
        # Check that refinement was performed
        for column in output_df.columns:
            # Check for missing values
            assert not output_df[column].isna().any()
            
            # Calculate z-scores to check for outliers
            z_scores = np.abs((output_df[column] - output_df[column].mean()) / output_df[column].std())
            assert np.all(z_scores <= 3.0)  # Allow some wiggle room
    
    def test_validate_command(
        self,
        sample_datasets: Dict[str, Dict[str, Any]],
        temp_dir: str
    ) -> None:
        """
        Test the validate command.
        
        Args:
            sample_datasets: Sample datasets
            temp_dir: Temporary directory for test outputs
        """
        # Get sample dataset
        dataset_info = sample_datasets["data_with_missing_values"]
        
        # Create input file
        input_path = os.path.join(temp_dir, "cli_validate_input.csv")
        
        # Convert dataset to DataFrame
        data = {}
        for var_name, values in dataset_info["variables"].items():
            # Convert None to np.nan
            float_values = [np.nan if v is None else float(v) for v in values]
            data[var_name] = float_values
        
        df = pd.DataFrame(data)
        df.to_csv(input_path, index=False)
        
        # Create output path
        output_path = os.path.join(temp_dir, "cli_validate_output.json")
        
        # Build command-line arguments
        args = [
            "--log-level", "debug",
            "validate",
            input_path,
            "--output-file", output_path,
            "--checks", "range,missing,outliers"
        ]
        
        # Parse arguments
        parsed_args = parse_args(args)
        
        # Run command
        result = run_command(parsed_args)
        
        # Check that the command was run
        assert result != 0  # Should return non-zero because validation fails
        
        # Check that the output file was created
        assert os.path.exists(output_path)
        
        # Read the output file
        with open(output_path, "r") as f:
            validation_results = json.load(f)
        
        # Check validation results
        assert isinstance(validation_results, dict)
        assert set(validation_results.keys()) == set(data.keys())
        
        # The data has missing values, so validation should fail
        for var_name, result in validation_results.items():
            if "missing" in var_name.lower():
                assert result["is_valid"] == False
                assert any("missing" in msg.lower() for msg in result["error_messages"])
    
    def test_convert_command(
        self,
        sample_datasets: Dict[str, Dict[str, Any]],
        temp_dir: str
    ) -> None:
        """
        Test the convert command.
        
        Args:
            sample_datasets: Sample datasets
            temp_dir: Temporary directory for test outputs
        """
        # Get sample dataset
        dataset_info = sample_datasets["clean_data"]
        
        # Create input file
        input_path = os.path.join(temp_dir, "cli_convert_input.csv")
        
        # Convert dataset to DataFrame
        data = {}
        for var_name, values in dataset_info["variables"].items():
            # Convert None to np.nan
            float_values = [np.nan if v is None else float(v) for v in values]
            data[var_name] = float_values
        
        df = pd.DataFrame(data)
        df.to_csv(input_path, index=False)
        
        # Create output path
        output_path = os.path.join(temp_dir, "cli_convert_output.json")
        
        # Build command-line arguments
        args = [
            "--log-level", "debug",
            "convert",
            input_path,
            output_path,
            "--input-format", "csv",
            "--output-format", "json"
        ]
        
        # Parse arguments
        parsed_args = parse_args(args)
        
        # Run command
        result = run_command(parsed_args)
        
        # Check that the command was successful
        assert result == 0
        
        # Check that the output file was created
        assert os.path.exists(output_path)
        
        # Read the output file
        with open(output_path, "r") as f:
            json_data = json.load(f)
        
        # Check conversion results
        assert isinstance(json_data, dict)
        assert set(json_data.keys()) == set(data.keys())


class TestAPICustomization:
    """Tests for API customization and extension."""
    
    def test_register_custom_route(self, api_server: Tuple[str, int]) -> None:
        """
        Test registering a custom route to the API.
        
        Args:
            api_server: API server fixture
        """
        from mdr.api.rest import register_route, HTTPMethod
        
        host, port = api_server
        server_url = get_server_url(host, port)
        
        # Define a custom route handler
        def custom_handler(
            request_body: Dict[str, Any], query_params: Dict[str, str]
        ) -> APIResponse:
            """
            Custom route handler for testing.
            
            Args:
                request_body: JSON request body
                query_params: Query parameters
                
            Returns:
                API response
            """
            # Echo the request
            return APIResponse(
                data={
                    "body": request_body,
                    "query": query_params
                },
                message="Custom handler called",
                status_code=200
            )
        
        # Register the custom route
        register_route(
            path="/api/custom",
            method=HTTPMethod.POST,
            handler=custom_handler
        )
        
        # Create request payload
        payload = {
            "test": "value",
            "number": 42
        }
        
        # Send request to the custom endpoint
        response = requests.post(
            f"{server_url}/api/custom?param1=value1&param2=value2",
            json=payload
        )
        
        # Check that the request was successful
        assert response.status_code == 200
        
        # Parse the response
        response_data = response.json()
        
        # Check response structure
        assert "success" in response_data
        assert response_data["success"] == True
        assert "data" in response_data
        assert "message" in response_data
        
        # Check custom handler response
        data = response_data["data"]
        assert "body" in data
        assert "query" in data
        
        # Check body
        assert data["body"] == payload
        
        # Check query params
        assert data["query"] == {"param1": "value1", "param2": "value2"}
    
    def test_custom_cli_command(self) -> None:
        """Test adding a custom CLI command."""
        from mdr.api.cli import CLICommand
        
        # Create a registry
        registry = CommandRegistry()
        
        # Define a custom command function
        def custom_command(arg1: str, arg2: float, flag: bool = False) -> int:
            """
            Custom command for testing.
            
            Args:
                arg1: First argument
                arg2: Second argument
                flag: Optional flag
                
            Returns:
                Exit code (0 for success)
            """
            assert isinstance(arg1, str), "arg1 must be a string"
            assert isinstance(arg2, float), "arg2 must be a floating-point number"
            assert isinstance(flag, bool), "flag must be a boolean"
            
            if flag:
                return 1
            else:
                return 0
        
        # Create a command
        command = CLICommand(
            name="custom",
            description="Custom command for testing",
            func=custom_command,
            arguments=[
                {
                    "name": "arg1",
                    "help": "First argument",
                    "type": str,
                    "required": True
                },
                {
                    "name": "arg2",
                    "help": "Second argument",
                    "type": float,
                    "required": True
                },
                {
                    "name": "--flag",
                    "flags": ["-f"],
                    "help": "Optional flag",
                    "action": "store_true"
                }
            ]
        )
        
        # Register the command
        registry.register(command)
        
        # Check that the command was registered
        assert "custom" in registry.commands
        
        # Get the command
        registered_command = registry.get_command("custom")
        assert registered_command is not None
        assert registered_command.name == "custom"
        
        # Test the command function
        result = registered_command.func(arg1="test", arg2=1.0, flag=False)
        assert result == 0
        
        result = registered_command.func(arg1="test", arg2=1.0, flag=True)
        assert result == 1
        
        # Test with invalid arguments
        with pytest.raises(AssertionError):
            registered_command.func(arg1=123, arg2=1.0)  # type: ignore
        
        with pytest.raises(AssertionError):
            registered_command.func(arg1="test", arg2="1.0")  # type: ignore
        
        with pytest.raises(AssertionError):
            registered_command.func(arg1="test", arg2=1.0, flag="true")  # type: ignore