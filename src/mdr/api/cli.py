"""
Command-line interface for Macrodata Refinement (MDR).

This module provides a command-line interface for accessing MDR functionality.
"""

import argparse
import sys
import os
import json
import time
from typing import Dict, List, Union, Optional, Any, Callable, Tuple, TypeVar, cast
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from enum import Enum, auto

from mdr.utils.logging import get_logger, setup_logger, LogLevel, log_execution_time


@dataclass
class CLICommand:
    """Command for the CLI interface."""
    
    name: str
    description: str
    func: Callable[..., int]
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate command parameters."""
        assert isinstance(self.name, str), "name must be a string"
        assert isinstance(self.description, str), "description must be a string"
        assert callable(self.func), "func must be callable"
        assert isinstance(self.arguments, list), "arguments must be a list"
        
        # Validate arguments
        for arg in self.arguments:
            assert isinstance(arg, dict), "Each argument must be a dictionary"
            assert "name" in arg, "Each argument must have a 'name' field"
            assert isinstance(arg["name"], str), "Argument name must be a string"


class CommandRegistry:
    """Registry of CLI commands."""
    
    def __init__(self) -> None:
        """Initialize the command registry."""
        self.commands: Dict[str, CLICommand] = {}
    
    def register(self, command: CLICommand) -> None:
        """
        Register a command.
        
        Args:
            command: Command to register
        """
        assert isinstance(command, CLICommand), "command must be a CLICommand object"
        
        # Register the command
        self.commands[command.name] = command
    
    def get_command(self, name: str) -> Optional[CLICommand]:
        """
        Get a command by name.
        
        Args:
            name: Name of the command
            
        Returns:
            Command object, or None if not found
        """
        assert isinstance(name, str), "name must be a string"
        
        return self.commands.get(name)
    
    def get_all_commands(self) -> List[CLICommand]:
        """
        Get all registered commands.
        
        Returns:
            List of all commands
        """
        return list(self.commands.values())


# Global command registry
_registry = CommandRegistry()


def create_cli(
    program_name: str = "mdr",
    description: str = "Macrodata Refinement (MDR) Command-Line Interface"
) -> argparse.ArgumentParser:
    """
    Create a CLI parser with all registered commands.
    
    Args:
        program_name: Name of the program
        description: Description of the program
        
    Returns:
        Configured ArgumentParser
    """
    assert isinstance(program_name, str), "program_name must be a string"
    assert isinstance(description, str), "description must be a string"
    
    # Create the top-level parser
    parser = argparse.ArgumentParser(prog=program_name, description=description)
    
    # Add common arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the log level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add command subparsers
    for command in _registry.get_all_commands():
        # Create a subparser for this command
        command_parser = subparsers.add_parser(command.name, help=command.description)
        
        # Add command arguments
        for arg in command.arguments:
            # Get argument properties with defaults
            name = arg["name"]
            flags = arg.get("flags", [])
            help_text = arg.get("help", "")
            type_func = arg.get("type", str)
            default = arg.get("default", None)
            choices = arg.get("choices", None)
            required = arg.get("required", False)
            action = arg.get("action", "store")
            
            # Build argument args and kwargs
            arg_args = [name]
            if flags:
                arg_args.extend(flags)
            
            arg_kwargs = {"help": help_text}
            
            if default is not None:
                arg_kwargs["default"] = default
            
            if choices is not None:
                arg_kwargs["choices"] = choices
            
            if action != "store":
                arg_kwargs["action"] = action
            elif type_func is not None:
                arg_kwargs["type"] = type_func
            
            if required:
                arg_kwargs["required"] = required
            
            # Add the argument to the parser
            command_parser.add_argument(*arg_args, **arg_kwargs)
    
    return parser


def parse_args(
    args: Optional[List[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None
) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments (None for sys.argv)
        parser: ArgumentParser to use (None to create a new one)
        
    Returns:
        Parsed arguments
    """
    # Create parser if not provided
    if parser is None:
        parser = create_cli()
    
    # Parse arguments
    return parser.parse_args(args)


def run_command(args: argparse.Namespace) -> int:
    """
    Run a command based on parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Set up logging
    log_level = LogLevel[args.log_level.upper()]
    
    if args.log_file:
        from mdr.utils.logging import LogHandler
        setup_logger(
            level=log_level,
            handlers=[LogHandler.CONSOLE, LogHandler.FILE],
            log_dir=os.path.dirname(args.log_file)
        )
    else:
        setup_logger(level=log_level)
    
    logger = get_logger()
    
    # Get the command
    command_name = args.command
    if not command_name:
        logger.error("No command specified")
        return 1
    
    command = _registry.get_command(command_name)
    if not command:
        logger.error(f"Unknown command: {command_name}")
        return 1
    
    # Run the command function with the parsed arguments
    try:
        # Convert args namespace to dictionary
        arg_dict = vars(args)
        
        # Remove common arguments
        for common_arg in ["command", "verbose", "log_level", "log_file"]:
            if common_arg in arg_dict:
                del arg_dict[common_arg]
        
        # Run the command
        start_time = time.time()
        
        if args.verbose:
            logger.info(f"Running command: {command_name}")
        
        result = command.func(**arg_dict)
        
        end_time = time.time()
        
        if args.verbose:
            logger.info(f"Command {command_name} completed in {end_time - start_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running command {command_name}: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        
        return 1


# Define example CLI commands
@log_execution_time
def refine_command(
    input_file: str,
    output_file: str,
    smoothing_factor: float = 0.2,
    outlier_threshold: float = 3.0,
    imputation_method: str = "mean",
    normalization_type: str = "minmax"
) -> int:
    """
    Refine a data file.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        smoothing_factor: Smoothing factor for data refinement
        outlier_threshold: Threshold for outlier detection
        imputation_method: Method for imputing missing values
        normalization_type: Type of normalization to apply
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    assert isinstance(input_file, str), "input_file must be a string"
    assert isinstance(output_file, str), "output_file must be a string"
    assert isinstance(smoothing_factor, float), "smoothing_factor must be a floating-point number"
    assert 0.0 < smoothing_factor <= 1.0, "smoothing_factor must be between 0 and 1"
    assert isinstance(outlier_threshold, float), "outlier_threshold must be a floating-point number"
    assert outlier_threshold > 0.0, "outlier_threshold must be greater than 0"
    assert isinstance(imputation_method, str), "imputation_method must be a string"
    assert isinstance(normalization_type, str), "normalization_type must be a string"
    
    logger = get_logger()
    
    try:
        # Import necessary modules
        from mdr.io import read_csv, write_csv
        from mdr.core.refinement import refine_data, RefinementConfig
        
        # Read the input file
        logger.info(f"Reading data from {input_file}")
        data_dict = read_csv(input_file)
        
        # Create refinement config
        config = RefinementConfig(
            smoothing_factor=smoothing_factor,
            outlier_threshold=outlier_threshold,
            imputation_method=imputation_method,
            normalization_type=normalization_type
        )
        
        # Refine each column
        logger.info("Refining data")
        refined_dict = {}
        for key, data in data_dict.items():
            refined_dict[key] = refine_data(data, config)
        
        # Write the output file
        logger.info(f"Writing refined data to {output_file}")
        write_csv(refined_dict, output_file)
        
        logger.info("Data refinement completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error refining data: {str(e)}")
        return 1


@log_execution_time
def validate_command(
    input_file: str,
    output_file: Optional[str] = None,
    checks: str = "range,missing,outliers"
) -> int:
    """
    Validate a data file.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file (optional)
        checks: Comma-separated list of checks to perform
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    assert isinstance(input_file, str), "input_file must be a string"
    
    if output_file is not None:
        assert isinstance(output_file, str), "output_file must be a string"
    
    assert isinstance(checks, str), "checks must be a string"
    check_list = checks.split(",")
    
    logger = get_logger()
    
    try:
        # Import necessary modules
        from mdr.io import read_csv, write_json
        from mdr.core.validation import validate_data
        
        # Read the input file
        logger.info(f"Reading data from {input_file}")
        data_dict = read_csv(input_file)
        
        # Validate the data
        logger.info("Validating data")
        validation_results = validate_data(data_dict, check_list)
        
        # Convert validation results to dictionary
        results_dict = {}
        for key, result in validation_results.items():
            results_dict[key] = {
                "is_valid": result.is_valid,
                "error_messages": result.error_messages,
                "statistics": result.statistics
            }
        
        # Write the output file if specified
        if output_file:
            logger.info(f"Writing validation results to {output_file}")
            
            # Convert results to JSON and write to file
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
        
        # Print summary to console
        valid_count = sum(1 for result in validation_results.values() if result.is_valid)
        total_count = len(validation_results)
        
        logger.info(f"Validation completed: {valid_count}/{total_count} variables passed")
        
        # Print details for invalid variables
        for key, result in validation_results.items():
            if not result.is_valid:
                logger.warning(f"Variable '{key}' failed validation:")
                for message in result.error_messages:
                    logger.warning(f"  - {message}")
        
        # Return success if all variables are valid, otherwise return error
        return 0 if valid_count == total_count else 2
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return 1


@log_execution_time
def convert_command(
    input_file: str,
    output_file: str,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None
) -> int:
    """
    Convert a file from one format to another.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        input_format: Input file format (auto-detect if not specified)
        output_format: Output file format (auto-detect if not specified)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    assert isinstance(input_file, str), "input_file must be a string"
    assert isinstance(output_file, str), "output_file must be a string"
    
    if input_format is not None:
        assert isinstance(input_format, str), "input_format must be a string"
    
    if output_format is not None:
        assert isinstance(output_format, str), "output_format must be a string"
    
    logger = get_logger()
    
    try:
        # Import necessary modules
        from mdr.io.formats import detect_format, convert_file_format, FormatType
        
        # Detect formats if not specified
        if input_format is None:
            logger.info(f"Auto-detecting input format for {input_file}")
            input_format_type = detect_format(input_file)
            input_format = input_format_type.name.lower()
            logger.info(f"Detected input format: {input_format}")
        else:
            input_format_type = FormatType[input_format.upper()]
        
        if output_format is None:
            # Detect from file extension
            _, ext = os.path.splitext(output_file)
            ext = ext.lower().lstrip('.')
            
            if ext in ['csv', 'tsv', 'txt']:
                output_format = 'csv'
            elif ext == 'json':
                output_format = 'json'
            elif ext in ['xls', 'xlsx']:
                output_format = 'excel'
            elif ext == 'parquet':
                output_format = 'parquet'
            elif ext in ['h5', 'hdf5']:
                output_format = 'hdf5'
            else:
                logger.warning(f"Could not detect output format from extension '{ext}', defaulting to CSV")
                output_format = 'csv'
            
            logger.info(f"Using output format: {output_format}")
        
        output_format_type = FormatType[output_format.upper()]
        
        # Convert the file
        logger.info(f"Converting {input_file} from {input_format} to {output_format}")
        convert_file_format(input_file, output_file)
        
        logger.info(f"Conversion completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        return 1


# Register example commands
_registry.register(CLICommand(
    name="refine",
    description="Refine a data file",
    func=refine_command,
    arguments=[
        {
            "name": "input_file",
            "help": "Path to the input file",
            "type": str,
            "required": True
        },
        {
            "name": "output_file",
            "help": "Path to the output file",
            "type": str,
            "required": True
        },
        {
            "name": "--smoothing-factor",
            "flags": ["-s"],
            "help": "Smoothing factor (0-1)",
            "type": float,
            "default": 0.2
        },
        {
            "name": "--outlier-threshold",
            "flags": ["-o"],
            "help": "Threshold for outlier detection",
            "type": float,
            "default": 3.0
        },
        {
            "name": "--imputation-method",
            "flags": ["-i"],
            "help": "Method for imputing missing values",
            "type": str,
            "choices": ["mean", "median", "linear", "forward"],
            "default": "mean"
        },
        {
            "name": "--normalization-type",
            "flags": ["-n"],
            "help": "Type of normalization to apply",
            "type": str,
            "choices": ["minmax", "zscore", "robust", "decimal_scaling"],
            "default": "minmax"
        }
    ]
))

_registry.register(CLICommand(
    name="validate",
    description="Validate a data file",
    func=validate_command,
    arguments=[
        {
            "name": "input_file",
            "help": "Path to the input file",
            "type": str,
            "required": True
        },
        {
            "name": "--output-file",
            "flags": ["-o"],
            "help": "Path to the output file",
            "type": str
        },
        {
            "name": "--checks",
            "flags": ["-c"],
            "help": "Comma-separated list of checks to perform",
            "type": str,
            "default": "range,missing,outliers"
        }
    ]
))

_registry.register(CLICommand(
    name="convert",
    description="Convert a file from one format to another",
    func=convert_command,
    arguments=[
        {
            "name": "input_file",
            "help": "Path to the input file",
            "type": str,
            "required": True
        },
        {
            "name": "output_file",
            "help": "Path to the output file",
            "type": str,
            "required": True
        },
        {
            "name": "--input-format",
            "flags": ["-i"],
            "help": "Input file format",
            "type": str,
            "choices": ["csv", "json", "excel", "parquet", "hdf5"]
        },
        {
            "name": "--output-format",
            "flags": ["-o"],
            "help": "Output file format",
            "type": str,
            "choices": ["csv", "json", "excel", "parquet", "hdf5"]
        }
    ]
))


# Main entry point for the CLI
def main() -> int:
    """
    Main entry point for the MDR CLI.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_args()
    
    # Run the specified command
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())