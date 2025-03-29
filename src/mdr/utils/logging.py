"""
Logging utilities for Macrodata Refinement (MDR).

This module provides logging setup, configuration, and utilities.
"""

import logging
import sys
import os
import time
import functools
from typing import Callable, Any, Optional, Dict, Union, List, TypeVar, cast
from enum import Enum, auto
import json


class LogLevel(Enum):
    """Log levels for the MDR logger."""
    
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class LogHandler(Enum):
    """Types of log handlers."""
    
    CONSOLE = auto()
    FILE = auto()
    JSON = auto()


# Map our log levels to Python's logging levels
_LOG_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL
}

# Global logger instance
_logger = None


def setup_logger(
    name: str = "mdr",
    level: Union[LogLevel, str, int] = LogLevel.INFO,
    handlers: List[LogHandler] = [LogHandler.CONSOLE],
    log_dir: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure the MDR logger.
    
    Args:
        name: Name of the logger
        level: Log level (can be a LogLevel enum, string name, or integer level)
        handlers: List of handlers to add to the logger
        log_dir: Directory for log files (for FILE handler)
        log_format: Log message format string
        date_format: Date format string for log messages
        
    Returns:
        Configured logger instance
    """
    assert isinstance(name, str), "name must be a string"
    
    # Convert string level to LogLevel enum if needed
    if isinstance(level, str):
        try:
            level = LogLevel[level.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level string: {level}")
    
    # Convert integer level to LogLevel enum if needed
    if isinstance(level, int):
        level_map_reversed = {v: k for k, v in _LOG_LEVEL_MAP.items()}
        if level in level_map_reversed:
            level = level_map_reversed[level]
        else:
            raise ValueError(f"Invalid log level integer: {level}")
    
    assert isinstance(level, LogLevel), "level must be a LogLevel enum, string, or integer"
    assert isinstance(handlers, list), "handlers must be a list"
    assert all(isinstance(h, LogHandler) for h in handlers), "All handlers must be LogHandler enums"
    
    if log_dir is not None:
        assert isinstance(log_dir, str), "log_dir must be a string"
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    if log_format is not None:
        assert isinstance(log_format, str), "log_format must be a string"
    
    if date_format is not None:
        assert isinstance(date_format, str), "date_format must be a string"
    
    # Set default formats if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL_MAP[level])
    
    # Remove any existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    # Add requested handlers
    for handler_type in handlers:
        if handler_type == LogHandler.CONSOLE:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(_LOG_LEVEL_MAP[level])
            
            # Create and set formatter
            console_formatter = logging.Formatter(log_format, date_format)
            console_handler.setFormatter(console_formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
            
        elif handler_type == LogHandler.FILE:
            # Check if log_dir is provided
            if log_dir is None:
                raise ValueError("log_dir must be provided for FILE handler")
            
            # File handler
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(_LOG_LEVEL_MAP[level])
            
            # Create and set formatter
            file_formatter = logging.Formatter(log_format, date_format)
            file_handler.setFormatter(file_formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
        elif handler_type == LogHandler.JSON:
            # Check if log_dir is provided
            if log_dir is None:
                raise ValueError("log_dir must be provided for JSON handler")
            
            # JSON handler (custom formatter)
            json_log_file = os.path.join(log_dir, f"{name}_json.log")
            json_handler = logging.FileHandler(json_log_file)
            json_handler.setLevel(_LOG_LEVEL_MAP[level])
            
            # Create and set formatter
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": self.formatTime(record, date_format),
                        "name": record.name,
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "module": record.module,
                        "line": record.lineno
                    }
                    
                    # Add exception info if available
                    if record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    
                    return json.dumps(log_data)
            
            json_formatter = JSONFormatter()
            json_handler.setFormatter(json_formatter)
            
            # Add handler to logger
            logger.addHandler(json_handler)
    
    # Store logger as global
    global _logger
    _logger = logger
    
    return logger


def get_logger() -> logging.Logger:
    """
    Get the MDR logger instance.
    
    Returns:
        Logger instance (creates a default one if not already set up)
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    
    return _logger


def set_log_level(level: Union[LogLevel, str, int]) -> None:
    """
    Set the log level for the MDR logger.
    
    Args:
        level: New log level (can be a LogLevel enum, string name, or integer level)
    """
    # Convert string level to LogLevel enum if needed
    if isinstance(level, str):
        try:
            level = LogLevel[level.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level string: {level}")
    
    # Convert integer level to LogLevel enum if needed
    if isinstance(level, int):
        level_map_reversed = {v: k for k, v in _LOG_LEVEL_MAP.items()}
        if level in level_map_reversed:
            level = level_map_reversed[level]
        else:
            raise ValueError(f"Invalid log level integer: {level}")
    
    assert isinstance(level, LogLevel), "level must be a LogLevel enum, string, or integer"
    
    # Get logger and set level
    logger = get_logger()
    logger.setLevel(_LOG_LEVEL_MAP[level])
    
    # Update level for all handlers
    for handler in logger.handlers:
        handler.setLevel(_LOG_LEVEL_MAP[level])


# Define a generic type for function
F = TypeVar('F', bound=Callable[..., Any])

def log_execution_time(func: F) -> F:
    """
    Decorator to log the execution time of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger()
        
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"Completed {func.__name__} in {execution_time:.6f} seconds")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Error in {func.__name__} after {execution_time:.6f} seconds: {str(e)}")
            raise
    
    return cast(F, wrapper)