"""
REST API for Macrodata Refinement (MDR).

This module provides a REST API interface for accessing MDR functionality.
"""

import json
import threading
import time
from typing import Dict, List, Union, Optional, Any, Callable, Tuple, TypeVar, cast
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from enum import Enum, auto
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import io
import traceback

from mdr.utils.logging import get_logger, log_execution_time


class HTTPMethod(Enum):
    """HTTP methods supported by the API."""
    
    GET = auto()
    POST = auto()
    PUT = auto()
    DELETE = auto()
    PATCH = auto()


class ResponseFormat(Enum):
    """Response formats supported by the API."""
    
    JSON = auto()
    CSV = auto()
    HTML = auto()
    TEXT = auto()


@dataclass
class APIConfig:
    """Configuration for the REST API server."""
    
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert isinstance(self.host, str), "host must be a string"
        assert isinstance(self.port, int), "port must be an integer"
        assert self.port > 0, "port must be a positive integer"
        assert isinstance(self.debug, bool), "debug must be a boolean"
        assert isinstance(self.cors_enabled, bool), "cors_enabled must be a boolean"
        assert isinstance(self.allowed_origins, list), "allowed_origins must be a list"
        assert all(isinstance(origin, str) for origin in self.allowed_origins), \
            "All allowed origins must be strings"


@dataclass
class APIResponse:
    """Response from an API endpoint."""
    
    data: Optional[Any] = None
    message: Optional[str] = None
    status_code: int = 200
    format: ResponseFormat = ResponseFormat.JSON
    headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate response parameters."""
        assert isinstance(self.status_code, int), "status_code must be an integer"
        assert 100 <= self.status_code <= 599, "status_code must be between 100 and 599"
        assert isinstance(self.format, ResponseFormat), "format must be a ResponseFormat enum"
        assert isinstance(self.headers, dict), "headers must be a dictionary"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in self.headers.items()), \
            "All header keys and values must be strings"
    
    def to_bytes(self) -> bytes:
        """
        Convert the response to bytes based on the format.
        
        Returns:
            Response data as bytes
        """
        if self.format == ResponseFormat.JSON:
            # Convert data to JSON
            response_dict = {
                "success": 200 <= self.status_code < 300,
                "data": self._convert_to_json_serializable(self.data),
                "message": self.message
            }
            
            return json.dumps(response_dict).encode('utf-8')
            
        elif self.format == ResponseFormat.CSV:
            # Convert data to CSV
            if isinstance(self.data, pd.DataFrame):
                csv_buffer = io.StringIO()
                self.data.to_csv(csv_buffer, index=False)
                return csv_buffer.getvalue().encode('utf-8')
            elif isinstance(self.data, dict) and all(isinstance(v, np.ndarray) for v in self.data.values()):
                # Convert dictionary of arrays to DataFrame
                csv_buffer = io.StringIO()
                pd.DataFrame(self.data).to_csv(csv_buffer, index=False)
                return csv_buffer.getvalue().encode('utf-8')
            else:
                raise ValueError("Data must be a DataFrame or dictionary of arrays for CSV format")
            
        elif self.format == ResponseFormat.HTML:
            # Return HTML data as-is
            if isinstance(self.data, str):
                return self.data.encode('utf-8')
            else:
                raise ValueError("Data must be a string for HTML format")
            
        elif self.format == ResponseFormat.TEXT:
            # Convert data to text
            if self.data is None:
                return (self.message or "").encode('utf-8')
            elif isinstance(self.data, str):
                return self.data.encode('utf-8')
            else:
                return str(self.data).encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported response format: {self.format}")
    
    def get_content_type(self) -> str:
        """
        Get the Content-Type header for the response.
        
        Returns:
            Content-Type string
        """
        if self.format == ResponseFormat.JSON:
            return "application/json"
        elif self.format == ResponseFormat.CSV:
            return "text/csv"
        elif self.format == ResponseFormat.HTML:
            return "text/html"
        elif self.format == ResponseFormat.TEXT:
            return "text/plain"
        else:
            return "application/octet-stream"
    
    def _convert_to_json_serializable(self, data: Any) -> Any:
        """
        Convert data to JSON-serializable format.
        
        Args:
            data: Data to convert
            
        Returns:
            JSON-serializable data
        """
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return {col: data[col].tolist() for col in data.columns}
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {str(k): self._convert_to_json_serializable(v) for k, v in data.items()}
        elif hasattr(data, "to_dict"):
            return data.to_dict()
        elif hasattr(data, "__dict__"):
            return self._convert_to_json_serializable(data.__dict__)
        else:
            return str(data)


# Type for route handler functions
RouteHandler = Callable[[Dict[str, Any], Dict[str, str]], APIResponse]

# Global server instance
_server = None
_server_thread = None

# Dictionary mapping routes to handlers
_routes: Dict[str, Dict[HTTPMethod, RouteHandler]] = {}


class MDRRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MDR API."""
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        self._handle_request(HTTPMethod.GET)
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        self._handle_request(HTTPMethod.POST)
    
    def do_PUT(self) -> None:
        """Handle PUT requests."""
        self._handle_request(HTTPMethod.PUT)
    
    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        self._handle_request(HTTPMethod.DELETE)
    
    def do_PATCH(self) -> None:
        """Handle PATCH requests."""
        self._handle_request(HTTPMethod.PATCH)
    
    def _handle_request(self, method: HTTPMethod) -> None:
        """
        Handle a request of any method.
        
        Args:
            method: HTTP method of the request
        """
        logger = get_logger()
        logger.debug(f"Handling {method.name} request: {self.path}")
        
        try:
            # Parse path and query parameters
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            
            # Parse query parameters
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Convert query parameters to simple dict (take first value for each parameter)
            query_dict = {k: v[0] for k, v in query_params.items()}
            
            # Parse request body for methods with a body
            request_body = {}
            if method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    body_data = self.rfile.read(content_length).decode('utf-8')
                    content_type = self.headers.get('Content-Type', '')
                    
                    if 'application/json' in content_type:
                        # Parse JSON body
                        try:
                            request_body = json.loads(body_data)
                        except json.JSONDecodeError:
                            self._send_error(400, "Invalid JSON in request body")
                            return
                    elif 'application/x-www-form-urlencoded' in content_type:
                        # Parse form data
                        form_data = urllib.parse.parse_qs(body_data)
                        request_body = {k: v[0] for k, v in form_data.items()}
            
            # Find the matching route handler
            if path in _routes and method in _routes[path]:
                handler = _routes[path][method]
                
                # Call the handler with parsed parameters
                try:
                    response = handler(request_body, query_dict)
                    
                    # Send the response
                    self._send_response(response)
                    
                except Exception as e:
                    # Log the error
                    logger.error(f"Error in route handler: {str(e)}")
                    if hasattr(self.server, 'config') and self.server.config.debug:
                        logger.error(traceback.format_exc())
                    
                    # Send error response
                    self._send_error(500, str(e))
            else:
                # No matching route found
                self._send_error(404, f"No handler for {method.name} {path}")
                
        except Exception as e:
            # Log the error
            logger.error(f"Error handling request: {str(e)}")
            if hasattr(self.server, 'config') and self.server.config.debug:
                logger.error(traceback.format_exc())
            
            # Send error response
            self._send_error(500, str(e))
    
    def _send_response(self, response: APIResponse) -> None:
        """
        Send an API response.
        
        Args:
            response: Response to send
        """
        # Set response status code
        self.send_response(response.status_code)
        
        # Set CORS headers if enabled
        if hasattr(self.server, 'config') and self.server.config.cors_enabled:
            origin = self.headers.get('Origin', '*')
            allowed_origins = self.server.config.allowed_origins
            
            if '*' in allowed_origins or origin in allowed_origins:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        
        # Set content type header
        self.send_header('Content-Type', response.get_content_type())
        
        # Set custom headers
        for name, value in response.headers.items():
            self.send_header(name, value)
        
        self.end_headers()
        
        # Send response data
        self.wfile.write(response.to_bytes())
    
    def _send_error(self, status_code: int, message: str) -> None:
        """
        Send an error response.
        
        Args:
            status_code: HTTP status code
            message: Error message
        """
        response = APIResponse(
            data=None,
            message=message,
            status_code=status_code,
            format=ResponseFormat.JSON
        )
        
        self._send_response(response)
    
    def log_message(self, format: str, *args: Any) -> None:
        """
        Log a server message.
        
        Args:
            format: Format string
            *args: Format arguments
        """
        logger = get_logger()
        logger.debug(format % args)


def register_route(
    path: str,
    method: HTTPMethod,
    handler: RouteHandler
) -> None:
    """
    Register a route handler.
    
    Args:
        path: URL path to handle
        method: HTTP method to handle
        handler: Function to handle the request
    """
    assert isinstance(path, str), "path must be a string"
    assert isinstance(method, HTTPMethod), "method must be an HTTPMethod enum"
    assert callable(handler), "handler must be callable"
    
    global _routes
    
    # Initialize route entry if needed
    if path not in _routes:
        _routes[path] = {}
    
    # Register the handler
    _routes[path][method] = handler


def start_server(config: APIConfig = APIConfig()) -> None:
    """
    Start the API server.
    
    Args:
        config: Server configuration
    """
    assert isinstance(config, APIConfig), "config must be an APIConfig object"
    
    global _server, _server_thread
    
    # Check if server is already running
    if _server is not None:
        raise RuntimeError("Server is already running")
    
    # Create and configure the server
    _server = HTTPServer((config.host, config.port), MDRRequestHandler)
    _server.config = config  # type: ignore
    
    # Start the server in a separate thread
    _server_thread = threading.Thread(target=_server.serve_forever)
    _server_thread.daemon = True
    _server_thread.start()
    
    logger = get_logger()
    logger.info(f"API server started on http://{config.host}:{config.port}")


def stop_server() -> None:
    """Stop the API server."""
    global _server, _server_thread
    
    # Check if server is running
    if _server is None:
        raise RuntimeError("Server is not running")
    
    # Stop the server
    _server.shutdown()
    _server_thread.join()
    
    _server = None
    _server_thread = None
    
    logger = get_logger()
    logger.info("API server stopped")


# Define some example API endpoints
@log_execution_time
def refine_data_endpoint(
    request_body: Dict[str, Any],
    query_params: Dict[str, str]
) -> APIResponse:
    """
    Endpoint for refining data.
    
    Args:
        request_body: JSON request body
        query_params: Query parameters
        
    Returns:
        API response with refined data
    """
    from mdr.core.refinement import refine_data, RefinementConfig
    
    # Parse request parameters
    try:
        # Get data from request
        if "data" not in request_body:
            return APIResponse(
                message="Missing 'data' field in request body",
                status_code=400
            )
        
        # Convert data to numpy array
        data = np.array(request_body["data"], dtype=float)
        
        # Get refinement configuration
        smoothing_factor = float(request_body.get("smoothing_factor", 0.2))
        outlier_threshold = float(request_body.get("outlier_threshold", 3.0))
        imputation_method = request_body.get("imputation_method", "mean")
        normalization_type = request_body.get("normalization_type", "minmax")
        
        # Create refinement config
        config = RefinementConfig(
            smoothing_factor=smoothing_factor,
            outlier_threshold=outlier_threshold,
            imputation_method=imputation_method,
            normalization_type=normalization_type
        )
        
        # Refine the data
        refined_data = refine_data(data, config)
        
        # Return the refined data
        return APIResponse(
            data=refined_data,
            message="Data refined successfully",
            status_code=200
        )
        
    except Exception as e:
        return APIResponse(
            message=f"Error refining data: {str(e)}",
            status_code=400
        )


@log_execution_time
def validate_data_endpoint(
    request_body: Dict[str, Any],
    query_params: Dict[str, str]
) -> APIResponse:
    """
    Endpoint for validating data.
    
    Args:
        request_body: JSON request body
        query_params: Query parameters
        
    Returns:
        API response with validation results
    """
    from mdr.core.validation import validate_data
    
    # Parse request parameters
    try:
        # Get data from request
        if "data" not in request_body:
            return APIResponse(
                message="Missing 'data' field in request body",
                status_code=400
            )
        
        # Convert data to dictionary of numpy arrays
        data_dict = {}
        for key, value in request_body["data"].items():
            data_dict[key] = np.array(value, dtype=float)
        
        # Get validation checks
        checks = request_body.get("checks", ["range", "missing", "outliers"])
        
        # Get validation parameters
        params = request_body.get("params", {})
        
        # Validate the data
        validation_results = validate_data(data_dict, checks, params)
        
        # Convert validation results to JSON-serializable format
        results_dict = {}
        for key, result in validation_results.items():
            results_dict[key] = {
                "is_valid": result.is_valid,
                "error_messages": result.error_messages,
                "statistics": result.statistics
            }
        
        # Return the validation results
        return APIResponse(
            data=results_dict,
            message="Data validated successfully",
            status_code=200
        )
        
    except Exception as e:
        return APIResponse(
            message=f"Error validating data: {str(e)}",
            status_code=400
        )


# Register example endpoints
register_route("/api/refine", HTTPMethod.POST, refine_data_endpoint)
register_route("/api/validate", HTTPMethod.POST, validate_data_endpoint)