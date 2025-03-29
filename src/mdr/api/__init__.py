"""
API module for Macrodata Refinement (MDR).

This module provides interfaces for accessing MDR functionality
through REST APIs and command-line interfaces.
"""

from mdr.api.rest import (
    start_server,
    stop_server,
    register_route,
    APIConfig,
    APIResponse
)
from mdr.api.cli import (
    create_cli,
    parse_args,
    run_command,
    CLICommand,
    CommandRegistry
)

__all__ = [
    # REST API
    "start_server",
    "stop_server",
    "register_route",
    "APIConfig",
    "APIResponse",
    
    # CLI
    "create_cli",
    "parse_args",
    "run_command",
    "CLICommand",
    "CommandRegistry"
]