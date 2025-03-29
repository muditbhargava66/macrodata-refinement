.. _api_utils_logging:

Logging Utilities
=================

.. automodule:: mdr.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``logging`` module provides a consistent logging framework for the MDR package.
It includes functions for setting up loggers, configuring log levels, formatting
log messages, and routing logs to appropriate outputs.

Core Components
---------------

LogLevel
~~~~~~~~

.. autoclass:: mdr.utils.logging.LogLevel
   :members:
   :no-index:

An enumeration of log levels used in MDR, including DEBUG, INFO, WARNING, ERROR, and CRITICAL.

Functions
---------

.. autofunction:: mdr.utils.logging.setup_logger
   :no-index:
.. autofunction:: mdr.utils.logging.get_logger
   :no-index:
.. autofunction:: mdr.utils.logging.set_log_level
   :no-index:


Usage Examples
--------------

Basic logging setup:

.. code-block:: python

    from mdr.utils.logging import setup_logger, get_logger, LogLevel
    
    # Set up the logger
    setup_logger(level=LogLevel.INFO)
    
    # Get a logger instance
    logger = get_logger()
    
    # Log messages at different levels
    logger.debug("This is a debug message")  # Won't be shown at INFO level
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

Using the LoggingMixin:

.. code-block:: python

    from mdr.utils.logging import LoggingMixin, setup_logger, LogLevel
    
    # Set up the logger
    setup_logger(level=LogLevel.DEBUG)
    
    # Create a class with logging capabilities
    class MyProcessor(LoggingMixin):
        def process_data(self, data):
            self.logger.info("Starting data processing")
            
            if len(data) == 0:
                self.logger.warning("Empty data received")
                return None
            
            self.logger.debug(f"Processing {len(data)} data points")
            
            # Process the data...
            result = data * 2
            
            self.logger.info("Data processing completed")
            return result
    
    # Use the class
    processor = MyProcessor()
    result = processor.process_data([1, 2, 3, 4, 5])

Advanced Configuration
----------------------

Configure logging with custom handlers and formatters:

.. code-block:: python

    import logging
    from mdr.utils.logging import setup_logger, LogLevel
    
    # Configure a file handler
    file_handler = logging.FileHandler("mdr.log")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Set up the logger with custom handlers
    setup_logger(
        level=LogLevel.DEBUG,
        handlers=[file_handler],
        format_string="%(levelname)s: %(message)s"
    )
