"""
Initialization module for the Aztec GDDT v2 simulation package.

This module sets up the logging configuration for the entire Aztec GDDT simulation framework.
It configures both file-based and console logging to track simulation execution,
performance metrics, and potential issues during runtime.

The package uses a standardized logger named 'aztec-gddt-v2' that can be accessed
by importing DEFAULT_LOGGER from this module.
"""

import logging

# Default logger name used throughout the package
DEFAULT_LOGGER = 'aztec-gddt-v2'


def setup_logging(
    filename='cadcad.log',
    level=logging.INFO,
    format='\n%(asctime)s - %(name)s - %(levelname)s\n%(message)s',
):
    # Create a logger
    logger = logging.getLogger(DEFAULT_LOGGER)
    logger.setLevel(level)  # Set the logging level

    # Remove any existing handlers to avoid duplicate logging
    # when setup_logging is called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler and set level to INFO
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)

    # Create a console (stream) handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(format, '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# Initialize logging when the package is imported
setup_logging()
