import logging
import sys

def setup_logger(log_level=logging.INFO):
    """
    Setup and configure a logger instance to be used across the project.
    """
    # Use the name of the module for the logger
    logger = logging.getLogger(__name__) 
    
    # --- Prevents setting up the same logger multiple times ---
    if logger.hasHandlers():
        return logger

    # Prevent messages from being propagated to the root logger
    logger.propagate = False
    
    # Set the logging level
    logger.setLevel(log_level)
    
    # Create console handler and set level
    # We use sys.stdout to ensure it works well in Databricks notebooks
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s > %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to the logger
    logger.addHandler(console_handler)
    
    return logger

LOGGER = setup_logger()