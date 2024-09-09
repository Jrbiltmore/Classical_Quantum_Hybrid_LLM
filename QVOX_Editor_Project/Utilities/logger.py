
# logger.py
# Logs editor actions, including file saves, state changes, and rendering updates.

import logging

class Logger:
    """Class responsible for logging actions and events in the QVOX Editor."""

    def __init__(self, log_file: str = "qvox_editor.log"):
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_action(self, action: str, details: str):
        """Logs a specific action with details."""
        logging.info(f"Action: {action} - Details: {details}")

    def log_error(self, error_message: str):
        """Logs an error message."""
        logging.error(f"Error: {error_message}")

    def log_warning(self, warning_message: str):
        """Logs a warning message."""
        logging.warning(f"Warning: {warning_message}")
