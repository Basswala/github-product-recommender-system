"""
Custom Exception Module
======================

This module provides custom exception classes for the product recommender system.
It includes enhanced error reporting with file and line number information.
"""

import sys


class CustomException(Exception):
    """
    Custom exception class with enhanced error reporting.

    Provides detailed error information including file location and line numbers
    to help with debugging and error tracking.
    """

    def __init__(self, message: str, error_detail: Exception = None, error_code: int = None):
        """
        Initialize CustomException with detailed error information.

        Args:
            message (str): The error message
            error_detail (Exception, optional): The original exception that caused this error
            error_code (int, optional): A numeric error code for categorization
        """
        self.message = message
        self.error_detail = error_detail
        self.error_code = error_code
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message, error_detail):
        """
        Generate detailed error message with file and line information.

        Args:
            message (str): The base error message
            error_detail (Exception): The original exception

        Returns:
            str: Formatted error message with file and line details
        """
        try:
            _, _, exc_tb = sys.exc_info()
            if exc_tb:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
            else:
                file_name = "Unknown File"
                line_number = "Unknown Line"

            return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"
        except Exception:
            # Fallback if sys.exc_info() fails
            return f"{message} | Error: {error_detail} | File: Unknown File | Line: Unknown Line"

    def __str__(self):
        """Return the formatted error message."""
        return self.error_message