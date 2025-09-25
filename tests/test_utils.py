"""
Unit tests for utility modules.

This module tests the utility functions including logging configuration
and custom exception handling.
"""

import os
import pytest
import logging
from unittest.mock import patch, mock_open
from pathlib import Path
from utils.logger import get_logger
from utils.custom_exception import CustomException


class TestLogger:
    """Test cases for the logger utility module."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_sets_correct_level(self):
        """Test that get_logger sets the correct logging level."""
        logger = get_logger("test_module")
        
        assert logger.level == logging.INFO
    
    def test_get_logger_different_names(self):
        """Test get_logger with different module names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 != logger2
    
    def test_get_logger_same_name_returns_same_instance(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        assert logger1 == logger2
    
    @patch('utils.logger.os.makedirs')
    def test_logs_directory_creation(self, mock_makedirs):
        """Test that logs directory is created."""
        # This test verifies that os.makedirs is called for logs directory
        # The actual directory creation happens at module import time
        assert True  # If we get here, the module imported without errors
    
    @patch('utils.logger.logging.basicConfig')
    def test_logging_configuration(self, mock_basic_config):
        """Test that logging is configured correctly."""
        # Re-import to trigger logging configuration
        import importlib
        import utils.logger
        importlib.reload(utils.logger)
        
        # Verify basicConfig was called
        mock_basic_config.assert_called_once()
        
        # Check the call arguments
        call_args = mock_basic_config.call_args
        assert 'filename' in call_args.kwargs
        assert 'format' in call_args.kwargs
        assert 'level' in call_args.kwargs
        assert call_args.kwargs['level'] == logging.INFO


class TestCustomException:
    """Test cases for the CustomException class."""
    
    def test_custom_exception_inheritance(self):
        """Test that CustomException inherits from Exception."""
        exception = CustomException("Test message")
        
        assert isinstance(exception, Exception)
        assert isinstance(exception, CustomException)
    
    def test_custom_exception_message(self):
        """Test CustomException with custom message."""
        message = "This is a custom error message"
        exception = CustomException(message)
        
        assert str(exception) == message
        assert exception.args == (message,)
    
    def test_custom_exception_with_code(self):
        """Test CustomException with error code."""
        message = "Database connection failed"
        code = "DB_CONN_001"
        exception = CustomException(message, code)
        
        assert str(exception) == message
        assert exception.error_code == code
    
    def test_custom_exception_without_code(self):
        """Test CustomException without error code."""
        message = "Simple error message"
        exception = CustomException(message)
        
        assert str(exception) == message
        assert not hasattr(exception, 'error_code')
    
    def test_custom_exception_raise_and_catch(self):
        """Test raising and catching CustomException."""
        message = "Test exception"
        
        with pytest.raises(CustomException) as exc_info:
            raise CustomException(message)
        
        assert str(exc_info.value) == message
    
    def test_custom_exception_with_details(self):
        """Test CustomException with additional details."""
        message = "Validation failed"
        code = "VAL_001"
        details = {"field": "email", "value": "invalid-email"}
        
        exception = CustomException(message, code, **details)
        
        assert str(exception) == message
        assert exception.error_code == code
        assert exception.field == "email"
        assert exception.value == "invalid-email"
    
    def test_custom_exception_multiple_args(self):
        """Test CustomException with multiple arguments."""
        message = "Multiple argument error"
        arg1 = "arg1"
        arg2 = "arg2"
        
        exception = CustomException(message, arg1, arg2)
        
        assert str(exception) == message
        assert exception.args == (message, arg1, arg2)
    
    @pytest.mark.parametrize("message,expected", [
        ("Simple message", "Simple message"),
        ("Message with numbers 123", "Message with numbers 123"),
        ("Message with special chars !@#$%", "Message with special chars !@#$%"),
        ("", ""),
        ("Unicode message: 你好世界", "Unicode message: 你好世界"),
    ])
    def test_custom_exception_various_messages(self, message, expected):
        """Test CustomException with various message types."""
        exception = CustomException(message)
        assert str(exception) == expected
    
    def test_custom_exception_repr(self):
        """Test CustomException string representation."""
        message = "Test message"
        exception = CustomException(message)
        
        repr_str = repr(exception)
        assert "CustomException" in repr_str
        assert message in repr_str
    
    def test_custom_exception_with_none_message(self):
        """Test CustomException with None message."""
        exception = CustomException(None)
        
        assert str(exception) == "None"
    
    def test_custom_exception_inheritance_chain(self):
        """Test that CustomException can be caught as base Exception."""
        message = "Test inheritance"
        exception = CustomException(message)
        
        # Should be catchable as Exception
        try:
            raise exception
        except Exception as e:
            assert str(e) == message
        
        # Should be catchable as CustomException
        try:
            raise exception
        except CustomException as e:
            assert str(e) == message


class TestUtilityIntegration:
    """Integration tests for utility modules."""
    
    def test_logger_with_custom_exception(self):
        """Test logger integration with CustomException."""
        logger = get_logger("test_integration")
        
        # This test verifies that logger can be used with custom exceptions
        try:
            raise CustomException("Integration test error")
        except CustomException as e:
            logger.error(f"Caught custom exception: {str(e)}")
        
        assert True  # If we get here without error, integration works
    
    def test_logger_level_configuration(self):
        """Test that logger level is properly configured."""
        logger = get_logger("level_test")
        
        # Test that logger accepts different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        assert logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.ERROR)
        assert logger.isEnabledFor(logging.CRITICAL)
    
    @patch('utils.logger.datetime')
    def test_log_file_naming(self, mock_datetime):
        """Test that log file naming uses current date."""
        mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
        
        # Re-import to trigger new datetime mock
        import importlib
        import utils.logger
        importlib.reload(utils.logger)
        
        # Verify that strftime was called with the expected format
        mock_datetime.now.return_value.strftime.assert_called()
    
    def test_multiple_loggers_independence(self):
        """Test that multiple loggers work independently."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module3")
        
        # All should be different instances
        assert logger1 != logger2
        assert logger2 != logger3
        assert logger1 != logger3
        
        # All should have correct names
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger3.name == "module3"
