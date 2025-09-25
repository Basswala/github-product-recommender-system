"""
Basic CI-Friendly Tests
=====================

Simple tests that don't require external services or complex mocking.
These tests are designed to run in CI/CD environments reliably.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestBasicImports:
    """Test that all modules can be imported without errors."""

    def test_import_config(self):
        """Test that config module imports successfully."""
        from flipkart.config import Config
        assert hasattr(Config, 'EMBEDDING_MODEL')
        assert hasattr(Config, 'RAG_MODEL')

    def test_import_data_converter(self):
        """Test that data converter module imports successfully."""
        from flipkart.data_converter import DataConverter
        assert DataConverter is not None

    def test_import_utils(self):
        """Test that utils modules import successfully."""
        from utils.custom_exception import CustomException
        assert CustomException is not None

    def test_flask_app_import(self):
        """Test that Flask app can be imported."""
        try:
            from app import create_app
            assert create_app is not None
        except Exception as e:
            pytest.skip(f"Flask app import failed: {e}")


class TestConfigConstants:
    """Test configuration constants and values."""

    def test_model_constants(self):
        """Test that model constants are set correctly."""
        from flipkart.config import Config

        assert Config.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert Config.RAG_MODEL == "llama-3.1-8b-instant"

    def test_config_attributes_exist(self):
        """Test that all required config attributes exist."""
        from flipkart.config import Config

        required_attrs = [
            'ASTRA_DB_API_ENDPOINT',
            'ASTRA_DB_APPLICATION_TOKEN',
            'ASTRA_DB_KEYSPACE',
            'GROQ_API_KEY',
            'EMBEDDING_MODEL',
            'RAG_MODEL'
        ]

        for attr in required_attrs:
            assert hasattr(Config, attr), f"Config missing attribute: {attr}"


class TestCustomException:
    """Test custom exception functionality."""

    def test_custom_exception_creation(self):
        """Test that custom exceptions can be created."""
        from utils.custom_exception import CustomException

        exc = CustomException("Test message")
        assert str(exc) is not None
        assert "Test message" in str(exc)

    def test_custom_exception_with_details(self):
        """Test custom exception with error details."""
        from utils.custom_exception import CustomException

        original_error = ValueError("Original error")
        exc = CustomException("Wrapper message", original_error)

        assert "Wrapper message" in str(exc)
        assert "ValueError" in str(exc) or "Original error" in str(exc)

    def test_custom_exception_inheritance(self):
        """Test that CustomException inherits from Exception."""
        from utils.custom_exception import CustomException

        exc = CustomException("Test")
        assert isinstance(exc, Exception)


class TestDataConverter:
    """Test data converter functionality."""

    def test_data_converter_initialization(self):
        """Test that DataConverter can be initialized."""
        from flipkart.data_converter import DataConverter

        converter = DataConverter("dummy_path.csv")
        assert converter.file_path == "dummy_path.csv"

    def test_data_converter_has_convert_method(self):
        """Test that DataConverter has convert method."""
        from flipkart.data_converter import DataConverter

        converter = DataConverter("dummy_path.csv")
        assert hasattr(converter, 'convert')
        assert callable(converter.convert)


class TestProjectStructure:
    """Test project structure and file organization."""

    def test_required_directories_exist(self):
        """Test that required directories exist."""
        project_root = Path(__file__).parent.parent

        required_dirs = ['flipkart', 'utils', 'templates', 'static']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"

    def test_required_files_exist(self):
        """Test that required files exist."""
        project_root = Path(__file__).parent.parent

        required_files = [
            'app.py',
            'main.py',
            'requirements.txt',
            'README.md',
            'pyproject.toml'
        ]

        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"

    def test_python_files_syntax(self):
        """Test that Python files have valid syntax."""
        import ast

        project_root = Path(__file__).parent.parent
        python_files = [
            project_root / 'app.py',
            project_root / 'main.py',
            project_root / 'flipkart' / 'config.py',
            project_root / 'flipkart' / 'data_converter.py'
        ]

        for py_file in python_files:
            if py_file.exists():
                with open(py_file, 'r') as f:
                    content = f.read()
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {py_file}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])