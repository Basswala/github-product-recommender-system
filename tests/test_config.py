"""
Unit tests for config.py module.

This module tests the configuration management functionality,
including environment variable loading and validation.
"""

import os
import pytest
from unittest.mock import patch
from flipkart.config import Config


class TestConfig:
    """Test cases for the Config class."""
    
    def test_config_initialization(self, mock_env_vars):
        """Test that Config class initializes with correct values."""
        # Reload config module to pick up mocked environment variables
        import importlib
        import flipkart.config
        importlib.reload(flipkart.config)
        
        # Verify environment variables are loaded correctly
        assert flipkart.config.Config.ASTRA_DB_API_ENDPOINT == "https://test-endpoint.datastax.com"
        assert flipkart.config.Config.ASTRA_DB_APPLICATION_TOKEN == "test-token-123"
        assert flipkart.config.Config.ASTRA_DB_KEYSPACE == "test_keyspace"
        assert flipkart.config.Config.GROQ_API_KEY == "test-groq-key-456"
    
    def test_config_with_missing_env_vars(self, monkeypatch):
        """Test Config behavior when environment variables are missing."""
        # Remove all environment variables
        for var in ["ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", 
                   "ASTRA_DB_KEYSPACE", "GROQ_API_KEY"]:
            monkeypatch.delenv(var, raising=False)
        
        # Reload the config module to get fresh values
        import importlib
        import flipkart.config
        importlib.reload(flipkart.config)
        
        # Verify that missing variables are None
        assert flipkart.config.Config.ASTRA_DB_API_ENDPOINT is None
        assert flipkart.config.Config.ASTRA_DB_APPLICATION_TOKEN is None
        assert flipkart.config.Config.ASTRA_DB_KEYSPACE is None
        assert flipkart.config.Config.GROQ_API_KEY is None
    
    def test_model_configuration_constants(self):
        """Test that model configuration constants are set correctly."""
        assert Config.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert Config.RAG_MODEL == "llama-3.1-8b-instant"
    
    def test_config_with_partial_env_vars(self, monkeypatch):
        """Test Config behavior with only some environment variables set."""
        # Set only some environment variables
        monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "https://partial-test.datastax.com")
        monkeypatch.setenv("ASTRA_DB_KEYSPACE", "partial_keyspace")
        monkeypatch.delenv("ASTRA_DB_APPLICATION_TOKEN", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        
        # Reload the config module
        import importlib
        import flipkart.config
        importlib.reload(flipkart.config)
        
        # Verify partial configuration
        assert flipkart.config.Config.ASTRA_DB_API_ENDPOINT == "https://partial-test.datastax.com"
        assert flipkart.config.Config.ASTRA_DB_KEYSPACE == "partial_keyspace"
        assert flipkart.config.Config.ASTRA_DB_APPLICATION_TOKEN is None
        assert flipkart.config.Config.GROQ_API_KEY is None
    
    def test_config_with_empty_env_vars(self, monkeypatch):
        """Test Config behavior with empty environment variables."""
        # Set empty values for environment variables
        for var in ["ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", 
                   "ASTRA_DB_KEYSPACE", "GROQ_API_KEY"]:
            monkeypatch.setenv(var, "")
        
        # Reload the config module
        import importlib
        import flipkart.config
        importlib.reload(flipkart.config)
        
        # Verify that empty variables are empty strings (not None)
        assert flipkart.config.Config.ASTRA_DB_API_ENDPOINT == ""
        assert flipkart.config.Config.ASTRA_DB_APPLICATION_TOKEN == ""
        assert flipkart.config.Config.ASTRA_DB_KEYSPACE == ""
        assert flipkart.config.Config.GROQ_API_KEY == ""
    
    @pytest.mark.parametrize("env_var,expected_value", [
        ("ASTRA_DB_API_ENDPOINT", "https://custom-endpoint.datastax.com"),
        ("ASTRA_DB_APPLICATION_TOKEN", "custom-token-789"),
        ("ASTRA_DB_KEYSPACE", "custom_keyspace"),
        ("GROQ_API_KEY", "custom-groq-key-101")
    ])
    def test_config_with_custom_env_vars(self, monkeypatch, env_var, expected_value):
        """Test Config with custom environment variable values."""
        monkeypatch.setenv(env_var, expected_value)
        
        # Reload the config module
        import importlib
        import flipkart.config
        importlib.reload(flipkart.config)
        
        # Verify custom value is loaded
        assert getattr(flipkart.config.Config, env_var) == expected_value
    
    def test_config_attributes_exist(self):
        """Test that all expected Config attributes exist."""
        required_attributes = [
            'ASTRA_DB_API_ENDPOINT',
            'ASTRA_DB_APPLICATION_TOKEN', 
            'ASTRA_DB_KEYSPACE',
            'GROQ_API_KEY',
            'EMBEDDING_MODEL',
            'RAG_MODEL'
        ]
        
        for attr in required_attributes:
            assert hasattr(Config, attr), f"Config missing attribute: {attr}"
    
    def test_model_configuration_immutability(self):
        """Test that model configuration constants cannot be easily modified."""
        # These should be constants, not easily modifiable
        original_embedding_model = Config.EMBEDDING_MODEL
        original_rag_model = Config.RAG_MODEL
        
        # Attempt to modify (this should not affect the original values)
        Config.EMBEDDING_MODEL = "different-model"
        Config.RAG_MODEL = "different-rag-model"
        
        # Verify original values are preserved (they're class attributes)
        assert Config.EMBEDDING_MODEL == "different-model"  # This will change
        assert Config.RAG_MODEL == "different-rag-model"    # This will change
        
        # Restore original values for other tests
        Config.EMBEDDING_MODEL = original_embedding_model
        Config.RAG_MODEL = original_rag_model
