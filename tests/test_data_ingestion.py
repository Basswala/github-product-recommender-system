"""
Unit tests for data_ingestion.py module.

This module tests the data ingestion functionality including
AstraDB integration, embedding generation, and document storage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import logging
from flipkart.data_ingestion import DataIngestor
from flipkart.config import Config


class TestDataIngestor:
    """Test cases for the DataIngestor class."""
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_init_with_valid_config(self, mock_astra_db, mock_embeddings, mock_env_vars):
        """Test DataIngestor initialization with valid configuration."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        # Create DataIngestor instance
        ingestor = DataIngestor()
        
        # Verify embeddings initialization
        mock_embeddings.assert_called_once_with(model=Config.EMBEDDING_MODEL)
        
        # Verify AstraDB initialization
        mock_astra_db.assert_called_once_with(
            embedding=mock_embeddings_instance,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )
        
        # Verify instance attributes
        assert ingestor.embedding == mock_embeddings_instance
        assert ingestor.vstore == mock_astra_instance
    
    def test_init_with_missing_env_vars(self, monkeypatch):
        """Test DataIngestor initialization with missing environment variables."""
        # Remove all required environment variables
        for var in ["ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", 
                   "ASTRA_DB_KEYSPACE"]:
            monkeypatch.delenv(var, raising=False)
        
        with pytest.raises(ValueError, match="Missing required environment variables"):
            DataIngestor()
    
    def test_init_with_partial_env_vars(self, monkeypatch):
        """Test DataIngestor initialization with only some environment variables."""
        # Set only some environment variables
        monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "https://test.datastax.com")
        monkeypatch.delenv("ASTRA_DB_APPLICATION_TOKEN", raising=False)
        monkeypatch.delenv("ASTRA_DB_KEYSPACE", raising=False)
        
        with pytest.raises(ValueError, match="Missing required environment variables"):
            DataIngestor()
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_validate_config_with_valid_vars(self, mock_astra_db, mock_embeddings, mock_env_vars):
        """Test _validate_config with all required environment variables."""
        ingestor = DataIngestor()
        # If we get here without exception, validation passed
        assert True
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_load_existing_true(self, mock_astra_db, mock_embeddings, mock_env_vars):
        """Test ingest method with load_existing=True."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        ingestor = DataIngestor()
        result = ingestor.ingest(load_existing=True)
        
        # Should return the vector store without adding documents
        assert result == mock_astra_instance
        mock_astra_instance.add_documents.assert_not_called()
    
    @patch('flipkart.data_ingestion.DataConverter')
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_load_existing_false_with_valid_csv(
        self, mock_astra_db, mock_embeddings, mock_data_converter, 
        mock_env_vars, sample_documents
    ):
        """Test ingest method with load_existing=False and valid CSV."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        # Mock DataConverter
        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = sample_documents
        mock_data_converter.return_value = mock_converter_instance
        
        # Create temporary CSV file
        with patch('pathlib.Path.exists', return_value=True):
            ingestor = DataIngestor()
            result = ingestor.ingest(load_existing=False)
        
        # Verify DataConverter was called with correct path
        mock_data_converter.assert_called_once_with("Data/flipkart_product_review.csv")
        mock_converter_instance.convert.assert_called_once()
        
        # Verify documents were added to vector store
        mock_astra_instance.add_documents.assert_called_once_with(sample_documents)
        
        # Verify result
        assert result == mock_astra_instance
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_csv_file_not_found(self, mock_astra_db, mock_embeddings, mock_env_vars):
        """Test ingest method when CSV file doesn't exist."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        # Mock Path.exists to return False
        with patch('pathlib.Path.exists', return_value=False):
            ingestor = DataIngestor()
            
            with pytest.raises(FileNotFoundError, match="CSV file not found"):
                ingestor.ingest(load_existing=False)
    
    @patch('flipkart.data_ingestion.DataConverter')
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_with_empty_documents(
        self, mock_astra_db, mock_embeddings, mock_data_converter, mock_env_vars
    ):
        """Test ingest method with empty document list."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        # Mock DataConverter to return empty list
        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = []
        mock_data_converter.return_value = mock_converter_instance
        
        with patch('pathlib.Path.exists', return_value=True):
            ingestor = DataIngestor()
            
            with pytest.raises(ValueError, match="No documents found"):
                ingestor.ingest(load_existing=False)
    
    @patch('flipkart.data_ingestion.DataConverter')
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_with_document_addition_error(
        self, mock_astra_db, mock_embeddings, mock_data_converter, 
        mock_env_vars, sample_documents
    ):
        """Test ingest method when document addition fails."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_instance.add_documents.side_effect = Exception("Database error")
        mock_astra_db.return_value = mock_astra_instance
        
        # Mock DataConverter
        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = sample_documents
        mock_data_converter.return_value = mock_converter_instance
        
        with patch('pathlib.Path.exists', return_value=True):
            ingestor = DataIngestor()
            
            with pytest.raises(Exception, match="Database error"):
                ingestor.ingest(load_existing=False)
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_with_converter_error(self, mock_astra_db, mock_embeddings, mock_env_vars):
        """Test ingest method when DataConverter fails."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('flipkart.data_ingestion.DataConverter') as mock_converter:
            
            mock_converter_instance = Mock()
            mock_converter_instance.convert.side_effect = Exception("CSV parsing error")
            mock_converter.return_value = mock_converter_instance
            
            ingestor = DataIngestor()
            
            with pytest.raises(Exception, match="CSV parsing error"):
                ingestor.ingest(load_existing=False)
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    def test_embedding_model_initialization_error(self, mock_embeddings, mock_env_vars):
        """Test DataIngestor initialization when embedding model fails."""
        mock_embeddings.side_effect = Exception("Model loading error")
        
        with pytest.raises(Exception, match="Model loading error"):
            DataIngestor()
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_astra_db_connection_error(self, mock_astra_db, mock_embeddings, mock_env_vars):
        """Test DataIngestor initialization when AstraDB connection fails."""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_db.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception, match="Connection error"):
            DataIngestor()
    
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_logging_behavior(self, mock_astra_db, mock_embeddings, mock_env_vars, caplog):
        """Test that appropriate logging messages are generated."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        with caplog.at_level(logging.INFO):
            ingestor = DataIngestor()
            ingestor.ingest(load_existing=True)
        
        # Verify logging messages
        assert "Initializing DataIngestor" in caplog.text
        assert "Initialized embedding model" in caplog.text
        assert "Successfully connected to AstraDB vector store" in caplog.text
        assert "Starting data ingestion with load_existing=True" in caplog.text
        assert "Returning existing vector store for production use" in caplog.text
    
    @pytest.mark.parametrize("load_existing", [True, False])
    @patch('flipkart.data_ingestion.HuggingFaceEndpointEmbeddings')
    @patch('flipkart.data_ingestion.AstraDBVectorStore')
    def test_ingest_return_type(self, mock_astra_db, mock_embeddings, mock_env_vars, load_existing):
        """Test that ingest method returns correct type."""
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_astra_instance = Mock()
        mock_astra_db.return_value = mock_astra_instance
        
        ingestor = DataIngestor()
        
        if load_existing:
            result = ingestor.ingest(load_existing=load_existing)
        else:
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    ingestor.ingest(load_existing=load_existing)
                return
        
        assert result == mock_astra_instance
