"""
Pytest configuration and shared fixtures for product recommender tests.

This module provides common test fixtures, mock configurations, and
test utilities used across all test modules.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_csv_data() -> pd.DataFrame:
    """Create sample CSV data for testing."""
    return pd.DataFrame({
        'Product_Name': [
            'iPhone 13 Pro Max',
            'Samsung Galaxy S21',
            'OnePlus 9 Pro',
            'MacBook Pro M1',
            'Dell XPS 13'
        ],
        'Review': [
            'Amazing phone with great camera quality and battery life.',
            'Good phone but camera could be better.',
            'Fast performance and good value for money.',
            'Excellent laptop for professional work.',
            'Solid laptop with good build quality.'
        ],
        'Rating': [5, 4, 4, 5, 4],
        'Sentiment': ['Positive', 'Positive', 'Positive', 'Positive', 'Positive']
    })


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample LangChain documents for testing."""
    return [
        Document(
            page_content="iPhone 13 Pro Max - Amazing phone with great camera quality and battery life.",
            metadata={"Product_Name": "iPhone 13 Pro Max", "Rating": 5}
        ),
        Document(
            page_content="Samsung Galaxy S21 - Good phone but camera could be better.",
            metadata={"Product_Name": "Samsung Galaxy S21", "Rating": 4}
        ),
        Document(
            page_content="OnePlus 9 Pro - Fast performance and good value for money.",
            metadata={"Product_Name": "OnePlus 9 Pro", "Rating": 4}
        )
    ]


@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Mock environment variables for testing."""
    env_vars = {
        "ASTRA_DB_API_ENDPOINT": "https://test-endpoint.datastax.com",
        "ASTRA_DB_APPLICATION_TOKEN": "test-token-123",
        "ASTRA_DB_KEYSPACE": "test_keyspace",
        "GROQ_API_KEY": "test-groq-key-456"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def mock_vector_store():
    """Mock AstraDB vector store for testing."""
    mock_store = Mock()
    mock_store.add_documents = Mock()
    mock_store.as_retriever = Mock()
    mock_store.similarity_search = Mock()
    return mock_store


@pytest.fixture
def mock_embedding():
    """Mock HuggingFace embedding model for testing."""
    mock_embedding = Mock()
    mock_embedding.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]] * 3)
    mock_embedding.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    return mock_embedding


@pytest.fixture
def mock_llm():
    """Mock Groq LLM for testing."""
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content="This is a test response"))
    return mock_llm


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing."""
    mock_chain = Mock()
    mock_chain.invoke = Mock(return_value={"answer": "Test RAG response"})
    return mock_chain


@pytest.fixture
def temp_csv_file(sample_csv_data, tmp_path) -> Path:
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_reviews.csv"
    sample_csv_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def flask_app():
    """Create a Flask test app."""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(flask_app):
    """Create a test client for Flask app."""
    return flask_app.test_client()


# Test markers
def pytest_configure(config):
    """Configure custom test markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Test data utilities
class TestDataFactory:
    """Factory class for creating test data."""
    
    @staticmethod
    def create_product_review(
        product_name: str = "Test Product",
        review: str = "Test review",
        rating: int = 4,
        sentiment: str = "Positive"
    ) -> Dict[str, Any]:
        """Create a single product review record."""
        return {
            "Product_Name": product_name,
            "Review": review,
            "Rating": rating,
            "Sentiment": sentiment
        }
    
    @staticmethod
    def create_multiple_reviews(count: int = 5) -> List[Dict[str, Any]]:
        """Create multiple product review records."""
        products = [
            "iPhone 13", "Samsung Galaxy", "OnePlus 9", "MacBook Pro", "Dell XPS"
        ]
        reviews = [
            "Great phone!", "Good quality", "Fast performance", 
            "Excellent laptop", "Solid build"
        ]
        
        return [
            TestDataFactory.create_product_review(
                product_name=products[i % len(products)],
                review=reviews[i % len(reviews)],
                rating=(i % 5) + 1
            )
            for i in range(count)
        ]


@pytest.fixture
def test_data_factory() -> TestDataFactory:
    """Provide test data factory."""
    return TestDataFactory()
