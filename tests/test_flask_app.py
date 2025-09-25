"""
Integration tests for Flask application.

This module tests the Flask web application including routes,
request handling, and response generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
from prometheus_client import generate_latest
from app import create_app


class TestFlaskApp:
    """Test cases for the Flask application."""
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_create_app_initialization(self, mock_rag_builder, mock_data_ingestor):
        """Test Flask app creation and initialization."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        # Create app
        app = create_app()
        
        # Verify app is Flask instance
        assert isinstance(app, Flask)
        
        # Verify DataIngestor was initialized and called
        mock_data_ingestor.assert_called_once()
        mock_ingestor_instance.ingest.assert_called_once_with(load_existing=True)
        
        # Verify RAGChainBuilder was initialized and called
        mock_rag_builder.assert_called_once_with(mock_vector_store)
        mock_rag_builder_instance.build_chain.assert_called_once()
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_index_route(self, mock_rag_builder, mock_data_ingestor, client):
        """Test the index route."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        # Create app with test client
        app = create_app()
        with app.test_client() as test_client:
            response = test_client.get('/')
            
            # Verify response
            assert response.status_code == 200
            assert response.content_type == 'text/html; charset=utf-8'
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_get_response_route_post(self, mock_rag_builder, mock_data_ingestor):
        """Test the get_response route with POST request."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_chain.invoke.return_value = {"answer": "Test response from RAG"}
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        # Create app
        app = create_app()
        with app.test_client() as client:
            response = client.post('/get', data={'msg': 'Test message'})
            
            # Verify response
            assert response.status_code == 200
            assert response.get_data(as_text=True) == "Test response from RAG"
            
            # Verify RAG chain was called correctly
            mock_rag_chain.invoke.assert_called_once_with(
                {"input": "Test message"},
                config={"configurable": {"session_id": "user-session"}}
            )
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_get_response_route_with_different_messages(self, mock_rag_builder, mock_data_ingestor):
        """Test get_response route with different message types."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        # Test different messages
        test_messages = [
            "What is the best smartphone?",
            "Tell me about laptops",
            "Recommend a product for gaming",
            "How is the camera quality?"
        ]
        
        app = create_app()
        with app.test_client() as client:
            for i, message in enumerate(test_messages):
                mock_rag_chain.invoke.return_value = {"answer": f"Response {i+1}"}
                
                response = client.post('/get', data={'msg': message})
                
                assert response.status_code == 200
                assert response.get_data(as_text=True) == f"Response {i+1}"
                
                # Verify RAG chain was called with correct message
                mock_rag_chain.invoke.assert_called_with(
                    {"input": message},
                    config={"configurable": {"session_id": "user-session"}}
                )
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_metrics_route(self, mock_rag_builder, mock_data_ingestor):
        """Test the metrics route."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        with app.test_client() as client:
            response = client.get('/metrics')
            
            # Verify response
            assert response.status_code == 200
            assert response.content_type == 'text/plain; charset=utf-8'
            
            # Verify response contains Prometheus metrics
            response_text = response.get_data(as_text=True)
            assert 'http_requests_total' in response_text
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_request_count_metric(self, mock_rag_builder, mock_data_ingestor):
        """Test that request count metric is incremented."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        with app.test_client() as client:
            # Make multiple requests to index route
            for _ in range(3):
                response = client.get('/')
                assert response.status_code == 200
            
            # Check metrics endpoint
            response = client.get('/metrics')
            metrics_text = response.get_data(as_text=True)
            
            # Should have at least 3 requests counted
            assert 'http_requests_total 3.0' in metrics_text or 'http_requests_total 3' in metrics_text
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_get_response_missing_msg_parameter(self, mock_rag_builder, mock_data_ingestor):
        """Test get_response route when msg parameter is missing."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        with app.test_client() as client:
            # POST without 'msg' parameter
            response = client.post('/get')
            
            # Should return 400 Bad Request
            assert response.status_code == 400
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_get_response_rag_chain_error(self, mock_rag_builder, mock_data_ingestor):
        """Test get_response route when RAG chain fails."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_chain.invoke.side_effect = Exception("RAG chain error")
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        with app.test_client() as client:
            response = client.post('/get', data={'msg': 'Test message'})
            
            # Should return 500 Internal Server Error
            assert response.status_code == 500
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_app_routes_exist(self, mock_rag_builder, mock_data_ingestor):
        """Test that all expected routes exist."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        
        # Check that routes exist
        with app.test_request_context():
            rules = [rule.rule for rule in app.url_map.iter_rules()]
            
            assert '/' in rules
            assert '/get' in rules
            assert '/metrics' in rules
    
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_app_configuration(self, mock_rag_builder, mock_data_ingestor):
        """Test Flask app configuration."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        
        # Verify app name
        assert app.name == 'app'
    
    @pytest.mark.parametrize("method,route,expected_status", [
        ("GET", "/", 200),
        ("GET", "/metrics", 200),
        ("POST", "/get", 400),  # Missing msg parameter
        ("GET", "/get", 405),   # Method not allowed
        ("POST", "/nonexistent", 404),  # Route not found
    ])
    @patch('app.DataIngestor')
    @patch('app.RAGChainBuilder')
    def test_route_methods_and_status_codes(
        self, mock_rag_builder, mock_data_ingestor, 
        method, route, expected_status
    ):
        """Test various routes with different methods and expected status codes."""
        # Setup mocks
        mock_ingestor_instance = Mock()
        mock_vector_store = Mock()
        mock_ingestor_instance.ingest.return_value = mock_vector_store
        mock_data_ingestor.return_value = mock_ingestor_instance
        
        mock_rag_builder_instance = Mock()
        mock_rag_chain = Mock()
        mock_rag_builder_instance.build_chain.return_value = mock_rag_chain
        mock_rag_builder.return_value = mock_rag_builder_instance
        
        app = create_app()
        with app.test_client() as client:
            if method == "POST" and route == "/get":
                response = client.post(route)  # No data
            else:
                response = client.open(route, method=method)
            
            assert response.status_code == expected_status
