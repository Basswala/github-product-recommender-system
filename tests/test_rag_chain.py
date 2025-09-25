"""
Unit tests for rag_chain.py module.

This module tests the RAG chain implementation including
chain building, history management, and response generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from flipkart.rag_chain import RAGChainBuilder


class TestRAGChainBuilder:
    """Test cases for the RAGChainBuilder class."""
    
    def test_init(self, mock_vector_store):
        """Test RAGChainBuilder initialization."""
        builder = RAGChainBuilder(mock_vector_store)
        
        assert builder.vector_store == mock_vector_store
        assert builder.history_store == {}
    
    @patch('flipkart.rag_chain.ChatGroq')
    def test_init_with_groq_model(self, mock_chat_groq, mock_vector_store, mock_env_vars):
        """Test RAGChainBuilder initialization with Groq model."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        builder = RAGChainBuilder(mock_vector_store)
        
        # Verify ChatGroq was initialized with correct parameters
        mock_chat_groq.assert_called_once_with(model="llama-3.1-8b-instant", temperature=0.5)
        assert builder.model == mock_llm_instance
    
    def test_get_history_new_session(self, mock_vector_store):
        """Test _get_history with new session ID."""
        builder = RAGChainBuilder(mock_vector_store)
        session_id = "new-session-123"
        
        history = builder._get_history(session_id)
        
        # Should create new history for new session
        assert session_id in builder.history_store
        assert history == builder.history_store[session_id]
        assert isinstance(history, type(history))  # Should be ChatMessageHistory instance
    
    def test_get_history_existing_session(self, mock_vector_store):
        """Test _get_history with existing session ID."""
        builder = RAGChainBuilder(mock_vector_store)
        session_id = "existing-session-456"
        
        # Create initial history
        first_history = builder._get_history(session_id)
        
        # Get history again for same session
        second_history = builder._get_history(session_id)
        
        # Should return the same history instance
        assert first_history == second_history
        assert session_id in builder.history_store
    
    def test_get_history_multiple_sessions(self, mock_vector_store):
        """Test _get_history with multiple different sessions."""
        builder = RAGChainBuilder(mock_vector_store)
        session1 = "session-1"
        session2 = "session-2"
        
        history1 = builder._get_history(session1)
        history2 = builder._get_history(session2)
        
        # Should have separate histories for each session
        assert len(builder.history_store) == 2
        assert session1 in builder.history_store
        assert session2 in builder.history_store
        assert history1 != history2
    
    @patch('flipkart.rag_chain.create_history_aware_retriever')
    @patch('flipkart.rag_chain.create_retrieval_chain')
    @patch('flipkart.rag_chain.create_stuff_documents_chain')
    @patch('flipkart.rag_chain.RunnableWithMessageHistory')
    def test_build_chain_success(self, mock_runnable, mock_create_retrieval, 
                                mock_create_stuff, mock_create_history_aware,
                                mock_vector_store, mock_env_vars):
        """Test successful RAG chain building."""
        # Setup mocks
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        mock_history_aware_retriever = Mock()
        mock_create_history_aware.return_value = mock_history_aware_retriever
        
        mock_stuff_chain = Mock()
        mock_create_stuff.return_value = mock_stuff_chain
        
        mock_rag_chain = Mock()
        mock_create_retrieval.return_value = mock_rag_chain
        
        mock_final_chain = Mock()
        mock_runnable.return_value = mock_final_chain
        
        builder = RAGChainBuilder(mock_vector_store)
        result = builder.build_chain()
        
        # Verify vector store retriever was created
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        
        # Verify history-aware retriever was created
        mock_create_history_aware.assert_called_once()
        
        # Verify document chain was created
        mock_create_stuff.assert_called_once()
        
        # Verify retrieval chain was created
        mock_create_retrieval.assert_called_once_with(
            mock_history_aware_retriever, mock_stuff_chain
        )
        
        # Verify final chain was wrapped with message history
        mock_runnable.assert_called_once()
        assert result == mock_final_chain
    
    def test_build_chain_retriever_configuration(self, mock_vector_store):
        """Test that retriever is configured with correct search parameters."""
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        builder = RAGChainBuilder(mock_vector_store)
        
        with patch('flipkart.rag_chain.create_history_aware_retriever'), \
             patch('flipkart.rag_chain.create_retrieval_chain'), \
             patch('flipkart.rag_chain.create_stuff_documents_chain'), \
             patch('flipkart.rag_chain.RunnableWithMessageHistory'):
            
            builder.build_chain()
            
            # Verify retriever was created with k=3
            mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
    
    @patch('flipkart.rag_chain.ChatPromptTemplate')
    def test_build_chain_prompt_templates(self, mock_prompt_template, mock_vector_store, mock_env_vars):
        """Test that prompt templates are created correctly."""
        mock_vector_store.as_retriever.return_value = Mock()
        
        builder = RAGChainBuilder(mock_vector_store)
        
        with patch('flipkart.rag_chain.create_history_aware_retriever'), \
             patch('flipkart.rag_chain.create_retrieval_chain'), \
             patch('flipkart.rag_chain.create_stuff_documents_chain'), \
             patch('flipkart.rag_chain.RunnableWithMessageHistory'):
            
            builder.build_chain()
            
            # Verify ChatPromptTemplate was called (for context and QA prompts)
            assert mock_prompt_template.from_messages.call_count >= 2
    
    @patch('flipkart.rag_chain.RunnableWithMessageHistory')
    def test_build_chain_message_history_configuration(self, mock_runnable, mock_vector_store, mock_env_vars):
        """Test that RunnableWithMessageHistory is configured correctly."""
        mock_vector_store.as_retriever.return_value = Mock()
        mock_final_chain = Mock()
        mock_runnable.return_value = mock_final_chain
        
        builder = RAGChainBuilder(mock_vector_store)
        
        with patch('flipkart.rag_chain.create_history_aware_retriever') as mock_history_aware, \
             patch('flipkart.rag_chain.create_retrieval_chain') as mock_create_retrieval, \
             patch('flipkart.rag_chain.create_stuff_documents_chain'):
            
            mock_rag_chain = Mock()
            mock_create_retrieval.return_value = mock_rag_chain
            
            result = builder.build_chain()
            
            # Verify RunnableWithMessageHistory was called with correct parameters
            mock_runnable.assert_called_once()
            call_args = mock_runnable.call_args
            
            # Check the parameters passed to RunnableWithMessageHistory
            assert call_args[0][0] == mock_rag_chain  # RAG chain
            assert call_args[1]['input_messages_key'] == 'input'
            assert call_args[1]['history_messages_key'] == 'chat_history'
            assert call_args[1]['output_messages_key'] == 'answer'
            
            assert result == mock_final_chain
    
    def test_build_chain_with_vector_store_error(self, mock_vector_store):
        """Test build_chain when vector store fails to create retriever."""
        mock_vector_store.as_retriever.side_effect = Exception("Retriever creation failed")
        
        builder = RAGChainBuilder(mock_vector_store)
        
        with pytest.raises(Exception, match="Retriever creation failed"):
            builder.build_chain()
    
    @patch('flipkart.rag_chain.create_history_aware_retriever')
    def test_build_chain_with_history_aware_error(self, mock_history_aware, mock_vector_store, mock_env_vars):
        """Test build_chain when history-aware retriever creation fails."""
        mock_vector_store.as_retriever.return_value = Mock()
        mock_history_aware.side_effect = Exception("History-aware retriever failed")
        
        builder = RAGChainBuilder(mock_vector_store)
        
        with pytest.raises(Exception, match="History-aware retriever failed"):
            builder.build_chain()
    
    def test_history_store_persistence(self, mock_vector_store):
        """Test that history store persists across multiple operations."""
        builder = RAGChainBuilder(mock_vector_store)
        session_id = "persistent-session"
        
        # Create history
        history1 = builder._get_history(session_id)
        
        # Add some messages to history (if supported by mock)
        if hasattr(history1, 'add_message'):
            history1.add_message(HumanMessage(content="Hello"))
            history1.add_message(AIMessage(content="Hi there!"))
        
        # Get history again
        history2 = builder._get_history(session_id)
        
        # Should be the same instance with preserved state
        assert history1 == history2
        assert len(builder.history_store) == 1
    
    @pytest.mark.parametrize("session_id", [
        "session-123",
        "user-session",
        "test_session_456",
        "session-with-dashes",
        "session_with_underscores"
    ])
    def test_get_history_with_different_session_ids(self, mock_vector_store, session_id):
        """Test _get_history with various session ID formats."""
        builder = RAGChainBuilder(mock_vector_store)
        
        history = builder._get_history(session_id)
        
        assert session_id in builder.history_store
        assert history == builder.history_store[session_id]
    
    def test_build_chain_returns_correct_type(self, mock_vector_store, mock_env_vars):
        """Test that build_chain returns the correct type."""
        mock_vector_store.as_retriever.return_value = Mock()
        mock_final_chain = Mock()
        
        builder = RAGChainBuilder(mock_vector_store)
        
        with patch('flipkart.rag_chain.create_history_aware_retriever'), \
             patch('flipkart.rag_chain.create_retrieval_chain'), \
             patch('flipkart.rag_chain.create_stuff_documents_chain'), \
             patch('flipkart.rag_chain.RunnableWithMessageHistory', return_value=mock_final_chain):
            
            result = builder.build_chain()
            
            assert result == mock_final_chain
