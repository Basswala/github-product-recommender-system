"""
Configuration Management for Product Recommender
===============================================

This module centralizes all configuration settings for the product recommender system.
It loads sensitive credentials from environment variables and defines model configurations.

Environment Variables Required:
    ASTRA_DB_API_ENDPOINT: DataStax Astra DB API endpoint URL
    ASTRA_DB_APPLICATION_TOKEN: Authentication token for Astra DB
    ASTRA_DB_KEYSPACE: Keyspace (namespace) in Astra DB
    GROQ_API_KEY: API key for Groq LLM service
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This should be called before accessing any environment variables
load_dotenv()


class Config:
    """
    Configuration class that holds all application settings.

    This class uses environment variables for sensitive data like API keys
    and database credentials, while keeping model configurations as constants.

    Attributes:
        ASTRA_DB_API_ENDPOINT: DataStax Astra DB connection endpoint
        ASTRA_DB_APPLICATION_TOKEN: Authentication token for database access
        ASTRA_DB_KEYSPACE: Database keyspace/namespace for data organization
        GROQ_API_KEY: API key for accessing Groq's LLM service
        EMBEDDING_MODEL: HuggingFace model for generating text embeddings
        RAG_MODEL: Groq model used for generating responses in RAG chain
    """

    # Database Configuration - loaded from environment variables
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

    # LLM Service Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model Configuration - these are constants and don't need environment variables
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # BGE model for high-quality embeddings
    RAG_MODEL = "llama-3.1-8b-instant"         # Fast Llama model for response generation