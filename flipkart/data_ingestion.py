"""
Data Ingestion Module for Product Recommender
===========================================

This module handles the ingestion of product review data into a vector database.
It processes CSV files, converts them to embeddings, and stores them in AstraDB
for similarity search and retrieval in the RAG system.

Key Components:
- DataIngestor: Main class that orchestrates the ingestion process
- AstraDB integration for vector storage
- HuggingFace embeddings for text vectorization
- Comprehensive error handling and logging

Flow:
1. Connect to AstraDB vector store
2. Load and validate CSV data
3. Convert reviews to LangChain documents
4. Generate embeddings and store in vector database
"""

import logging
from pathlib import Path
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.vectorstores import VectorStore
from flipkart.data_converter import DataConverter
from flipkart.config import Config

# Set up logger for this module
logger = logging.getLogger(__name__)

class DataIngestor:
    """
    Data ingestion class for loading product reviews into AstraDB vector store.

    This class handles the complete pipeline from CSV data to vector embeddings:
    1. Establishes connection to AstraDB vector store
    2. Initializes HuggingFace embedding model
    3. Converts CSV product reviews to LangChain documents
    4. Generates vector embeddings and stores them in the database

    The class supports both fresh data ingestion and loading existing vector stores,
    making it suitable for both initial setup and production use.
    """
    
    def __init__(self) -> None:
        """
        Initialize the DataIngestor with embedding model and vector store.

        This constructor:
        1. Validates all required environment variables are present
        2. Initializes HuggingFace embedding model for text vectorization
        3. Establishes connection to AstraDB vector store
        4. Sets up the collection for storing product review embeddings

        Raises:
            ValueError: If required environment variables are missing
            ConnectionError: If unable to connect to AstraDB
            Exception: If embedding model initialization fails
        """
        try:
            logger.info("Initializing DataIngestor")

            # Validate all required environment variables are present
            self._validate_config()

            # Initialize HuggingFace embedding model
            # This model will convert text reviews into dense vector representations
            self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model: {Config.EMBEDDING_MODEL}")

            # Set up AstraDB vector store connection
            # This will store and enable similarity search on review embeddings
            self.vstore = AstraDBVectorStore(
                embedding=self.embedding,                          # Embedding model instance
                collection_name="flipkart_database",              # Collection name in AstraDB
                api_endpoint=Config.ASTRA_DB_API_ENDPOINT,        # Database endpoint
                token=Config.ASTRA_DB_APPLICATION_TOKEN,          # Authentication token
                namespace=Config.ASTRA_DB_KEYSPACE                # Database keyspace
            )
            logger.info("Successfully connected to AstraDB vector store")

        except Exception as e:
            logger.error(f"Failed to initialize DataIngestor: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """
        Validate that all required configuration variables are present.

        This method checks that all necessary environment variables for AstraDB
        connection are properly set before attempting to connect.

        Raises:
            ValueError: If any required environment variables are missing
        """
        # List of required environment variables for AstraDB connection
        required_vars = [
            ("ASTRA_DB_API_ENDPOINT", Config.ASTRA_DB_API_ENDPOINT),
            ("ASTRA_DB_APPLICATION_TOKEN", Config.ASTRA_DB_APPLICATION_TOKEN),
            ("ASTRA_DB_KEYSPACE", Config.ASTRA_DB_KEYSPACE)
        ]

        # Check which variables are missing (None or empty)
        missing_vars = [var_name for var_name, var_value in required_vars if var_value is None]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def ingest(self, load_existing: bool = True) -> VectorStore:
        """
        Ingest data into the vector store.

        This method handles two scenarios:
        1. load_existing=True: Returns existing vector store for production use
        2. load_existing=False: Processes CSV file and adds new documents to vector store

        The ingestion process:
        1. Loads CSV file containing product reviews
        2. Converts reviews to LangChain Document objects
        3. Generates embeddings for each review
        4. Stores embeddings in AstraDB for similarity search

        Args:
            load_existing (bool): If True, return existing vector store without adding new documents.
                                If False, load and add documents from CSV file.

        Returns:
            VectorStore: The configured vector store instance ready for retrieval

        Raises:
            FileNotFoundError: If the CSV file doesn't exist at expected location
            ValueError: If the CSV file is empty or malformed
            Exception: If document ingestion or embedding generation fails
        """
        try:
            logger.info(f"Starting data ingestion with load_existing={load_existing}")

            # For production use, return existing vector store without processing new data
            if load_existing:
                logger.info("Returning existing vector store for production use")
                return self.vstore

            # Path to the CSV file containing product reviews
            # Note: Case-sensitive path - ensure 'Data' folder exists with correct capitalization
            csv_path = Path("Data/flipkart_product_review.csv")

            # Verify the CSV file exists before processing
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found at: {csv_path}")

            logger.info(f"Loading and converting documents from: {csv_path}")
            # Convert CSV data to LangChain Document objects
            docs = DataConverter(str(csv_path)).convert()

            # Ensure we have documents to process
            if not docs:
                raise ValueError("No documents found in the CSV file")

            logger.info(f"Successfully converted {len(docs)} documents to LangChain format")
            logger.info("Generating embeddings and adding documents to vector store...")

            # Add documents to vector store in batches for better performance
            # This will: 1. Generate embeddings for each review using HuggingFace model
            # 2. Store embeddings in AstraDB for similarity search
            batch_size = 50  # Process in smaller batches to avoid timeouts
            total_docs = len(docs)

            for i in range(0, total_docs, batch_size):
                batch = docs[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_docs + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                self.vstore.add_documents(batch)

            logger.info("Successfully added all documents to vector store")

            return self.vstore
            
        except FileNotFoundError as e:
            logger.error(f"CSV file not found: {str(e)}")
            logger.error("Please ensure Data/flipkart_product_review.csv exists")
            raise
        except ValueError as e:
            logger.error(f"Data validation error: {str(e)}")
            logger.error("Please check CSV file format and content")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data ingestion: {str(e)}")
            logger.error("This could be due to network issues, API limits, or database connectivity")
            raise
        
