"""
Data Ingestion Test Script
==========================

This script tests the complete data ingestion pipeline for the product recommender system.
It loads Flipkart product review data from CSV, converts it to vector embeddings,
and stores them in AstraDB for similarity search.

Usage:
    python main.py

Prerequisites:
    - .env file with AstraDB credentials
    - Data/flipkart_product_review.csv file
    - Internet connection for model downloads
"""

import logging
from flipkart.data_ingestion import DataIngestor

# Configure logging to show detailed information during the ingestion process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to test the data ingestion process.

    This function:
    1. Initializes the DataIngestor with AstraDB connection
    2. Processes the CSV file and converts reviews to embeddings
    3. Stores the embeddings in the vector database
    4. Provides troubleshooting information if errors occur

    Raises:
        Exception: Re-raises any exceptions after logging them for debugging
    """
    try:
        print("Starting product recommender data ingestion...")

        # Initialize data ingestor - connects to AstraDB and sets up embedding model
        ingestor = DataIngestor()

        # Test with load_existing=False to actually ingest new data from CSV
        # This will read the CSV, convert reviews to embeddings, and store in vector DB
        print("Ingesting data from CSV file...")
        vector_store = ingestor.ingest(load_existing=False)

        print("✅ Data ingestion completed successfully!")
        print(f"Vector store type: {type(vector_store)}")

    except Exception as e:
        print(f"❌ Error during data ingestion: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a .env file with valid AstraDB credentials")
        print("2. Check that Data/flipkart_product_review.csv exists")
        print("3. Verify your internet connection for HuggingFace model download")
        raise

# Entry point for script execution
if __name__ == "__main__":
    main()
