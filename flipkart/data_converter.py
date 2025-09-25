"""
CSV to LangChain Document Converter
=================================

This module handles the conversion of CSV product review data into LangChain Document objects
that can be processed by the vector store and RAG chain.

The conversion process:
1. Reads CSV file containing product reviews
2. Extracts relevant columns (product_title, review)
3. Creates LangChain Document objects with review text as content
4. Adds product title as metadata for retrieval context
"""

import pandas as pd
from langchain_core.documents import Document


class DataConverter:
    """
    Converts CSV product review data to LangChain Document format.

    This class reads a CSV file containing product reviews and converts each review
    into a LangChain Document object suitable for vector storage and retrieval.

    Attributes:
        file_path (str): Path to the CSV file containing product review data
    """

    def __init__(self, file_path: str):
        """
        Initialize the DataConverter with a CSV file path.

        Args:
            file_path (str): Path to the CSV file containing product reviews
                           Expected columns: 'product_title', 'review'
        """
        self.file_path = file_path

    def convert(self):
        """
        Convert CSV data to LangChain Document objects.

        This method:
        1. Reads the CSV file using pandas
        2. Filters to keep only relevant columns (product_title, review)
        3. Creates Document objects with review text as content
        4. Adds product title as metadata for context during retrieval

        Returns:
            List[Document]: List of LangChain Document objects, where each document
                          contains a product review as content and product title as metadata

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            KeyError: If required columns are missing from the CSV
            pandas.errors.EmptyDataError: If the CSV file is empty
        """
        # Read CSV and select only the columns we need for the recommender system
        df = pd.read_csv(self.file_path)[["product_title", "review"]]

        # Filter out empty or null reviews for better quality
        df = df.dropna(subset=['review', 'product_title'])
        df = df[df['review'].str.strip() != '']

        # Convert each row to a LangChain Document
        # The review text becomes the searchable content
        # The product title is stored as metadata for context
        docs = [
            Document(
                page_content=str(row['review']).strip(),
                metadata={
                    "product_name": str(row["product_title"]).strip(),
                    "source": "flipkart_reviews"
                }
            )
            for _, row in df.iterrows()
            if len(str(row['review']).strip()) > 10  # Filter very short reviews
        ]

        return docs