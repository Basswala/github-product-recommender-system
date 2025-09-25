"""
Unit tests for data_converter.py module.

This module tests the CSV to LangChain Document conversion functionality,
including file handling, data validation, and document creation.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from langchain_core.documents import Document
from flipkart.data_converter import DataConverter


class TestDataConverter:
    """Test cases for the DataConverter class."""
    
    def test_converter_initialization(self):
        """Test DataConverter initialization with file path."""
        file_path = "test_data.csv"
        converter = DataConverter(file_path)
        
        assert converter.file_path == file_path
    
    def test_convert_with_valid_csv(self, temp_csv_file, sample_csv_data):
        """Test conversion with valid CSV data."""
        # Create a proper CSV with expected columns
        csv_data = pd.DataFrame({
            'product_title': sample_csv_data['Product_Name'],
            'review': sample_csv_data['Review'],
            'extra_column': ['extra'] * len(sample_csv_data)  # Should be ignored
        })
        csv_data.to_csv(temp_csv_file, index=False)
        
        converter = DataConverter(str(temp_csv_file))
        documents = converter.convert()
        
        # Verify correct number of documents
        assert len(documents) == len(sample_csv_data)
        
        # Verify document structure
        for i, doc in enumerate(documents):
            assert isinstance(doc, Document)
            assert doc.page_content == sample_csv_data.iloc[i]['Review']
            assert doc.metadata['product_name'] == sample_csv_data.iloc[i]['Product_Name']
    
    def test_convert_with_minimal_csv(self, tmp_path):
        """Test conversion with minimal valid CSV."""
        csv_file = tmp_path / "minimal.csv"
        csv_data = pd.DataFrame({
            'product_title': ['Test Product'],
            'review': ['Great product!']
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        assert len(documents) == 1
        assert documents[0].page_content == 'Great product!'
        assert documents[0].metadata['product_name'] == 'Test Product'
    
    def test_convert_with_empty_csv(self, tmp_path):
        """Test conversion with empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_data = pd.DataFrame(columns=['product_title', 'review'])
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        assert len(documents) == 0
    
    def test_convert_with_missing_columns(self, tmp_path):
        """Test conversion with missing required columns."""
        csv_file = tmp_path / "missing_columns.csv"
        csv_data = pd.DataFrame({
            'product_title': ['Test Product'],
            'rating': [5]  # Missing 'review' column
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        
        with pytest.raises(KeyError, match="review"):
            converter.convert()
    
    def test_convert_with_nonexistent_file(self):
        """Test conversion with non-existent file."""
        converter = DataConverter("nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            converter.convert()
    
    def test_convert_with_malformed_csv(self, tmp_path):
        """Test conversion with malformed CSV file."""
        csv_file = tmp_path / "malformed.csv"
        csv_file.write_text("product_title,review\nInvalid,CSV\nContent\n")
        
        converter = DataConverter(str(csv_file))
        
        # This should raise a pandas parsing error
        with pytest.raises(Exception):  # Could be various pandas exceptions
            converter.convert()
    
    def test_convert_preserves_data_types(self, tmp_path):
        """Test that conversion preserves different data types in metadata."""
        csv_file = tmp_path / "data_types.csv"
        csv_data = pd.DataFrame({
            'product_title': ['Product 1', 'Product 2'],
            'review': ['Review 1', 'Review 2']
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        for doc in documents:
            assert isinstance(doc.metadata['product_name'], str)
            assert isinstance(doc.page_content, str)
    
    def test_convert_with_special_characters(self, tmp_path):
        """Test conversion with special characters in data."""
        csv_file = tmp_path / "special_chars.csv"
        csv_data = pd.DataFrame({
            'product_title': ['Product with "quotes" & symbols'],
            'review': ['Review with émojis 🎉 and ñ characters']
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        assert len(documents) == 1
        assert 'émojis 🎉 and ñ characters' in documents[0].page_content
        assert 'Product with "quotes" & symbols' in documents[0].metadata['product_name']
    
    def test_convert_with_unicode_data(self, tmp_path):
        """Test conversion with Unicode data."""
        csv_file = tmp_path / "unicode.csv"
        csv_data = pd.DataFrame({
            'product_title': ['商品名称'],
            'review': ['这是一个很好的产品评论']
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        assert len(documents) == 1
        assert documents[0].page_content == '这是一个很好的产品评论'
        assert documents[0].metadata['product_name'] == '商品名称'
    
    def test_convert_filters_columns_correctly(self, tmp_path):
        """Test that conversion only uses required columns."""
        csv_file = tmp_path / "multiple_columns.csv"
        csv_data = pd.DataFrame({
            'product_title': ['Product 1', 'Product 2'],
            'review': ['Review 1', 'Review 2'],
            'rating': [5, 4],
            'price': [100, 200],
            'category': ['Electronics', 'Books']
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        # Verify that only product_name is in metadata
        for doc in documents:
            assert 'product_name' in doc.metadata
            assert 'rating' not in doc.metadata
            assert 'price' not in doc.metadata
            assert 'category' not in doc.metadata
    
    @pytest.mark.parametrize("file_path", [
        "relative_path.csv",
        "/absolute/path.csv",
        "path/with/spaces.csv",
        "path-with-dashes.csv"
    ])
    def test_converter_with_different_paths(self, file_path):
        """Test DataConverter with different file path formats."""
        converter = DataConverter(file_path)
        assert converter.file_path == file_path
    
    def test_convert_with_whitespace_in_data(self, tmp_path):
        """Test conversion with whitespace in product titles and reviews."""
        csv_file = tmp_path / "whitespace.csv"
        csv_data = pd.DataFrame({
            'product_title': ['  Product with spaces  ', '\tProduct with tabs\t'],
            'review': ['  Review with leading spaces  ', 'Review with\ttabs\t']
        })
        csv_data.to_csv(csv_file, index=False)
        
        converter = DataConverter(str(csv_file))
        documents = converter.convert()
        
        # Note: pandas.read_csv doesn't strip whitespace by default
        # This test verifies current behavior
        assert len(documents) == 2
        assert documents[0].metadata['product_name'] == '  Product with spaces  '
        assert documents[0].page_content == '  Review with leading spaces  '
