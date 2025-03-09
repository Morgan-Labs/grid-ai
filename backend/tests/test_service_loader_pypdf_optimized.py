"""Tests for the optimized PyPDF loader service."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.schema import Document as LangchainDocument

from app.services.loaders.pypdf_service import PDFLoader, PDF_TYPE_TEXT, PDF_TYPE_SCANNED, PDF_TYPE_MIXED


@pytest.fixture
def pdf_loader():
    """Create a PDF loader for testing."""
    return PDFLoader()


@pytest.mark.asyncio
@patch("app.services.loaders.pypdf_service.pypdf.PdfReader")
async def test_load_pdf_optimized(mock_pdf_reader, pdf_loader):
    """Test loading a PDF with optimized PyPDF."""
    # Mock the PDF reader
    mock_reader_instance = MagicMock()
    mock_pdf_reader.return_value = mock_reader_instance
    
    # Mock pages
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Test content page 1"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Test content page 2"
    
    mock_reader_instance.pages = [mock_page1, mock_page2]
    
    # Mock file hash to avoid caching issues
    with patch.object(pdf_loader, "_get_file_hash", return_value="test_hash"):
        # Mock PDF type detection
        with patch.object(pdf_loader, "_detect_pdf_type", return_value=PDF_TYPE_TEXT):
            # Call the load method
            result = await pdf_loader.load("test.pdf")
    
    # Verify the result
    assert len(result) == 2
    assert result[0].page_content == "Test content page 1"
    assert result[1].page_content == "Test content page 2"


@pytest.mark.asyncio
@patch("app.services.loaders.pypdf_service.partition_pdf")
async def test_load_with_optimized_unstructured(mock_partition_pdf, pdf_loader):
    """Test loading a PDF with optimized Unstructured."""
    # Mock the partition_pdf function
    mock_element1 = MagicMock()
    mock_element1.text = "Element 1 content"
    mock_element1.metadata = {"page_number": 1}
    
    mock_element2 = MagicMock()
    mock_element2.text = "Element 2 content"
    mock_element2.metadata = {"page_number": 2}
    
    mock_partition_pdf.return_value = [mock_element1, mock_element2]
    
    # Mock file hash to avoid caching issues
    with patch.object(pdf_loader, "_get_file_hash", return_value="test_hash"):
        # Mock PDF type detection
        with patch.object(pdf_loader, "_detect_pdf_type", return_value=PDF_TYPE_SCANNED):
            # Call the load method
            result = await pdf_loader.load("test.pdf")
    
    # Verify the result
    assert len(result) == 2
    assert result[0].page_content == "Element 1 content"
    assert result[1].page_content == "Element 2 content"
    
    # Verify that partition_pdf was called with the correct parameters
    mock_partition_pdf.assert_called_once_with(
        "test.pdf",
        strategy="fast",
        infer_table_structure=False,
        extract_images_in_pdf=False,
    )


@pytest.mark.asyncio
@patch("app.services.loaders.pypdf_service.fitz.open")
async def test_detect_pdf_type_with_pymupdf(mock_fitz_open, pdf_loader):
    """Test detecting PDF type with PyMuPDF."""
    # Mock the fitz.open function
    mock_doc = MagicMock()
    mock_fitz_open.return_value = mock_doc
    
    # Mock pages
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "This is a text page with lots of content " * 10
    mock_page1.get_images.return_value = []
    
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Another text page"
    mock_page2.get_images.return_value = [("image1", 0, 0, 100, 100)]
    
    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
    mock_doc.__len__.return_value = 2
    
    # Call the detect_pdf_type method
    result = await pdf_loader._detect_pdf_type("test.pdf")
    
    # Verify the result
    assert result == PDF_TYPE_MIXED  # Should be mixed because it has both text and images


@pytest.mark.asyncio
async def test_determine_optimal_batch_size(pdf_loader):
    """Test determining optimal batch size based on document size."""
    # Test with different page counts
    assert pdf_loader._determine_optimal_batch_size(5) == 5
    assert pdf_loader._determine_optimal_batch_size(15) == 15
    assert pdf_loader._determine_optimal_batch_size(25) == 10
    assert pdf_loader._determine_optimal_batch_size(60) == 15
    assert pdf_loader._determine_optimal_batch_size(120) == 20


@pytest.mark.asyncio
@patch("builtins.open", new_callable=MagicMock)
@patch("hashlib.md5")
async def test_get_file_hash(mock_md5, mock_open, pdf_loader):
    """Test generating a file hash for caching."""
    # Mock the md5 hash
    mock_md5_instance = MagicMock()
    mock_md5_instance.hexdigest.return_value = "test_hash_value"
    mock_md5.return_value = mock_md5_instance
    
    # Mock the file read
    mock_file = MagicMock()
    mock_file.read.return_value = b"test file content"
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Call the get_file_hash method
    result = await pdf_loader._get_file_hash("test.pdf")
    
    # Verify the result
    assert result == "test_hash_value"
    mock_file.read.assert_called_once_with(8192)  # Should read first 8KB


@pytest.mark.asyncio
@patch("app.services.loaders.pypdf_service.PDF_CACHE")
def test_update_cache(mock_cache, pdf_loader):
    """Test updating the cache with new documents."""
    # Mock the cache
    mock_cache.__len__.return_value = 100  # Cache is full
    mock_cache.items.return_value = [("old_hash", (0, [])), ("newer_hash", (1, []))]
    
    # Create test documents
    documents = [
        LangchainDocument(page_content="Test content", metadata={"page": 1})
    ]
    
    # Call the update_cache method
    pdf_loader._update_cache("test_hash", documents)
    
    # Verify the cache was updated
    mock_cache.__setitem__.assert_called_once()
    
    # When cache is full, it should remove the oldest entry
    if mock_cache.__len__.return_value >= 100:
        mock_cache.pop.assert_called_once_with("old_hash")
