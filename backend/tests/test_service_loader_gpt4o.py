"""Tests for the GPT-4o PDF loader service."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.schema import Document as LangchainDocument

from app.core.config import Settings
from app.services.loaders.gpt4o_pdf_service import GPT4OPDFLoader


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.openai_api_key = "test-api-key"
    return settings


@pytest.fixture
def gpt4o_loader(mock_settings):
    """Create a GPT-4o PDF loader for testing."""
    return GPT4OPDFLoader(settings=mock_settings)


@pytest.mark.asyncio
@patch("langchain_community.document_loaders.PyMuPDFLoader")
@patch("langchain_openai.ChatOpenAI")
@patch("langchain_community.document_loaders.parsers.LLMImageBlobParser")
async def test_load_with_table_extraction(
    mock_image_parser, mock_chat_openai, mock_pymupdf_loader, gpt4o_loader
):
    """Test loading a PDF with table extraction."""
    # Mock the table extraction loader
    mock_table_loader_instance = MagicMock()
    mock_pymupdf_loader.return_value = mock_table_loader_instance
    
    # Create mock documents
    mock_docs = [
        LangchainDocument(
            page_content="Test content page 1",
            metadata={"source": "test.pdf", "page": 1}
        ),
        LangchainDocument(
            page_content="Test content page 2",
            metadata={"source": "test.pdf", "page": 2}
        )
    ]
    mock_table_loader_instance.load.return_value = mock_docs
    
    # Call the load method
    result = await gpt4o_loader.load("test.pdf")
    
    # Verify the result
    assert len(result) == 2
    assert result[0].page_content == "Test content page 1"
    assert result[1].page_content == "Test content page 2"
    
    # Verify that the table extraction loader was created with the correct parameters
    mock_pymupdf_loader.assert_called_once_with(
        "test.pdf",
        extract_tables="markdown"
    )


@pytest.mark.asyncio
@patch("langchain_community.document_loaders.PyMuPDFLoader")
@patch("langchain_openai.ChatOpenAI")
@patch("langchain_community.document_loaders.parsers.LLMImageBlobParser")
async def test_load_with_gpt4o_image_extraction(
    mock_image_parser, mock_chat_openai, mock_pymupdf_loader, gpt4o_loader
):
    """Test loading a PDF with GPT-4o image extraction when table extraction fails."""
    # Mock the table extraction loader to fail
    mock_table_loader_instance = MagicMock()
    mock_pymupdf_loader.return_value = mock_table_loader_instance
    mock_table_loader_instance.load.side_effect = Exception("Table extraction failed")
    
    # Mock the image extraction loader
    mock_image_loader_instance = MagicMock()
    mock_pymupdf_loader.return_value = mock_image_loader_instance
    
    # Create mock documents for image extraction
    mock_docs = [
        LangchainDocument(
            page_content="Image extracted content page 1",
            metadata={"source": "test.pdf", "page": 1}
        ),
        LangchainDocument(
            page_content="Image extracted content page 2",
            metadata={"source": "test.pdf", "page": 2}
        )
    ]
    mock_image_loader_instance.load.return_value = mock_docs
    
    # Mock the ChatOpenAI and LLMImageBlobParser
    mock_chat_instance = MagicMock()
    mock_chat_openai.return_value = mock_chat_instance
    
    mock_parser_instance = MagicMock()
    mock_image_parser.return_value = mock_parser_instance
    
    # Call the load method
    with patch.object(gpt4o_loader, "extract_tables", False):  # Skip table extraction
        result = await gpt4o_loader.load("test.pdf")
    
    # Verify the result
    assert len(result) == 2
    assert "Image extracted content" in result[0].page_content
    
    # Verify that the image extraction was attempted
    mock_chat_openai.assert_called_once_with(
        api_key="test-api-key",
        model="gpt-4o",
        max_tokens=1024,
        temperature=0
    )
    mock_image_parser.assert_called_once_with(model=mock_chat_instance)


@pytest.mark.asyncio
@patch("langchain_community.document_loaders.PyMuPDFLoader")
async def test_load_with_standard_pymupdf_fallback(
    mock_pymupdf_loader, gpt4o_loader
):
    """Test loading a PDF with standard PyMuPDF when other methods fail."""
    # Mock all extraction methods to fail except the last standard one
    mock_pymupdf_loader.side_effect = [
        MagicMock(load=MagicMock(side_effect=Exception("Table extraction failed"))),
        MagicMock(load=MagicMock(side_effect=Exception("Image extraction failed"))),
        MagicMock(load=MagicMock(return_value=[
            LangchainDocument(
                page_content="Standard PyMuPDF content",
                metadata={"source": "test.pdf", "page": 1}
            )
        ]))
    ]
    
    # Call the load method
    with patch.object(gpt4o_loader, "extract_tables", False), \
         patch.object(gpt4o_loader, "extract_images", False):
        result = await gpt4o_loader.load("test.pdf")
    
    # Verify the result
    assert len(result) == 1
    assert "Standard PyMuPDF content" in result[0].page_content


@pytest.mark.asyncio
@patch("langchain_community.document_loaders.PyMuPDFLoader")
async def test_load_with_all_methods_failing(
    mock_pymupdf_loader, gpt4o_loader
):
    """Test loading a PDF when all extraction methods fail."""
    # Mock all extraction methods to fail
    mock_pymupdf_loader.side_effect = [
        MagicMock(load=MagicMock(side_effect=Exception("Table extraction failed"))),
        MagicMock(load=MagicMock(side_effect=Exception("Image extraction failed"))),
        MagicMock(load=MagicMock(side_effect=Exception("Standard extraction failed")))
    ]
    
    # Call the load method
    with patch.object(gpt4o_loader, "extract_tables", False), \
         patch.object(gpt4o_loader, "extract_images", False):
        result = await gpt4o_loader.load("test.pdf")
    
    # Verify the result is an empty list
    assert len(result) == 0


@pytest.mark.asyncio
async def test_load_unsupported_file_type(gpt4o_loader):
    """Test loading an unsupported file type."""
    with pytest.raises(ValueError) as excinfo:
        await gpt4o_loader.load("test.txt")
    
    assert "Unsupported file type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_configure(gpt4o_loader):
    """Test configuring the loader."""
    # Default configuration
    assert gpt4o_loader.extract_tables is True
    assert gpt4o_loader.extract_images is True
    assert gpt4o_loader.images_format == "markdown"
    
    # Configure the loader
    gpt4o_loader.configure(
        extract_tables=False,
        extract_images=True,
        images_format="html"
    )
    
    # Verify the configuration
    assert gpt4o_loader.extract_tables is False
    assert gpt4o_loader.extract_images is True
    assert gpt4o_loader.images_format == "html"
