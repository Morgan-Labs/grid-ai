"""Test the TextractLoader service with S3 integration."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.schema import Document as LangchainDocument

from app.core.config import Settings
from app.services.loaders.textract_service import TextractLoader


@pytest.fixture
def settings():
    """Create a settings object for testing."""
    return Settings(
        aws_region="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_session_token="test-session-token",
        s3_bucket_name="test-bucket",
        s3_prefix="test-documents"
    )


@pytest.fixture
def textract_loader(settings):
    """Create a TextractLoader instance for testing."""
    return TextractLoader(settings=settings)


@pytest.mark.asyncio
async def test_load_unsupported_file_type(textract_loader):
    """Test loading an unsupported file type."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        await textract_loader.load("test.docx")


@pytest.mark.asyncio
@patch("app.services.loaders.textract_service.TextractLoader._upload_to_s3")
@patch("app.services.loaders.textract_service.TextractLoader._process_with_textract_s3")
async def test_load_pdf_with_textract(mock_process_with_textract_s3, mock_upload_to_s3, textract_loader):
    """Test loading a PDF file with Textract using S3."""
    # Setup mocks
    mock_upload_to_s3.return_value = "test-documents/test-uuid/test.pdf"
    
    # Mock the process_with_textract_s3 method to return documents
    mock_documents = [
        LangchainDocument(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1}
        )
    ]
    mock_process_with_textract_s3.return_value = mock_documents
    
    # Call the method
    result = await textract_loader.load("test.pdf")
    
    # Assertions
    assert result == mock_documents
    mock_upload_to_s3.assert_called_once_with("test.pdf")
    mock_process_with_textract_s3.assert_called_once_with(
        "test-documents/test-uuid/test.pdf", 
        "test.pdf"
    )


@pytest.mark.asyncio
@patch("app.services.loaders.textract_service.boto3.client")
@patch("app.services.loaders.textract_service.asyncio.get_event_loop")
async def test_upload_to_s3(mock_get_event_loop, mock_boto3_client, textract_loader):
    """Test uploading a file to S3."""
    # Setup mocks
    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client
    
    mock_loop = MagicMock()
    mock_get_event_loop.return_value = mock_loop
    mock_loop.run_in_executor.return_value = None
    
    # Call the method
    with patch("app.services.loaders.textract_service.uuid.uuid4", return_value="test-uuid"):
        s3_key = await textract_loader._upload_to_s3("test.pdf")
    
    # Assertions
    assert s3_key == "test-documents/test-uuid/test.pdf"
    mock_boto3_client.assert_called_once_with(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_session_token="test-session-token"
    )
    mock_loop.run_in_executor.assert_called_once()


@pytest.mark.asyncio
@patch("app.services.loaders.textract_service.boto3.client")
@patch("app.services.loaders.textract_service.AmazonTextractPDFLoader")
@patch("app.services.loaders.textract_service.asyncio.get_event_loop")
async def test_process_with_textract_s3(mock_get_event_loop, mock_textract_loader, mock_boto3_client, textract_loader):
    """Test processing a document with Textract using S3."""
    # Setup mocks
    mock_textract_client = MagicMock()
    mock_boto3_client.return_value = mock_textract_client
    
    mock_loader_instance = MagicMock()
    mock_textract_loader.return_value = mock_loader_instance
    
    mock_loop = MagicMock()
    mock_get_event_loop.return_value = mock_loop
    
    # Mock the load method to return documents
    mock_documents = [
        LangchainDocument(
            page_content="Test content",
            metadata={"source": "s3://test-bucket/test-documents/test-uuid/test.pdf", "page": 1}
        )
    ]
    mock_loop.run_in_executor.return_value = mock_documents
    
    # Call the method
    result = await textract_loader._process_with_textract_s3(
        "test-documents/test-uuid/test.pdf", 
        "test.pdf"
    )
    
    # Assertions
    assert len(result) == 1
    assert result[0].page_content == "Test content"
    assert result[0].metadata["source"] == "test.pdf"  # Should be updated to original path
    assert result[0].metadata["page"] == 1
    
    mock_boto3_client.assert_called_once_with(
        "textract",
        region_name="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_session_token="test-session-token"
    )
    mock_textract_loader.assert_called_once_with(
        "s3://test-bucket/test-documents/test-uuid/test.pdf",
        client=mock_textract_client
    )


@pytest.mark.asyncio
@patch("app.services.loaders.textract_service.TextractLoader._upload_to_s3")
@patch("app.services.loaders.textract_service.TextractLoader._process_with_textract_s3")
async def test_load_pdf_with_cache(mock_process_with_textract_s3, mock_upload_to_s3, textract_loader):
    """Test loading a PDF file with cache."""
    # Setup mocks
    mock_upload_to_s3.return_value = "test-documents/test-uuid/test.pdf"
    
    # Mock the process_with_textract_s3 method to return documents
    mock_documents = [
        LangchainDocument(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1}
        )
    ]
    mock_process_with_textract_s3.return_value = mock_documents
    
    # Call the method twice
    result1 = await textract_loader.load("test.pdf")
    result2 = await textract_loader.load("test.pdf")
    
    # Assertions
    assert result1 == mock_documents
    assert result2 == mock_documents
    # Upload and process should only be called once due to caching
    mock_upload_to_s3.assert_called_once()
    mock_process_with_textract_s3.assert_called_once()


@pytest.mark.asyncio
@patch("app.services.loaders.textract_service.boto3.client")
@patch("app.services.loaders.textract_service.AmazonTextractPDFLoader")
@patch("app.services.loaders.textract_service.asyncio.get_event_loop")
async def test_process_with_textract_s3_empty_result(mock_get_event_loop, mock_textract_loader, mock_boto3_client, textract_loader):
    """Test processing a document with Textract using S3 with empty result."""
    # Setup mocks
    mock_textract_client = MagicMock()
    mock_boto3_client.return_value = mock_textract_client
    
    mock_loader_instance = MagicMock()
    mock_textract_loader.return_value = mock_loader_instance
    
    mock_loop = MagicMock()
    mock_get_event_loop.return_value = mock_loop
    
    # Mock the load method to return an empty list
    mock_loop.run_in_executor.return_value = []
    
    # Call the method
    result = await textract_loader._process_with_textract_s3(
        "test-documents/test-uuid/test.pdf", 
        "test.pdf"
    )
    
    # Assertions
    assert len(result) == 1
    assert "Empty document" in result[0].page_content
    assert result[0].metadata["source"] == "test.pdf"
    assert result[0].metadata["page"] == 1


@pytest.mark.asyncio
@patch("app.services.loaders.textract_service.boto3.client")
@patch("app.services.loaders.textract_service.AmazonTextractPDFLoader")
@patch("app.services.loaders.textract_service.asyncio.get_event_loop")
async def test_process_with_textract_s3_exception(mock_get_event_loop, mock_textract_loader, mock_boto3_client, textract_loader):
    """Test processing a document with Textract using S3 with exception."""
    # Setup mocks
    mock_textract_client = MagicMock()
    mock_boto3_client.return_value = mock_textract_client
    
    mock_loader_instance = MagicMock()
    mock_textract_loader.return_value = mock_loader_instance
    
    mock_loop = MagicMock()
    mock_get_event_loop.return_value = mock_loop
    
    # Mock the load method to raise an exception
    mock_loop.run_in_executor.side_effect = Exception("Test error")
    
    # Call the method
    result = await textract_loader._process_with_textract_s3(
        "test-documents/test-uuid/test.pdf", 
        "test.pdf"
    )
    
    # Assertions
    assert len(result) == 1
    assert "Error processing document with Textract" in result[0].page_content
    assert result[0].metadata["source"] == "test.pdf"
    assert result[0].metadata["page"] == 1
    assert result[0].metadata["error"] == "Test error"
