"""Enhanced PyMuPDF loader service with OCR capabilities."""

import logging
import os
import importlib.util
from typing import List, Optional, Literal

from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser

from app.services.loaders.base import LoaderService
from app.core.config import Settings

logger = logging.getLogger(__name__)

class PyMuPDFLoaderService(LoaderService):
    """Enhanced PDF loader service using PyMuPDF with OCR capabilities."""

    def __init__(self, settings: Settings):
        """Initialize the PyMuPDF loader service."""
        self.settings = settings
        self.extract_images = True  # Default to extracting images
        self.ocr_enabled = True  # Default to enabling OCR
        self.images_format = "markdown"  # Default format for images

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path with enhanced image extraction and OCR."""
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading file with extension: {file_extension}")

        if file_extension == ".pdf":
            try:
                logger.info(f"Attempting to load PDF with PyMuPDFLoader: {file_path}")
                
                # Check if we should use OCR for images
                if self.ocr_enabled:
                    # Check if tesseract is installed
                    tesseract_available = self._check_tesseract_installed()
                    
                    if tesseract_available:
                        logger.info("Using Tesseract OCR for image extraction")
                        try:
                            # Create a loader with image extraction and OCR
                            loader = PyMuPDFLoader(
                                file_path,
                                extract_images=self.extract_images,
                                mode="page",  # Extract by page
                                images_inner_format=f"{self.images_format}-img",
                                images_parser=TesseractBlobParser()
                            )
                        except Exception as ocr_error:
                            logger.warning(f"Error setting up OCR: {str(ocr_error)}. Falling back to standard loader.")
                            loader = PyMuPDFLoader(file_path)
                    else:
                        logger.warning("Tesseract not available. Using standard PyMuPDFLoader.")
                        loader = PyMuPDFLoader(file_path)
                else:
                    # Use standard loader without OCR
                    loader = PyMuPDFLoader(file_path)
                
                # Load the documents
                documents = loader.load()
                
                # Check if we got any text content
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info(f"Successfully loaded PDF with PyMuPDFLoader: {len(documents)} pages")
                    return documents
                else:
                    logger.warning("PyMuPDFLoader returned empty content.")
                    
                    # Try with table extraction as a fallback
                    logger.info("Attempting to load with table extraction")
                    try:
                        table_loader = PyMuPDFLoader(
                            file_path,
                            extract_tables="markdown"
                        )
                        table_documents = table_loader.load()
                        
                        if table_documents and any(doc.page_content.strip() for doc in table_documents):
                            logger.info(f"Successfully loaded PDF with table extraction: {len(table_documents)} pages")
                            return table_documents
                    except Exception as table_error:
                        logger.error(f"Error extracting tables: {str(table_error)}")
                    
                    return []
            except Exception as e:
                logger.error(f"Error using PyMuPDFLoader: {str(e)}")
                return []
                
        elif file_extension == ".txt":
            try:
                logger.info(f"Loading text file: {file_path}")
                loader = TextLoader(file_path)
                documents = loader.load()
                logger.info(f"Successfully loaded text file: {len(documents)} documents")
                return documents
            except Exception as e:
                logger.error(f"Error loading text file: {str(e)}")
                return []
        else:
            error_msg = f"Unsupported file type: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _check_tesseract_installed(self) -> bool:
        """Check if Tesseract is installed and available."""
        try:
            # Check if pytesseract is installed
            if importlib.util.find_spec("pytesseract") is None:
                logger.warning("pytesseract package is not installed")
                return False
                
            # Try to import and check tesseract
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract is not properly installed: {str(e)}")
            return False
    
    def configure(self, 
                 extract_images: bool = True, 
                 ocr_enabled: bool = True,
                 images_format: Literal["markdown", "html", "text"] = "markdown") -> None:
        """Configure the loader settings.
        
        Args:
            extract_images: Whether to extract images from PDFs
            ocr_enabled: Whether to use OCR for extracted images
            images_format: Format for extracted images (markdown, html, or text)
        """
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.images_format = images_format
        logger.info(f"PyMuPDFLoader configured: extract_images={extract_images}, ocr_enabled={ocr_enabled}, images_format={images_format}")
