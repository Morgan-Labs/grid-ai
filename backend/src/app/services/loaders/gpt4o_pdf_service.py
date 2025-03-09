"""GPT-4o powered PDF loader service with intelligent extraction capabilities."""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI

from app.services.loaders.base import LoaderService
from app.core.config import Settings

logger = logging.getLogger(__name__)

# Cache for processed documents to avoid reprocessing
_document_cache: Dict[str, List[LangchainDocument]] = {}

class GPT4OPDFLoader(LoaderService):
    """PDF loader service using PyMuPDF with intelligent extraction method selection."""

    def __init__(self, settings: Settings):
        """Initialize the GPT-4o PDF loader service."""
        self.settings = settings
        self.openai_api_key = settings.openai_api_key
        self.extract_tables = True
        self.extract_images = True
        self.images_format = "markdown"
        self.use_cache = True
        self.max_cache_size = 100  # Maximum number of documents to cache
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path with intelligent extraction method selection.
        
        This loader analyzes the PDF content and chooses the most appropriate
        extraction method based on the document characteristics.
        """
        start_time = time.time()
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading file with extension: {file_extension} using GPT-4o enhanced loader")

        if file_extension != ".pdf":
            error_msg = f"Unsupported file type for GPT-4o PDF loader: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check cache first if enabled
        if self.use_cache and file_path in _document_cache:
            cache_entry = _document_cache[file_path]
            logger.info(f"Using cached document: {file_path}")
            return cache_entry
        
        try:
            logger.info(f"Attempting to load PDF with GPT-4o enhanced loader: {file_path}")
            
            # Analyze the PDF to determine the best extraction method
            has_images, has_tables, is_scanned = await self._analyze_pdf(file_path)
            
            documents = []
            
            # Choose extraction method based on document characteristics
            if is_scanned:
                # For scanned documents, use GPT-4o for best results
                documents = await self._extract_with_gpt4o(file_path)
            elif has_tables and self.extract_tables:
                # For documents with tables, try table extraction first
                try:
                    documents = await self._extract_with_tables(file_path)
                except Exception as e:
                    logger.warning(f"Table extraction failed: {str(e)}")
                    # Fall back to standard extraction if table extraction fails
                    if not documents:
                        documents = await self._extract_standard(file_path)
            elif has_images and self.extract_images:
                # For documents with images but no tables, use GPT-4o
                try:
                    documents = await self._extract_with_gpt4o(file_path)
                except Exception as e:
                    logger.warning(f"GPT-4o extraction failed: {str(e)}")
                    # Fall back to standard extraction if GPT-4o fails
                    if not documents:
                        documents = await self._extract_standard(file_path)
            else:
                # For simple text documents, use standard extraction
                documents = await self._extract_standard(file_path)
            
            # If we still don't have any content, return a placeholder
            if not documents or not any(doc.page_content.strip() for doc in documents):
                logger.warning(f"Could not extract content from PDF: {file_path}")
                documents = [LangchainDocument(
                    page_content=f"No content could be extracted from this document.",
                    metadata={"source": file_path, "page": 1}
                )]
            
            # Cache the result if caching is enabled
            if self.use_cache:
                # Limit cache size by removing oldest entries if needed
                if len(_document_cache) >= self.max_cache_size:
                    # Remove the first item (oldest)
                    _document_cache.pop(next(iter(_document_cache)))
                
                _document_cache[file_path] = documents
            
            processing_time = time.time() - start_time
            logger.info(f"PDF processing completed in {processing_time:.2f} seconds")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error using GPT-4o PDF loader: {str(e)}")
            return [LangchainDocument(
                page_content=f"Error processing document: {str(e)}",
                metadata={"source": file_path, "page": 1, "error": str(e)}
            )]
    
    async def _analyze_pdf(self, file_path: str) -> Tuple[bool, bool, bool]:
        """Analyze PDF to determine if it contains images, tables, or is scanned.
        
        Returns:
            Tuple of (has_images, has_tables, is_scanned)
        """
        try:
            # Run this in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._analyze_pdf_sync, file_path)
        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            # Default to conservative estimates
            return True, True, False
    
    def _analyze_pdf_sync(self, file_path: str) -> Tuple[bool, bool, bool]:
        """Synchronous version of _analyze_pdf for thread pool execution."""
        has_images = False
        has_tables = False
        is_scanned = False
        
        try:
            doc = fitz.open(file_path)
            
            # Check first few pages (up to 3) to determine document characteristics
            max_pages = min(3, len(doc))
            text_length = 0
            image_count = 0
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # Check for images
                images = page.get_images(full=True)
                image_count += len(images)
                
                # Check for text
                text = page.get_text()
                text_length += len(text)
                
                # Check for tables (simple heuristic based on text layout)
                if "table" in text.lower() or text.count("\n") > 10:
                    has_tables = True
            
            # Determine if document has images
            has_images = image_count > 0
            
            # Determine if document is likely scanned (high image count, low text)
            is_scanned = image_count > 0 and text_length < 500
            
            doc.close()
            
            logger.info(f"PDF analysis: has_images={has_images}, has_tables={has_tables}, is_scanned={is_scanned}")
            return has_images, has_tables, is_scanned
            
        except Exception as e:
            logger.error(f"Error in PDF analysis: {str(e)}")
            return False, False, False
    
    async def _extract_with_tables(self, file_path: str) -> List[LangchainDocument]:
        """Extract content from PDF with table extraction."""
        logger.info("Attempting to load with table extraction")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        table_loader = PyMuPDFLoader(
            file_path,
            extract_tables="markdown"
        )
        
        documents = await loop.run_in_executor(None, table_loader.load)
        
        if documents and any(doc.page_content.strip() for doc in documents):
            logger.info(f"Successfully loaded PDF with table extraction: {len(documents)} pages")
            return documents
        
        logger.warning("Table extraction returned empty content")
        return []
    
    async def _extract_with_gpt4o(self, file_path: str) -> List[LangchainDocument]:
        """Extract content from PDF with GPT-4o image extraction."""
        logger.info("Using GPT-4o for image extraction")
        
        try:
            # Create a ChatOpenAI model with the API key
            chat_model = ChatOpenAI(
                api_key=self.openai_api_key,
                model="gpt-4o",
                max_tokens=1024,
                temperature=0
            )
            
            # Create the image parser with the model
            image_parser = LLMImageBlobParser(model=chat_model)
            
            # Run in a try-except block to catch image processing errors
            try:
                # Create the loader with image extraction
                loader = PyMuPDFLoader(
                    file_path,
                    mode="page",  # Extract by page
                    images_inner_format=f"{self.images_format}-img",
                    images_parser=image_parser
                )
                
                # Load the documents (this is already async internally)
                documents = loader.load()
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info(f"Successfully loaded PDF with GPT-4o image extraction: {len(documents)} pages")
                    return documents
                
                logger.warning("GPT-4o image extraction returned empty content")
            except Exception as img_error:
                # Handle specific image processing errors
                logger.error(f"Error in GPT-4o image processing: {str(img_error)}")
                
                # Try with a simpler approach - just text extraction without images
                logger.info("Falling back to text-only extraction with GPT-4o")
                
                # Create a simpler loader without image extraction
                simple_loader = PyMuPDFLoader(
                    file_path,
                    mode="page"  # Extract by page, no images
                )
                
                # Load the documents
                documents = simple_loader.load()
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info(f"Successfully loaded PDF with text-only extraction: {len(documents)} pages")
                    return documents
                
                logger.warning("Text-only extraction returned empty content")
            
            # If we get here, both approaches failed
            return await self._extract_standard(file_path)
            
        except Exception as e:
            logger.error(f"All GPT-4o extraction methods failed: {str(e)}")
            # Fall back to standard extraction
            return await self._extract_standard(file_path)
    
    async def _extract_standard(self, file_path: str) -> List[LangchainDocument]:
        """Extract content from PDF with standard PyMuPDF."""
        logger.info("Attempting to load with standard PyMuPDF")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        standard_loader = PyMuPDFLoader(file_path)
        
        documents = await loop.run_in_executor(None, standard_loader.load)
        
        if documents and any(doc.page_content.strip() for doc in documents):
            logger.info(f"Successfully loaded PDF with standard PyMuPDF: {len(documents)} pages")
            return documents
        
        logger.warning("Standard PyMuPDF extraction returned empty content")
        return []
    
    def configure(self, 
                 extract_tables: bool = True,
                 extract_images: bool = True,
                 images_format: str = "markdown",
                 use_cache: bool = True,
                 max_cache_size: int = 100,
                 cache_ttl: int = 3600) -> None:
        """Configure the loader settings.
        
        Args:
            extract_tables: Whether to extract tables from PDFs
            extract_images: Whether to extract images from PDFs
            images_format: Format for extracted images (markdown, html, or text)
            use_cache: Whether to cache processed documents
            max_cache_size: Maximum number of documents to cache
            cache_ttl: Cache TTL in seconds
        """
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.images_format = images_format
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        
        logger.info(
            f"GPT-4o PDF loader configured: extract_tables={extract_tables}, "
            f"extract_images={extract_images}, images_format={images_format}, "
            f"use_cache={use_cache}, max_cache_size={max_cache_size}, cache_ttl={cache_ttl}"
        )
