"""Simple PDF loader service with optimized performance."""

import asyncio
import functools
import logging
import os
from typing import Dict, List, Optional, Union
import time

from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import TextLoader
import pypdf

from app.services.loaders.base import LoaderService

logger = logging.getLogger(__name__)

# Try to import UnstructuredPDFLoader, but don't fail if it's not available
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("unstructured package not found, please install it with `pip install unstructured`")

# Simple in-memory cache for PDF documents
# Key: file_path, Value: (timestamp, documents)
PDF_CACHE: Dict[str, tuple[float, List[LangchainDocument]]] = {}
# Cache expiration time in seconds (5 minutes)
CACHE_EXPIRATION = 300

class SimplePDFLoader(LoaderService):
    """Simple PDF and Text loader service with optimized performance."""

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path with optimized performance."""
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading file with extension: {file_extension}")

        if file_extension == ".pdf":
            # Check cache first
            if file_path in PDF_CACHE:
                timestamp, documents = PDF_CACHE[file_path]
                if time.time() - timestamp < CACHE_EXPIRATION:
                    logger.info(f"Using cached PDF: {file_path}")
                    return documents
            
            # Not in cache or cache expired, load the PDF
            start_time = time.time()
            documents = await self._load_pdf_optimized(file_path)
            elapsed_time = time.time() - start_time
            logger.info(f"PDF loading completed in {elapsed_time:.2f} seconds")
            
            # Cache the result
            PDF_CACHE[file_path] = (time.time(), documents)
            return documents
                
        elif file_extension == ".txt":
            try:
                logger.info(f"Loading text file: {file_path}")
                # Run TextLoader in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                loader = TextLoader(file_path)
                documents = await loop.run_in_executor(None, loader.load)
                logger.info(f"Successfully loaded text file: {len(documents)} documents")
                return documents
            except Exception as e:
                logger.error(f"Error loading text file: {str(e)}")
                return []
        else:
            error_msg = f"Unsupported file type: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    async def _load_pdf_optimized(self, file_path: str) -> List[LangchainDocument]:
        """Load PDF with optimized performance using direct PyPDF access and parallel processing."""
        try:
            logger.info(f"Loading PDF with optimized PyPDF: {file_path}")
            
            # Use PyPDF directly for better performance
            loop = asyncio.get_event_loop()
            
            # Run PDF opening in a thread pool to avoid blocking
            pdf_reader = await loop.run_in_executor(
                None, 
                functools.partial(pypdf.PdfReader, file_path, strict=False)
            )
            
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            if num_pages == 0:
                logger.warning(f"PDF has no pages: {file_path}")
                return []
            
            # Process pages in parallel but in smaller batches to avoid memory issues
            batch_size = 10  # Process 10 pages at a time
            all_documents = []
            
            for i in range(0, num_pages, batch_size):
                batch_end = min(i + batch_size, num_pages)
                logger.info(f"Processing batch of pages {i} to {batch_end-1}")
                
                # Create tasks for this batch
                tasks = []
                for page_num in range(i, batch_end):
                    tasks.append(self._process_page(loop, pdf_reader, page_num, file_path))
                
                # Process this batch
                batch_documents = await asyncio.gather(*tasks)
                
                # Filter out empty documents
                batch_documents = [doc for doc in batch_documents if doc and doc.page_content.strip()]
                all_documents.extend(batch_documents)
            
            if all_documents:
                logger.info(f"Successfully extracted {len(all_documents)} pages with content")
                return all_documents
            
            # If PyPDF didn't extract any text, try UnstructuredPDFLoader as fallback
            if UNSTRUCTURED_AVAILABLE:
                logger.warning("PyPDF returned empty content. Trying UnstructuredPDFLoader as fallback.")
                return await self._load_with_unstructured(file_path)
            else:
                logger.warning("No text extracted from PDF and Unstructured not available")
                return [LangchainDocument(
                    page_content=f"PDF document: {os.path.basename(file_path)}",
                    metadata={"source": file_path, "page": 1}
                )]
                
        except Exception as e:
            logger.error(f"Error using optimized PyPDF: {str(e)}")
            
            # Try UnstructuredPDFLoader as fallback if available
            if UNSTRUCTURED_AVAILABLE:
                logger.info("Trying UnstructuredPDFLoader as fallback")
                return await self._load_with_unstructured(file_path)
            else:
                logger.error("PDF loading failed and Unstructured not available")
                return [LangchainDocument(
                    page_content=f"Error processing PDF: {os.path.basename(file_path)}",
                    metadata={"source": file_path, "page": 1, "error": str(e)}
                )]
    
    async def _process_page(self, loop, pdf_reader, page_num: int, source: str) -> Optional[LangchainDocument]:
        """Process a single PDF page asynchronously."""
        try:
            # Extract text from page in a thread pool
            page = pdf_reader.pages[page_num]
            text = await loop.run_in_executor(None, page.extract_text)
            
            if not text or not text.strip():
                logger.warning(f"No text extracted from page {page_num}")
                return None
            
            # Create LangChain document
            return LangchainDocument(
                page_content=text,
                metadata={"page": page_num + 1, "source": source}
            )
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return None
    
    async def _load_with_unstructured(self, file_path: str) -> List[LangchainDocument]:
        """Load PDF with UnstructuredPDFLoader."""
        if not UNSTRUCTURED_AVAILABLE:
            return []
        
        try:
            logger.info(f"Attempting to load PDF with UnstructuredPDFLoader: {file_path}")
            loop = asyncio.get_event_loop()
            
            # Run UnstructuredPDFLoader in a thread pool
            unstructured_loader = UnstructuredPDFLoader(file_path)
            unstructured_documents = await loop.run_in_executor(None, unstructured_loader.load)
            
            if unstructured_documents:
                logger.info(f"Successfully loaded PDF with UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                return unstructured_documents
            else:
                logger.warning("UnstructuredPDFLoader returned empty content")
                return []
        except Exception as e:
            logger.error(f"Error using UnstructuredPDFLoader: {str(e)}")
            return []
