"""PyPDF loader service with optimized performance and smart PDF type detection."""

import asyncio
import functools
import hashlib
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

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

# Try to import PyMuPDF (fitz) for PDF analysis
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) not found, PDF analysis will be limited")

# Enhanced in-memory cache for PDF documents
# Key: file_hash, Value: (timestamp, documents)
PDF_CACHE: Dict[str, tuple[float, List[LangchainDocument]]] = {}
# Cache expiration time in seconds (30 minutes)
CACHE_EXPIRATION = 1800
# Maximum cache size
MAX_CACHE_SIZE = 100

# PDF type constants
PDF_TYPE_TEXT = "text"
PDF_TYPE_SCANNED = "scanned"
PDF_TYPE_MIXED = "mixed"
PDF_TYPE_UNKNOWN = "unknown"

class PDFLoader(LoaderService):
    """PDF and Text loader service with optimized performance and smart PDF type detection."""

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path with optimized performance."""
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading file with extension: {file_extension}")

        if file_extension == ".pdf":
            # Check cache first using file hash instead of path
            file_hash = await self._get_file_hash(file_path)
            if file_hash in PDF_CACHE:
                timestamp, documents = PDF_CACHE[file_hash]
                if time.time() - timestamp < CACHE_EXPIRATION:
                    logger.info(f"Using cached PDF: {file_path}")
                    # Only use cache if it contains actual content
                    if documents and len(documents) > 0:
                        return documents
                    else:
                        logger.info("Cached result was empty, reprocessing document")
            
            # Not in cache or cache expired, load the PDF
            start_time = time.time()
            
            # Detect PDF type to choose the best extraction method
            pdf_type = await self._detect_pdf_type(file_path)
            logger.info(f"Detected PDF type: {pdf_type} for {file_path}")
            
            # Choose extraction method based on PDF type
            if pdf_type == PDF_TYPE_SCANNED:
                # For scanned documents, skip PyPDF and go straight to Unstructured
                if UNSTRUCTURED_AVAILABLE:
                    logger.info(f"Using UnstructuredPDFLoader for scanned document: {file_path}")
                    documents = await self._load_with_optimized_unstructured(file_path)
                else:
                    logger.warning("Scanned document detected but Unstructured not available")
                    documents = await self._load_pdf_optimized(file_path)
            else:
                # For text-based or mixed documents, try PyPDF first
                documents = await self._load_pdf_optimized(file_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"PDF loading completed in {elapsed_time:.2f} seconds")
            
            # Only cache non-empty results
            if documents and len(documents) > 0:
                self._update_cache(file_hash, documents)
                logger.info(f"Cached {len(documents)} documents for future use")
            else:
                logger.warning("Not caching empty result")
            
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
    
    async def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash of the file for caching."""
        loop = asyncio.get_event_loop()
        
        # Run file hashing in a thread pool
        def hash_file():
            with open(file_path, 'rb') as f:
                # Read first 8KB for quick hashing
                return hashlib.md5(f.read(8192)).hexdigest()
        
        return await loop.run_in_executor(None, hash_file)
    
    def _update_cache(self, file_hash: str, documents: List[LangchainDocument]) -> None:
        """Update the cache with new documents."""
        # Add to cache
        PDF_CACHE[file_hash] = (time.time(), documents)
        
        # Limit cache size by removing oldest entries if needed
        if len(PDF_CACHE) > MAX_CACHE_SIZE:
            # Sort by timestamp and remove oldest
            oldest_key = sorted(PDF_CACHE.items(), key=lambda x: x[1][0])[0][0]
            PDF_CACHE.pop(oldest_key)
            logger.info(f"Cache full, removed oldest entry: {oldest_key}")
    
    async def _detect_pdf_type(self, file_path: str) -> str:
        """Detect if a PDF is text-based, scanned, or mixed."""
        # Quick check based on file size
        file_size = os.path.getsize(file_path)
        if file_size > 10_000_000:  # 10MB
            logger.info(f"Large PDF detected ({file_size} bytes), likely scanned or image-heavy")
            return PDF_TYPE_SCANNED
        
        # Use PyMuPDF for more accurate detection if available
        if PYMUPDF_AVAILABLE:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._analyze_pdf_with_pymupdf, file_path)
        
        # Fallback to basic detection with PyPDF
        return await self._basic_pdf_detection(file_path)
    
    def _analyze_pdf_with_pymupdf(self, file_path: str) -> str:
        """Analyze PDF using PyMuPDF to determine its type."""
        try:
            doc = fitz.open(file_path)
            
            # Check first few pages (up to 5) to determine document characteristics
            max_pages = min(5, len(doc))
            text_pages = 0
            image_pages = 0
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # Check for images
                images = page.get_images(full=True)
                
                # Check for text
                text = page.get_text()
                
                # Determine if page is text-based or image-based
                if len(text.strip()) > 100:  # Significant text content
                    text_pages += 1
                if len(images) > 0:  # Has images
                    image_pages += 1
            
            doc.close()
            
            # Determine document type based on page analysis
            if text_pages == 0 and image_pages > 0:
                return PDF_TYPE_SCANNED  # No text pages, only images
            elif text_pages > 0 and image_pages == 0:
                return PDF_TYPE_TEXT  # Only text pages
            elif text_pages > 0 and image_pages > 0:
                return PDF_TYPE_MIXED  # Mix of text and images
            else:
                return PDF_TYPE_UNKNOWN
                
        except Exception as e:
            logger.error(f"Error analyzing PDF with PyMuPDF: {str(e)}")
            return PDF_TYPE_UNKNOWN
    
    async def _basic_pdf_detection(self, file_path: str) -> str:
        """Basic PDF type detection using PyPDF."""
        try:
            loop = asyncio.get_event_loop()
            
            # Check for text markers in PDF
            def check_pdf():
                with open(file_path, 'rb') as f:
                    # Read first 5KB to check for text
                    data = f.read(5120)
                    
                # Check for text markers in PDF
                if b'/Text' in data or b'/Font' in data:
                    return PDF_TYPE_TEXT
                elif b'/Image' in data or b'/XObject' in data:
                    return PDF_TYPE_SCANNED
                else:
                    return PDF_TYPE_UNKNOWN
            
            return await loop.run_in_executor(None, check_pdf)
        except Exception as e:
            logger.error(f"Error in basic PDF detection: {str(e)}")
            return PDF_TYPE_UNKNOWN
    
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
            
            # Determine optimal batch size based on document size
            batch_size = self._determine_optimal_batch_size(num_pages)
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
                return await self._load_with_optimized_unstructured(file_path)
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
                return await self._load_with_optimized_unstructured(file_path)
            else:
                logger.error("PDF loading failed and Unstructured not available")
                return [LangchainDocument(
                    page_content=f"Error processing PDF: {os.path.basename(file_path)}",
                    metadata={"source": file_path, "page": 1, "error": str(e)}
                )]
    
    def _determine_optimal_batch_size(self, total_pages: int) -> int:
        """Determine optimal batch size based on document size."""
        if total_pages > 100:
            return 20  # Larger batches for very large documents
        elif total_pages > 50:
            return 15
        elif total_pages > 20:
            return 10
        else:
            return total_pages  # Process all pages at once for small documents
    
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
    
    async def _load_with_optimized_unstructured(self, file_path: str) -> List[LangchainDocument]:
        """Load PDF with optimized Unstructured approach."""
        if not UNSTRUCTURED_AVAILABLE:
            return []
        
        try:
            logger.info(f"Attempting to load PDF with optimized Unstructured: {file_path}")
            loop = asyncio.get_event_loop()
            
            # Try different configurations to see what works
            try:
                # First try with default parameters
                logger.info("Trying UnstructuredPDFLoader with default parameters")
                unstructured_loader = UnstructuredPDFLoader(file_path)
                unstructured_documents = await loop.run_in_executor(None, unstructured_loader.load)
                
                if unstructured_documents:
                    logger.info(f"Successfully loaded document with default UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                    return unstructured_documents
                else:
                    logger.warning("Default UnstructuredPDFLoader returned empty content")
            except Exception as e1:
                logger.error(f"Error using default UnstructuredPDFLoader: {str(e1)}")
                logger.error(f"Error type: {type(e1).__name__}")
                logger.error(f"Error details: {repr(e1)}")
            
            try:
                # Try with elements mode and fast strategy
                logger.info("Trying UnstructuredPDFLoader with elements mode and fast strategy")
                
                def run_elements_fast_loader():
                    loader = UnstructuredPDFLoader(
                        file_path,
                        mode="elements",
                        strategy="fast"
                    )
                    return loader.load()
                
                # Execute in thread pool to avoid blocking
                unstructured_documents = await loop.run_in_executor(None, run_elements_fast_loader)
                
                if unstructured_documents:
                    logger.info(f"Successfully loaded document with elements/fast UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                    return unstructured_documents
                else:
                    logger.warning("Elements/fast UnstructuredPDFLoader returned empty content")
            except Exception as e2:
                logger.error(f"Error using elements/fast UnstructuredPDFLoader: {str(e2)}")
                logger.error(f"Error type: {type(e2).__name__}")
                logger.error(f"Error details: {repr(e2)}")
            
            try:
                # Try with paged mode
                logger.info("Trying UnstructuredPDFLoader with paged mode")
                
                def run_paged_loader():
                    loader = UnstructuredPDFLoader(
                        file_path,
                        mode="paged"
                    )
                    return loader.load()
                
                # Execute in thread pool to avoid blocking
                unstructured_documents = await loop.run_in_executor(None, run_paged_loader)
                
                if unstructured_documents:
                    logger.info(f"Successfully loaded document with paged UnstructuredPDFLoader: {len(unstructured_documents)} elements")
                    return unstructured_documents
                else:
                    logger.warning("Paged UnstructuredPDFLoader returned empty content")
            except Exception as e3:
                logger.error(f"Error using paged UnstructuredPDFLoader: {str(e3)}")
                logger.error(f"Error type: {type(e3).__name__}")
                logger.error(f"Error details: {repr(e3)}")
            
            # If all attempts failed, return empty list
            logger.error("All UnstructuredPDFLoader configurations failed")
            return []
            
        except Exception as e:
            logger.error(f"Error in _load_with_optimized_unstructured: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {repr(e)}")
            return []
