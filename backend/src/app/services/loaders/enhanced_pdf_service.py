"""Enhanced PDF text extraction service with multiple extraction backends.

This service provides robust PDF text extraction by trying multiple extraction methods
in sequence, ensuring maximum content recovery from different PDF types.
"""

import os
import logging
import io
from typing import List, Dict, Any, Optional, Union

from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Import fitz (PyMuPDF) - this is more robust than PyPDF for many PDFs
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning(
        "PyMuPDF is not available. Install it using 'pip install pymupdf' for improved PDF extraction."
    )

# Import spaCy for text cleaning if available
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load a small model for text cleaning
    nlp = spacy.blank("en")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning(
        "spaCy is not available. Install it using 'pip install spacy' for better text cleaning."
    )

class EnhancedPDFExtractor:
    """Enhanced PDF extraction service with multiple backends."""

    def __init__(self, min_text_length: int = 100, chunk_overlap: int = 200):
        """Initialize the extractor.
        
        Args:
            min_text_length: Minimum text length to consider extraction successful
            chunk_overlap: Overlap between page chunks for context preservation
        """
        self.min_text_length = min_text_length
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
    async def extract_text(self, file_path: str) -> List[LangchainDocument]:
        """Extract text from a PDF file using multiple extraction methods.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangchainDocument objects with extracted text
        """
        extraction_methods = [
            self._extract_with_pymupdf,
            self._extract_with_pypdf,
            # Add other extraction methods as needed
        ]
        
        for method in extraction_methods:
            try:
                documents = await method(file_path)
                
                # Validate extraction result
                if documents and self._is_extraction_successful(documents):
                    self.logger.info(f"Successfully extracted text with {method.__name__}")
                    return documents
                else:
                    self.logger.warning(f"Extraction with {method.__name__} was unsuccessful or returned empty content")
            except Exception as e:
                self.logger.warning(f"Error in {method.__name__}: {str(e)}")
                
        # If all methods failed, return a document with error info
        self.logger.error(f"All extraction methods failed for {file_path}")
        return [LangchainDocument(
            page_content="Error: Unable to extract text from this PDF. The file might be password-protected, corrupted, or contain only images.",
            metadata={"source": file_path, "extraction_error": True}
        )]
    
    def _is_extraction_successful(self, documents: List[LangchainDocument]) -> bool:
        """Check if extraction was successful based on content length and quality.
        
        Args:
            documents: List of extracted documents
            
        Returns:
            Boolean indicating if extraction was successful
        """
        if not documents:
            return False
            
        # Check total text length
        total_text = "".join([doc.page_content for doc in documents])
        if len(total_text.strip()) < self.min_text_length:
            return False
            
        # Check if the text appears to be meaningful (not just garbage)
        # This is a simple heuristic; you might want to improve it
        words = total_text.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        if avg_word_length > 15:  # Likely garbage text
            return False
            
        return True
        
    async def _extract_with_pymupdf(self, file_path: str) -> List[LangchainDocument]:
        """Extract text using PyMuPDF (fitz).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangchainDocument objects
        """
        if not PYMUPDF_AVAILABLE:
            self.logger.warning("PyMuPDF not available, skipping this extraction method")
            return []
            
        documents = []
        try:
            # Open the PDF
            pdf_document = fitz.open(file_path)
            
            for page_num, page in enumerate(pdf_document):
                # Extract text with special options for better results
                text = page.get_text("text")
                
                # Clean the text
                text = self._clean_text(text)
                
                if text.strip():
                    documents.append(LangchainDocument(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "total_pages": len(pdf_document),
                            "extraction_method": "pymupdf"
                        }
                    ))
            
            pdf_document.close()
            
            # If no text was extracted, try again with different options
            if not documents:
                return await self._extract_with_pymupdf_advanced(file_path)
                
            return documents
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction error: {str(e)}")
            return []
            
    async def _extract_with_pymupdf_advanced(self, file_path: str) -> List[LangchainDocument]:
        """Extract text using PyMuPDF with advanced options.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangchainDocument objects
        """
        if not PYMUPDF_AVAILABLE:
            return []
            
        documents = []
        try:
            # Open the PDF
            pdf_document = fitz.open(file_path)
            
            for page_num, page in enumerate(pdf_document):
                # Try different extraction modes
                extraction_modes = ["text", "html", "dict", "xhtml", "json"]
                
                for mode in extraction_modes:
                    try:
                        if mode == "text":
                            text = page.get_text(mode)
                        elif mode == "html" or mode == "xhtml":
                            html = page.get_text(mode)
                            # Remove HTML tags for plain text (simple approach)
                            text = html.replace(r"<[^>]*>", "")
                        elif mode == "dict" or mode == "json":
                            # Extract text blocks and join them
                            data = page.get_text(mode)
                            if isinstance(data, str):
                                text = data
                            else:
                                blocks = data.get("blocks", [])
                                text_parts = []
                                for block in blocks:
                                    if "lines" in block:
                                        for line in block["lines"]:
                                            if "spans" in line:
                                                for span in line["spans"]:
                                                    if "text" in span:
                                                        text_parts.append(span["text"])
                                text = " ".join(text_parts)
                        else:
                            continue
                            
                        # Clean the text
                        text = self._clean_text(text)
                        
                        if text.strip():
                            documents.append(LangchainDocument(
                                page_content=text,
                                metadata={
                                    "source": file_path,
                                    "page": page_num + 1,
                                    "total_pages": len(pdf_document),
                                    "extraction_method": f"pymupdf_{mode}"
                                }
                            ))
                            # If this mode worked, break the loop
                            break
                    except Exception as e:
                        self.logger.debug(f"Error with PyMuPDF mode {mode}: {str(e)}")
                        continue
            
            pdf_document.close()
            return documents
            
        except Exception as e:
            self.logger.error(f"PyMuPDF advanced extraction error: {str(e)}")
            return []
    
    async def _extract_with_pypdf(self, file_path: str) -> List[LangchainDocument]:
        """Extract text using PyPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangchainDocument objects
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Clean each document's text
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)
                doc.metadata["extraction_method"] = "pypdf"
                
            return documents
            
        except Exception as e:
            self.logger.error(f"PyPDF extraction error: {str(e)}")
            return []
            
    def _clean_text(self, text: str) -> str:
        """Clean extracted text to improve quality.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Basic cleaning
        cleaned_text = text
        
        # Remove repeated whitespace
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove fax headers that often appear in medical records
        cleaned_text = re.sub(r'\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+\d{3}-\d{3}-\d{4}\s+->\s+[^\n]+\s+Page\s+\d+', '', cleaned_text)
        
        # Use spaCy for better text cleaning if available
        if SPACY_AVAILABLE and cleaned_text:
            try:
                doc = nlp(cleaned_text)
                # Remove non-alphanumeric tokens and normalize whitespace
                cleaned_tokens = [token.text for token in doc if token.is_alpha or token.is_digit or token.is_punct]
                cleaned_text = " ".join(cleaned_tokens)
            except Exception as e:
                self.logger.debug(f"spaCy cleaning error: {str(e)}")
                
        return cleaned_text.strip()

# Main loader service implementation
class EnhancedPDFLoader:
    """Enhanced PDF and Text loader service with multiple extraction backends."""

    def __init__(self):
        """Initialize the loader."""
        self.pdf_extractor = EnhancedPDFExtractor()
        self.logger = logging.getLogger(__name__)

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path with robust extraction.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of LangchainDocument objects
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".pdf":
            documents = await self.pdf_extractor.extract_text(file_path)
            
            # Log diagnostic information
            if documents:
                self.logger.info(f"Loaded PDF: {file_path}, extracted {len(documents)} pages")
                
                # Check content quality
                total_content = sum(len(doc.page_content.strip()) for doc in documents)
                if total_content == 0:
                    self.logger.warning(f"PDF loaded but no content extracted: {file_path}")
                else:
                    self.logger.info(f"Extracted {total_content} characters from {file_path}")
                    
            return documents
            
        elif file_extension == ".txt":
            try:
                loader = TextLoader(file_path)
                return loader.load()
            except Exception as e:
                self.logger.error(f"Error loading text file {file_path}: {str(e)}")
                return [LangchainDocument(
                    page_content=f"Error loading text file: {str(e)}",
                    metadata={"source": file_path, "error": str(e)}
                )]
        else:
            error_msg = f"Unsupported file type: {file_path}"
            self.logger.error(error_msg)
            return [LangchainDocument(
                page_content=f"Error: {error_msg}",
                metadata={"source": file_path, "error": error_msg}
            )]