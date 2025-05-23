"""Document service."""

import asyncio
import logging
import os
import tempfile
import time
import uuid
import requests
import aiofiles
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.services.loaders.factory import LoaderFactory
from app.services.vector_db.base import VectorDBService
from app.services.document_persistence_service import DocumentPersistenceService
from app.models.document import Document

logger = logging.getLogger(__name__)


class DocumentService:
    """Document service."""

    def __init__(
        self,
        vector_db_service: VectorDBService,
        llm_service: CompletionService,
        settings: Settings,
    ):
        """Document service."""
        self.vector_db_service = vector_db_service
        self.llm_service = llm_service
        self.settings = settings
        self.loader_factory = LoaderFactory()
        
        # Use larger chunks for big documents to reduce processing time
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        
        # Larger chunk size, smaller overlap for big documents
        self.large_doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size * 3,  # 3x larger chunks
            chunk_overlap=int(self.settings.chunk_overlap * 0.5),  # 50% less overlap
        )
        
        # Remove all in-memory status tracking
        # self.document_statuses = {}
        # self._load_document_statuses()

    async def upload_document(
        self,
        filename: str,
        file_content: bytes,
        author: str = "unknown",
        tag: str = "",
        page_count: int = 0,
    ) -> Optional[str]:
        """
        Upload a document.
        
        WARNING: This method performs synchronous processing which may cause timeouts with large documents.
        For production use with large files, consider using process_document_background method instead.
        
        Parameters
        ----------
        filename : str
            The name of the document to upload.
        file_content : bytes
            The content of the document to upload.
            
        Returns
        -------
        Optional[str]
            The document ID if successful, None otherwise.
        """
        try:
            document_id = self._generate_document_id()
            logger.info(f"Created document_id: {document_id}")

            documents_dir = os.path.join(tempfile.gettempdir(), "ai_grid_documents")
            os.makedirs(documents_dir, exist_ok=True)
            file_path = os.path.join(documents_dir, f"{document_id}{os.path.splitext(filename)[1]}")
            logger.info(f"Saving document to: {file_path}")
            with open(file_path, 'wb') as f:
                f.write(file_content)

            # Try to extract real metadata (page count, author, tag, etc.)
            extracted_page_count = 0
            extracted_author = author
            extracted_tag = tag
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                extracted_page_count = doc.page_count
                meta = doc.metadata
                if meta and meta.get("author"):
                    extracted_author = meta["author"]
                doc.close()
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {e}")

            doc_meta = Document(
                id=document_id,
                name=filename,
                author=extracted_author,
                tag=extracted_tag,
                page_count=extracted_page_count,
                status="processing"
            )
            await DocumentPersistenceService.insert_document(doc_meta)

            try:
                chunks = await self._process_document(file_path)
                logger.info(f"Processed document into {len(chunks)} chunks")
                await DocumentPersistenceService.update_status(document_id, "completed")
                return document_id
            except Exception as e:
                logger.error(f"Error processing document: {e}", exc_info=True)
                await DocumentPersistenceService.update_status(document_id, "failed")
                return None

        except Exception as e:
            logger.error(f"Error uploading document: {e}", exc_info=True)
            return None

    async def _process_document(
        self, file_path: str
    ) -> List[LangchainDocument]:
        """Process a document with optimized performance and parallel processing."""
        start_time = time.time()
        
        # Load the document asynchronously
        docs = await self._load_document(file_path)
        load_time = time.time() - start_time
        logger.info(f"Document loading completed in {load_time:.2f} seconds")
        
        if not docs:
            logger.warning(f"No content loaded from document: {file_path}")
            # Create a single empty document to avoid downstream issues
            return [LangchainDocument(
                page_content="No content could be extracted from this document.",
                metadata={"page": 1, "source": file_path, "error": "Content extraction failed"}
            )]

        # Check file size to determine which splitter to use
        file_size = os.path.getsize(file_path)
        large_file = file_size > 5_000_000  # 5MB threshold
        
        if large_file:
            logger.info(f"Using larger chunk size for document ({file_size} bytes)")
            splitter = self.large_doc_splitter
        else:
            splitter = self.default_splitter

        # Split the document into chunks using a thread pool to avoid blocking
        chunk_start = time.time()
        loop = asyncio.get_event_loop()
        
        # Process documents in parallel for better performance
        # Determine optimal batch size based on document count
        if len(docs) > 50:
            batch_size = 5  # Smaller batches for many documents
        elif len(docs) > 20:
            batch_size = 10
        else:
            batch_size = 20  # Larger batches for fewer documents
        
        all_chunks = []
        chunk_tasks = []
        
        # Create tasks for each batch
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            # Create a task for each batch
            task = loop.run_in_executor(
                None, 
                lambda b=batch: splitter.split_documents(b)
            )
            chunk_tasks.append(task)
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*chunk_tasks)
        
        # Combine results
        for batch_chunks in batch_results:
            all_chunks.extend(batch_chunks)
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Document chunking completed in {chunk_time:.2f} seconds, created {len(all_chunks)} chunks")
        
        if not all_chunks:
            logger.warning(f"Document was loaded but no chunks were created: {file_path}")
            # Create a single chunk with the original content to ensure we have something
            return [LangchainDocument(
                page_content="Document was processed but no meaningful chunks could be extracted.",
                metadata={"page": 1, "source": file_path, "error": "Chunking failed"}
            )]
        
        total_time = time.time() - start_time
        logger.info(f"Total document processing completed in {total_time:.2f} seconds")
        return all_chunks

    async def _load_document(self, file_path: str) -> List[LangchainDocument]:
        """Load a document with smart fallback mechanisms for robust processing."""
        # Store the original loader type
        original_loader_type = self.settings.loader
        
        # Detect file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # For non-PDF files, just use the primary loader
        if file_extension != ".pdf":
            primary_loader = self.loader_factory.create_loader(self.settings)
            if primary_loader is None:
                raise ValueError(
                    f"No loader available for configured loader type: {original_loader_type}"
                )
            return await primary_loader.load(file_path)
        
        # For PDF files, implement a smart fallback strategy based on PDF type
        try:
            # First, detect PDF type to choose the best extraction method
            pdf_type = await self._detect_pdf_type(file_path)
            logger.info(f"Smart fallback: Detected PDF type: {pdf_type} for {file_path}")
            
            # Choose extraction strategy based on PDF type
            if pdf_type == "scanned":
                # For scanned documents, try Unstructured first, then Textract, then GPT-4o
                return await self._load_scanned_pdf(file_path, original_loader_type)
            elif pdf_type == "text":
                # For text-based documents, try PyPDF first, then Unstructured
                return await self._load_text_pdf(file_path, original_loader_type)
            else:  # mixed or unknown
                # For mixed documents, try PyPDF first, then Unstructured, then Textract, then GPT-4o
                return await self._load_mixed_pdf(file_path, original_loader_type)
        except Exception as e:
            logger.error(f"Smart PDF loading failed: {e}. Falling back to standard approach.")
            # Fall back to the original approach if the smart approach fails
            return await self._load_document_with_fallbacks(file_path, original_loader_type)
    
    async def _detect_pdf_type(self, file_path: str) -> str:
        """Detect if a PDF is text-based, scanned, or mixed."""
        # Quick check based on file size
        file_size = os.path.getsize(file_path)
        if file_size > 10_000_000:  # 10MB
            logger.info(f"Large PDF detected ({file_size} bytes), likely scanned or image-heavy")
            return "scanned"
        
        # Try to import PyMuPDF (fitz) for PDF analysis
        try:
            import fitz
            
            # Use PyMuPDF for more accurate detection
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
                return "scanned"  # No text pages, only images
            elif text_pages > 0 and image_pages == 0:
                return "text"  # Only text pages
            elif text_pages > 0 and image_pages > 0:
                return "mixed"  # Mix of text and images
            else:
                return "unknown"
                
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available for PDF analysis")
            # Fallback to basic detection
            return await self._basic_pdf_detection(file_path)
        except Exception as e:
            logger.error(f"Error analyzing PDF with PyMuPDF: {str(e)}")
            return "unknown"
    
    async def _basic_pdf_detection(self, file_path: str) -> str:
        """Basic PDF type detection using file markers."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 5KB to check for text
                data = f.read(5120)
                
            # Check for text markers in PDF
            if b'/Text' in data or b'/Font' in data:
                return "text"
            elif b'/Image' in data or b'/XObject' in data:
                return "scanned"
            else:
                return "unknown"
        except Exception as e:
            logger.error(f"Error in basic PDF detection: {str(e)}")
            return "unknown"
    
    async def _load_scanned_pdf(self, file_path: str, original_loader_type: str) -> List[LangchainDocument]:
        """Load a scanned PDF with optimized fallback chain."""
        # Try GPT-4o loader first for scanned documents
        try:
            logger.info("Smart fallback: Trying GPT-4o loader first for scanned document")
            self.settings.loader = "gpt4o_pdf"
            gpt4o_loader = self.loader_factory.create_loader(self.settings)
            
            if gpt4o_loader is not None:
                documents = await gpt4o_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded scanned document with GPT-4o")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("GPT-4o loader returned empty content for scanned document")
            else:
                logger.warning("GPT-4o loader could not be created")
        except Exception as e:
            logger.error(f"GPT-4o loader failed for scanned document: {e}")
        finally:
            self.settings.loader = original_loader_type
        
        # Then try Unstructured as fallback
        try:
            logger.info("Smart fallback: Trying Unstructured loader for scanned document")
            self.settings.loader = "unstructured"
            unstructured_loader = self.loader_factory.create_loader(self.settings)
            
            if unstructured_loader is not None:
                documents = await unstructured_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded scanned document with Unstructured")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("Unstructured loader returned empty content for scanned document")
            else:
                logger.warning("Unstructured loader could not be created")
        except Exception as e:
            logger.error(f"Unstructured loader failed for scanned document: {e}")
        finally:
            self.settings.loader = original_loader_type
            
        # If all loaders failed, return an empty list
        logger.error(f"All loaders failed for scanned document: {file_path}")
        return []
    
    async def _load_text_pdf(self, file_path: str, original_loader_type: str) -> List[LangchainDocument]:
        """Load a text-based PDF with optimized fallback chain."""
        # For text documents, try PyPDF first
        try:
            logger.info("Smart fallback: Trying PyPDF loader for text document")
            self.settings.loader = "pypdf"
            pypdf_loader = self.loader_factory.create_loader(self.settings)
            
            if pypdf_loader is not None:
                documents = await pypdf_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded text document with PyPDF")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("PyPDF loader returned empty content for text document")
            else:
                logger.warning("PyPDF loader could not be created")
        except Exception as e:
            logger.error(f"PyPDF loader failed for text document: {e}")
        finally:
            self.settings.loader = original_loader_type
        
        # Then try Unstructured
        try:
            logger.info("Smart fallback: Trying Unstructured loader for text document")
            self.settings.loader = "unstructured"
            unstructured_loader = self.loader_factory.create_loader(self.settings)
            
            if unstructured_loader is not None:
                documents = await unstructured_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded text document with Unstructured")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("Unstructured loader returned empty content for text document")
            else:
                logger.warning("Unstructured loader could not be created")
        except Exception as e:
            logger.error(f"Unstructured loader failed for text document: {e}")
        finally:
            self.settings.loader = original_loader_type
        
        # If all loaders failed, return an empty list
        logger.error(f"All loaders failed for text document: {file_path}")
        return []
    
    async def _load_mixed_pdf(self, file_path: str, original_loader_type: str) -> List[LangchainDocument]:
        """Load a mixed PDF with optimized fallback chain."""
        # For mixed documents, try PyPDF first
        try:
            logger.info("Smart fallback: Trying PyPDF loader for mixed document")
            self.settings.loader = "pypdf"
            pypdf_loader = self.loader_factory.create_loader(self.settings)
            
            if pypdf_loader is not None:
                documents = await pypdf_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded mixed document with PyPDF")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("PyPDF loader returned empty content for mixed document")
            else:
                logger.warning("PyPDF loader could not be created")
        except Exception as e:
            logger.error(f"PyPDF loader failed for mixed document: {e}")
        finally:
            self.settings.loader = original_loader_type
        
        # Then try Unstructured
        try:
            logger.info("Smart fallback: Trying Unstructured loader for mixed document")
            self.settings.loader = "unstructured"
            unstructured_loader = self.loader_factory.create_loader(self.settings)
            
            if unstructured_loader is not None:
                documents = await unstructured_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded mixed document with Unstructured")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("Unstructured loader returned empty content for mixed document")
            else:
                logger.warning("Unstructured loader could not be created")
        except Exception as e:
            logger.error(f"Unstructured loader failed for mixed document: {e}")
        finally:
            self.settings.loader = original_loader_type
        
        # Then try Textract - COMMENTED OUT DUE TO EXPIRED CREDENTIALS
        # try:
        #     logger.info("Smart fallback: Trying Textract loader for mixed document")
        #     self.settings.loader = "textract"
        #     textract_loader = self.loader_factory.create_loader(self.settings)
        #     
        #     if textract_loader is not None:
        #         try:
        #             documents = await textract_loader.load(file_path)
        #             
        #             if documents and any(doc.page_content.strip() for doc in documents):
        #                 logger.info("Successfully loaded mixed document with Textract")
        #                 self.settings.loader = original_loader_type
        #                 return documents
        #             else:
        #                 logger.warning("Textract loader returned empty content for mixed document")
        #         except ValueError as ve:
        #             # Check for the special error code for expired AWS credentials
        #             if str(ve) == "AWS_CREDENTIALS_EXPIRED":
        #                 logger.warning("AWS credentials have expired, skipping Textract and trying GPT-4o")
        #                 # Skip to GPT-4o immediately
        #                 raise ValueError("AWS_CREDENTIALS_EXPIRED")
        #             else:
        #                 # Re-raise other ValueError exceptions
        #                 raise
        #     else:
        #         logger.warning("Textract loader could not be created")
        # except ValueError as ve:
        #     # Special handling for expired AWS credentials
        #     if str(ve) == "AWS_CREDENTIALS_EXPIRED":
        #         logger.warning("Skipping Textract due to expired AWS credentials")
        #         # Continue to GPT-4o
        #     else:
        #         logger.error(f"Textract loader failed with ValueError: {ve}")
        # except Exception as e:
        #     logger.error(f"Textract loader failed for mixed document: {e}")
        # finally:
        #     self.settings.loader = original_loader_type
        
        # Finally try GPT-4o as last resort
        try:
            logger.info("Smart fallback: Trying GPT-4o loader for mixed document")
            self.settings.loader = "gpt4o_pdf"
            gpt4o_loader = self.loader_factory.create_loader(self.settings)
            
            if gpt4o_loader is not None:
                documents = await gpt4o_loader.load(file_path)
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded mixed document with GPT-4o")
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("GPT-4o loader returned empty content for mixed document")
            else:
                logger.warning("GPT-4o loader could not be created")
        except Exception as e:
            logger.error(f"GPT-4o loader failed for mixed document: {e}")
        finally:
            self.settings.loader = original_loader_type
        
        # If all loaders failed, return an empty list
        logger.error(f"All loaders failed for mixed document: {file_path}")
        return []
    
    async def _load_document_with_fallbacks(self, file_path: str, original_loader_type: str) -> List[LangchainDocument]:
        """Load a document with the original fallback mechanism."""
        # Create the primary loader
        primary_loader = self.loader_factory.create_loader(self.settings)
        if primary_loader is None:
            raise ValueError(
                f"No loader available for configured loader type: {original_loader_type}"
            )

        # Try the primary loader first
        try:
            logger.info(f"Attempting to load document with primary loader: {original_loader_type}")
            documents = await primary_loader.load(file_path)
            
            # Check if we got meaningful content
            if documents and any(doc.page_content.strip() for doc in documents):
                logger.info(f"Successfully loaded document with {original_loader_type}")
                return documents
            else:
                logger.warning(f"Primary loader {original_loader_type} returned empty content. Trying fallbacks.")
                # Fall through to fallbacks
        except Exception as e:
            logger.error(f"Primary loader {original_loader_type} failed: {e}. Trying fallbacks.")
            # Fall through to fallbacks
        
        # Try GPT-4o loader as first fallback
        try:
            logger.info("Attempting to load document with GPT-4o loader as fallback")
            # Temporarily change the loader type to gpt4o_pdf
            self.settings.loader = "gpt4o_pdf"
            gpt4o_loader = self.loader_factory.create_loader(self.settings)
            
            if gpt4o_loader is not None:
                documents = await gpt4o_loader.load(file_path)
                
                # Check if we got meaningful content
                if documents and any(doc.page_content.strip() for doc in documents):
                    logger.info("Successfully loaded document with GPT-4o loader")
                    # Restore original loader type
                    self.settings.loader = original_loader_type
                    return documents
                else:
                    logger.warning("GPT-4o loader returned empty content")
            else:
                logger.warning("GPT-4o loader could not be created")
        except Exception as e:
            logger.error(f"GPT-4o loader failed: {e}")
        finally:
            # Restore original loader type
            self.settings.loader = original_loader_type
        
        # Try Textract loader as second fallback - COMMENTED OUT DUE TO EXPIRED CREDENTIALS
        # try:
        #     logger.info("Attempting to load document with Textract loader as fallback")
        #     # Temporarily change the loader type to textract
        #     self.settings.loader = "textract"
        #     textract_loader = self.loader_factory.create_loader(self.settings)
        #     
        #     if textract_loader is not None:
        #         try:
        #             documents = await textract_loader.load(file_path)
        #             
        #             # Check if we got meaningful content
        #             if documents and any(doc.page_content.strip() for doc in documents):
        #                 logger.info("Successfully loaded document with Textract loader")
        #                 # Restore original loader type
        #                 self.settings.loader = original_loader_type
        #                 return documents
        #             else:
        #                 logger.warning("Textract loader returned empty content")
        #         except ValueError as ve:
        #             # Check for the special error code for expired AWS credentials
        #             if str(ve) == "AWS_CREDENTIALS_EXPIRED":
        #                 logger.warning("AWS credentials have expired, skipping Textract")
        #                 # Skip to next fallback
        #                 raise ValueError("AWS_CREDENTIALS_EXPIRED")
        #             else:
        #                 # Re-raise other ValueError exceptions
        #                 raise
        #     else:
        #         logger.warning("Textract loader could not be created")
        # except ValueError as ve:
        #     # Special handling for expired AWS credentials
        #     if str(ve) == "AWS_CREDENTIALS_EXPIRED":
        #         logger.warning("Skipping Textract due to expired AWS credentials")
        #         # Continue to next fallback
        #     else:
        #         logger.error(f"Textract loader failed with ValueError: {ve}")
        # except Exception as e:
        #     logger.error(f"Textract loader failed: {e}")
        # finally:
        #     # Restore original loader type
        #     self.settings.loader = original_loader_type
        
        # If all loaders failed, return an empty list
        logger.error(f"All loaders failed for document: {file_path}")
        return []

    @staticmethod
    def _generate_document_id() -> str:
        return uuid.uuid4().hex

    async def delete_document(self, document_id: str, parent_run_id: str = None) -> Dict[str, str]:
        """Delete a document."""
        try:
            # The parent_run_id will be handled by the traceable decorator
            result = await self.vector_db_service.delete_document(document_id, parent_run_id)
            return result
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    async def fetch_document_text_by_id(self, document_id: str) -> Optional[str]:
        """Fetch document text from the external API using document ID.
        
        Parameters
        ----------
        document_id : str
            The ID of the document to fetch
            
        Returns
        -------
        Optional[str]
            The document text if successful, None otherwise
        """
        try:
            logger.info(f"Fetching document text for document_id: {document_id}")
            
            # Format the API URL with document ID and token
            api_url = self.settings.document_api_endpoint.format(
                document_id, 
                self.settings.document_api_token
            )
            
            # Set headers
            headers = {
                'accept': 'application/json',
            }
            
            # Make the request
            response = requests.get(api_url, headers=headers)
            
            # Check for successful response
            if response.status_code == 200:
                # Extract and return the text from the JSON response
                text_content = response.json().get('text')
                
                if text_content:
                    logger.info(f"Successfully fetched text for document_id: {document_id} ({len(text_content)} characters)")
                    return text_content
                else:
                    logger.warning(f"Document API returned empty text for document_id: {document_id}")
            else:
                logger.error(f"Failed to fetch document text, status code: {response.status_code}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error fetching document text: {e}", exc_info=True)
            return None

    async def fetch_document_metadata_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch document metadata from the external API using document ID.

        Parameters
        ----------
        document_id : str
            The ID of the document to fetch metadata for

        Returns
        -------
        Optional[Dict[str, Any]]
            The document metadata if successful, None otherwise
        """
        try:
            logger.info(f"Fetching document metadata for document_id: {document_id}")

            # Use the dedicated metadata endpoint URL from settings
            api_url = self.settings.document_metadata_api_endpoint.format(document_id)

            # Set headers, including the API token if required by the external API
            headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {self.settings.document_api_token}' # Assuming same token works
            }

            # Make the request
            response = requests.get(api_url, headers=headers)

            # Check for successful response
            if response.status_code == 200:
                metadata = response.json()
                logger.info(f"Successfully fetched metadata for document_id: {document_id}")
                return metadata
            else:
                logger.error(f"Failed to fetch document metadata, status code: {response.status_code}, response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error fetching document metadata: {e}", exc_info=True)
            return None

    async def process_document_text(self, document_id: str, text_content: str) -> Optional[str]:
        """Process document text content and store as chunks in the vector database.
        
        Parameters
        ----------
        document_id : str
            The ID to use for the document
        text_content : str
            The text content of the document
            
        Returns
        -------
        Optional[str]
            The document ID if successful, None otherwise
        """
        try:
            logger.info(f"Processing text content for document_id: {document_id}")
            
            # Create a LangchainDocument from the text content
            langchain_doc = LangchainDocument(
                page_content=text_content,
                metadata={
                    "source": f"external-api-document-{document_id}",
                    "document_id": document_id,
                    "page": 1
                }
            )
            
            # Split the document into chunks
            chunks = self.splitter.split_documents([langchain_doc])
            
            if not chunks:
                logger.warning(f"No chunks were created from document text: {document_id}")
                return document_id  # Return the ID even if no chunks were created
            
            logger.info(f"Document text split into {len(chunks)} chunks")
            
            # Prepare chunks for vector storage
            parent_run_id = None  # The traceable decorator will handle this
            prepared_chunks = await self.vector_db_service.prepare_chunks(
                document_id, chunks, parent_run_id
            )
            
            if not prepared_chunks:
                logger.warning(f"No prepared chunks for document: {document_id}")
                return document_id
            
            # Add document_id to each prepared chunk
            for chunk in prepared_chunks:
                chunk["document_id"] = document_id
            
            # Store chunks in vector database
            result = await self.vector_db_service.upsert_vectors(prepared_chunks, parent_run_id)
            logger.info(f"Vector upsert result: {result}")
            
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document text: {e}", exc_info=True)
            return None
            
    async def fetch_and_process_document_by_id(self, document_id: str) -> Optional[str]:
        """Fetch document from external API by ID and process it.
        
        This method combines fetching and processing in a single call.
        
        Parameters
        ----------
        document_id : str
            The ID of the document to fetch and process
            
        Returns
        -------
        Optional[str]
            The document ID if successful, None otherwise
        """
        try:
            # Fetch document text from external API
            text_content = await self.fetch_document_text_by_id(document_id)
            
            if not text_content:
                logger.error(f"Failed to fetch text for document_id: {document_id}")
                return None
                
            # Process the text content
            return await self.process_document_text(document_id, text_content)
            
        except Exception as e:
            logger.error(f"Error in fetch and process flow: {e}", exc_info=True)
            return None
    
    async def batch_process_documents_by_ids(self, document_ids: List[str], 
                                           max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple documents by IDs in parallel with controlled concurrency.
        
        Parameters
        ----------
        document_ids : List[str]
            List of document IDs to process
        max_concurrent : int
            Maximum number of concurrent requests
            
        Returns
        -------
        List[Dict[str, Any]]
            List of results with status for each document ID
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one_id(doc_id: str) -> Dict[str, Any]:
            """Process a single document ID with semaphore control."""
            async with semaphore:
                try:
                    processed_id = await self.fetch_and_process_document_by_id(doc_id)
                    
                    if processed_id:
                        return {
                            "input_id": doc_id,
                            "status": "success",
                            "processed_id": processed_id
                        }
                    else:
                        return {
                            "input_id": doc_id,
                            "status": "error",
                            "message": "Failed to process document"
                        }
                except Exception as e:
                    logger.error(f"Error processing document ID {doc_id}: {str(e)}")
                    return {
                        "input_id": doc_id,
                        "status": "error",
                        "message": str(e)
                    }
        
        # Create tasks for all document IDs
        tasks = [process_one_id(doc_id) for doc_id in document_ids]
        
        # Process all document IDs with controlled concurrency
        results = await asyncio.gather(*tasks)
        
        # Generate summary counts
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        return {
            "results": results,
            "summary": {
                "total_requested": len(document_ids),
                "successful": successful,
                "failed": failed
            }
        }
    
    async def get_document_chunks(self, document_id: str, parent_run_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document from the vector database.
        
        This method queries the vector database for all chunks associated with
        the given document ID and returns them.
        
        Parameters
        ----------
        document_id : str
            The ID of the document to retrieve chunks for.
            
        Returns
        -------
        List[Dict[str, Any]]
            A list of document chunks, each containing text and metadata.
            
        Raises
        ------
        ValueError
            If the document is not found or if no chunks are available.
        """
        try:
            logger.info(f"Getting document chunks for document_id: {document_id}")
            
            # The parent_run_id will be handled by the traceable decorator
            
            # Query the vector database for all chunks with this document_id
            chunks = await self.vector_db_service.get_document_chunks(document_id, parent_run_id)
            
            if not chunks:
                logger.warning(f"No chunks found for document_id: {document_id}")
                return []
                
            logger.info(f"Retrieved {len(chunks)} chunks for document_id: {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {e}", exc_info=True)
            raise

    async def process_document_background(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        document_id: str,
    ) -> Optional[str]:
        """
        Process a document in the background.
        """
        logger.info(f"Starting background processing for document: {filename} ({document_id})")
        try:
            # Initial document record with 'processing' status is now inserted by the upload endpoint.
            # No need for an initial update_status(document_id, "processing") here.
            
            documents_dir = os.path.join(tempfile.gettempdir(), "ai_grid_documents")
            os.makedirs(documents_dir, exist_ok=True)
            file_extension = os.path.splitext(filename)[1] or ".pdf"
            file_path = os.path.join(documents_dir, f"{document_id}{file_extension}")
            logger.info(f"Saving document to: {file_path}")
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            try:
                chunks = await self._process_document(file_path)
                logger.info(f"Processed document into {len(chunks)} chunks")
                if not chunks:
                    logger.warning(f"No chunks were extracted from document: {filename}")
                    await DocumentPersistenceService.update_status(document_id, "completed")
                    return document_id
                if chunks:
                    logger.info(f"First chunk sample: {chunks[0].page_content[:100]}...")
                for chunk in chunks:
                    chunk.metadata["file_path"] = file_path
                parent_run_id = None
                prepared_chunks = await self.vector_db_service.prepare_chunks(
                    document_id, chunks, parent_run_id
                )
                logger.info(f"Prepared {len(prepared_chunks)} chunks for vector storage")
                if not prepared_chunks:
                    logger.warning(f"No prepared chunks for document: {filename}")
                    await DocumentPersistenceService.update_status(document_id, "completed")
                    return document_id
                for chunk in prepared_chunks:
                    chunk["file_path"] = file_path
                result = await self.vector_db_service.upsert_vectors(prepared_chunks, parent_run_id)
                logger.info(f"Upsert result: {result}")
                await DocumentPersistenceService.update_status(document_id, "completed")
                logger.info(f"Background processing completed successfully for document: {filename} ({document_id})")
                return document_id
            except Exception as e:
                logger.error(f"Error processing document: {e}", exc_info=True)
                await DocumentPersistenceService.update_status(document_id, "failed")
                return None
        except Exception as e:
            logger.error(f"Error in background processing for document: {filename} ({document_id}): {e}", exc_info=True)
            await DocumentPersistenceService.update_status(document_id, "failed")
            return None
        finally:
            pass

    async def get_document_status(self, document_id: str) -> str:
        """
        Get document processing status.
        Returns 'completed' if document exists in vector store but not in status tracking.
        """
        # First check if document exists in the status tracking database
        doc = await DocumentPersistenceService.get_document(document_id)
        if doc:
            return doc.status
            
        # If not found in DB but exists in vector store, mark as completed
        # This is for backward compatibility with documents loaded before DB tracking
        try:
            # Check if document exists in vector database by querying for at least one chunk
            chunks = await self.vector_db_service.get_document_chunks(document_id)
            if chunks and len(chunks) > 0:
                # Document exists in vector store, mark it as completed
                logger.info(f"Document {document_id} found in vector store but not in status DB. Marking as completed.")
                return "completed"
        except Exception as e:
            logger.error(f"Error checking document existence in vector store: {e}")
            
        # Default status if not found anywhere
        return "unknown"
