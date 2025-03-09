"""Amazon Textract PDF loader service with S3 integration and optimizations."""

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple
import time
import boto3
import io
import threading

from langchain.schema import Document as LangchainDocument
from langchain_community.document_loaders import AmazonTextractPDFLoader

from app.services.loaders.base import LoaderService
from app.core.config import Settings

logger = logging.getLogger(__name__)

# Simple in-memory cache for documents
# Key: file_hash, Value: (timestamp, documents)
TEXTRACT_CACHE: Dict[str, tuple[float, List[LangchainDocument]]] = {}
# Cache expiration time in seconds (1 hour)
CACHE_EXPIRATION = 3600

# Map of local file paths to S3 keys
# This helps with cleanup if needed
S3_FILE_MAP: Dict[str, str] = {}

# Thread pool for parallel processing
THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# S3 client pool
S3_CLIENTS = {}
S3_CLIENT_LOCK = threading.Lock()

# Textract client pool
TEXTRACT_CLIENTS = {}
TEXTRACT_CLIENT_LOCK = threading.Lock()

class TextractLoader(LoaderService):
    """Amazon Textract PDF loader service with S3 integration and optimizations."""

    def __init__(self, settings: Settings):
        """Initialize the Textract loader service."""
        self.settings = settings
        self.aws_region = settings.aws_region
        self.aws_access_key = settings.aws_access_key_id
        self.aws_secret_key = settings.aws_secret_access_key
        self.aws_session_token = settings.aws_session_token
        self.s3_bucket = settings.s3_bucket_name
        self.s3_prefix = settings.s3_prefix

    async def load(self, file_path: str) -> List[LangchainDocument]:
        """Load document from file path using Amazon Textract with S3."""
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading file with extension: {file_extension}")

        supported_extensions = [".pdf", ".tiff", ".tif", ".png", ".jpg", ".jpeg"]
        
        if file_extension not in supported_extensions:
            error_msg = f"Unsupported file type for Textract: {file_path}. Supported types: {', '.join(supported_extensions)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Generate a hash of the file for caching
        file_hash = await self._get_file_hash(file_path)
        
        # Check cache first
        if file_hash in TEXTRACT_CACHE:
            timestamp, documents = TEXTRACT_CACHE[file_hash]
            if time.time() - timestamp < CACHE_EXPIRATION:
                logger.info(f"Using cached Textract result for: {file_path}")
                return documents
        
        # Not in cache or cache expired, process with Textract
        start_time = time.time()
        
        # Upload to S3 first
        s3_key = await self._upload_to_s3(file_path)
        
        # Process with Textract using the S3 location
        documents = await self._process_with_textract_s3(s3_key, file_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Textract processing completed in {elapsed_time:.2f} seconds")
        
        # Cache the result using the file hash
        TEXTRACT_CACHE[file_hash] = (time.time(), documents)
        return documents

    async def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash of the file for caching."""
        try:
            loop = asyncio.get_event_loop()
            
            # Use a small buffer size for large files
            buffer_size = 65536  # 64kb chunks
            
            # Use SHA256 for hashing
            sha256 = hashlib.sha256()
            
            def _hash_file():
                with open(file_path, 'rb') as f:
                    while True:
                        data = f.read(buffer_size)
                        if not data:
                            break
                        sha256.update(data)
                return sha256.hexdigest()
            
            # Run the hashing in a thread pool
            file_hash = await loop.run_in_executor(THREAD_POOL, _hash_file)
            return file_hash
            
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            # Fall back to using the file path and modification time
            return f"{file_path}_{os.path.getmtime(file_path)}"

    def _get_s3_client(self):
        """Get or create an S3 client from the pool."""
        thread_id = threading.get_ident()
        
        with S3_CLIENT_LOCK:
            if thread_id not in S3_CLIENTS:
                # Create an S3 client
                s3_client_args = {
                    "region_name": self.aws_region,
                    "aws_access_key_id": self.aws_access_key,
                    "aws_secret_access_key": self.aws_secret_key
                }
                
                # Add session token if available (required for temporary credentials)
                if self.aws_session_token:
                    s3_client_args["aws_session_token"] = self.aws_session_token
                    
                # Configure for faster uploads
                config = boto3.session.Config(
                    signature_version='s3v4',
                    s3={'use_accelerate_endpoint': False},
                    retries={'max_attempts': 3, 'mode': 'standard'}
                )
                
                S3_CLIENTS[thread_id] = boto3.client("s3", **s3_client_args, config=config)
                
            return S3_CLIENTS[thread_id]

    def _get_textract_client(self):
        """Get or create a Textract client from the pool."""
        thread_id = threading.get_ident()
        
        with TEXTRACT_CLIENT_LOCK:
            if thread_id not in TEXTRACT_CLIENTS:
                # Create boto3 clients
                textract_client_args = {
                    "region_name": self.aws_region,
                    "aws_access_key_id": self.aws_access_key,
                    "aws_secret_access_key": self.aws_secret_key
                }
                
                # Add session token if available (required for temporary credentials)
                if self.aws_session_token:
                    textract_client_args["aws_session_token"] = self.aws_session_token
                    
                # Configure for faster processing
                config = boto3.session.Config(
                    retries={'max_attempts': 3, 'mode': 'standard'}
                )
                
                TEXTRACT_CLIENTS[thread_id] = boto3.client("textract", **textract_client_args, config=config)
                
            return TEXTRACT_CLIENTS[thread_id]

    async def _upload_to_s3(self, file_path: str) -> str:
        """Upload a file to S3 and return the S3 key."""
        try:
            logger.info(f"Uploading file to S3: {file_path}")
            
            # Create a unique key for the file
            file_name = os.path.basename(file_path)
            file_id = str(uuid.uuid4())
            s3_key = f"{self.s3_prefix}/{file_id}/{file_name}"
            
            # Get an S3 client from the pool
            s3_client = self._get_s3_client()
            
            # Upload the file to S3 with optimized settings
            loop = asyncio.get_event_loop()
            
            def _upload_file():
                # Use TransferConfig for multipart uploads
                transfer_config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,  # 8MB
                    max_concurrency=4,
                    multipart_chunksize=8 * 1024 * 1024,  # 8MB
                    use_threads=True
                )
                
                # Upload with the transfer manager
                s3_client.upload_file(
                    file_path, 
                    self.s3_bucket, 
                    s3_key,
                    Config=transfer_config
                )
            
            # Run the upload in a thread pool
            await loop.run_in_executor(THREAD_POOL, _upload_file)
            
            logger.info(f"Successfully uploaded file to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Store the mapping for potential cleanup
            S3_FILE_MAP[file_path] = s3_key
            
            return s3_key
            
        except Exception as e:
            # Check for expired token error
            if "ExpiredToken" in str(e):
                logger.error(f"AWS credentials have expired: {str(e)}")
                # Return a special error code that can be handled by the caller
                raise ValueError("AWS_CREDENTIALS_EXPIRED")
            else:
                logger.error(f"Error uploading file to S3: {str(e)}")
                raise

    async def _process_with_textract_s3(self, s3_key: str, original_file_path: str) -> List[LangchainDocument]:
        """Process document with Amazon Textract using S3 location."""
        try:
            logger.info(f"Processing with Amazon Textract from S3: {s3_key}")
            
            # Get a Textract client from the pool
            textract_client = self._get_textract_client()
            
            # Create the loader with S3 path
            s3_path = f"s3://{self.s3_bucket}/{s3_key}"
            
            # Run Textract processing in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create the loader with appropriate configuration
            loader = AmazonTextractPDFLoader(
                s3_path,
                client=textract_client
            )
            
            # Run the loader in a thread pool
            documents = await loop.run_in_executor(THREAD_POOL, loader.load)
            
            # Update metadata to point to the original file path
            for doc in documents:
                doc.metadata["source"] = original_file_path
            
            if documents:
                logger.info(f"Successfully extracted {len(documents)} pages with Textract from S3")
                return documents
            else:
                logger.warning("Textract returned empty content")
                return [LangchainDocument(
                    page_content=f"Empty document: {os.path.basename(original_file_path)}",
                    metadata={"source": original_file_path, "page": 1}
                )]
                
        except Exception as e:
            logger.error(f"Error using Amazon Textract with S3: {str(e)}")
            return [LangchainDocument(
                page_content=f"Error processing document with Textract: {os.path.basename(original_file_path)}",
                metadata={"source": original_file_path, "page": 1, "error": str(e)}
            )]
