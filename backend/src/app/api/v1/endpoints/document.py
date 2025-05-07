"""Document router with optimized performance."""

import asyncio
import logging
import time
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Request, Response, Body

from app.core.dependencies import get_document_service
from app.models.document import Document
from app.schemas.document_api import (
    DeleteDocumentResponseSchema,
    DocumentResponseSchema,
    BatchUploadResponseSchema,
    DocumentPreviewResponseSchema,
    DocumentByIdSchema,
    BatchFetchByIdsSchema,
    BatchFetchResponseSchema,
)
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document"])

@router.options(
    "",
    status_code=200,
)
async def options_document_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for document uploads.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}

@router.options(
    "/batch",
    status_code=200,
)
async def options_batch_document_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for batch document uploads.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.options(
    "/fetch-by-id",
    status_code=200,
)
async def options_fetch_by_id_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for fetch by ID endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.options(
    "/batch-fetch-by-ids",
    status_code=200,
)
async def options_batch_fetch_by_ids_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for batch fetch by IDs endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.post(
    "",
    response_model=DocumentResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_endpoint(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponseSchema:
    """
    Upload a document and process it.

    Parameters
    ----------
    file : UploadFile
        The file to be uploaded and processed.
    document_service : DocumentService
        The document service for processing the file.

    Returns
    -------
    DocumentResponse
        The processed document information.

    Raises
    ------
    HTTPException
        If the file name is missing or if an error occurs during processing.
    """
    # Ensure CORS headers are present with additional debug information
    origin = request.headers.get("Origin")
    # Add Vary header to help with caching
    response.headers["Vary"] = "Origin"
    
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers, DNT, If-Modified-Since, Cache-Control, Range"
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name is missing",
        )

    logger.info(
        f"Endpoint received file: {file.filename}, content type: {file.content_type}"
    )

    start_time = time.time()
    try:
        # Read file content
        file_content = await file.read()
        read_time = time.time() - start_time
        logger.info(f"File read completed in {read_time:.2f} seconds")
        
        # Log the file size
        logger.info(f"Processing file of size: {len(file_content)} bytes")
        
        # Process document
        process_start = time.time()
        document_id = await document_service.upload_document(
            file.filename, file_content
        )
        process_time = time.time() - process_start
        logger.info(f"Document processing completed in {process_time:.2f} seconds")

        if document_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while processing the document",
            )

        # TODO: Fetch actual document details from a database
        document = Document(
            id=document_id,
            name=file.filename,
            author="author_name",  # TODO: Determine this dynamically
            tag="document_tag",  # TODO: Determine this dynamically
            page_count=10,  # TODO: Determine this dynamically
        )
        
        total_time = time.time() - start_time
        logger.info(f"Total upload time: {total_time:.2f} seconds")
        return DocumentResponseSchema(**document.model_dump())

    except ValueError as ve:
        logger.error(f"ValueError in upload_document_endpoint: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Unexpected error in upload_document_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post(
    "/batch",
    response_model=BatchUploadResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def batch_upload_documents_endpoint(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> BatchUploadResponseSchema:
    """
    Upload multiple documents in parallel and process them.

    Parameters
    ----------
    files : List[UploadFile]
        The files to be uploaded and processed.
    document_service : DocumentService
        The document service for processing the files.

    Returns
    -------
    BatchUploadResponseSchema
        Information about the processed documents.

    Raises
    ------
    HTTPException
        If an error occurs during processing.
    """
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    logger.info(f"Batch upload endpoint received {len(files)} files")
    start_time = time.time()

    # Read all files first to avoid timeout issues
    file_data = []
    for file in files:
        if file.filename:
            try:
                content = await file.read()
                file_data.append((file.filename, content))
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {str(e)}")
    
    logger.info(f"Read {len(file_data)} files, starting processing")
    
    # Process files in parallel with concurrency control
    # Limit concurrency to avoid overwhelming the system
    semaphore = asyncio.Semaphore(10)  # Process up to 10 files concurrently
    
    async def process_file(filename: str, content: bytes):
        async with semaphore:
            try:
                document_id = await document_service.upload_document(filename, content)
                
                if document_id:
                    return Document(
                        id=document_id,
                        name=filename,
                        author="author_name",
                        tag="document_tag",
                        page_count=10,
                    )
                return None
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                return None

    # Create tasks for all files
    tasks = [process_file(filename, content) for filename, content in file_data]
    
    # Process all files with controlled concurrency
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    documents = [doc for doc in results if doc is not None]
    
    total_time = time.time() - start_time
    logger.info(f"Batch upload completed in {total_time:.2f} seconds, processed {len(documents)}/{len(file_data)} documents")
    
    return BatchUploadResponseSchema(
        documents=[DocumentResponseSchema(**doc.model_dump()) for doc in documents],
        total_files=len(files),
        successful_files=len(documents),
        failed_files=len(files) - len(documents),
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponseSchema)
async def delete_document_endpoint(
    request: Request,
    response: Response,
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponseSchema:
    """
    Delete a document.

    Parameters
    ----------
    document_id : str
        The ID of the document to be deleted.
    document_service : DocumentService
        The document service for deleting the document.

    Returns
    -------
    DeleteDocumentResponse
        A response containing the deletion status and message.

    Raises
    ------
    HTTPException
        If an error occurs during the deletion process.
    """
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    try:
        result = await document_service.delete_document(document_id)
        if result:
            return DeleteDocumentResponseSchema(
                id=document_id,
                status="success",
                message="Document deleted successfully",
            )
        else:
            return DeleteDocumentResponseSchema(
                id=document_id,
                status="error",
                message="Failed to delete document",
            )
    except ValueError as ve:
        logger.error(f"ValueError in delete_document_endpoint: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in delete_document_endpoint: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred"
        )


@router.post(
    "/fetch-by-id",
    response_model=DocumentResponseSchema,
    status_code=status.HTTP_200_OK,
)
async def fetch_document_by_id_endpoint(
    request: Request,
    response: Response,
    data: DocumentByIdSchema,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponseSchema:
    """
    Fetch a document by ID from the external API, process it, and store it.
    
    Parameters
    ----------
    data : DocumentByIdSchema
        The document ID to fetch.
    document_service : DocumentService
        The document service for processing the document.
        
    Returns
    -------
    DocumentResponseSchema
        The processed document information.
        
    Raises
    ------
    HTTPException
        If the document cannot be fetched or processed.
    """
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    
    logger.info(f"Fetching document by ID: {data.document_id}")
    start_time = time.time()
    
    try:
        # Fetch document text from external API and process it
        document_id = await document_service.fetch_and_process_document_by_id(data.document_id)
        
        if document_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to fetch or process document with ID: {data.document_id}",
            )
        
        # Create a document response
        document = Document(
            id=document_id,
            name=f"external-document-{data.document_id}",
            author="external-api",
            tag="fetched-by-id",
            page_count=1,  # Default since we can't determine the actual page count
        )
        
        total_time = time.time() - start_time
        logger.info(f"Document fetch and processing completed in {total_time:.2f} seconds")
        
        return DocumentResponseSchema(**document.model_dump())
        
    except Exception as e:
        logger.error(f"Error fetching document by ID: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )


@router.options(
    "/fetch-by-id",
    status_code=200,
)
async def options_fetch_by_id_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for fetch by ID endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.options(
    "/batch-fetch-by-ids",
    status_code=200,
)
async def options_batch_fetch_by_ids_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for batch fetch by IDs endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.get(
    "/get-metadata/{document_id}",
    status_code=status.HTTP_200_OK,
)
async def get_document_metadata_endpoint(
    request: Request,
    response: Response,
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Fetch metadata of a document by its ID from the external OCR API.
    This acts as a proxy to avoid CORS issues in the frontend.

    Parameters
    ----------
    document_id : str
        The document ID to fetch metadata for.
    document_service : DocumentService
        The document service for fetching document metadata.

    Returns
    -------
    dict
        Metadata of the document.

    Raises
    ------
    HTTPException
        If the document metadata cannot be fetched.
    """
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"

    logger.info(f"Fetching metadata for document ID: {document_id}")

    try:
        # Fetch document metadata from external API via the service
        metadata = await document_service.fetch_document_metadata_by_id(document_id)

        if metadata is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to fetch metadata for document ID: {document_id}",
            )

        # Return the raw metadata
        return metadata

    except Exception as e:
        logger.error(f"Error fetching document metadata by ID: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )


@router.options(
    "/get-metadata/{document_id}",
    status_code=200,
)
async def options_get_document_metadata_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for get document metadata endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.get(
    "/get-text/{document_id}",
    status_code=status.HTTP_200_OK,
)
async def get_document_text_endpoint(
    request: Request,
    response: Response,
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    """
    Fetch text content of a document by its ID without processing it.
    This is a simple proxy to the external OCR API.
    
    Parameters
    ----------
    document_id : str
        The document ID to fetch text for.
    document_service : DocumentService
        The document service for fetching document text.
        
    Returns
    -------
    dict
        Text content of the document.
        
    Raises
    ------
    HTTPException
        If the document text cannot be fetched.
    """
    origin = request.headers.get("Origin")
    
    # Add Vary header to help with caching
    response.headers["Vary"] = "Origin"
    
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    
    logger.info(f"Fetching text for document ID: {document_id}")
    
    try:
        # Fetch document text from external API
        text = await document_service.fetch_document_text_by_id(document_id)
        
        if text is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to fetch text for document ID: {document_id}",
            )
        
        # Return the raw text
        return {"document_id": document_id, "text": text}
        
    except Exception as e:
        logger.error(f"Error fetching document text by ID: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )


@router.options(
    "/{path:path}",
    status_code=200,
)
async def options_any_document_endpoint(
    request: Request,
    response: Response,
    path: str,
):
    """
    Handle preflight OPTIONS requests for any document endpoint.
    This universal OPTIONS handler will catch all preflight requests for any document paths.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers, DNT, If-Modified-Since, Cache-Control, Range"
        response.headers["Access-Control-Max-Age"] = "3600"  # Cache preflight for 60 minutes    
    
    return {}


@router.post(
    "/batch-fetch-by-ids",
    response_model=BatchFetchResponseSchema,
    status_code=status.HTTP_200_OK,
)
async def batch_fetch_documents_by_ids_endpoint(
    request: Request,
    response: Response,
    data: BatchFetchByIdsSchema,
    document_service: DocumentService = Depends(get_document_service),
) -> BatchFetchResponseSchema:
    """
    Fetch and process multiple documents by their IDs in parallel.
    
    Parameters
    ----------
    data : BatchFetchByIdsSchema
        The list of document IDs to fetch and process.
    document_service : DocumentService
        The document service for processing the documents.
        
    Returns
    -------
    BatchFetchResponseSchema
        Information about the processed documents.
        
    Raises
    ------
    HTTPException
        If an error occurs during batch processing.
    """
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    
    if not data.document_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document IDs provided",
        )
    
    logger.info(f"Batch fetching {len(data.document_ids)} documents by ID")
    start_time = time.time()
    
    try:
        # Process all document IDs in parallel with controlled concurrency
        results = await document_service.batch_process_documents_by_ids(
            data.document_ids,
            max_concurrent=data.max_concurrent
        )
        
        total_time = time.time() - start_time
        logger.info(f"Batch document fetch completed in {total_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch document fetch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during batch processing: {str(e)}",
        )


@router.get("/{document_id}/preview", response_model=DocumentPreviewResponseSchema)
async def preview_document_text(
    request: Request,
    response: Response,
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentPreviewResponseSchema:
    """
    Get document preview as text by retrieving chunks from the vector database.

    Parameters
    ----------
    document_id : str
        The ID of the document to preview.
    document_service : DocumentService
        The document service for retrieving the document chunks.

    Returns
    -------
    DocumentPreviewResponseSchema
        The preview content of the document.

    Raises
    ------
    HTTPException
        If an error occurs during the preview process.
    """
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    logger.info(f"Document text preview requested for document_id: {document_id}")
    try:
        logger.info("Retrieving document chunks from vector database")
        chunks = await document_service.get_document_chunks(document_id)
        
        if not chunks:
            logger.warning(f"No chunks found for document_id: {document_id}")
            return DocumentPreviewResponseSchema(content="No content available for this document.")
        
        # Sort chunks by chunk number to maintain document order
        sorted_chunks = sorted(chunks, key=lambda x: x.get("chunk_number", 0))
        
        # Concatenate all chunk texts
        content = "\n\n".join(chunk.get("text", "") for chunk in sorted_chunks)
        logger.info(f"Retrieved document content, length: {len(content) if content else 0}")
        
        return DocumentPreviewResponseSchema(content=content)
    except ValueError as ve:
        logger.error(f"ValueError in preview_document_text: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in preview_document_text: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
