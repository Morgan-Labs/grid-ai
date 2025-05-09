"""Query router."""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.core.dependencies import get_llm_service, get_vector_db_service, get_settings, get_document_service
from app.core.config import Settings
from app.schemas.query_api import (
    QueryAnswer,
    QueryAnswerResponse,
    QueryRequestSchema,
    QueryResult,
)
from app.services.llm.base import CompletionService
from app.services.query_service import (
    process_query_with_retry,
    schedule_query_queue_processing,
    check_document_readiness
)
from app.services.vector_db.base import VectorDBService
from app.services.document_service import DocumentService
from app.models.query_core import Chunk, FormatType, QueryType, Rule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])
logger.info("Query router initialized")

# Add utility function for inference queries
async def inference_query(
    query: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
) -> QueryResult:
    """Generate a response directly from LLM without document retrieval."""
    # Use process_query_with_retry with a special document ID for inference
    dummy_doc_id = "00000000000000000000000000000000"  # Special ID for inference
    return await process_query_with_retry(
        query_type="inference",
        query=query,
        document_id=dummy_doc_id,
        rules=rules,
        format=format,
        llm_service=llm_service,
        vector_db_service=None
    )


@router.options(
    "",
    status_code=200,
)
async def options_query_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for query endpoint.
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
async def options_batch_query_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for batch query endpoint.
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
    "/test-error",
    status_code=200,
)
async def options_test_error_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for test error endpoint.
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


def configure_llm_service(
    llm_service: CompletionService, 
    settings: Settings,
    model: str,
    provider: str = None,
) -> CompletionService:
    """Configure the LLM service based on model and provider.
    
    Parameters
    ----------
    llm_service : CompletionService
        The language model service to configure.
    settings : Settings
        Application settings.
    model : str
        The model to use.
    provider : str, optional
        The provider to use. If None, it will be detected from the model name.
        
    Returns
    -------
    CompletionService
        The configured LLM service.
    """
    # Set model in settings
    settings.llm_model = model
    
    # Auto-detect provider from model name if not provided
    if not provider:
        if "gpt" in model.lower():
            provider = "openai"
        # elif "claude" in model.lower():
        #     provider = "anthropic"
        elif "gemini" in model.lower():
            provider = "gemini"
        elif "anthropic" in model.lower():
            provider = "bedrock"
        else:
            provider = "openai"  # Default to OpenAI
    
    # Get the appropriate virtual key based on provider
    provider_keys = {
        "anthropic": "anthropic-a27fda",
        "gemini": "gemini-3161fc", 
        "openai": "openai-6a3e17",
        "bedrock": "bedrock-77eded"
    }
    
    virtual_key = provider_keys.get(provider)
    
    # For non-OpenAI providers, always use Portkey
    if provider != "openai":
        # Set provider to portkey in settings
        settings.llm_provider = "portkey"
        
        # Create a new Portkey service
        from app.services.llm.factory import CompletionServiceFactory
        llm_service = CompletionServiceFactory.create_service(settings)
        
        # Configure virtual key and model
        if hasattr(llm_service, 'update_virtual_key'):
            try:
                logger.info(f"Setting virtual key for {provider}: {virtual_key} with model: {model}")
                llm_service.update_virtual_key(virtual_key, provider, model)
            except Exception as e:
                logger.error(f"Error setting virtual key: {str(e)}")
    
    # For OpenAI, we can either use direct OpenAI integration or Portkey
    else:
        # If we want to use Portkey for OpenAI too (for tracking/routing)
        if settings.portkey_enabled:
            settings.llm_provider = "portkey"
            from app.services.llm.factory import CompletionServiceFactory
            llm_service = CompletionServiceFactory.create_service(settings)
            
            # Configure virtual key and model for OpenAI through Portkey
            if hasattr(llm_service, 'update_virtual_key'):
                try:
                    logger.info(f"Setting virtual key for OpenAI: {virtual_key} with model: {model}")
                    llm_service.update_virtual_key(virtual_key, provider, model)
                except Exception as e:
                    logger.error(f"Error setting virtual key: {str(e)}")
        else:
            # Use direct OpenAI integration
            settings.llm_provider = "openai"
            from app.services.llm.factory import CompletionServiceFactory
            llm_service = CompletionServiceFactory.create_service(settings)
    
    return llm_service


@router.post("", response_model=QueryAnswerResponse)
async def run_query(
    request: QueryRequestSchema,
    req: Request,
    resp: Response,
    llm_service: CompletionService = Depends(get_llm_service),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
    document_service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> QueryAnswerResponse:
    # Ensure CORS headers are present
    origin = req.headers.get("Origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    """
    Run a query and generate a response.

    This endpoint processes incoming query requests, determines the appropriate
    query type, and executes the corresponding query function. It supports
    vector, hybrid, and decomposition query types.

    Parameters
    ----------
    request : QueryRequestSchema
        The incoming query request.
    llm_service : CompletionService
        The language model service.
    vector_db_service : VectorDBService
        The vector database service.
    document_service : DocumentService
        The document service for checking document readiness.

    Returns
    -------
    QueryResponseSchema
        The generated response to the query.

    Raises
    ------
    HTTPException
        If there's an error processing the query.
    """
    # Log the incoming request parameters
    logger.info("📝 QUERY REQUEST PARAMETERS:")
    logger.info(f"📝 Document ID: {request.document_id}")
    logger.info(f"📝 Query: {request.prompt.query}")
    logger.info(f"📝 Type: {request.prompt.type}")
    logger.info(f"📝 LLM Model: {request.prompt.llm_model or 'Default'}")
    logger.info(f"📝 LLM Provider: {request.prompt.llm_provider or 'Auto-detected'}")
    logger.info(f"📝 Rules: {len(request.prompt.rules)} rules")
    
    # Store original settings to restore later
    original_provider = settings.llm_provider
    original_model = settings.llm_model
    
    # Configure LLM service if custom model is specified
    if request.prompt.llm_model:
        logger.info(f"🔄 Configuring LLM service for model: {request.prompt.llm_model}")
        llm_service = configure_llm_service(
            llm_service, settings, 
            request.prompt.llm_model, 
            request.prompt.llm_provider
        )
    
    try:
        query_start_time = time.time()
        request_id = str(uuid.uuid4())
        
        query_type = request.prompt.type
        query = request.prompt.query
        document_id = request.document_id
        rules = request.prompt.rules
        format = request.prompt.format or "str"
        
        # Log request details with request ID for tracking
        logger.info(f"💬 [{request_id}] Processing query: '{query[:50]}...' (Type: {query_type})")
        
        # Prepare result based on query type
        result = None
        
        # Check if this is an inference query (no document needed)
        if query_type == "inference":
            try:
                result = await inference_query(query, rules, format, llm_service)
            except Exception as e:
                logger.error(f"🔴 [{request_id}] Error in inference query: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error in inference query: {str(e)}",
                )
        # For document-based queries, process with document readiness check
        elif document_id:
            try:
                result = await process_query_with_retry(
                    query_type,
                    query,
                    document_id,
                    rules,
                    format,
                    llm_service,
                    vector_db_service,
                    document_service=document_service  # Pass document service for readiness check
                )
                
                # Check if the result indicates the query was queued
                if hasattr(result, 'queued') and result.queued:
                    # The query was queued because document isn't ready yet
                    logger.info(f"🟡 [{request_id}] Query queued for document {document_id}, waiting for processing to complete")
                    
                    # Handle the queued query differently
                    if hasattr(result, 'queued_query') and result.queued_query:
                        try:
                            # Wait for the future to be resolved (with timeout)
                            # We'll wait up to 1 second for now, and return a temporary response
                            # The client should poll the results endpoint
                            timeout = 1.0  # seconds
                            try:
                                # Try to get result with timeout
                                actual_result = await asyncio.wait_for(result.queued_query.future, timeout)
                                # If we got the result, use it
                                result = actual_result
                                logger.info(f"✅ [{request_id}] Document became ready while waiting, got result immediately")
                            except asyncio.TimeoutError:
                                # If it times out, we'll return the queued status
                                logger.info(f"⏳ [{request_id}] Returning queued status, document still processing")
                                # Return a response indicating document is still processing
                                return QueryAnswerResponse(
                                    query=query,
                                    result=QueryAnswer(
                                        answer="Document is still processing, please try again shortly",
                                        processing=True  # Indicate document is still processing
                                    ),
                                    document_id=document_id,
                                    retrieval_type=query_type,
                                    timing={
                                        "total_seconds": time.time() - query_start_time,
                                    }
                                )
                        except Exception as future_error:
                            logger.error(f"🔴 [{request_id}] Error waiting for queued query result: {str(future_error)}")
                            # Continue with regular query flow
            except Exception as e:
                logger.error(f"🔴 [{request_id}] Error in document query: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error processing query: {str(e)}",
                )
        else:
            # No document ID and not an inference query
            logger.error(f"🔴 [{request_id}] Document ID is required for non-inference queries")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document ID is required for this query type",
            )
        
        # Create response from the result
        response_data = create_response_from_result(request, result)
        
        # Add timing information
        response_data.timing = {
            "total_seconds": time.time() - query_start_time,
        }
        
        # Log completion
        answer_preview = str(response_data.result.answer)[:50]
        answer_preview = answer_preview + "..." if len(str(response_data.result.answer)) > 50 else answer_preview
        logger.info(f"✅ [{request_id}] Query completed in {response_data.timing['total_seconds']:.2f}s, answer: {answer_preview}")
        
        return response_data
    except Exception as e:
        logger.error(f"🔴 Unexpected error in run_query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )
    finally:
        # Restore original settings
        settings.llm_provider = original_provider
        settings.llm_model = original_model


@router.post("/batch", response_model=List[QueryAnswerResponse])
async def run_batch_queries(
    request: Request,
    response: Response,
    requests: List[QueryRequestSchema],
    llm_service: CompletionService = Depends(get_llm_service),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
    document_service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> List[QueryAnswerResponse]:
    """
    Run multiple queries in parallel with optimized processing for faster initial responses.
    
    This endpoint processes multiple query requests in parallel, with optimizations to
    return initial results as quickly as possible.

    Parameters
    ----------
    requests : List[QueryRequestSchema]
        The list of query requests to process in parallel.
    llm_service : CompletionService
        The language model service.
    vector_db_service : VectorDBService
        The vector database service.
    document_service : DocumentService
        The document service for checking readiness.

    Returns
    -------
    List[QueryAnswerResponse]
        A list of query responses in the same order as the input requests.

    Raises
    ------
    HTTPException
        If there's an error processing the queries.
    """
    # Ensure CORS headers are set even for error responses
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin"
    
    try:
        start_time = time.time()
        logger.info(f"Received batch query request with {len(requests)} queries")
        
        # Store original settings to restore later
        original_provider = settings.llm_provider
        original_model = settings.llm_model
        
        # Create tasks for all queries
        tasks = []
        for req in requests:
            # Configure custom LLM if specified
            custom_llm = llm_service
            if req.prompt.llm_model:
                custom_llm = configure_llm_service(
                    llm_service,
                    settings,
                    req.prompt.llm_model,
                    req.prompt.llm_provider
                )
            
            # Create appropriate query task based on type
            if req.prompt.type == "inference":
                task = inference_query(
                    req.prompt.query,
                    req.prompt.rules,
                    req.prompt.format or "str",
                    custom_llm
                )
            else:
                # Regular document-based query
                task = process_query_with_retry(
                    req.prompt.type,
                    req.prompt.query,
                    req.document_id,
                    req.prompt.rules,
                    req.prompt.format or "str", 
                    custom_llm,
                    vector_db_service,
                    document_service=document_service
                )
            
            tasks.append((req, task))
        
        # Execute all tasks in parallel with a semaphore to limit concurrency
        results = [None] * len(requests)
        
        # Process queries with some concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent queries
        
        async def process_query_with_semaphore(index, req, task):
            async with semaphore:
                try:
                    result = await task
                    return index, result
                except Exception as e:
                    logger.error(f"Error processing query {index}: {str(e)}")
                    return index, None
        
        # Create and gather all tasks
        gather_tasks = [
            process_query_with_semaphore(i, req, task) 
            for i, (req, task) in enumerate(tasks)
        ]
        
        # Execute all tasks and collect results
        query_results = await asyncio.gather(*gather_tasks)
        
        # Process results
        for index, result in query_results:
            if result is not None:
                results[index] = create_response_from_result(requests[index], result)
            else:
                results[index] = create_fallback_response(requests[index])
        
        elapsed = time.time() - start_time
        logger.info(f"Batch query processing completed in {elapsed:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        return [create_fallback_response(req) for req in requests]
    finally:
        # Restore original settings
        settings.llm_provider = original_provider
        settings.llm_model = original_model


def create_response_from_result(req: QueryRequestSchema, result: Any) -> QueryAnswerResponse:
    """
    Create a response object from a query result.
    
    Parameters
    ----------
    req : QueryRequestSchema
        The original query request
    result : Any
        The query result object
        
    Returns
    -------
    QueryAnswerResponse
        The formatted response
    """
    # Handle the case where result might have the 'queued' attribute
    processing = hasattr(result, 'queued') and result.queued

    # Create answer with processing flag
    answer = QueryAnswer(
        id=str(uuid.uuid4()),
        document_id=req.document_id,
        prompt_id=req.prompt.id,
        answer=result.answer,
        type=req.prompt.type,
        processing=processing
    )
    
    # Create response object
    return QueryAnswerResponse(
        query=req.prompt.query,
        result=answer,
        document_id=req.document_id,
        retrieval_type=req.prompt.type,
        chunks=[] if processing else [chunk.model_dump() for chunk in (result.chunks or [])],
        resolved_entities=[] if processing else [entity.model_dump() for entity in (result.resolved_entities or [])]
    )


def create_fallback_response(req: QueryRequestSchema) -> QueryAnswerResponse:
    """Create a fallback response for a failed query."""
    # Create a type-appropriate fallback
    if req.prompt.type == "int":
        fallback_answer = 0
    elif req.prompt.type == "bool":
        fallback_answer = False
    elif req.prompt.type.endswith("_array"):
        fallback_answer = []
    else:
        fallback_answer = "Error processing query"
    
    # Create answer object
    answer = QueryAnswer(
        id=str(uuid.uuid4()),
        document_id=req.document_id,
        prompt_id=req.prompt.id,
        answer=fallback_answer,
        type=req.prompt.type,
    )
    
    # Create response with the new schema
    return QueryAnswerResponse(
        query=req.prompt.query,
        result=answer,
        document_id=req.document_id,
        retrieval_type=req.prompt.type,
        chunks=[],
        resolved_entities=[]
    )


@router.get("/test-error", response_model=Dict[str, Any])
async def test_error(
    request: Request,
    response: Response,
    error_type: str = "timeout"
) -> Dict[str, Any]:
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
    """
    Test endpoint to simulate different types of errors.
    
    This endpoint is useful for testing error handling in the frontend.
    
    Parameters
    ----------
    error_type : str
        The type of error to simulate. Options: "timeout", "validation", "server".
        
    Returns
    -------
    Dict[str, Any]
        A message indicating the error was simulated.
        
    Raises
    ------
    HTTPException
        The simulated error.
    """
    if error_type == "timeout":
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Simulated timeout error"
        )
    elif error_type == "validation":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Simulated validation error"
        )
    elif error_type == "server":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulated server error"
        )
    else:
        return {"message": f"Unknown error type: {error_type}"}
