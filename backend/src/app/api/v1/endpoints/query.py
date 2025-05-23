"""Query router."""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List
from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.core.dependencies import get_llm_service, get_vector_db_service, get_settings
from app.core.config import Settings
from app.schemas.query_api import (
    QueryAnswer,
    QueryAnswerResponse,
    QueryRequestSchema,
    QueryResult,
)
from app.services.llm.base import CompletionService
from app.services.query_service import (
    decomposition_query,
    hybrid_query,
    inference_query,
    process_queries_in_parallel,
    simple_vector_query,
)
from app.services.vector_db.base import VectorDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])
logger.info("Query router initialized")


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
    Creates a new service instance if provider or critical settings change.
    
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
    # Create a copy of settings to avoid modifying the global cached instance
    settings_copy = deepcopy(settings)
    
    # Set model in the copied settings
    settings_copy.llm_model = model
    
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
        settings_copy.llm_provider = "portkey"
        from app.services.llm.factory import CompletionServiceFactory
        # Pass the copied and modified settings to the factory
        configured_llm_service = CompletionServiceFactory.create_service(settings_copy)
        
        if hasattr(configured_llm_service, 'update_virtual_key'):
            try:
                logger.info(f"Setting virtual key for {provider}: {virtual_key} with model: {model} using Portkey (non-OpenAI)")
                configured_llm_service.update_virtual_key(virtual_key, provider, model)
            except Exception as e:
                logger.error(f"Error setting virtual key (non-OpenAI): {str(e)}")
    else: # OpenAI provider
        if settings_copy.portkey_enabled: # Check copied settings for portkey_enabled
            settings_copy.llm_provider = "portkey"
            from app.services.llm.factory import CompletionServiceFactory
            configured_llm_service = CompletionServiceFactory.create_service(settings_copy)
            
            if hasattr(configured_llm_service, 'update_virtual_key'):
                try:
                    logger.info(f"Setting virtual key for OpenAI via Portkey: {virtual_key} with model: {model}")
                    configured_llm_service.update_virtual_key(virtual_key, provider, model)
                except Exception as e:
                    logger.error(f"Error setting virtual key (OpenAI via Portkey): {str(e)}")
        else:
            # Use direct OpenAI integration
            settings_copy.llm_provider = "openai"
            from app.services.llm.factory import CompletionServiceFactory
            configured_llm_service = CompletionServiceFactory.create_service(settings_copy)
    
    return configured_llm_service


@router.post("", response_model=QueryAnswerResponse)
async def run_query(
    request: QueryRequestSchema,
    req: Request,
    resp: Response,
    llm_service: CompletionService = Depends(get_llm_service),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
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
    
    # Check document status if document_id is provided
    if request.document_id:
        # Get document service
        from app.core.dependencies import get_document_service
        document_service = get_document_service(req, settings)
        
        # Get document status
        document_status = await document_service.get_document_status(request.document_id)
        
        # Only allow querying if document is fully processed
        # TEMPORARY FIX: Allow 'unknown' status since documents.db migration
        if document_status != "completed" and document_status != "unknown":
            error_message = f"Document {request.document_id} is not ready for querying (status: {document_status})"
            logger.warning(error_message)
            return QueryAnswerResponse(
                id=str(uuid.uuid4()),
                answer=QueryAnswer(
                    id=uuid.uuid4().hex,
                    document_id=request.document_id,
                    prompt_id=request.prompt.id,
                    answer=f"The document is still being processed. Please try again in a moment.",
                    type=request.prompt.type
                ),
                document_id=request.document_id,
                prompt=request.prompt,
                chunks=[],
                resolved_entities=[],
            )
    
    # Store original settings to restore later
    original_provider = settings.llm_provider
    original_model = settings.llm_model
    
    # Configure LLM service if custom model is specified
    if request.prompt.llm_model:
        logger.info(f"🔄 Configuring LLM service for model: {request.prompt.llm_model}")
        llm_service = configure_llm_service(
            llm_service, 
            settings, 
            request.prompt.llm_model, 
            request.prompt.llm_provider
        )
        logger.info(f"✅ LLM service configured - Provider: {settings.llm_provider}, Model: {settings.llm_model}")
    
    try:
        # Handle inference queries (no document)
        if request.document_id == "00000000000000000000000000000000":
            query_response = await inference_query(
                request.prompt.query,
                request.prompt.rules,
                request.prompt.type,
                llm_service,
            )
        else:
            # Determine query type
            query_type = (
                "hybrid"
                if request.prompt.rules or request.prompt.type == "bool"
                else "vector"
            )

            query_functions = {
                "decomposed": decomposition_query,
                "hybrid": hybrid_query,
                "vector": simple_vector_query,
            }

            query_response = await query_functions[query_type](
                request.prompt.query,
                request.document_id,
                request.prompt.rules,
                request.prompt.type,
                llm_service,
                vector_db_service,
            )

        # Convert to QueryResult if needed
        if not isinstance(query_response, QueryResult):
            query_response = QueryResult(**query_response)

        # Create answer
        answer = QueryAnswer(
            id=uuid.uuid4().hex,
            document_id=request.document_id,
            prompt_id=request.prompt.id,
            answer=query_response.answer,
            type=request.prompt.type,
        )
        
        # Create response
        response_data = QueryAnswerResponse(
            answer=answer,
            chunks=query_response.chunks or [],
            resolved_entities=query_response.resolved_entities or [],
        )

        return response_data

    except asyncio.TimeoutError:
        logger.error("Timeout occurred while processing the query")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out while waiting for a response from the language model"
        )
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        
        # Create a type-appropriate fallback
        if request.prompt.type == "int":
            fallback = 0
        elif request.prompt.type == "bool":
            fallback = False
        elif request.prompt.type == "int_array" or request.prompt.type == "str_array":
            fallback = []
        else:
            fallback = f"Error processing query"
            
        answer = QueryAnswer(
            id=uuid.uuid4().hex,
            document_id=request.document_id,
            prompt_id=request.prompt.id,
            answer=fallback,
            type=request.prompt.type,
        )
        return QueryAnswerResponse(answer=answer, chunks=[], resolved_entities=[])
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
    settings: Settings = Depends(get_settings),
) -> List[QueryAnswerResponse]:
    """
    Run a batch of queries and generate responses.

    This endpoint processes multiple query requests in parallel, improving
    efficiency for batch processing.

    Parameters
    ----------
    requests : List[QueryRequestSchema]
        A list of incoming query requests.
    llm_service : CompletionService
        The language model service.
    vector_db_service : VectorDBService
        The vector database service.

    Returns
    -------
    List[QueryResponseSchema]
        A list of generated responses to the queries.

    Raises
    ------
    HTTPException
        If there's an error processing the queries.
    """
    try:
        # Log the number of queries in the batch
        logger.info(f"Received batch query request with {len(requests)} queries")
        
        # Get document service
        from app.core.dependencies import get_document_service
        document_service = get_document_service(request, settings)
        
        # Check document status for all unique document IDs
        document_ids = {req.document_id for req in requests if req.document_id}
        document_statuses = {}
        
        for doc_id in document_ids:
            document_statuses[doc_id] = await document_service.get_document_status(doc_id)
            
        # Create a map of query functions by query type
        query_functions = {
            "decomposition": decomposition_query,
            "hybrid": hybrid_query,
            "simple_vector": simple_vector_query,
            "inference": inference_query,
        }
        
        # Create a list of query parameters for processing
        query_params = []
        final_responses = []
        
        start_time = time.time()

        # Check document status and prepare valid queries
        for req in requests:
            # Skip requests with document IDs that are not completed
            # TEMPORARY FIX: Allow 'unknown' status since documents.db migration
            if req.document_id and document_statuses.get(req.document_id) != "completed" and document_statuses.get(req.document_id) != "unknown":
                error_message = f"Document {req.document_id} is not ready for querying (status: {document_statuses.get(req.document_id)})"
                logger.warning(error_message)
                
                # Create a response with an appropriate message
                response = QueryAnswerResponse(
                    id=str(uuid.uuid4()),
                    answer=QueryAnswer(
                        id=uuid.uuid4().hex,
                        document_id=req.document_id,
                        prompt_id=req.prompt.id,
                        answer=f"The document is still being processed. Please try again in a moment.",
                        type=req.prompt.type
                    ),
                    document_id=req.document_id,
                    prompt=req.prompt,
                    chunks=[],
                    resolved_entities=[],
                )
                final_responses.append(response)
                continue
                
            # Configure LLM service if custom model is specified
            current_llm_service = llm_service
            if req.prompt.llm_model:
                current_llm_service = configure_llm_service(
                    llm_service, 
                    settings, 
                    req.prompt.llm_model, 
                    req.prompt.llm_provider
                )
            
            # Check if this is an inference-only query (placeholder ID)
            # Aligning with the check in the single query endpoint
            if req.document_id == "00000000000000000000000000000000":
                query_type = "inference"
                query_params.append({
                    "query_type": query_type,
                    "query": req.prompt.query,
                    "document_id": None, # Pass None downstream for clarity, type is already set
                    "rules": req.prompt.rules,
                    "format": req.prompt.type,
                    "llm_service": current_llm_service,
                    "vector_db_service": None, # No vector DB for inference
                    "request": req, # Pass original request for response creation
                })
            elif req.document_id is None:
                # Handle cases where document_id might genuinely be None (if that's valid)
                # Currently treating this also as inference, but could be an error case depending on requirements.
                logger.warning(f"Query received with document_id=None for request: {req.prompt.id}. Treating as inference.")
                query_type = "inference"
                query_params.append({
                    "query_type": query_type,
                    "query": req.prompt.query,
                    "document_id": None,
                    "rules": req.prompt.rules,
                    "format": req.prompt.type,
                    "llm_service": current_llm_service,
                    "vector_db_service": None,
                    "request": req,
                })
            else:
                # Determine query type for actual document-based query
                query_type = (
                    "hybrid"
                    if req.prompt.rules or req.prompt.type == "bool"
                    else "vector" # Default to vector if document_id is present but no rules/bool type
                )
                
                # Add parameters for document-based query
                query_params.append({
                    "query_type": query_type,
                    "query": req.prompt.query,
                    "document_id": req.document_id,
                    "rules": req.prompt.rules,
                    "format": req.prompt.type,
                    "llm_service": current_llm_service,
                    "vector_db_service": vector_db_service, # Pass vector DB service
                    "request": req, # Pass original request for response creation
                })
        
        # If we have valid queries to process
        if query_params:
            # Process queries in parallel
            logger.info(f"Processing {len(query_params)} queries in parallel")
            results = await process_queries_in_parallel(query_params, llm_service, vector_db_service)
            
            # Create responses for valid results
            for i, result in enumerate(results):
                # Get the original request
                req = query_params[i]["request"]
                
                response = QueryAnswerResponse(
                    id=str(uuid.uuid4()),
                    answer=QueryAnswer(
                        id=uuid.uuid4().hex,
                        document_id=req.document_id,
                        prompt_id=req.prompt.id,
                        answer=result.answer,
                        type=req.prompt.type
                    ),
                    document_id=req.document_id,
                    prompt=req.prompt,
                    chunks=result.chunks,
                    resolved_entities=result.resolved_entities,
                )
                final_responses.append(response)
        
        end_time = time.time()
        logger.info(f"Batch query processing completed in {end_time - start_time:.2f}s")
        
        return final_responses
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch queries: {str(e)}",
        )


def create_response_from_result(req: QueryRequestSchema, result: Any) -> QueryAnswerResponse:
    """Create a QueryAnswerResponse from a query result."""
    # Handle exceptions
    if isinstance(result, Exception):
        return create_fallback_response(req)
    
    # Convert to QueryResult if needed
    if not isinstance(result, QueryResult):
        try:
            result = QueryResult(**result)
        except Exception:
            return create_fallback_response(req)
    
    # Create answer object
    answer = QueryAnswer(
        id=uuid.uuid4().hex,
        document_id=req.document_id,
        prompt_id=req.prompt.id,
        answer=result.answer,
        type=req.prompt.type,
    )
    
    # Create response
    return QueryAnswerResponse(
        answer=answer,
        chunks=result.chunks or [],
        resolved_entities=result.resolved_entities or [],
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
        fallback_answer = ""
    
    # Create answer object
    answer = QueryAnswer(
        id=uuid.uuid4().hex,
        document_id=req.document_id,
        prompt_id=req.prompt.id,
        answer=fallback_answer,
        type=req.prompt.type,
    )
    
    # Create response
    return QueryAnswerResponse(
        answer=answer,
        chunks=[],
        resolved_entities=[],
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
