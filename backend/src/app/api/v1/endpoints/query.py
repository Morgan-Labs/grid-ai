"""Query router."""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List

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
        "anthropic": "bedrock-d2433f",
        "gemini": "gemini-3161fc", 
        "openai": "openai-6a3e17",
        "bedrock": "bedrock-d2433f"
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
        
        # Categorize queries by complexity for prioritized processing
        inference_requests = []
        simple_vector_requests = []
        complex_vector_requests = []
        
        for req in requests:
            if req.document_id == "00000000000000000000000000000000":
                inference_requests.append(req)
            elif not req.prompt.rules and req.prompt.type != "bool":
                simple_vector_requests.append(req)
            else:
                complex_vector_requests.append(req)
        
        # Pre-allocate results array with the exact size needed
        results = [None] * len(requests)
        request_to_index = {id(req): i for i, req in enumerate(requests)}
        
        # Process priority queries first
        priority_requests = inference_requests + simple_vector_requests
        remaining_requests = complex_vector_requests
        
        if priority_requests:
            # Prepare inference tasks
            inference_tasks = []
            for req in inference_requests:
                # Configure LLM service if custom model is specified
                if req.prompt.llm_model:
                    logger.info(f"📝 BATCH QUERY - Model: {req.prompt.llm_model}, Provider: {req.prompt.llm_provider or 'Auto-detected'}")
                    
                    # Create a copy of the settings to avoid modifying the global settings
                    req_settings = Settings()
                    req_settings.llm_model = req.prompt.llm_model
                    req_settings.llm_provider = settings.llm_provider
                    req_settings.portkey_api_key = settings.portkey_api_key
                    req_settings.portkey_enabled = settings.portkey_enabled
                    req_settings.openai_api_key = settings.openai_api_key
                    
                    # Configure LLM service with the request-specific settings
                    current_llm = configure_llm_service(
                        llm_service, 
                        req_settings, 
                        req.prompt.llm_model, 
                        req.prompt.llm_provider
                    )
                    logger.info(f"✅ Batch query LLM configured - Provider: {req_settings.llm_provider}, Model: {req_settings.llm_model}")
                else:
                    current_llm = llm_service
                
                task = inference_query(
                    req.prompt.query,
                    req.prompt.rules,
                    req.prompt.type,
                    current_llm,
                )
                inference_tasks.append((req, task))
            
            # Prepare simple vector query parameters
            simple_vector_params = []
            for req in simple_vector_requests:
                simple_vector_params.append({
                    "query_type": "simple_vector",
                    "query": req.prompt.query,
                    "document_id": req.document_id,
                    "rules": req.prompt.rules,
                    "format": req.prompt.type,
                    "_original_req": req,
                })
            
            # Execute inference queries
            if inference_tasks:
                for req, task in inference_tasks:
                    try:
                        result = await task
                        idx = request_to_index[id(req)]
                        results[idx] = create_response_from_result(req, result)
                    except Exception as e:
                        logger.error(f"Error in inference query: {str(e)}")
                        idx = request_to_index[id(req)]
                        results[idx] = create_fallback_response(req)
            
            # Execute simple vector queries
            if simple_vector_params:
                try:
                    vector_results = await process_queries_in_parallel(
                        simple_vector_params, llm_service, vector_db_service
                    )
                    
                    # Map results back to original indices
                    for i, result in enumerate(vector_results):
                        req = simple_vector_params[i]["_original_req"]
                        idx = request_to_index[id(req)]
                        results[idx] = create_response_from_result(req, result)
                except Exception as e:
                    for param in simple_vector_params:
                        req = param["_original_req"]
                        idx = request_to_index[id(req)]
                        results[idx] = create_fallback_response(req)
        
        # Process remaining complex queries
        if remaining_requests:
            complex_params = []
            for req in remaining_requests:
                complex_params.append({
                    "query_type": "hybrid",
                    "query": req.prompt.query,
                    "document_id": req.document_id,
                    "rules": req.prompt.rules,
                    "format": req.prompt.type,
                    "_original_req": req,
                })
            
            try:
                complex_results = await process_queries_in_parallel(
                    complex_params, llm_service, vector_db_service
                )
                
                # Map results back to original indices
                for i, result in enumerate(complex_results):
                    req = complex_params[i]["_original_req"]
                    idx = request_to_index[id(req)]
                    results[idx] = create_response_from_result(req, result)
            except Exception as e:
                for param in complex_params:
                    req = param["_original_req"]
                    idx = request_to_index[id(req)]
                    results[idx] = create_fallback_response(req)
        
        # Fill any remaining None values with fallbacks
        for i, result in enumerate(results):
            if result is None:
                results[i] = create_fallback_response(requests[i])
        
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
