"""Query router."""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.dependencies import get_llm_service, get_vector_db_service
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


@router.post("", response_model=QueryAnswerResponse)
async def run_query(
    request: QueryRequestSchema,
    llm_service: CompletionService = Depends(get_llm_service),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
) -> QueryAnswerResponse:
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
    if request.document_id == "00000000000000000000000000000000":
        try:
            query_response = await inference_query(
                request.prompt.query,
                request.prompt.rules,
                request.prompt.type,
                llm_service,
            )

            if not isinstance(query_response, QueryResult):
                query_response = QueryResult(**query_response)

            answer = QueryAnswer(
                id=uuid.uuid4().hex,
                document_id=request.document_id,
                prompt_id=request.prompt.id,
                answer=query_response.answer,
                type=request.prompt.type,
            )
            response_data = QueryAnswerResponse(
                answer=answer, 
                chunks=query_response.chunks or [],
                resolved_entities=query_response.resolved_entities or []
            )

            return response_data
        except Exception as e:
            logger.error(f"Error in inference query: {str(e)}")
            # Create a type-appropriate fallback
            if request.prompt.type == "int":
                fallback = 0
            elif request.prompt.type == "bool":
                fallback = False
            elif request.prompt.type == "int_array":
                fallback = []
            elif request.prompt.type == "str_array":
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

    try:
        logger.info(f"Received query request: {request}")

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

        if not isinstance(query_response, QueryResult):
            query_response = QueryResult(**query_response)

        answer = QueryAnswer(
            id=uuid.uuid4().hex,
            document_id=request.document_id,
            prompt_id=request.prompt.id,
            answer=query_response.answer,
            type=request.prompt.type,
        )
        # Include resolved_entities in the response
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
        error_detail = str(e) if str(e) else "Internal server error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


@router.post("/batch", response_model=List[QueryAnswerResponse])
async def run_batch_queries(
    request: Request,
    response: Response,
    requests: List[QueryRequestSchema],
    llm_service: CompletionService = Depends(get_llm_service),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
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
        
        # Prioritize and categorize queries for optimal processing
        inference_requests = []
        simple_vector_requests = []
        complex_vector_requests = []
        
        # Categorize queries by complexity for prioritized processing
        for req in requests:
            if req.document_id == "00000000000000000000000000000000":
                inference_requests.append(req)
            elif not req.prompt.rules and req.prompt.type != "bool":
                # Simple vector queries are faster
                simple_vector_requests.append(req)
            else:
                # Complex queries with rules or boolean outputs
                complex_vector_requests.append(req)
        
        # Pre-allocate results array with the exact size needed
        results = [None] * len(requests)
        request_to_index = {}
        
        # Map requests to their original indices
        for i, req in enumerate(requests):
            request_to_index[id(req)] = i
        
        # Process simple queries first to get initial results faster
        priority_requests = inference_requests + simple_vector_requests
        remaining_requests = complex_vector_requests
        
        # Process priority queries
        if priority_requests:
            # Prepare inference tasks
            inference_tasks = []
            for req in inference_requests:
                task = inference_query(
                    req.prompt.query,
                    req.prompt.rules,
                    req.prompt.type,
                    llm_service,
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
                    logger.error(f"Error processing simple vector queries: {str(e)}")
                    # Create fallbacks for failed queries
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
                logger.error(f"Error processing complex queries: {str(e)}")
                # Create fallbacks for failed queries
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
        # Create fallback responses for all requests
        return [create_fallback_response(req) for req in requests]

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
async def test_error(error_type: str = "timeout") -> Dict[str, Any]:
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
    