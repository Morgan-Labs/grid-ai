"""Query service."""

import asyncio
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from app.models.query_core import Chunk, FormatType, QueryType, Rule
from app.schemas.query_api import (
    QueryResult,
    ResolvedEntitySchema,
    SearchResponse,
)
from app.services.llm_service import (
    CompletionService,
    generate_inferred_response,
    generate_response,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SearchMethod = Callable[[str, str, List[Rule]], Awaitable[SearchResponse]]

# Concurrency control - increased from 5 to 10 for better throughput
# This can be adjusted based on server capacity
MAX_CONCURRENT_QUERIES = 10
QUERY_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

# Retry configuration - reduced delay for faster failure recovery
MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds


def get_search_method(
    query_type: QueryType, vector_db_service: Any
) -> SearchMethod:
    """Get the search method based on the query type."""
    if query_type == "decomposition":
        return vector_db_service.decomposed_search
    elif query_type == "hybrid":
        return vector_db_service.hybrid_search
    else:  # simple_vector
        return lambda q, d, r: vector_db_service.vector_search([q], d)


def extract_chunks(search_response: SearchResponse) -> List[Chunk]:
    """Extract chunks from the search response."""
    return (
        search_response["chunks"]
        if isinstance(search_response, dict)
        else search_response.chunks
    )


def replace_keywords(
    text: Union[str, List[str]], keyword_replacements: Dict[str, str]
) -> tuple[
    Union[str, List[str]], Dict[str, Union[str, List[str]]]
]:  # Changed return type
    """Replace keywords in text and return both the modified text and transformation details."""
    if not text or not keyword_replacements:
        return text, {
            "original": text,
            "resolved": text,
        }  # Return dict instead of TransformationDict

    # Handle list of strings
    if isinstance(text, list):
        original_text = text.copy()
        result = []
        modified = False

        # Create a single regex pattern for all keywords
        pattern = "|".join(map(re.escape, keyword_replacements.keys()))
        regex = re.compile(f"\\b({pattern})\\b")

        for item in text:
            # Single pass replacement for all keywords
            new_item = regex.sub(
                lambda m: keyword_replacements[m.group()], item
            )
            result.append(new_item)
            if new_item != item:
                modified = True

        if modified:
            return result, {"original": original_text, "resolved": result}
        return result, {"original": original_text, "resolved": result}

    # Handle single string
    return replace_keywords_in_string(text, keyword_replacements)


def replace_keywords_in_string(
    text: str, keyword_replacements: Dict[str, str]
) -> tuple[str, Dict[str, Union[str, List[str]]]]:  # Changed return type
    """Keywords for single string."""
    if not text:
        return text, {"original": text, "resolved": text}

    # Create a single regex pattern for all keywords
    pattern = "|".join(map(re.escape, keyword_replacements.keys()))
    regex = re.compile(f"\\b({pattern})\\b")

    # Single pass replacement
    result = regex.sub(lambda m: keyword_replacements[m.group()], text)

    # Only return transformation if something changed
    if result != text:
        return result, {"original": text, "resolved": result}
    return text, {"original": text, "resolved": text}


async def process_query_with_retry(
    query_type: QueryType,
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
    retries: int = MAX_RETRIES,
) -> QueryResult:
    """Process a query with optimized retry logic for faster response times."""
    last_exception = None
    
    # Truncate query for logging to avoid excessive log size
    query_preview = query[:30] + "..." if len(query) > 30 else query
    
    for attempt in range(retries + 1):
        try:
            # Use the semaphore to limit concurrency
            async with QUERY_SEMAPHORE:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{retries} for query: {query_preview}")
                
                start_time = time.time()
                
                # Process the query
                result = await process_query(
                    query_type,
                    query,
                    document_id,
                    rules,
                    format,
                    llm_service,
                    vector_db_service,
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Query processed in {elapsed:.2f}s")
                return result
                
        except Exception as e:
            last_exception = e
            
            if attempt < retries:
                # Simplified retry delay with minimal jitter
                jitter = RETRY_DELAY * (0.9 + 0.2 * attempt)
                await asyncio.sleep(jitter)
    
    # If we get here, all retries failed - create appropriate fallback
    logger.error(f"Query failed after {retries+1} attempts: {str(last_exception)}")
    
    # Return a fallback result based on the expected format
    if format == "int":
        fallback = 0
    elif format == "bool":
        fallback = False
    elif format in ["int_array", "str_array"]:
        fallback = []
    else:
        fallback = ""
        
    return QueryResult(
        answer=fallback,
        chunks=[],
        resolved_entities=[]
    )


async def process_query(
    query_type: QueryType,
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the specified type."""
    search_method = get_search_method(query_type, vector_db_service)

    # Step 1: Get search response
    search_response = await search_method(query, document_id, rules)
    chunks = extract_chunks(search_response)
    concatenated_chunks = " ".join(chunk.content for chunk in chunks)

    # Step 2: Generate response from LLM
    answer = await generate_response(
        llm_service, query, concatenated_chunks, rules, format
    )
    answer_value = answer["answer"]

    transformations: Dict[str, Union[str, List[str]]] = {
        "original": "",
        "resolved": "",
    }

    result_chunks = []

    if format in ["str", "str_array"]:
        # Extract and apply keyword replacements from all resolve_entity rules
        resolve_entity_rules = [
            rule for rule in rules if rule.type == "resolve_entity"
        ]

        result_chunks = (
            []
            if answer_value in ("not found", None)
            and query_type != "decomposition"
            else chunks
        )

        # First populate the replacements dictionary
        replacements: Dict[str, str] = {}
        if resolve_entity_rules and answer_value:
            for rule in resolve_entity_rules:
                if rule.options:
                    rule_replacements = dict(
                        option.split(":") for option in rule.options
                    )
                    replacements.update(rule_replacements)

            # Then apply the replacements if we have any
            if replacements:
                print(f"Resolving entities in answer: {answer_value}")
                if isinstance(answer_value, list):
                    transformed_list, transform_dict = replace_keywords(
                        answer_value, replacements
                    )
                    transformations = transform_dict
                    answer_value = transformed_list
                else:
                    transformed_value, transform_dict = replace_keywords(
                        answer_value, replacements
                    )
                    transformations = transform_dict
                    answer_value = transformed_value

    return QueryResult(
        answer=answer_value,
        chunks=result_chunks[:10],
        resolved_entities=(
            [
                ResolvedEntitySchema(
                    original=transformations["original"],
                    resolved=transformations["resolved"],
                    source={"type": "column", "id": "some-id"},
                    entityType="some-type",
                )
            ]
            if transformations["original"] or transformations["resolved"]
            else None
        ),
    )


async def process_queries_in_parallel(
    queries: List[Dict[str, Any]],
    llm_service: CompletionService,
    vector_db_service: Any,
) -> List[QueryResult]:
    """
    Process multiple queries in parallel with optimized concurrency control.
    
    Parameters
    ----------
    queries : List[Dict[str, Any]]
        List of query parameters, each containing:
        - query_type: QueryType
        - query: str
        - document_id: str
        - rules: List[Rule]
        - format: FormatType
    llm_service : CompletionService
        The language model service.
    vector_db_service : Any
        The vector database service.
        
    Returns
    -------
    List[QueryResult]
        List of query results in the same order as the input queries.
    """
    logger.info(f"Processing {len(queries)} queries in parallel")
    
    # Prioritize queries - put simpler queries first for faster initial results
    prioritized_queries = sorted(
        enumerate(queries), 
        key=lambda x: 0 if x[1]["query_type"] == "simple_vector" else 1
    )
    original_indices = [idx for idx, _ in prioritized_queries]
    sorted_queries = [q for _, q in prioritized_queries]
    
    # Create tasks with retry logic
    tasks = [
        process_query_with_retry(
            q["query_type"],
            q["query"],
            q["document_id"],
            q["rules"],
            q["format"],
            llm_service,
            vector_db_service,
        )
        for q in sorted_queries
    ]
    
    # Use a larger batch size for the first batch to get initial results faster
    first_batch_size = min(len(tasks), MAX_CONCURRENT_QUERIES * 2)
    remaining_tasks = tasks[first_batch_size:]
    
    # Process first batch with higher priority
    logger.info(f"Processing first batch with {first_batch_size} queries")
    first_batch_results = await asyncio.gather(*tasks[:first_batch_size], return_exceptions=True)
    
    # Process remaining batches
    remaining_results = []
    if remaining_tasks:
        # Process remaining tasks with standard concurrency
        logger.info(f"Processing remaining {len(remaining_tasks)} queries")
        remaining_results = await asyncio.gather(*remaining_tasks, return_exceptions=True)
    
    # Combine results
    batch_results = first_batch_results + remaining_results
    
    # Process results, converting exceptions to fallback values
    results = [None] * len(queries)  # Pre-allocate result list
    
    for i, result in enumerate(batch_results):
        original_idx = original_indices[i]
        q = queries[original_idx]
        
        if isinstance(result, Exception):
            logger.error(f"Query {original_idx} failed: {str(result)}")
            # Create fallback result based on format
            if q["format"] == "int":
                fallback_answer = 0
            elif q["format"] == "bool":
                fallback_answer = False
            elif q["format"].endswith("_array"):
                fallback_answer = []
            else:
                fallback_answer = ""
            
            results[original_idx] = QueryResult(
                answer=fallback_answer,
                chunks=[],
                resolved_entities=[]
            )
        else:
            results[original_idx] = result
    
    return results


# Convenience functions for specific query types
async def decomposition_query(
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the decomposition type."""
    return await process_query_with_retry(
        "decomposition",
        query,
        document_id,
        rules,
        format,
        llm_service,
        vector_db_service,
    )


async def hybrid_query(
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the hybrid type."""
    return await process_query_with_retry(
        "hybrid",
        query,
        document_id,
        rules,
        format,
        llm_service,
        vector_db_service,
    )


async def simple_vector_query(
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the simple vector type."""
    return await process_query_with_retry(
        "simple_vector",
        query,
        document_id,
        rules,
        format,
        llm_service,
        vector_db_service,
    )


async def inference_query(
    query: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
) -> QueryResult:
    """Generate a response with optimized processing, no need for vector retrieval."""
    # Truncate query for logging
    # query_preview = query[:30] + "..." if len(query) > 30 else query
    # logger.info(f"Processing inference query: {query_preview}")
    
    try:
        # Use the semaphore to limit concurrency
        async with QUERY_SEMAPHORE:
            start_time = time.time()
            
            # Generate response from LLM
            answer = await generate_inferred_response(
                llm_service, query, rules, format
            )
            answer_value = answer["answer"]
            
            # Fast path for simple types
            if format in ["int", "bool"] and not isinstance(answer_value, (list, dict)):
                return QueryResult(answer=answer_value, chunks=[])
            
            # Optimized array handling
            if format.endswith("_array"):
                # Check if this is a tag query - use empty arrays for errors
                is_tag_query = any(keyword in query.lower() for keyword in 
                                ["tag", "categor", "injur", "type", "list"])
                
                # Convert string to array if needed
                if isinstance(answer_value, str):
                    try:
                        # Try to parse as a Python list
                        import ast
                        cleaned_value = answer_value.strip()
                        if cleaned_value.startswith('[') and cleaned_value.endswith(']'):
                            try:
                                parsed_value = ast.literal_eval(cleaned_value)
                                if isinstance(parsed_value, list):
                                    answer_value = parsed_value
                            except (ValueError, SyntaxError):
                                # Fallback to simple parsing
                                items = cleaned_value[1:-1].split(',')
                                items = [item.strip().strip('\'"') for item in items if item.strip()]
                                
                                if format == 'int_array':
                                    try:
                                        answer_value = [int(item) for item in items]
                                    except ValueError:
                                        answer_value = [] if is_tag_query else [0]
                                else:
                                    answer_value = items
                    except Exception:
                        # If all parsing fails, use default
                        answer_value = [] if is_tag_query else ([0] if format == 'int_array' else [])
                
                # Ensure we have a list
                if not isinstance(answer_value, list):
                    answer_value = [] if is_tag_query else ([0] if format == 'int_array' else [])
            
            # Entity resolution if needed
            resolve_entity_rules = [rule for rule in rules if rule.type == "resolve_entity"]
            if resolve_entity_rules and answer_value:
                # Build replacements dictionary
                replacements = {}
                for rule in resolve_entity_rules:
                    if rule.options:
                        try:
                            rule_replacements = dict(option.split(":") for option in rule.options)
                            replacements.update(rule_replacements)
                        except ValueError:
                            continue
                
                # Apply replacements if any
                if replacements:
                    if isinstance(answer_value, list):
                        transformed_value, _ = replace_keywords(answer_value, replacements)
                        answer_value = transformed_value
                    else:
                        transformed_value, _ = replace_keywords_in_string(str(answer_value), replacements)
                        answer_value = transformed_value
            
            elapsed = time.time() - start_time
            logger.info(f"Inference query processed in {elapsed:.2f}s")
            return QueryResult(answer=answer_value, chunks=[])
        
    except Exception as e:
        logger.error(f"Error in inference query: {str(e)}")
        
        # Fast fallback creation based on format
        is_tag_query = format.endswith('_array') and any(
            keyword in query.lower() for keyword in ["tag", "categor", "injur", "type", "list"]
        )
        
        if format == 'int':
            fallback_value = 0
        elif format == 'bool':
            fallback_value = False
        elif format.endswith('_array'):
            fallback_value = []
        else:
            fallback_value = ""
            
        return QueryResult(answer=fallback_value, chunks=[])
