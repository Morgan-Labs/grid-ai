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
MAX_CONCURRENT_QUERIES = 5
QUERY_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

# Retry configuration - reduced delay for faster failure recovery
MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds

# Document readiness check configuration
MAX_READINESS_CHECKS = 3
READINESS_CHECK_INTERVAL = 2.0  # seconds

# Query queue for documents that are still processing
DOCUMENT_QUERY_QUEUE = {}  # document_id -> List[QueuedQuery]

# Pending queries by document ID
class QueuedQuery:
    """A class to represent a query that's waiting for a document to be ready"""
    
    def __init__(self, query_type: QueryType, query: str, document_id: str, 
                 rules: List[Rule], format: FormatType, retries_left: int = MAX_READINESS_CHECKS):
        self.query_type = query_type
        self.query = query
        self.document_id = document_id
        self.rules = rules
        self.format = format
        self.retries_left = retries_left
        self.future = asyncio.get_event_loop().create_future()
        self.created_at = time.time()
    
    async def execute(self, llm_service: CompletionService, vector_db_service: Any) -> QueryResult:
        """Execute the query and resolve its future"""
        try:
            result = await process_query_with_retry(
                self.query_type, self.query, self.document_id, 
                self.rules, self.format, llm_service, vector_db_service
            )
            self.future.set_result(result)
            return result
        except Exception as e:
            self.future.set_exception(e)
            raise

async def check_document_readiness(document_id: str, document_service) -> bool:
    """
    Check if a document is ready for querying.
    
    Parameters
    ----------
    document_id : str
        The ID of the document to check
    document_service : DocumentService
        The document service for checking status
        
    Returns
    -------
    bool
        True if the document is ready, False otherwise
    """
    try:
        status = await document_service.get_document_status(document_id)
        logger.info(f"Document {document_id} status: {status}")
        return status == "completed"
    except Exception as e:
        logger.error(f"Error checking document status: {e}")
        # If we can't check status, assume it's not ready
        return False

async def queue_query_for_document(
    query_type: QueryType,
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    document_service,
) -> QueuedQuery:
    """
    Queue a query for a document that might not be ready yet.
    
    Parameters
    ----------
    query_type : QueryType
        The type of query to perform
    query : str
        The query string
    document_id : str
        The document ID to query against
    rules : List[Rule]
        Rules to apply to the query
    format : FormatType
        The format to return the result in
    document_service : DocumentService
        Service for checking document readiness
        
    Returns
    -------
    QueuedQuery
        A queued query object with a future that will be resolved when the query completes
    """
    # Initialize document queue if it doesn't exist
    if document_id not in DOCUMENT_QUERY_QUEUE:
        DOCUMENT_QUERY_QUEUE[document_id] = []
    
    # Create a new queued query and add it to the queue
    queued_query = QueuedQuery(query_type, query, document_id, rules, format)
    DOCUMENT_QUERY_QUEUE[document_id].append(queued_query)
    
    # Log the queuing
    logger.info(f"Queued query for document {document_id}: {query[:30]}...")
    
    return queued_query


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
    document_service = None,
) -> QueryResult:
    """Process a query with optimized retry logic for faster response times."""
    last_exception = None
    
    # Truncate query for logging to avoid excessive log size
    query_preview = query[:30] + "..." if len(query) > 30 else query
    
    # Check document readiness if document_service is provided
    if document_service:
        is_ready = await check_document_readiness(document_id, document_service)
        if not is_ready:
            logger.warning(f"Document {document_id} is not ready for query: {query_preview}")
            
            # Create a queued query that will be processed when document is ready
            queued_query = await queue_query_for_document(
                query_type, query, document_id, rules, format, document_service
            )
            
            # Return a placeholder result indicating the query is queued
            # The caller should check the future for the actual result
            return QueryResult(
                answer="Document is still processing, query has been queued",
                chunks=[],
                resolved_entities=[],
                queued=True,
                queued_query=queued_query
            )
    
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
    # Handle inference queries directly with the LLM
    if query_type == "inference":
        try:
            from app.services.llm_service import generate_inferred_response
            # Generate response directly from LLM without document retrieval
            answer = await generate_inferred_response(llm_service, query, rules, format)
            answer_value = answer["answer"]
            
            # Create a basic result with just the answer
            return QueryResult(
                answer=answer_value,
                chunks=[],
                resolved_entities=[]
            )
        except Exception as e:
            logger.error(f"Error in inference query: {e}")
            # Return a fallback result
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
    
    # For non-inference queries, continue with normal document-based processing
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

    # Construct the QueryResult with resolved entities
    result = QueryResult(
        answer=answer_value,
        chunks=result_chunks if format in ["str", "str_array"] else [],
        resolved_entities=[
                ResolvedEntitySchema(
                    original=transformations["original"],
                    resolved=transformations["resolved"],
            )
        ]
        if transformations["original"] != transformations["resolved"]
        else [],
    )

    return result

# Add a background task to process query queue
async def process_document_query_queue(document_id: str, document_service, llm_service: CompletionService, vector_db_service: Any):
    """
    Process all queued queries for a document once it's ready.
    
    Parameters
    ----------
    document_id : str
        The document ID to process queued queries for
    document_service : DocumentService
        The document service for checking readiness
    llm_service : CompletionService
        The LLM service for processing queries
    vector_db_service : Any
        The vector DB service for retrieving document chunks
    """
    if document_id not in DOCUMENT_QUERY_QUEUE or not DOCUMENT_QUERY_QUEUE[document_id]:
        logger.info(f"No queries queued for document {document_id}")
        return
    
    # Check if document is ready
    is_ready = await check_document_readiness(document_id, document_service)
    if not is_ready:
        logger.info(f"Document {document_id} still not ready, will retry queued queries later")
        return
    
    logger.info(f"Processing {len(DOCUMENT_QUERY_QUEUE[document_id])} queued queries for document {document_id}")
    
    # Process all queued queries
    queued_queries = DOCUMENT_QUERY_QUEUE[document_id].copy()
    DOCUMENT_QUERY_QUEUE[document_id] = []  # Clear the queue to avoid reprocessing
    
    for queued_query in queued_queries:
        try:
            result = await process_query_with_retry(
                queued_query.query_type,
                queued_query.query,
                queued_query.document_id,
                queued_query.rules,
                queued_query.format,
                llm_service,
                vector_db_service
            )
            
            # Set the result in the future
            if not queued_query.future.done():
                queued_query.future.set_result(result)
                
            logger.info(f"Processed queued query for document {document_id}: {queued_query.query[:30]}...")
            
        except Exception as e:
            logger.error(f"Error processing queued query: {e}")
            # Set exception in the future if it's not already done
            if not queued_query.future.done():
                queued_query.future.set_exception(e)

# Add a function to schedule periodic processing of queued queries
async def schedule_query_queue_processing(document_service, llm_service: CompletionService, vector_db_service: Any):
    """
    Periodically process queued queries for documents that might have finished processing.
    
    Parameters
    ----------
    document_service : DocumentService
        The document service for checking readiness
    llm_service : CompletionService
        The LLM service for processing queries
    vector_db_service : Any
        The vector DB service for retrieving document chunks
    """
    while True:
        try:
            # Process queues for all documents
            documents_with_queues = list(DOCUMENT_QUERY_QUEUE.keys())
            
            for document_id in documents_with_queues:
                if DOCUMENT_QUERY_QUEUE[document_id]:  # Only process if there are queued queries
                    await process_document_query_queue(document_id, document_service, llm_service, vector_db_service)
            
            # Sleep before next check
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in query queue processing: {e}")
            await asyncio.sleep(10)  # Sleep longer on error
