"""Vector index implementation using Qdrant."""

# mypy: disable-error-code="index"

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.models.query_core import Chunk, Rule
from app.schemas.query_api import VectorResponseSchema
from app.services.embedding.base import EmbeddingService
from app.services.llm_service import CompletionService
from app.services.vector_db.base import VectorDBService

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantMetadata(BaseModel, extra="forbid"):
    """Metadata for Qdrant documents."""

    text: str
    page_number: int
    chunk_number: int
    document_id: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QdrantService(VectorDBService):
    """Vector service implementation using Qdrant."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: CompletionService,
        settings: Settings,
    ):
        self.settings = settings
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.collection_name = settings.index_name
        self.dimensions = settings.dimensions
        qdrant_config = settings.qdrant.model_dump(exclude_none=True)
        self.client = QdrantClient(**qdrant_config)

    # Collection existence cache
    _collection_exists = False
    
    async def upsert_vectors(
        self, vectors: List[Dict[str, Any]], parent_run_id: str = None
    ) -> Dict[str, str]:
        """Add vectors to a Qdrant collection with optimized batching and caching."""
        logger.info(f"Upserting {len(vectors)} chunks")
        
        # Only check collection existence once per application lifecycle
        if not QdrantService._collection_exists:
            await self.ensure_collection_exists()
            QdrantService._collection_exists = True
        
        # Convert vectors to points
        points = [
            models.PointStruct(
                id=entry.pop("id"), vector=entry.pop("vector"), payload=entry
            )
            for entry in vectors
        ]
        
        # Optimize batch size based on vector count
        # Smaller batches for larger vectors to prevent timeouts
        if len(vectors) > 100:
            batch_size = 30
        elif len(vectors) > 50:
            batch_size = 40
        else:
            batch_size = 50
            
        # Split into batches
        batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]
        
        logger.info(f"Split {len(points)} points into {len(batches)} batches with size {batch_size}")
        
        success_count = 0
        error_count = 0
        
        # Process each batch with exponential backoff retry
        for i, batch in enumerate(batches):
            max_retries = 3
            retry_delay = 1.0  # Start with 1 second delay
            
            for retry in range(max_retries):
                try:
                    logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} points (attempt {retry+1})")
                    self.client.upsert(self.collection_name, points=batch, wait=True)
                    success_count += len(batch)
                    logger.info(f"Successfully processed batch {i+1}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry == max_retries - 1:  # Last retry attempt
                        error_count += len(batch)
                        logger.error(f"Error processing batch {i+1} after {max_retries} attempts: {str(e)}")
                        
                        # Try with even smaller batches as a last resort
                        if len(batch) > 10:
                            logger.info(f"Attempting final recovery with smaller batches for batch {i+1}")
                            smaller_batch_size = 5
                            smaller_batches = [batch[j:j + smaller_batch_size] for j in range(0, len(batch), smaller_batch_size)]
                            
                            for k, small_batch in enumerate(smaller_batches):
                                try:
                                    self.client.upsert(self.collection_name, points=small_batch, wait=True)
                                    success_count += len(small_batch)
                                    error_count -= len(small_batch)
                                    logger.info(f"Successfully processed micro-batch {k+1}/{len(smaller_batches)}")
                                except Exception as e2:
                                    logger.error(f"Error processing micro-batch {k+1}: {str(e2)}")
                    else:
                        # Not the last retry, wait and try again
                        logger.warning(f"Batch {i+1} failed, retrying in {retry_delay}s: {str(e)}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
        
        if error_count > 0:
            return {
                "message": f"Partially upserted vectors. Success: {success_count}, Failed: {error_count}",
                "warning": "Some vectors could not be uploaded due to timeout or other errors."
            }
        
        return {"message": f"Successfully upserted {success_count} chunks."}

    async def vector_search(
        self, queries: List[str], document_id: str, parent_run_id: str = None
    ) -> VectorResponseSchema:
        """Perform a vector search on the Qdrant collection."""
        logger.info(f"Retrieving vectors for {len(queries)} queries.")

        final_chunks: List[Dict[str, Any]] = []

        for query in queries:
            logger.info("Generating embedding.")
            embedded_query = await self.get_single_embedding(query, parent_run_id)
            logger.info("Searching...")

            query_response = self.client.query_points(
                self.collection_name,
                query=embedded_query,
                limit=40,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
            ).points

            final_chunks.extend(
                [point.payload for point in query_response if point.payload]
            )

        seen_chunks, formatted_output = set(), []

        for chunk in final_chunks:
            if chunk["chunk_number"] not in seen_chunks:
                seen_chunks.add(chunk["chunk_number"])
                formatted_output.append(
                    {"content": chunk["text"], "page": chunk["page_number"]}
                )

        logger.info(f"Retrieved {len(formatted_output)} unique chunks.")
        return VectorResponseSchema(
            message="Query processed successfully.",
            chunks=[Chunk(**chunk) for chunk in formatted_output],
        )

    async def hybrid_search(
        self,
        query: str,
        document_id: str,
        rules: list[Rule],
        parent_run_id: str = None
    ) -> VectorResponseSchema:
        """Perform a hybrid search on the Qdrant collection."""
        logger.info("Performing hybrid search.")

        sorted_keyword_chunks = []
        keywords = await self.extract_keywords(query, rules, self.llm_service)

        if keywords:
            like_conditions: Sequence[models.FieldCondition] = [
                models.FieldCondition(
                    key="text", match=models.MatchText(text=keyword)
                )
                for keyword in keywords
            ]
            _filter = models.Filter(
                must=models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                should=like_conditions,  # type: ignore
            )

            logger.info("Running query with keyword filters.")
            keyword_response = self.client.query_points(
                collection_name=self.collection_name,
                query_filter=_filter,
                with_payload=True,
            ).points
            keyword_response = [
                point.payload for point in keyword_response if point.payload  # type: ignore
            ]

            def count_keywords(text: str, keywords: List[str]) -> int:
                return sum(
                    text.lower().count(keyword.lower()) for keyword in keywords
                )

            sorted_keyword_chunks = sorted(
                keyword_response,
                key=lambda chunk: count_keywords(
                    chunk["text"], keywords or []
                ),
                reverse=True,
            )

        embedded_query = await self.get_single_embedding(query, parent_run_id)
        logger.info("Running semantic similarity search.")

        semantic_response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedded_query,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=40,
            with_payload=True,
        ).points

        semantic_response = [
            point.payload for point in semantic_response if point.payload  # type: ignore
        ]

        print(f"Found {len(semantic_response)} semantic chunks.")

        # Combine the top results from keyword and semantic searches
        combined_chunks = sorted_keyword_chunks[:20] + semantic_response

        # Sort the combined results by chunk number
        combined_sorted_chunks = sorted(
            combined_chunks, key=lambda chunk: chunk["chunk_number"]
        )

        # Eliminate duplicate chunks
        seen_chunks = set()
        formatted_output = []

        for chunk in combined_sorted_chunks:
            if chunk["chunk_number"] not in seen_chunks:
                formatted_output.append(
                    {"content": chunk["text"], "page": chunk["page_number"]}
                )
                seen_chunks.add(chunk["chunk_number"])

        logger.info(f"Retrieved {len(formatted_output)} unique chunks.")

        return VectorResponseSchema(
            message="Query processed successfully.",
            chunks=[Chunk(**chunk) for chunk in formatted_output],
        )

    # Decomposition query
    async def decomposed_search(
        self,
        query: str,
        document_id: str,
        rules: List[Rule],
        parent_run_id: str = None
    ) -> Dict[str, Any]:
        """Perform a decomposed search on a Qdrant collection."""
        logger.info("Decomposing query into smaller sub-queries.")
        decomposition_response = await self.llm_service.decompose_query(query, parent_run_id)
        sub_query_chunks = await self.vector_search(
            decomposition_response["sub-queries"], document_id, parent_run_id
        )
        return {
            "sub_queries": decomposition_response["sub-queries"],
            "chunks": sub_query_chunks.chunks,
        }

    async def keyword_search(
        self, query: str, document_id: str, keywords: List[str], parent_run_id: str = None
    ) -> VectorResponseSchema:
        """Perform a keyword search."""
        # Not being used currently
        raise NotImplementedError("Keyword search is not implemented yet.")

    async def ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE
                ),
            )

    async def get_document_chunks(self, document_id: str, parent_run_id: str = None) -> List[Dict[str, Any]]:
        """Get all chunks for a document from the Qdrant database.
        
        Parameters
        ----------
        document_id : str
            The ID of the document to retrieve chunks for.
            
        Returns
        -------
        List[Dict[str, Any]]
            A list of document chunks, each containing text and metadata.
        """
        try:
            logger.info(f"Retrieving chunks for document_id: {document_id} from Qdrant")
            
            # Query the collection for all chunks with this document_id
            scroll_response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1000,  # Set a high limit to get all chunks
                with_payload=True,
            )
            
            # The scroll method returns a tuple (records, next_page_offset)
            # Extract the records from the tuple
            records = scroll_response[0] if isinstance(scroll_response, tuple) else scroll_response
            
            if not records:
                logger.warning(f"No chunks found for document_id: {document_id}")
                return []
            
            # Convert the records to dictionaries
            chunks = []
            for record in records:
                if hasattr(record, 'payload') and record.payload:
                    # Create a dictionary with the payload data
                    chunk_dict = record.payload.copy()
                    chunks.append(chunk_dict)
            
            logger.info(f"Retrieved {len(chunks)} chunks from Qdrant")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving document chunks from Qdrant: {e}", exc_info=True)
            return []

    async def delete_document(self, document_id: str, parent_run_id: str = None) -> Dict[str, str]:
        """Delete a document from a Qdrant collection."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )
        return {
            "status": "success",
            "message": "Document deleted successfully.",
        }
