"""The base class for the vector database services."""

import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from langchain.schema import Document
from pydantic import BaseModel, Field

from app.models.query_core import Rule
from app.schemas.query_api import VectorResponseSchema
from app.services.embedding.base import EmbeddingService
from app.services.embedding.cache import EmbeddingCache
from app.services.llm.base import CompletionService
from app.services.llm_service import get_keywords

logger = logging.getLogger(__name__)


class Metadata(BaseModel, extra="forbid"):
    """Metadata stored in vector storage."""

    text: str
    page_number: int
    chunk_number: int
    document_id: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class VectorDBService(ABC):
    """The base class for the vector database services."""

    embedding_service: EmbeddingService
    
    # Shared embedding cache for all vector DB service instances
    # Will be initialized in __init__ with settings
    embedding_cache = None
    
    def __init__(self, embedding_service: EmbeddingService, settings):
        """Initialize the vector DB service with embedding service and settings."""
        self.embedding_service = embedding_service
        
        # Initialize the embedding cache if not already initialized
        if VectorDBService.embedding_cache is None:
            VectorDBService.embedding_cache = EmbeddingCache(max_size=settings.embedding_cache_size)
            logger.info(f"Initialized embedding cache with max size {settings.embedding_cache_size}")

    @abstractmethod
    async def upsert_vectors(
        self, vectors: List[Dict[str, Any]], parent_run_id: str = None
    ) -> Dict[str, str]:
        """Upsert the vectors into the vector database."""
        pass

    @abstractmethod
    async def vector_search(
        self, queries: List[str], document_id: str, parent_run_id: str = None
    ) -> VectorResponseSchema:
        """Perform a vector search."""
        pass

    # Update other methods if they also return VectorResponse
    @abstractmethod
    async def keyword_search(
        self, query: str, document_id: str, keywords: List[str], parent_run_id: str = None
    ) -> VectorResponseSchema:
        """Perform a keyword search."""
        pass

    @abstractmethod
    async def hybrid_search(
        self, query: str, document_id: str, rules: List[Rule], parent_run_id: str = None
    ) -> VectorResponseSchema:
        """Perform a hybrid search."""
        pass

    @abstractmethod
    async def decomposed_search(
        self, query: str, document_id: str, rules: List[Rule], parent_run_id: str = None
    ) -> Dict[str, Any]:
        """Decomposition query."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str, parent_run_id: str = None) -> Dict[str, str]:
        """Delete the document from the vector database."""
        pass

    @abstractmethod
    async def ensure_collection_exists(self) -> None:
        """Ensure the collection exists in the vector database."""
        pass

    async def get_embeddings(
        self, texts: Union[str, List[str]], parent_run_id: str = None
    ) -> List[List[float]]:
        """Get embeddings for the given text(s) using the embedding service with caching."""
        if isinstance(texts, str):
            texts = [texts]
            
        # Check cache first
        result = []
        texts_to_embed = []
        cache_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get(text)
            if cached_embedding:
                result.append(cached_embedding)
            else:
                texts_to_embed.append(text)
                cache_indices.append(i)
        
        # If all embeddings were cached, return immediately
        if not texts_to_embed:
            logger.info(f"All {len(texts)} embeddings found in cache")
            return result
        
        # Get embeddings for texts not in cache
        cache_hit_rate = (len(texts) - len(texts_to_embed)) / len(texts)
        logger.info(f"Getting embeddings for {len(texts_to_embed)} texts (cache hit rate: {cache_hit_rate:.2%})")
        new_embeddings = await self.embedding_service.get_embeddings(texts_to_embed, parent_run_id)
        
        # Update cache with new embeddings
        for i, embedding in enumerate(new_embeddings):
            self.embedding_cache.set(texts_to_embed[i], embedding)
        
        # Merge cached and new embeddings in the correct order
        final_result = [None] * len(texts)
        
        # Place cached embeddings
        cached_count = 0
        for i in range(len(texts)):
            if i not in cache_indices:
                final_result[i] = result[cached_count]
                cached_count += 1
                
        # Place new embeddings
        for i, embedding in zip(cache_indices, new_embeddings):
            final_result[i] = embedding
            
        return final_result

    async def get_single_embedding(self, text: str, parent_run_id: str = None) -> List[float]:
        """Get a single embedding for the given text."""
        embeddings = await self.get_embeddings(text, parent_run_id)
        return embeddings[0]

    async def prepare_chunks(
        self, document_id: str, chunks: List[Document], parent_run_id: str = None
    ) -> List[Dict[str, Any]]:
        """Prepare chunks for insertion into the vector database."""
        logger.info(f"Preparing {len(chunks)} chunks")

        # Clean the chunks
        cleaned_texts = [
            re.sub(r"\s+", " ", chunk.page_content.strip()) for chunk in chunks
        ]

        logger.info("Generating embeddings.")

        # Embed all chunks at once
        embedded_chunks = await self.get_embeddings(cleaned_texts, parent_run_id)

        # Prepare the data for insertion
        return [
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "text": text,
                "page_number": chunk.metadata.get("page", i // 5 + 1),
                "chunk_number": i,
                "document_id": document_id,
            }
            for i, (chunk, text, embedding) in enumerate(
                zip(chunks, cleaned_texts, embedded_chunks)
            )
        ]
    
    @abstractmethod
    async def get_document_chunks(self, document_id: str, parent_run_id: str = None) -> List[Dict[str, Any]]:
        """Get all chunks for a document from the vector database.
        
        Parameters
        ----------
        document_id : str
            The ID of the document to retrieve chunks for.
            
        Returns
        -------
        List[Dict[str, Any]]
            A list of document chunks, each containing text and metadata.
        """
        pass
        
    async def extract_keywords(
        self, query: str, rules: list[Rule], llm_service: CompletionService
    ) -> list[str]:
        """Extract keywords from a user query."""
        keywords = []
        if rules:
            for rule in rules:
                if rule.type in ["must_return", "may_return"]:
                    if rule.options:
                        if isinstance(rule.options, list):
                            keywords.extend(rule.options)
                        elif isinstance(rule.options, dict):
                            for value in rule.options.values():
                                if isinstance(value, list):
                                    keywords.extend(value)
                                elif isinstance(value, str):
                                    keywords.append(value)

        if not keywords:
            extracted_keywords = await get_keywords(llm_service, query)
            if extracted_keywords and isinstance(extracted_keywords, list):
                keywords = extracted_keywords

        return keywords
