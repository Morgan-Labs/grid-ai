"""Optimized OpenAI embedding service implementation with parallel processing."""

import asyncio
import logging
from typing import List

from openai import AsyncOpenAI

from app.core.config import Settings
from app.services.embedding.base import EmbeddingService

logger = logging.getLogger(__name__)


class OptimizedOpenAIEmbeddingService(EmbeddingService):
    """Optimized OpenAI embedding service with parallel processing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.embedding_model
        self.max_parallel_requests = settings.embedding_max_parallel  # Use setting from config
        
        if not settings.openai_api_key:
            logger.error("OpenAI API key is required but not set")
            raise ValueError("OpenAI API key is required but not set")
        
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info(f"Initialized optimized OpenAI embedding service with model {self.model} and max parallel requests {self.max_parallel_requests}")

    async def get_embeddings(self, texts: List[str], parent_run_id: str = None) -> List[List[float]]:
        """Get embeddings for text with parallel batching to handle token limits."""
        if not texts:
            return []

        # Optimize batch size based on number of chunks
        if len(texts) > 5000:
            batch_size = 250
        elif len(texts) > 1000:
            batch_size = 150
        else:
            batch_size = 100

        logger.info(f"Processing {len(texts)} text chunks in parallel batches of {batch_size}")

        all_embeddings = [None] * len(texts)  # Pre-allocate result list
        semaphore = asyncio.Semaphore(self.max_parallel_requests)
        total_batches = (len(texts) + batch_size - 1) // batch_size

        async def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_num = batch_idx + 1
            
            async with semaphore:  # Limit concurrent API calls
                try:
                    # Create embeddings for this batch
                    response = await self.client.embeddings.create(
                        input=batch_texts, model=self.model
                    )
                    
                    # Extract the embeddings and place in correct positions
                    for i, embedding_data in enumerate(response.data):
                        all_embeddings[start_idx + i] = embedding_data.embedding
                    
                    logger.info(f"Successfully processed embedding batch {batch_num}/{total_batches}")
                    
                except Exception as e:
                    logger.error(f"Error processing embedding batch {batch_num}: {str(e)}")
                    # For failed batches, insert placeholder embeddings
                    dim = 1536  # Default dimension for text-embedding-3-small
                    
                    # Try to get dimension from successful embeddings
                    for emb in all_embeddings:
                        if emb is not None:
                            dim = len(emb)
                            break
                    
                    # Fill placeholders for this batch
                    for i in range(len(batch_texts)):
                        all_embeddings[start_idx + i] = [0.0] * dim
                    
                    logger.warning(f"Added placeholder embeddings for batch {batch_num} due to error")

        # Create tasks for all batches
        tasks = [process_batch(i) for i in range(total_batches)]
        
        # Run all batch processing tasks concurrently
        await asyncio.gather(*tasks)
        
        logger.info(f"Completed processing {len(all_embeddings)} embeddings in {total_batches} batches")
        
        return all_embeddings