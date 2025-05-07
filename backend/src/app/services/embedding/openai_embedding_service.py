"""OpenAI embedding service implementation."""

import logging
from typing import List

from openai import OpenAI

from app.core.config import Settings
from app.services.embedding.base import EmbeddingService

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.embedding_model
        
        # Check if API key is set before initializing the client
        if not settings.openai_api_key:
            logger.error("OpenAI API key is required but not set")
            raise ValueError("OpenAI API key is required but not set")
        
        # Initialize the client after checking the API key
        self.client = OpenAI(api_key=settings.openai_api_key)

    async def get_embeddings(self, texts: List[str], parent_run_id: str = None) -> List[List[float]]:
        """Get embeddings for text with batching to handle token limits."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping embeddings."
            )
            return []

        if not texts:
            return []

        # Maximum number of texts to send in one batch
        # Assuming max ~2000 tokens per text chunk for safety
        # OpenAI has a 300K token limit, so 100 chunks should be safe
        batch_size = 50

        # Log summary info only
        logger.info(f"Processing {len(texts)} text chunks in batches of {batch_size}")

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Create embeddings for this batch
                batch_embeddings = self.client.embeddings.create(
                    input=batch_texts, model=self.model
                ).data
                
                # Extract the embeddings
                for embedding in batch_embeddings:
                    all_embeddings.append(embedding.embedding)
                
                logger.info(f"Successfully processed embedding batch {batch_num}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error processing embedding batch {batch_num}: {str(e)}")
                # For failed batches, insert placeholder embeddings of the same dimension
                # This prevents misalignment between texts and embeddings
                if len(all_embeddings) > 0:
                    # Use the dimension of the first successful embedding
                    dim = len(all_embeddings[0])
                    for _ in range(len(batch_texts)):
                        all_embeddings.append([0.0] * dim)
                else:
                    # If no successful embeddings yet, use a standard dimension for text-embedding-3-small
                    for _ in range(len(batch_texts)):
                        all_embeddings.append([0.0] * 1536)
                
                logger.warning(f"Added placeholder embeddings for batch {batch_num} due to error")

        logger.info(f"Completed processing {len(all_embeddings)} embeddings in {total_batches} batches")
        
        return all_embeddings
