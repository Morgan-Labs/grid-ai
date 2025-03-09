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
        """Get embeddings for text."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping embeddings."
            )
            return []

        return [
            embedding.embedding
            for embedding in self.client.embeddings.create(
                input=texts, model=self.model
            ).data
        ]
