"""Factory for creating embedding services."""

import logging
from typing import Optional

from app.core.config import Settings
from app.services.embedding.base import EmbeddingService
from app.services.embedding.openai_embedding_service import (
    OpenAIEmbeddingService,
)
from app.services.embedding.optimized_openai_embedding_service import (
    OptimizedOpenAIEmbeddingService,
)

logger = logging.getLogger(__name__)


class EmbeddingServiceFactory:
    """Factory for creating embedding services."""

    @staticmethod
    def create_service(settings: Settings) -> Optional[EmbeddingService]:
        """Create an embedding service."""
        logger.info(
            f"Creating embedding service for provider: {settings.embedding_provider}"
        )
        
        # Check if optimized mode is enabled
        use_optimized = getattr(settings, "use_optimized_embedding", True)
        
        if settings.embedding_provider == "openai":
            if use_optimized:
                logger.info("Using optimized OpenAI embedding service with parallel processing")
                return OptimizedOpenAIEmbeddingService(settings)
            else:
                logger.info("Using standard OpenAI embedding service")
                return OpenAIEmbeddingService(settings)
        # Add more providers here when needed
        return None
