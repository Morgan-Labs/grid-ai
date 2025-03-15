"""Factory for creating language model completion services."""

import logging
from typing import Optional

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.services.llm.openai_llm_service import OpenAICompletionService
from app.services.llm.portkey_llm_service import PortkeyLLMService

logger = logging.getLogger(__name__)


class CompletionServiceFactory:
    """Factory for creating completion services."""

    @staticmethod
    def create_service(settings: Settings) -> Optional[CompletionService]:
        """Create a completion service."""
        logger.info(
            f"Creating completion service for provider: {settings.llm_provider}"
        )
        
        if settings.llm_provider == "portkey":
            logger.info(f"Using Portkey LLM service with model: {settings.llm_model}")
            return PortkeyLLMService(settings)
        elif settings.llm_provider == "openai":
            logger.info(f"Using OpenAI LLM service with model: {settings.llm_model}")
            return OpenAICompletionService(settings)
        
        return None
