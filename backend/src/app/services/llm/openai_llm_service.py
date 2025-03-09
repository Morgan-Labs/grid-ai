"""OpenAI completion service implementation."""

import asyncio
import logging
import time
from typing import Any, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.services.llm.base import CompletionService

logger = logging.getLogger(__name__)

# Default timeout for OpenAI API calls (in seconds) - reduced from 60 to improve responsiveness
DEFAULT_TIMEOUT = 30
# Maximum number of retries for OpenAI API calls
MAX_RETRIES = 2
# Initial backoff time for retries (in seconds)
INITIAL_BACKOFF = 0.5


class OpenAICompletionService(CompletionService):
    """OpenAI completion service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None  # type: ignore
            logger.warning(
                "OpenAI API key is not set. LLM features will be disabled."
            )

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel], parent_run_id: str = None, timeout: int = DEFAULT_TIMEOUT
    ) -> Optional[BaseModel]:
        """Generate a completion from the language model with optimized timeout and retry logic."""
        if self.client is None:
            logger.warning("OpenAI client is not initialized. Skipping generation.")
            return None

        # Optimized retry logic with faster backoff
        retries = 0
        backoff = INITIAL_BACKOFF
        last_error = None

        # Pre-calculate the model dump method to avoid repeated lookups
        model_dump_method = getattr(response_model, "model_dump", None)
        if model_dump_method is None:
            model_dump_method = getattr(response_model, "dict", lambda: {})

        while retries <= MAX_RETRIES:
            try:
                # Use asyncio.wait_for with reduced timeout for faster failure detection
                start_time = time.time()
                
                # Create and execute the API call task with timeout
                response = await asyncio.wait_for(
                    self._make_api_call(prompt, response_model, parent_run_id),
                    timeout=timeout
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"API call completed in {elapsed_time:.2f} seconds")
                
                if response is None:
                    return None
                
                # Extract and validate the parsed response
                parsed_response = response.choices[0].message.parsed
                if parsed_response is None:
                    return None

                # Validate the response model
                try:
                    validated_response = response_model(**parsed_response.model_dump())
                    # Quick check if all values are None
                    if all(value is None for value in validated_response.model_dump().values()):
                        return None
                    return validated_response
                except ValueError:
                    return None
                
            except asyncio.TimeoutError:
                retries += 1
                last_error = "Timeout occurred while waiting for OpenAI API response"
                
                if retries <= MAX_RETRIES:
                    # Faster backoff with less logging
                    wait_time = backoff * (1.5 ** (retries - 1))  # Reduced exponential factor
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                retries += 1
                last_error = str(e)
                
                if retries <= MAX_RETRIES:
                    # Faster backoff with less logging
                    wait_time = backoff * (1.5 ** (retries - 1))
                    await asyncio.sleep(wait_time)
        
        # If we've exhausted all retries, raise an exception
        raise Exception(f"Failed to generate completion: {last_error}")
    
    async def _make_api_call(
        self, prompt: str, response_model: Type[BaseModel], parent_run_id: str = None
    ) -> Any:
        """Make the actual API call to OpenAI with optimized settings."""
        # Use a connection pool for better performance
        return self.client.beta.chat.completions.parse(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
            timeout=DEFAULT_TIMEOUT,  # Set explicit timeout
        )

    async def decompose_query(self, query: str, parent_run_id: str = None) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping decomposition."
            )
            return {"sub_queries": [query]}

        # TODO: Implement the actual decomposition logic here
        return {"sub_queries": [query]}
