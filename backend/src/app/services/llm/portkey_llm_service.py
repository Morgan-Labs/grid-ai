"""Portkey LLM service implementation for multiple providers using virtual keys."""

import asyncio
import logging
import json
from typing import Any, Dict, Optional, Type

import instructor
from pydantic import BaseModel
from portkey_ai import Portkey, PORTKEY_GATEWAY_URL, createHeaders
from openai import OpenAI

from app.core.config import Settings
from app.services.llm.base import CompletionService

logger = logging.getLogger(__name__)

# Default timeout for API calls (in seconds)
DEFAULT_TIMEOUT = 30
# Maximum number of retries for API calls
MAX_RETRIES = 2
# Initial backoff time for retries (in seconds)
INITIAL_BACKOFF = 0.5


class PortkeyLLMService(CompletionService):
    """Portkey LLM service implementation using virtual keys for multiple providers."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the Portkey LLM service."""
        self.settings = settings
        
        # Check if Portkey API key is set
        if not settings.portkey_api_key:
            logger.warning("Portkey API key is not set. LLM features will be disabled.")
            self.client = None
            return
        
        # Initialize default Portkey client
        self.client = Portkey(api_key=settings.portkey_api_key)
        logger.info(f"Portkey client initialized with model: {settings.llm_model}")
        
        # Store the current virtual key and provider for routing
        self.current_virtual_key = settings.llm_virtual_key
        
        # Use the settings provider if available, otherwise default to openai
        self.current_provider = settings.llm_provider if settings.llm_provider and settings.llm_provider != "portkey" else "openai"
        
        # Virtual keys for different providers
        self.provider_keys = {
            "openai": "openai-6a3e17",
            "anthropic": "anthropic-a27fda",
            "gemini": "gemini-3161fc"
        }

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel], parent_run_id: str = None, timeout: int = DEFAULT_TIMEOUT
    ) -> Optional[BaseModel]:
        """Generate a completion from the language model with optimized timeout and retry logic."""
        if self.client is None:
            logger.warning("Portkey client is not initialized. Skipping generation.")
            return None

        # Retry logic with backoff
        retries = 0
        backoff = INITIAL_BACKOFF
        last_error = None

        while retries <= MAX_RETRIES:
            try:
                # Create and execute the API call task with timeout
                response = await asyncio.wait_for(
                    self._make_api_call(prompt, response_model, parent_run_id),
                    timeout=timeout
                )
                
                if response is None:
                    return None
                
                # Handle different response types
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    # This is a structured response from the beta.parse method
                    if hasattr(response.choices[0].message, 'parsed'):
                        return response.choices[0].message.parsed
                
                # If we got a direct model instance
                if isinstance(response, response_model):
                    return response
                
                # If we got a string (from direct content extraction)
                if isinstance(response, str):
                    try:
                        # Create a model instance with the 'answer' field
                        return response_model(answer=response)
                    except Exception as e:
                        logger.warning(f"Could not create model instance from string: {str(e)}")
                        return None
                
                logger.warning(f"Unhandled response type: {type(response)}")
                return None
                
            except asyncio.TimeoutError:
                retries += 1
                last_error = "Timeout occurred while waiting for API response"
                
                if retries <= MAX_RETRIES:
                    wait_time = backoff * (1.5 ** (retries - 1))
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                retries += 1
                last_error = str(e)
                logger.error(f"Error in API call: {str(e)}")
                
                if retries <= MAX_RETRIES:
                    wait_time = backoff * (1.5 ** (retries - 1))
                    await asyncio.sleep(wait_time)
        
        # If we've exhausted all retries, raise an exception
        raise Exception(f"Failed to generate completion: {last_error}")
    
    async def _make_api_call(
        self, prompt: str, response_model: Type[BaseModel], parent_run_id: str = None
    ) -> Any:
        """Make the API call using the appropriate Portkey client with virtual key routing."""
        # Prepare the messages
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Detect provider from model name
            model = self.settings.llm_model.lower()
            provider = self.current_provider
            
            # Auto-detect provider based on model name
            if "claude" in model:
                provider = "anthropic"
            elif "gemini" in model:
                provider = "gemini"
            elif "gpt" in model:
                provider = "openai"
            
            # Get the appropriate virtual key for the provider
            virtual_key = self.provider_keys.get(provider, self.current_virtual_key)
            
            # Create a Portkey client with the virtual key
            portkey_client = Portkey(
                api_key=self.settings.portkey_api_key,
                virtual_key=virtual_key
            )
            
            # Update current provider and virtual key
            self.current_provider = provider
            self.current_virtual_key = virtual_key
            
            # Use instructor for structured responses
            if provider == "openai" or provider == "gemini":
                # Create an OpenAI client that points to Portkey's API
                openai_client = OpenAI(
                    api_key=self.settings.openai_api_key,
                    base_url="https://api.portkey.ai/v1",
                    default_headers={
                        "X-Portkey-Api-Key": self.settings.portkey_api_key,
                        "X-Portkey-Provider": provider,
                        "X-Portkey-Virtual-Key": virtual_key,
                        "X-Portkey-Metadata": json.dumps({'_user': 'grid', 'service': 'ai-grid'}),
                    }
                )
                
                # Use instructor for structured responses
                instructor_client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
                response = instructor_client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    response_model=response_model,
                    timeout=DEFAULT_TIMEOUT,
                )
                return response
            elif provider == "anthropic":
                from anthropic import Anthropic
                import os
                # Use the API key from settings if available, otherwise fall back to environment variable
                if self.settings.anthropic_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = self.settings.anthropic_api_key
                client = Anthropic()
                instructor_client = instructor.from_anthropic(client, mode=instructor.Mode.ANTHROPIC_TOOLS)
                # Use Portkey's beta chat completions API
                response = instructor_client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    response_model=response_model,
                    timeout=DEFAULT_TIMEOUT,
                    max_tokens=8000,
                ) 
                return response      
            else: # other providers
                # Use Portkey's beta chat completions API
                response = portkey_client.beta.chat.completions.parse(
                    model=self.settings.llm_model,
                    messages=messages,
                    response_format=response_model,
                    timeout=DEFAULT_TIMEOUT,
                )
                return response
                
        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
            raise

    def update_virtual_key(self, virtual_key: str, provider: str, model: str = None) -> tuple:
        """Update the virtual key and provider for the next API call."""
        # Validate provider
        if not provider or provider.lower().strip() not in ["openai", "anthropic", "gemini"]:
            provider = "openai"
        
        # Set the provider and virtual key
        self.current_provider = provider.lower().strip()
        
        # If virtual key is empty but provider is specified, use the default virtual key for that provider
        if (not virtual_key or virtual_key.strip() == "") and self.current_provider in self.provider_keys:
            self.current_virtual_key = self.provider_keys[self.current_provider]
        else:
            self.current_virtual_key = virtual_key
        
        # Update model if provided
        if model:
            self.settings.llm_model = model
        
        # Return values for verification
        return self.current_provider, self.current_virtual_key

    async def decompose_query(self, query: str, parent_run_id: str = None) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning("Portkey client is not initialized. Skipping decomposition.")
            return {"sub_queries": [query]}

        # TODO: Implement the actual decomposition logic here
        return {"sub_queries": [query]}
