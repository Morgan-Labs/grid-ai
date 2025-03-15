"""Portkey LLM service implementation for multiple providers using virtual keys."""

import asyncio
import json
import logging
import time
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
        
        # Initialize default Portkey client without a virtual key
        self.client = Portkey(
            api_key=settings.portkey_api_key
        )
        
        logger.info(f"Portkey client initialized with model: {settings.llm_model}")
        
        # Store the current virtual key and provider for routing
        # If llm_provider and llm_virtual_key are provided in settings, use those as defaults
        self.current_virtual_key = settings.llm_virtual_key
        
        # Use the settings provider if available, otherwise default to openai
        self.current_provider = settings.llm_provider if settings.llm_provider and settings.llm_provider != "portkey" else "openai"
        
        # Log initial provider and virtual key state
        logger.info(f"Initial provider set to: {self.current_provider}")
        logger.info(f"Initial virtual key set to: {self.current_virtual_key}")
        
        # Create client mapping for different providers using virtual keys
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

        # Super detailed logging of current state
        logger.info(f"🔍🔍 CURRENT STATE:")
        logger.info(f"🔍🔍 Provider: '{self.current_provider}'")
        logger.info(f"🔍🔍 Virtual key: '{self.current_virtual_key}'")
        logger.info(f"🔍🔍 Model: '{self.settings.llm_model}'")

        # Optimized retry logic with faster backoff
        retries = 0
        backoff = INITIAL_BACKOFF
        last_error = None

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
                
                # Handle different response types
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    # This is a structured response from the beta.parse method
                    if hasattr(response.choices[0].message, 'parsed'):
                        return response.choices[0].message.parsed
                
                # If we got a direct model instance (from our custom handling)
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
                
                # If we got something else, log it and return None
                logger.warning(f"Unhandled response type: {type(response)}")
                return None
                
            except asyncio.TimeoutError:
                retries += 1
                last_error = "Timeout occurred while waiting for API response"
                
                if retries <= MAX_RETRIES:
                    # Faster backoff with less logging
                    wait_time = backoff * (1.5 ** (retries - 1))  # Reduced exponential factor
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                retries += 1
                last_error = str(e)
                logger.error(f"Error in API call: {str(e)}")
                
                if retries <= MAX_RETRIES:
                    # Faster backoff with less logging
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
        
        # Enhanced logging to show all relevant context
        logger.info(f"🚀 PORTKEY API CALL - Provider: '{self.current_provider}', Model: '{self.settings.llm_model}', Virtual Key: '{self.current_virtual_key}'")
        logger.info(f"🧠 Request Context - Prompt length: {len(prompt)} chars, Response model: {response_model.__name__}")
        
        try:
            # CRITICAL: Enforce model-provider consistency
            # If we detect a model from a specific provider, ensure we're using the right provider
            model = self.settings.llm_model.lower()
            forced_provider = None
            
            # Force provider based on model name patterns
            if "claude" in model:
                forced_provider = "anthropic"
                logger.info(f"🔒 Model '{model}' detected as Anthropic - forcing provider to 'anthropic'")
            elif "gemini" in model:
                forced_provider = "gemini"
                logger.info(f"🔒 Model '{model}' detected as Google - forcing provider to 'gemini'")
            elif "gpt" in model:
                forced_provider = "openai"
                logger.info(f"🔒 Model '{model}' detected as OpenAI - forcing provider to 'openai'")
            
            # Set provider based on forced detection or current setting
            provider = forced_provider or self.current_provider
            
            # Log if there's a provider mismatch
            if forced_provider and forced_provider != self.current_provider:
                logger.warning(f"⚠️ Provider mismatch detected! Model '{model}' requires '{forced_provider}' but current provider was '{self.current_provider}'")
                # Force the provider to match the model
                provider = forced_provider
            
            # Always get the appropriate virtual key for the provider
            virtual_key = None
            if provider in self.provider_keys:
                virtual_key = self.provider_keys[provider]
                logger.info(f"📌 Using virtual key for {provider}: {virtual_key}")
            
            # Validate provider is one of the supported providers
            if provider not in ["openai", "anthropic", "gemini"]:
                logger.warning(f"⚠️ Unsupported provider: {provider}. Defaulting to openai.")
                provider = "openai"
                virtual_key = self.provider_keys["openai"]
            
            # Always create a new Portkey client with the virtual key for this specific request
            if virtual_key:
                logger.info(f"🔑 Creating new Portkey client with provider: {provider}, virtual key: {virtual_key}")
                portkey_client = Portkey(
                    api_key=self.settings.portkey_api_key,
                    virtual_key=virtual_key
                )
            else:
                # Fallback to the default client
                logger.warning(f"⚠️ No virtual key available for provider: {provider}. Using default client.")
                portkey_client = self.client
            
            # Log the exact model and provider being used
            logger.info(f"📤 FINAL PARAMETERS - Provider: {provider}, Model: {self.settings.llm_model}, Virtual Key: {virtual_key}")
            
            # Always update current provider and virtual key to match what we're actually using
            self.current_provider = provider
            self.current_virtual_key = virtual_key
            logger.info(f"🔄 Updated internal state - Provider: {provider}, Virtual Key: {virtual_key}")
            
            # Choose the appropriate method based on provider and response_model
            # Use regular completions for all providers - more reliable
            logger.info(f"Using simple completions for provider: {self.current_provider}")
                
            try:
                # First determine if we should use instructor or regular API
                use_instructor = self.current_provider in ["openai", "gemini"]
                
                if use_instructor:
                    # Log which provider we're using with instructor
                    logger.info(f"Using instructor with Portkey for {self.current_provider.capitalize()}")
                    
                    if self.current_provider == "openai":
                        # For OpenAI, use the approach suggested by Portkey docs
                        logger.info("Creating OpenAI client with Portkey gateway URL")
                        
                        # Create an OpenAI client that points to Portkey's API
                        openai_client = OpenAI(
                            api_key=self.settings.openai_api_key,  # Use actual OpenAI key
                            base_url="https://api.portkey.ai/v1",  # Direct URL instead of constant
                            default_headers={
                                "X-Portkey-Api-Key": self.settings.portkey_api_key,
                                "X-Portkey-Provider": self.current_provider,
                                "X-Portkey-Virtual-Key": self.current_virtual_key
                            }
                        )
                        
                        # Use this client with instructor
                        instructor_client = instructor.from_openai(openai_client)
                    # elif self.current_provider == "anthropic":
                    #     # openai_client = OpenAI(
                    #     #     api_key=self.settings.openai_api_key,
                    #     #     base_url="https://api.portkey.ai/v1",
                    #     #     default_headers={
                    #     #         "X-Portkey-Api-Key": self.settings.portkey_api_key,
                    #     #         "X-Portkey-Provider": self.current_provider,
                    #     #         "X-Portkey-Virtual-Key": self.current_virtual_key
                    #     #     }
                    #     # )
                    #     instructor_client = instructor.from_anthropic(portkey_client, mode=instructor.Mode.ANTHROPIC_TOOLS)                    

                    elif self.current_provider == "gemini":
                        openai_client = OpenAI(
                            api_key=self.settings.openai_api_key,  # Use actual OpenAI key
                            base_url="https://api.portkey.ai/v1",  # Direct URL instead of constant
                            default_headers={
                                "X-Portkey-Api-Key": self.settings.portkey_api_key,
                                "X-Portkey-Provider": self.current_provider,  
                                "X-Portkey-Virtual-Key": self.current_virtual_key
                            }
                        )
                        # For other providers, use the original approach
                        instructor_client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
                    
                    # Use instructor to handle the structured response
                    response = instructor_client.chat.completions.create(
                        model=self.settings.llm_model,
                        messages=messages,
                        response_model=response_model,
                        timeout=DEFAULT_TIMEOUT,
                    )
                    
                    # Instructor returns the model instance directly
                    return response
                else:
                    # Use the most basic, common API format for other providers
                    response = portkey_client.beta.chat.completions.parse(
                        model=self.settings.llm_model,
                        messages=messages,
                        response_format=response_model,
                        timeout=DEFAULT_TIMEOUT,
                        max_tokens=8000,
                    )
                
                # For unstructured responses, try to extract the content
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                        content = response.choices[0].message.content
                        logger.info(f"Extracted content from response: {content[:50]}...")
                        
                        # If we have a response model, create a model instance with 'answer' field (not 'text')
                        if response_model and hasattr(response_model, '__name__'):
                            try:
                                # Try to create an instance of the response model
                                model_instance = response_model(answer=content)
                                return model_instance
                            except Exception as model_error:
                                logger.warning(f"Could not create model instance: {str(model_error)}")
                        
                        # No response model, just return the content
                        return content
                
                # Return the raw response if we couldn't extract content
                return response
                
            except Exception as e:
                logger.error(f"Error in completions.create: {str(e)}")
                
                # Try an alternative approach for Anthropic if the regular method fails
                if self.current_provider == "anthropic":
                    try:
                        logger.info("Trying alternative approach for Anthropic...")
                        # Use a more direct approach without all the Portkey parameters
                        from langchain_anthropic import ChatAnthropic
                        from langchain_core.messages import HumanMessage
                        
                        # Create a simple Anthropic client
                        anthropic_client = ChatAnthropic(model=self.settings.llm_model)
                        
                        # Extract the prompt content
                        prompt_content = messages[0]['content']
                        
                        # Call the model directly
                        ai_message = anthropic_client.invoke([HumanMessage(content=prompt_content)])
                        content = ai_message.content
                        
                        # Return result in the appropriate format
                        if response_model and hasattr(response_model, '__name__'):
                            try:
                                model_instance = response_model(answer=content)
                                return model_instance
                            except Exception as model_error:
                                logger.warning(f"Could not create model instance after alternative: {str(model_error)}")
                        
                        return content
                    except Exception as alt_e:
                        logger.error(f"Alternative approach for Anthropic also failed: {str(alt_e)}")
                
                # Provide a fallback response for debugging
                if response_model and hasattr(response_model, '__name__'):
                    try:
                        model_instance = response_model(answer=f"Error response: {str(e)}")
                        return model_instance
                    except:
                        pass
                
                # Re-raise the error if we can't create a fallback
                raise
            # This code is redundant because we already return from the try block
            # logger.info(f"📥 Response successfully received from model: {self.settings.llm_model}")
            # return response
            
        except Exception as e:
            logger.error(f"❌ Error calling API: {str(e)}")
            # Log more details about the error
            if hasattr(e, '__dict__'):
                logger.error(f"Error details: {str(e.__dict__)}")
            raise

    def update_virtual_key(self, virtual_key: str, provider: str, model: str = None) -> None:
        """Update the virtual key and provider for the next API call."""
        logger.info(f"🔄 UPDATING ROUTING - Previous: provider={self.current_provider}, key={self.current_virtual_key}, model={self.settings.llm_model}")
        
        # Validate provider
        if not provider or provider.lower().strip() not in ["openai", "anthropic", "gemini"]:
            logger.warning(f"⚠️ Invalid provider: '{provider}'. Valid options are 'openai', 'anthropic', or 'gemini'.")
            if not provider or provider.strip() == "":
                logger.warning("Empty provider specified. Defaulting to 'openai'.")
                provider = "openai"
        
        # Set the provider and virtual key
        self.current_provider = provider.lower().strip()
        
        # If virtual key is empty but provider is specified, use the default virtual key for that provider
        if (not virtual_key or virtual_key.strip() == "") and self.current_provider in self.provider_keys:
            default_key = self.provider_keys[self.current_provider]
            logger.info(f"No virtual key provided. Using default key for {self.current_provider}: {default_key}")
            self.current_virtual_key = default_key
        else:
            self.current_virtual_key = virtual_key
        
        # Update model if provided
        if model:
            logger.info(f"🔄 Updating model from {self.settings.llm_model} to {model}")
            self.settings.llm_model = model
        
        # Log the final update
        logger.info(f"🔄 ROUTING UPDATED to: provider='{self.current_provider}', key='{self.current_virtual_key}', model='{self.settings.llm_model}'")
        
        # Additional verification to ensure settings won't be overriden later
        if self.settings.llm_provider != "portkey":
            logger.warning(f"⚠️ Settings provider ({self.settings.llm_provider}) is not set to 'portkey'. "
                          f"This may cause conflicts. Current provider={self.current_provider}")
            
        # Return values for verification
        return self.current_provider, self.current_virtual_key

    async def decompose_query(self, query: str, parent_run_id: str = None) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning(
                "Portkey client is not initialized. Skipping decomposition."
            )
            return {"sub_queries": [query]}

        # TODO: Implement the actual decomposition logic here
        return {"sub_queries": [query]}
