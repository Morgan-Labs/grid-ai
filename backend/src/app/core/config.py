"""Configuration settings for the application.

This module defines the configuration settings using Pydantic's
SettingsConfigDict to load environment variables from a .env file.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("uvicorn")


class Qdrant(BaseSettings):
    """Qdrant connection configuration."""
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_", 
        extra="ignore",
        env_file=["../../../backend/.env", "../../.env", "../.env", ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    location: Optional[str] = None
    url: Optional[str] = None
    port: Optional[int] = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    https: Optional[bool] = None
    api_key: Optional[str] = None
    prefix: Optional[str] = None
    timeout: Optional[int] = None
    host: Optional[str] = None
    path: Optional[str] = None


class Settings(BaseSettings):
    """Settings class for the application."""

    # ENVIRONMENT CONFIG
    environment: str = "dev"
    testing: bool = bool(0)
    
    # AWS CONFIG
    aws_region: Optional[str] = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None  # Required for temporary credentials
    
    # S3 CONFIG
    s3_bucket_name: str = "ai-grid-deep"
    s3_prefix: str = "documents"  # Folder prefix for documents in the bucket

    # API CONFIG
    project_name: str = "AI Grid API"
    api_v1_str: str = "/api/v1"
    backend_cors_origins: List[str] = [
        "https://ai-grid-ik5o.onrender.com",
        "https://ai-grid-backend-jvru.onrender.com",
        "https://grid.mx2.dev",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8001",
    ]

    # LLM CONFIG
    dimensions: int = 1536
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    use_optimized_embedding: bool = True  # Use optimized embedding service with parallel processing
    embedding_cache_size: int = 50000  # Maximum number of embeddings to cache
    embedding_max_parallel: int = 5  # Maximum number of parallel embedding requests
    llm_provider: str = "portkey"  # Options: openai, anthropic, gemini
    llm_model: str = "gpt-4.1"
    llm_virtual_key: Optional[str] = None  # Portkey virtual key for the selected provider/model
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None  # API key for Anthropic models
    
    # PORTKEY CONFIG
    portkey_api_key: Optional[str] = None
    portkey_gateway_url: Optional[str] = None
    portkey_enabled: bool = False

    # VECTOR DATABASE CONFIG
    vector_db_provider: str = "milvus"
    index_name: str = "milvus"

    # DATABASE CONFIG
    milvus_db_uri: str = "/data/milvus_db.db"
    milvus_db_token: str = "root:Milvus"
    table_states_db_uri: str = "/data/table_states.db"

    # QDRANT CONFIG
    qdrant: Qdrant = Field(default_factory=lambda: Qdrant())

    # QUERY CONFIG
    query_type: str = "hybrid"

    # DOCUMENT PROCESSING CONFIG
    loader: str = "pypdf"
    chunk_size: int = 1024
    chunk_overlap: int = 256

    # UNSTRUCTURED CONFIG
    unstructured_api_key: Optional[str] = None
    
    # AUTHENTICATION CONFIG
    auth_password: Optional[str] = None
    jwt_secret: Optional[str] = None
    
    # DOCUMENT API CONFIG
    document_api_endpoint: str = "https://ocr.api.mx2.law/doc/{}/text?token={}" # Endpoint for fetching text
    document_metadata_api_endpoint: str = "https://ocr.api.mx2.law/doc/{}" # Endpoint for fetching metadata
    document_api_token: Optional[str] = None
    
    # LANGSMITH CONFIG
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "ai-grid"
    langsmith_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=["../../../backend/.env", "../../.env", "../.env", ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="_",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get the settings for the application."""
    logger.info("Loading config settings from the environment...")
    
    # # Debug: Check for .env files
    # env_paths = ["../../../backend/.env", "../../.env", "../.env", ".env"]
    # for env_path in env_paths:
    #     path = Path(env_path)
    #     if path.exists():
    #         logger.info(f"Found .env file at: {path.absolute()}")
    #     else:
    #         logger.info(f".env file not found at: {path.absolute()}")
    
    # # Debug: Current working directory
    # logger.info(f"Current working directory: {os.getcwd()}")
    
    settings = Settings()
    
    # Debug log to check if the OpenAI API key is loaded
    if settings.openai_api_key:
        logger.info("OpenAI API key is set")
    else:
        logger.warning("OpenAI API key is not set")
    
    # Check if Anthropic API key is loaded
    if settings.anthropic_api_key:
        logger.info("Anthropic API key is set")
    else:
        logger.warning("Anthropic API key is not set")
    
    # Check if Portkey is configured
    if settings.portkey_api_key:
        logger.info("Portkey API key is set")
        settings.portkey_enabled = True
    else:
        logger.info("Portkey API key is not set, Portkey integration is disabled")
    
    # # Debug log to check vector db provider and Qdrant configuration
    # logger.info(f"Vector DB provider: {settings.vector_db_provider}")
    # if settings.vector_db_provider == "qdrant":
    #     logger.info(f"Qdrant configuration: {settings.qdrant.model_dump()}")
    
    return settings
