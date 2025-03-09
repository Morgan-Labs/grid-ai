"""Dependencies for the application using FastAPI app state for singletons."""

import logging
from typing import Any, Callable, Type, TypeVar

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.document_service import DocumentService
from app.services.embedding.base import EmbeddingService
from app.services.llm.base import CompletionService
from app.services.vector_db.base import VectorDBService

logger = logging.getLogger(__name__)


def get_llm_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> CompletionService:
    """Get the LLM service from application state."""
    if not hasattr(request.app.state, "llm_service"):
        raise ValueError("LLM service not initialized in application state")
    
    return request.app.state.llm_service


def get_embedding_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Get the embedding service from application state."""
    if not hasattr(request.app.state, "embedding_service"):
        raise ValueError("Embedding service not initialized in application state")
    
    return request.app.state.embedding_service


def get_vector_db_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> VectorDBService:
    """Get the vector database service from application state."""
    if not hasattr(request.app.state, "vector_db_service"):
        raise ValueError("Vector DB service not initialized in application state")
    
    return request.app.state.vector_db_service


def get_document_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """Get the document service from application state."""
    if not hasattr(request.app.state, "document_service"):
        raise ValueError("Document service not initialized in application state")
    
    return request.app.state.document_service
