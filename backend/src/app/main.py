"""Main module for the AI Grid API service with optimized service initialization."""

import logging
import os
from typing import Any, Dict

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
# from app.core.auth import decode_token 
from app.core.config import Settings, get_settings
from app.core.middleware import AuthMiddleware
from app.services.document_service import DocumentService
from app.services.embedding.factory import EmbeddingServiceFactory
from app.services.llm.factory import CompletionServiceFactory
from app.services.vector_db.factory import VectorDBFactory
from app.services.table_state_service import init_db_async # Import async init

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    openapi_url=f"{settings.api_v1_str}/openapi.json",
    redirect_slashes=False,  # Disable automatic redirects for trailing slashes
)

# Configure CORS with enhanced settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-grid-ik5o.onrender.com",
        "https://ai-grid-backend-jvru.onrender.com",
        "https://grid.mx2.dev",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Access-Control-Allow-Origin"],
    max_age=3600,  # Cache preflight requests for 60 minutes
)


# Add the authentication middleware
app.add_middleware(AuthMiddleware)

# Include the API router
app.include_router(api_router, prefix=settings.api_v1_str)


@app.on_event("startup")
async def startup_event():
    """Initialize services once at application startup."""
    logger.info("Initializing application services...")
    
    # Initialize database schema asynchronously first
    try:
        await init_db_async()
        logger.info("Async database initialization complete.")
    except Exception as e:
        logger.critical(f"Failed to initialize async database: {e}", exc_info=True)
        # Depending on requirements, you might want to prevent startup
        # raise RuntimeError(f"Failed to initialize database: {e}") from e 
        # Or set a flag similar to services_initialized
        app.state.db_initialized = False 
        return # Stop further initialization if DB fails
    else:
        app.state.db_initialized = True

    # Ensure data directory exists with proper permissions
    # Use the directory from the table_states_db_uri setting
    data_dir = os.path.dirname(os.path.abspath(settings.table_states_db_uri))
    try:
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Ensured data directory exists: {data_dir}")
        
        # Try to set directory permissions
        try:
            # os.chmod(data_dir, 0o777) # Removed permission setting
            logger.info(f"Ensured data directory exists: {data_dir}")
        except Exception as e:
            logger.warning(f"Could not set permissions on data directory: {e}")

        # Create database files if they don't exist
        table_states_db = os.path.join(data_dir, "table_states.db")
        if not os.path.exists(table_states_db):
            open(table_states_db, 'a').close()
            logger.info(f"Created table states database file: {table_states_db}")

    except Exception as e:
        logger.error(f"Error setting up data directory: {e}")
    
    # Initialize LangSmith tracing if enabled
    if settings.langsmith_tracing and settings.langsmith_api_key:
        logger.info("Initializing LangSmith tracing")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        
        # Log successful LangSmith configuration
        logger.info(f"LangSmith tracing enabled with project: {settings.langsmith_project}")
    
    try:
        # Initialize embedding service
        logger.info(f"Creating embedding service for provider: {settings.embedding_provider}")
        app.state.embedding_service = EmbeddingServiceFactory.create_service(settings)
        if app.state.embedding_service is None:
            logger.error(f"Failed to create embedding service for provider: {settings.embedding_provider}")
            app.state.services_initialized = False
            return
        
        # Initialize LLM service
        logger.info(f"Creating LLM service for provider: {settings.llm_provider}")
        app.state.llm_service = CompletionServiceFactory.create_service(settings)
        if app.state.llm_service is None:
            logger.error(f"Failed to create LLM service for provider: {settings.llm_provider}")
            app.state.services_initialized = False
            return
        
        # Initialize vector database service
        logger.info(f"Creating vector database service for provider: {settings.vector_db_provider}")
        app.state.vector_db_service = VectorDBFactory.create_vector_db_service(
            app.state.embedding_service, 
            app.state.llm_service, 
            settings
        )
        if app.state.vector_db_service is None:
            logger.error(f"Failed to create vector database service for provider: {settings.vector_db_provider}")
            app.state.services_initialized = False
            return
        
        # Initialize document service
        logger.info("Creating document service")
        app.state.document_service = DocumentService(
            app.state.vector_db_service,
            app.state.llm_service,
            settings
        )
        
        app.state.services_initialized = True
        logger.info("All application services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        app.state.services_initialized = False


@app.get("/ping")
async def pong(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Ping the API to check if it's running."""
    return {
        "ping": "pong!",
        "environment": settings.environment,
        "testing": settings.testing,
    }
