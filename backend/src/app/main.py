"""Main module for the AI Grid API service with optimized service initialization."""

import logging
import os
from typing import Any, Dict

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.api import api_router
from app.core.auth import decode_token
from app.core.config import Settings, get_settings
from app.services.document_service import DocumentService
from app.services.embedding.factory import EmbeddingServiceFactory
from app.services.llm.factory import CompletionServiceFactory
from app.services.vector_db.factory import VectorDBFactory

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
        # Do not use wildcard with credentials - browsers reject it
        # "*"
    ],
    allow_origin_regex=r"https://(.*\.)?ai-grid\.onrender\.com",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With", 
                  "Access-Control-Request-Method", "Access-Control-Request-Headers",
                  "DNT", "If-Modified-Since", "Cache-Control", "Range"],
    expose_headers=["Content-Length", "Content-Range", "Access-Control-Allow-Origin"],
    max_age=3600,  # Cache preflight requests for 60 minutes
)


# Add middleware to ensure CORS headers are always present
class EnsureCORSMiddleware(BaseHTTPMiddleware):
    """Middleware to ensure CORS headers are always present in responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Add CORS headers to all responses.
        
        Args:
            request: The FastAPI request object.
            call_next: The next middleware or endpoint handler.
            
        Returns:
            Response: The response with CORS headers.
        """
        # Special handling for OPTIONS requests (preflight)
        if request.method == "OPTIONS":
            # Get the origin header
            origin = request.headers.get("Origin")
            
            
            # Create a new response with CORS headers
            response = Response(status_code=200)
            
            # Set CORS headers - be permissive for OPTIONS requests
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers, DNT, If-Modified-Since, Cache-Control, Range"
                response.headers["Access-Control-Max-Age"] = "3600" # 1 hour cache for preflight
            
            if "/api/v1/document" in request.url.path:
                logger.info(f"Special handling for document endpoint: {request.url.path}")
                # Ensure all necessary headers are present for document endpoints
                response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers, DNT, If-Modified-Since, Cache-Control, Range"
                response.headers["Vary"] = "Origin"
                
            return response
            
        try:
            # Process non-OPTIONS requests and get the response
            response = await call_next(request)
            
            # Ensure CORS headers are present for all responses
            origin = request.headers.get("Origin")
            if origin:
                # Only set specific origin, not wildcard, when credentials are used
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                
                # Add these headers for non-OPTIONS requests too
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range, Access-Control-Allow-Origin"
                
                # Special handling for document endpoints in regular requests too
                if "/api/v1/document" in request.url.path:
                    response.headers["Vary"] = "Origin"
            
            return response
        except RuntimeError as e:
            # Handle the "No response returned" error
            if str(e) == "No response returned.":
                # Create a new response with CORS headers
                response = Response(status_code=204)  # No Content
                origin = request.headers.get("Origin")
                if origin:
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
                return response
            else:
                # Re-raise other runtime errors
                raise

# Add the CORS middleware
app.add_middleware(EnsureCORSMiddleware)

# Authentication middleware
class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to check authentication for protected routes."""
    
    async def dispatch(self, request: Request, call_next):
        """Check authentication for protected routes.
        
        Args:
            request: The FastAPI request object.
            call_next: The next middleware or endpoint handler.
            
        Returns:
            Response: The response from the next middleware or endpoint.
        """
        # Public paths that don't require authentication
        public_paths = [
            "/ping",
            "/docs",
            "/redoc",
            f"{settings.api_v1_str}/auth/login",
            f"{settings.api_v1_str}/auth/verify",
            f"{settings.api_v1_str}/openapi.json", # OpenAPI schema
        ]
        
        # Check if the path is public
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)
        
        # Check for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get the Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(
                content='{"detail":"Not authenticated"}',
                status_code=403,
                media_type="application/json"
            )
        
        # Extract and validate the token
        token = auth_header.replace("Bearer ", "")
        payload = decode_token(token)
        if payload is None:
            return Response(
                content='{"detail":"Invalid or expired token"}',
                status_code=403,
                media_type="application/json"
            )
        
        # Token is valid, proceed with the request
        return await call_next(request)

# Add the authentication middleware
app.add_middleware(AuthMiddleware)

# Include the API router
app.include_router(api_router, prefix=settings.api_v1_str)


@app.on_event("startup")
async def startup_event():
    """Initialize services once at application startup."""
    logger.info("Initializing application services...")
    
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
