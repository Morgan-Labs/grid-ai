import logging
from typing import Any

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Settings, get_settings
from app.core.auth import decode_token # For AuthMiddleware

logger = logging.getLogger(__name__)
settings = get_settings() # Loaded once

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
        public_paths = [
            "/ping",
            "/docs",
            "/redoc",
            f"{settings.api_v1_str}/auth/login",
            f"{settings.api_v1_str}/auth/verify",
            f"{settings.api_v1_str}/openapi.json",
        ]
        
        if any(request.url.path.startswith(path) for path in public_paths) or request.method == "OPTIONS":
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=403,
                detail="Not authenticated",
            )
        
        token = auth_header.replace("Bearer ", "")
        payload = decode_token(token)
        if payload is None:
            raise HTTPException(
                status_code=403,
                detail="Invalid or expired token",
            )
        
        # Optionally, attach user/payload to request state if needed by route handlers
        # request.state.user = payload 
        
        return await call_next(request) 