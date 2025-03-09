"""Authentication module for the application.

This module provides functions for password verification and JWT token generation.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.config import get_settings

# Get settings
settings = get_settings()

# Get authentication settings from config
AUTH_PASSWORD = settings.auth_password if hasattr(settings, 'auth_password') else os.environ.get("AUTH_PASSWORD", "secure-ai-grid-password")

# JWT settings
JWT_SECRET = settings.jwt_secret if hasattr(settings, 'jwt_secret') else os.environ.get("JWT_SECRET", "ai-grid-jwt-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 30  # Token valid for 30 days


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token data model."""
    exp: float  


def verify_password(password: str) -> bool:
    """Verify if the provided password matches the configured password.

    Args:
        password: The password to verify.

    Returns:
        bool: True if the password is correct, False otherwise.
    """
    return password == AUTH_PASSWORD


def create_access_token() -> str:
    """Create a new JWT access token.

    Returns:
        str: The JWT access token.
    """
    # Set token expiration
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS)
    
    # Create token payload
    to_encode = {"exp": expire.timestamp()}
    
    # Encode the JWT
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict]:
    """Decode and validate a JWT token.

    Args:
        token: The JWT token to decode.

    Returns:
        Optional[Dict]: The decoded token payload if valid, None otherwise.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Check if token has expired
        if payload["exp"] < time.time():
            return None
            
        return payload
    except jwt.PyJWTError:
        return None


# Bearer token authentication
class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication."""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> TokenData:
        """Validate the JWT token in the Authorization header.

        Args:
            request: The FastAPI request object.

        Returns:
            TokenData: The decoded token data.

        Raises:
            HTTPException: If the token is invalid or missing.
        """
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme."
                )
                
            payload = decode_token(credentials.credentials)
            if payload is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or expired token."
                )
                
            return TokenData(exp=payload["exp"])
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization credentials."
            )


# Dependency for protected routes
jwt_auth = JWTBearer()
