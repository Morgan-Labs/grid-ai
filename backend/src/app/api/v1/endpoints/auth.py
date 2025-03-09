"""Authentication endpoints for the API."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.auth import Token, create_access_token, verify_password

router = APIRouter()


class PasswordRequest(BaseModel):
    """Password request model."""
    password: str


@router.post("/login", response_model=Token)
async def login(request: PasswordRequest) -> Token:
    """Authenticate with password and return a JWT token.

    Args:
        request: The password request.

    Returns:
        Token: The JWT token.

    Raises:
        HTTPException: If the password is incorrect.
    """
    if not verify_password(request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT token
    access_token = create_access_token()
    
    return Token(access_token=access_token)


from fastapi import Request

@router.post("/verify")
async def verify_token(request: Request) -> dict:
    """Verify that the token is valid.
    
    This endpoint is protected by the JWT authentication and will
    return a 403 error if the token is invalid.

    Returns:
        dict: A success message.
    """
    return {
        "status": "authenticated",
        "services_initialized": getattr(request.app.state, "services_initialized", False)
    }
