"""Authentication endpoints for the API."""

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from app.core.auth import Token, create_access_token, verify_password

router = APIRouter()


@router.options(
    "/login",
    status_code=200,
)
async def options_login_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for login endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


@router.options(
    "/verify",
    status_code=200,
)
async def options_verify_endpoint(
    request: Request,
    response: Response,
):
    """
    Handle preflight OPTIONS requests for verify endpoint.
    """
    # Set CORS headers for preflight request
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
        response.headers["Access-Control-Max-Age"] = "1800"  # Cache preflight for 30 minutes
    return {}


class PasswordRequest(BaseModel):
    """Password request model."""
    password: str


@router.post("/login", response_model=Token)
async def login(
    request: PasswordRequest,
    req: Request,
    resp: Response
) -> Token:
    # Ensure CORS headers are present
    origin = req.headers.get("Origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
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


@router.post("/verify")
async def verify_token(
    request: Request,
    response: Response
) -> dict:
    # Ensure CORS headers are present
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With, Access-Control-Request-Method, Access-Control-Request-Headers"
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
