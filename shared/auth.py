import os
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import settings

security = HTTPBearer()

# Dev bypass: if SECRET_KEY is the default and token is "demo-token", allow it
_DEV_DEMO_TOKEN = "demo-token"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validates the JWT token and returns the subject (user_id).
    In development mode (default SECRET_KEY), also accepts 'demo-token' directly.
    """
    token = credentials.credentials

    # Dev-mode passthrough for the frontend demo token
    if token == _DEV_DEMO_TOKEN and settings.SECRET_KEY == "super-secret-key-change-in-production":
        return "demo-user"

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub", "unknown_user")
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )
