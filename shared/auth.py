import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import settings

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validates the JWT token and returns the subject (user_id).
    Used in gateway endpoints as a dependency.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub", "unknown_user")
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )
