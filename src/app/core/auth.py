import logging
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

logger = logging.getLogger(__name__)

# FastAPI security scheme for Bearer token validation
bearer_scheme = HTTPBearer(auto_error=False)

def verify_a2a_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> None:
    """FastAPI dependency to conditionally verify A2A API Key.
    
    If `A2A_API_KEY` is not set in settings/environment, validation is skipped (open for local dev).
    If set, requests must include a valid 'Authorization: Bearer <key>' header.
    """
    if not settings.a2a_api_key:
        # Open for local development
        return

    if credentials is None:
        logger.warning("Authentication failed: Missing Authorization header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization: Bearer token header."
        )

    if credentials.credentials != settings.a2a_api_key:
        logger.warning("Authentication failed: Invalid API key credentials provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )
