# src/api/training/dependencies.py
#
# Auth for the training service.
# Uses the same pattern as the core API — queries the shared
# MySQL instance directly. No network hop, no JWT, no duplication.
# Supports both X-API-Key and Authorization: Bearer headers.

from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (APIKeyHeader, HTTPAuthorizationCredentials,
                              HTTPBearer)
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.models import ApiKey
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db

logging_utility = UtilsInterface.LoggingUtility()

API_KEY_NAME = "X-API-Key"
_api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    api_key_header: Optional[str] = Security(_api_key_header_scheme),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    db: Session = Depends(get_db),
) -> str:
    """
    Validates API Key from either 'X-API-Key' or 'Authorization: Bearer' headers.
    Directly queries the shared MySQL api_keys table.
    Returns the user_id string on success.
    """

    # 1. Resolve which API Key to use
    api_key = None

    if api_key_header:
        api_key = api_key_header.strip()
    elif bearer_token:
        api_key = bearer_token.credentials.strip()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide 'X-API-Key' or 'Authorization: Bearer' header.",
            headers={"WWW-Authenticate": "Bearer, APIKey"},
        )

    # 2. Extract prefix and validate format
    prefix = api_key[:8]

    if len(api_key) <= len(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key format.",
            headers={"WWW-Authenticate": "Bearer, APIKey"},
        )

    # 3. Query the shared database
    key = (
        db.query(ApiKey)
        .filter(
            ApiKey.prefix == prefix,
            ApiKey.is_active.is_(True),
        )
        .first()
    )

    if not key or not hasattr(key, "verify_key") or not key.verify_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key.",
            headers={"WWW-Authenticate": "Bearer, APIKey"},
        )

    if key.expires_at and key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key has expired.",
            headers={"WWW-Authenticate": "Bearer, APIKey"},
        )

    logging_utility.info("Training API — authenticated user: %s", key.user_id)

    return key.user_id
