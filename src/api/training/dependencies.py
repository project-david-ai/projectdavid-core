# src/api/training/dependencies.py
#
# Auth for the training service.
# Uses the same X-API-Key pattern as the core API — queries the shared
# MySQL instance directly. No network hop, no JWT, no duplication.

from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from projectdavid_common import UtilsInterface
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.models.models import ApiKey

logging_utility = UtilsInterface.LoggingUtility()

API_KEY_NAME = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_current_user_id(
    api_key_header: Optional[str] = Security(_api_key_scheme),
    db: Session = Depends(get_db),
) -> str:
    """
    Validates X-API-Key directly against the shared MySQL api_keys table.
    Mirrors entities_api/dependencies.py get_api_key exactly.
    Returns the user_id string on success.
    """
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in 'X-API-Key' header.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    prefix = api_key_header[:8]
    if len(api_key_header) <= len(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key format.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    key = db.query(ApiKey).filter(ApiKey.prefix == prefix, ApiKey.is_active.is_(True)).first()

    if not key or not key.verify_key(api_key_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    if key.expires_at and key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key has expired.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    logging_utility.info("Training API — authenticated user: %s", key.user_id)
    return key.user_id
