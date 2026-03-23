# src/api/training/dependencies.py

from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
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
    Direct Auth: Validates API Key against the shared MySQL database.
    Prevents Bcrypt crashes by validating token length before verification.
    """

    # 1. Resolve API Key from headers
    api_key = None
    if api_key_header:
        api_key = api_key_header.strip()
    elif bearer_token:
        api_key = bearer_token.credentials.strip()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key.",
            headers={"WWW-Authenticate": "Bearer, APIKey"},
        )

    # 2. Bcrypt Safety Check
    # The 'bcrypt' algorithm used by the ORM has a hard limit of 72 bytes.
    # If the provided string is longer, it is mathematically impossible to be a valid key.
    if len(api_key.encode('utf-8')) > 72:
        logging_utility.warning("Auth attempt with over-sized token (>72 bytes). Rejecting.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key format (too long).",
        )

    # 3. Extract prefix and validate basic format
    prefix = api_key[:8]
    if len(api_key) <= 8:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key format.",
        )

    # 4. Query the shared database directly
    key_record = (
        db.query(ApiKey)
        .filter(
            ApiKey.prefix == prefix,
            ApiKey.is_active.is_(True),
        )
        .first()
    )

    # 5. Verify the key hash
    if not key_record or not hasattr(key_record, "verify_key"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key.",
        )

    try:
        # This call reaches out to the passlib/bcrypt logic
        if not key_record.verify_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key.",
            )
    except ValueError as e:
        # Catch-all for any other bcrypt-level string errors
        logging_utility.error("Bcrypt verification failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Error validating API Key format.",
        )

    # 6. Check Expiration
    if key_record.expires_at and key_record.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key has expired.",
        )

    logging_utility.info("Training API — Direct Auth Successful: %s", key_record.user_id)

    return key_record.user_id
