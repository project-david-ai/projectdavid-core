# src/api/training/dependencies.py
#
# FastAPI dependencies for the training service.
# Auth uses the same JWT secret as sandbox (SANDBOX_AUTH_SECRET)
# but via Authorization: Bearer header rather than WebSocket query param.

import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()

SECRET_KEY = os.getenv("SANDBOX_AUTH_SECRET")
ALGORITHM = "HS256"

bearer_scheme = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """
    Validates the Bearer JWT and returns the user_id (sub claim).
    Raises 401 on any auth failure.
    """
    token = credentials.credentials

    if not SECRET_KEY:
        logging_utility.error("SANDBOX_AUTH_SECRET is not set.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server auth configuration error.",
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload — missing sub claim.",
            )

        return str(user_id)

    except jwt.ExpiredSignatureError:
        logging_utility.warning("Rejected request: token expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except jwt.InvalidTokenError as e:
        logging_utility.warning("Rejected request: invalid token — %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )
    except Exception as e:
        logging_utility.error("Unexpected auth error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error.",
        )
