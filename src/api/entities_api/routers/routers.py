from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_db

logging_utility = LoggingUtility()
router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Check application health by verifying database connectivity.
    """
    try:
        db.execute(text("SELECT 1"))

        logging_utility.info("Health check passed.")

        return {
            "database": True,
            "status": "healthy",
        }

    except Exception as exc:
        logging_utility.error(f"Health check failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Database connection failed.",
        ) from exc
