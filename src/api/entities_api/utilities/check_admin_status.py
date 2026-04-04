# src/api/entities_api/utilities/check_admin_status.py
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import User as UserModel


def _is_admin(user_id: str, db: Session) -> bool:
    """Return True if the user exists and has is_admin=True."""
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    return bool(user and user.is_admin)
