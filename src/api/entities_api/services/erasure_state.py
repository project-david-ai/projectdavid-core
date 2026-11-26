"""Durable account-erasure state backed by the authoritative ORM."""

import hashlib
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import AccountErasure, User

PENDING = "pending"
COMPLETE = "complete"

erasures = AccountErasure.__table__


def subject_hash(user_id: str) -> str:
    """Return a stable pseudonymous identifier for erasure reconciliation."""
    return hashlib.sha256(
        ("core-account-erasure:v1:" + user_id).encode("utf-8")
    ).hexdigest()


def erasure_state(
    db: Session,
    user_id: str,
    *,
    for_update: bool = False,
) -> Optional[str]:
    """Return durable erasure state, or None if no receipt exists."""
    statement = select(erasures.c.state).where(
        erasures.c.subject_hash == subject_hash(user_id)
    )

    if for_update:
        statement = statement.with_for_update()

    return db.execute(statement).scalar_one_or_none()


def lock_user(db: Session, user_id: str) -> Optional[User]:
    """Acquire the canonical user-row lock within the caller's transaction."""
    return db.execute(
        select(User)
        .where(User.id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


def require_mutable_user(db: Session, user_id: str) -> User:
    """
    Serialize account mutations against the erasure transaction.

    The caller must retain this transaction until its mutation
    has committed or rolled back.
    """
    user = lock_user(db, user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    if erasure_state(db, user_id, for_update=True) is not None:
        raise HTTPException(
            status_code=409,
            detail="Account deletion is pending or complete.",
        )

    return user
