"""Tenant-owned SQL resource service for Scratchpads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from projectdavid_common import UtilsInterface, ValidationInterface
from sqlalchemy.exc import IntegrityError

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Scratchpad, Thread

validator = ValidationInterface()


class ScratchpadResourceService:
    """Own Scratchpad identity, ownership, metadata, and SQL lifecycle."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
    ) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _scratchpad_read(
        row: Scratchpad,
    ) -> validator.ScratchpadRead:
        return validator.ScratchpadRead.model_validate(row)

    @staticmethod
    def _owned_thread(
        db: Any,
        *,
        thread_id: str,
        user_id: str,
    ) -> Thread:
        thread = (
            db.query(Thread)
            .filter(
                Thread.id == thread_id,
                Thread.owner_id == user_id,
            )
            .first()
        )

        if thread is None:
            raise HTTPException(
                status_code=404,
                detail="Thread not found",
            )

        return thread

    @staticmethod
    def _owned_scratchpad(
        db: Any,
        *,
        scratchpad_id: str,
        user_id: str,
    ) -> Scratchpad:
        scratchpad = (
            db.query(Scratchpad)
            .filter(
                Scratchpad.id == scratchpad_id,
                Scratchpad.owner_id == user_id,
            )
            .first()
        )

        if scratchpad is None:
            raise HTTPException(
                status_code=404,
                detail="Scratchpad not found",
            )

        return scratchpad

    def create_scratchpad(
        self,
        scratchpad: validator.ScratchpadCreate,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        """Create the canonical Scratchpad for an owned Thread."""

        with self._session_factory() as db:
            self._owned_thread(
                db,
                thread_id=scratchpad.thread_id,
                user_id=user_id,
            )

            existing = (
                db.query(Scratchpad)
                .filter(
                    Scratchpad.thread_id == scratchpad.thread_id,
                    Scratchpad.owner_id == user_id,
                )
                .first()
            )

            if existing is not None:
                raise HTTPException(
                    status_code=409,
                    detail=("Scratchpad already exists for this thread"),
                )

            row = Scratchpad(
                id=(UtilsInterface.IdentifierService.generate_scratchpad_id()),
                owner_id=user_id,
                thread_id=scratchpad.thread_id,
                meta_data={},
            )

            db.add(row)

            try:
                db.commit()
            except IntegrityError as exc:
                db.rollback()

                raise HTTPException(
                    status_code=409,
                    detail=("Scratchpad already exists for this thread"),
                ) from exc

            db.refresh(row)

            return self._scratchpad_read(row)

    def ensure_scratchpad_for_thread(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        """
        Resolve the canonical Scratchpad for an owned Thread,
        creating it when the Thread predates first-class
        Scratchpad resources.

        This is an internal compatibility primitive.
        """

        with self._session_factory() as db:
            self._owned_thread(
                db,
                thread_id=thread_id,
                user_id=user_id,
            )

            existing = (
                db.query(Scratchpad)
                .filter(
                    Scratchpad.thread_id == thread_id,
                    Scratchpad.owner_id == user_id,
                )
                .first()
            )

            if existing is not None:
                return self._scratchpad_read(existing)

            row = Scratchpad(
                id=(UtilsInterface.IdentifierService.generate_scratchpad_id()),
                owner_id=user_id,
                thread_id=thread_id,
                meta_data={},
            )

            db.add(row)

            try:
                db.commit()

            except IntegrityError:
                # A concurrent caller may have won the
                # one-Scratchpad-per-Thread race.
                db.rollback()

                existing = (
                    db.query(Scratchpad)
                    .filter(
                        Scratchpad.thread_id == thread_id,
                        Scratchpad.owner_id == user_id,
                    )
                    .first()
                )

                if existing is not None:
                    return self._scratchpad_read(existing)

                raise

            db.refresh(row)

            return self._scratchpad_read(row)

    def retrieve_scratchpad(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        """Retrieve one Scratchpad owned by the authenticated user."""

        with self._session_factory() as db:
            row = self._owned_scratchpad(
                db,
                scratchpad_id=scratchpad_id,
                user_id=user_id,
            )

            return self._scratchpad_read(row)

    def retrieve_scratchpad_by_thread(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        """Resolve the canonical Scratchpad for an owned Thread."""

        with self._session_factory() as db:
            self._owned_thread(
                db,
                thread_id=thread_id,
                user_id=user_id,
            )

            row = (
                db.query(Scratchpad)
                .filter(
                    Scratchpad.thread_id == thread_id,
                    Scratchpad.owner_id == user_id,
                )
                .first()
            )

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail="Scratchpad not found",
                )

            return self._scratchpad_read(row)

    def list_scratchpads(
        self,
        *,
        user_id: str,
    ) -> validator.ScratchpadList:
        """List Scratchpads owned by one authenticated user."""

        with self._session_factory() as db:
            rows = (
                db.query(Scratchpad)
                .filter(
                    Scratchpad.owner_id == user_id,
                )
                .order_by(
                    Scratchpad.created_at,
                    Scratchpad.id,
                )
                .all()
            )

            return validator.ScratchpadList(
                data=[self._scratchpad_read(row) for row in rows]
            )

    def update_scratchpad(
        self,
        scratchpad_id: str,
        scratchpad: validator.ScratchpadUpdate,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        """Update mutable Scratchpad metadata."""

        data = scratchpad.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

        with self._session_factory() as db:
            row = self._owned_scratchpad(
                db,
                scratchpad_id=scratchpad_id,
                user_id=user_id,
            )

            if "meta_data" in data:
                # Assign a fresh dict so SQLAlchemy's mutable JSON
                # instrumentation always observes the replacement.
                row.meta_data = dict(data["meta_data"])

                db.commit()
                db.refresh(row)

            return self._scratchpad_read(row)

    def delete_record(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadDeleted:
        """Delete only the tenant-owned Scratchpad SQL record."""

        with self._session_factory() as db:
            row = self._owned_scratchpad(
                db,
                scratchpad_id=scratchpad_id,
                user_id=user_id,
            )

            db.delete(row)
            db.commit()

        return validator.ScratchpadDeleted(
            id=scratchpad_id,
        )


__all__ = ["ScratchpadResourceService"]
