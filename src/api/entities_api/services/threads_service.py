import json
import time
from typing import Any, Dict, List

from fastapi import HTTPException
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.orm import Session

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Message, Thread, User
from src.api.entities_api.utilities.cache_utils import get_sync_message_cache

logging_utility = LoggingUtility()
validator = ValidationInterface()


class ThreadService:
    """CRUD logic for Thread objects."""

    def __init__(self):
        pass

    # ──────────────────────────────────────────────────────────
    # Ownership guard  (mirrors AssistantService._assert_owner)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _assert_owner(db_thread: Thread, user_id: str) -> None:
        """
        Raise 403 if *user_id* is not the canonical owner of *db_thread*.

        Fast path: owner_id column (add via migration to unlock this).
        Fallback:  thread_participants association table (works today,
                   remove once owner_id is NOT NULL and back-filled).
        """
        if db_thread.owner_id is not None:
            if db_thread.owner_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to modify this thread.",
                )
        else:
            # Fallback during back-fill window.
            participant_ids = {u.id for u in db_thread.participants}
            if user_id not in participant_ids:
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to modify this thread.",
                )

    # ──────────────────────────────────────────────────────────
    # Public CRUD
    # ──────────────────────────────────────────────────────────

    def create_thread(
        self,
        thread: validator.ThreadCreate,
        user_id: str,  # ← NEW: canonical owner
    ) -> validator.ThreadReadDetailed:
        """Create a new thread, record the creator as owner, and attach participants."""
        with SessionLocal() as db:
            existing_users = (
                db.query(User).filter(User.id.in_(thread.participant_ids)).all()
            )
            if len(existing_users) != len(thread.participant_ids):
                raise HTTPException(status_code=400, detail="Invalid user IDs")

            # Ensure the creator is always a participant even if omitted from the list.
            creator = db.query(User).filter(User.id == user_id).first()
            if not creator:
                raise HTTPException(status_code=404, detail="Owning user not found")

            db_thread = Thread(
                id=UtilsInterface.IdentifierService.generate_thread_id(),
                created_at=int(time.time()),
                meta_data=json.dumps({}),
                object="thread",
                tool_resources=json.dumps({}),
                owner_id=user_id,  # ← set canonical owner at creation time
            )
            db.add(db_thread)

            participant_set = {u.id: u for u in existing_users}
            participant_set[user_id] = creator  # idempotent creator inclusion
            for user in participant_set.values():
                db_thread.participants.append(user)

            db.commit()
            db.refresh(db_thread)
            return self._create_thread_read_detailed(db_thread)

    def get_thread(self, thread_id: str) -> validator.ThreadReadDetailed:
        with SessionLocal() as db:
            db_thread = self._get_thread_or_404(thread_id, db)
            return self._create_thread_read_detailed(db_thread)

    def delete_thread(
        self,
        thread_id: str,
        user_id: str,  # ← NEW: ownership required to delete
    ) -> validator.ThreadDeleted:
        logging_utility.info("Deleting thread and clearing history: %s", thread_id)
        with SessionLocal() as db:
            try:
                db_thread = self._get_thread_or_404(thread_id, db)

                # ── Ownership check ──────────────────────────────────────────
                self._assert_owner(db_thread, user_id)

                # 1. DB Cleanup
                db.query(Message).filter(Message.thread_id == thread_id).delete()
                db_thread.participants = []
                db.delete(db_thread)
                db.commit()

                # 2. Cache Invalidation
                try:
                    msg_cache = get_sync_message_cache()
                    msg_cache.delete_history_sync(thread_id)
                    logging_utility.info(
                        f"Invalidated message cache for thread: {thread_id}"
                    )
                except Exception as e:
                    logging_utility.error(f"Failed to invalidate message cache: {e}")

                return validator.ThreadDeleted(id=thread_id)

            except HTTPException:
                raise
            except Exception as e:
                db.rollback()
                logging_utility.error("Error deleting thread: %s", str(e))
                raise HTTPException(status_code=500, detail="Delete failed")

    def list_threads_by_user(self, user_id: str) -> List[str]:
        with SessionLocal() as db:
            threads = (
                db.query(Thread)
                .join(Thread.participants)
                .filter(User.id == user_id)
                .all()
            )
            return [thread.id for thread in threads]

    def update_thread_metadata(
        self,
        thread_id: str,
        new_metadata: Dict[str, Any],
        user_id: str,  # ← NEW
    ) -> validator.ThreadReadDetailed:
        with SessionLocal() as db:
            db_thread = self._get_thread_or_404(thread_id, db)

            # ── Ownership check ──────────────────────────────────────────────
            self._assert_owner(db_thread, user_id)

            db_thread.meta_data = json.dumps(new_metadata)
            db.commit()
            db.refresh(db_thread)
            return self._create_thread_read_detailed(db_thread)

    def update_thread(
        self,
        thread_id: str,
        thread_update: validator.ThreadUpdate,
        user_id: str,  # ← NEW
    ) -> validator.ThreadReadDetailed:
        logging_utility.info(f"Attempting to update thread with id: {thread_id}")
        logging_utility.info(f"Update data: {thread_update.dict()}")

        with SessionLocal() as db:
            db_thread = self._get_thread_or_404(thread_id, db)

            # ── Ownership check ──────────────────────────────────────────────
            self._assert_owner(db_thread, user_id)

            update_data = thread_update.dict(exclude_unset=True)

            if "meta_data" in update_data and update_data["meta_data"] is not None:
                current_metadata = self._ensure_dict(db_thread.meta_data)
                current_metadata.update(update_data["meta_data"])
                db_thread.meta_data = current_metadata

            if (
                "tool_resources" in update_data
                and update_data["tool_resources"] is not None
            ):
                db_thread.tool_resources = update_data["tool_resources"]

            for key, value in update_data.items():
                if key not in ("meta_data", "tool_resources"):
                    setattr(db_thread, key, value)

            db.commit()
            db.refresh(db_thread)
            logging_utility.info(f"Successfully updated thread with id: {thread_id}")
            return self._create_thread_read_detailed(db_thread)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _get_thread_or_404(self, thread_id: str, db: Session) -> Thread:
        db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not db_thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return db_thread

    @staticmethod
    def _ensure_dict(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return {}
        return {}

    def _create_thread_read_detailed(
        self, db_thread: Thread
    ) -> validator.ThreadReadDetailed:
        participants = [
            ValidationInterface.UserBase.from_orm(user)
            for user in db_thread.participants
        ]
        return validator.ThreadReadDetailed(
            id=db_thread.id,
            created_at=db_thread.created_at,
            meta_data=self._ensure_dict(db_thread.meta_data),
            object=db_thread.object,
            tool_resources=self._ensure_dict(db_thread.tool_resources),
            participants=participants,
        )
