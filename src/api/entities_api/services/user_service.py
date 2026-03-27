import os
from datetime import datetime
from typing import List, Optional

from fastapi import HTTPException, status
from projectdavid.clients.vector_store_manager import VectorStoreManager
from projectdavid_common import UtilsInterface, ValidationInterface
from sqlalchemy import orm
from sqlalchemy.orm import Session

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import (
    Assistant,
    AuditLog,
    File,
    FileStorage,
    Message,
    Thread,
    User,
    VectorStore,
)
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.utils.samba_client import SambaClient

logging_utility = LoggingUtility()


class UserService:

    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # GDPR — Right to Erasure
    # ─────────────────────────────────────────────────────────────────────────

    def erase_user(self, user_id: str) -> None:
        """
        GDPR right-to-erasure.  Permanently and completely removes a user and
        all of their data, including physical assets that DB cascades cannot
        reach.

        Deletion order
        ──────────────
        1.  Physical files from Samba       (must precede DB row deletion)
        2.  Qdrant vector store collections (must precede DB row deletion)
        3.  Messages in user's threads      (no FK — must be explicit)
        4.  Soft-delete exclusively-owned assistants
        5.  Write immutable AuditLog entry  (survives with user_id = NULL)
        6.  db.delete(user) + commit        (DB cascades clean up the rest)

        After step 6 the following are removed by cascade:
            api_keys, runs → actions, files → file_storage,
            vector_stores → vector_store_files,
            sandboxes, batfish_snapshots,
            thread_participants rows, user_assistants rows

        And the following are nullified by SET NULL:
            threads.owner_id, assistants.owner_id, audit_logs.user_id
        """
        with SessionLocal() as db:
            db_user = db.query(User).filter(User.id == user_id).first()
            if not db_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found",
                )

            # ── 1. Physical file deletion from Samba ─────────────────────────
            self._erase_physical_files(db, user_id)

            # ── 2. Qdrant collection deletion ────────────────────────────────
            self._erase_vector_store_collections(db, user_id)

            # ── 3. Delete messages in user's threads ─────────────────────────
            # messages.thread_id has no FK — orphans must be cleaned explicitly.
            self._erase_messages_for_user_threads(db, user_id)

            # ── 4. Soft-delete exclusively-owned assistants ──────────────────
            self._soft_delete_exclusive_assistants(db, user_id)

            # ── 5. Write immutable audit log entry ───────────────────────────
            # Written before the user row is deleted so the DB session is still
            # valid. user_id will be SET NULL by cascade when user row is gone.
            self._write_erasure_audit_log(db, user_id)

            # ── 6. Delete the user row — cascades handle the rest ────────────
            db.delete(db_user)
            db.commit()

            logging_utility.info("GDPR erasure complete for user_id=%s", user_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Erasure helpers (private)
    # ─────────────────────────────────────────────────────────────────────────

    def _erase_physical_files(self, db: Session, user_id: str) -> None:
        """
        Delete every physical file from Samba for all files owned by user_id.
        Errors are logged but do not abort the erasure — a missing physical
        file should never block a legal deletion request.
        """
        samba = SambaClient(
            os.getenv("SMBCLIENT_SERVER"),
            os.getenv("SMBCLIENT_SHARE"),
            os.getenv("SMBCLIENT_USERNAME"),
            os.getenv("SMBCLIENT_PASSWORD"),
        )

        storage_rows = (
            db.query(FileStorage)
            .join(File, File.id == FileStorage.file_id)
            .filter(File.user_id == user_id)
            .all()
        )

        for row in storage_rows:
            if row.storage_system == "samba":
                try:
                    samba.delete_file(row.storage_path)
                    logging_utility.info(
                        "Erased physical file: %s (file_id=%s)",
                        row.storage_path,
                        row.file_id,
                    )
                except Exception as exc:
                    logging_utility.error(
                        "Failed to delete physical file %s during erasure of user %s: %s",
                        row.storage_path,
                        user_id,
                        exc,
                    )

    def _erase_vector_store_collections(self, db: Session, user_id: str) -> None:
        """
        Delete every Qdrant collection owned by user_id.
        Errors are logged but do not abort the erasure.
        """
        qdrant_host = os.getenv("VECTOR_STORE_HOST", "qdrant")
        vector_manager = VectorStoreManager(vector_store_host=qdrant_host)

        stores = db.query(VectorStore).filter(VectorStore.user_id == user_id).all()

        for store in stores:
            try:
                vector_manager.delete_store(store.collection_name)
                logging_utility.info(
                    "Erased Qdrant collection: %s (store_id=%s)",
                    store.collection_name,
                    store.id,
                )
            except Exception as exc:
                logging_utility.error(
                    "Failed to delete Qdrant collection %s during erasure of user %s: %s",
                    store.collection_name,
                    user_id,
                    exc,
                )

    def _erase_messages_for_user_threads(self, db: Session, user_id: str) -> None:
        """
        Delete all messages in threads owned by user_id.

        messages.thread_id is a plain string with no FK constraint — there is
        no cascade that reaches it.  We must delete explicitly before the
        thread rows are removed (which happens via SET NULL on owner_id, not
        hard delete — but we still want the content gone for erasure).
        """
        owned_thread_ids = db.query(Thread.id).filter(Thread.owner_id == user_id).all()
        thread_id_list = [row.id for row in owned_thread_ids]

        if not thread_id_list:
            return

        deleted = (
            db.query(Message)
            .filter(Message.thread_id.in_(thread_id_list))
            .delete(synchronize_session=False)
        )
        db.flush()
        logging_utility.info(
            "Erased %d message(s) across %d thread(s) for user_id=%s",
            deleted,
            len(thread_id_list),
            user_id,
        )

    def _soft_delete_exclusive_assistants(self, db: Session, user_id: str) -> None:
        """
        Soft-delete any assistant where user_id is the canonical owner AND
        no other users are associated via user_assistants.

        Assistants shared with other users are left intact — their owner_id
        will be SET NULL by the cascade when the user row is deleted.
        """
        now = int(datetime.utcnow().timestamp())

        owned_assistants = (
            db.query(Assistant)
            .filter(
                Assistant.owner_id == user_id,
                Assistant.deleted_at.is_(None),
            )
            .all()
        )

        for asst in owned_assistants:
            # Count other users associated with this assistant
            other_users = [u for u in asst.users if u.id != user_id]
            if not other_users:
                asst.deleted_at = now
                logging_utility.info(
                    "Soft-deleted exclusively-owned assistant %s for user_id=%s",
                    asst.id,
                    user_id,
                )
            else:
                logging_utility.info(
                    "Skipped soft-delete of shared assistant %s (has %d other user(s))",
                    asst.id,
                    len(other_users),
                )

        db.flush()

    def _write_erasure_audit_log(self, db: Session, user_id: str) -> None:
        """
        Write an immutable compliance record of the erasure.

        The AuditLog row is written while the user still exists.
        After db.delete(user), the user_id FK is SET NULL by cascade —
        the record survives anonymised as the legal proof of erasure.
        """
        log_entry = AuditLog(
            user_id=user_id,
            action="ERASE",
            entity_type="User",
            entity_id=user_id,
            timestamp=datetime.utcnow(),
            details={
                "reason": "GDPR right-to-erasure request",
                "erased_at": datetime.utcnow().isoformat(),
            },
        )
        db.add(log_entry)
        db.flush()
        logging_utility.info(
            "Audit log entry written for erasure of user_id=%s", user_id
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Standard CRUD (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def create_user(
        self, user_create: ValidationInterface.UserCreate
    ) -> ValidationInterface.UserRead:
        with SessionLocal() as db:
            if user_create.email:
                existing_user = (
                    db.query(User).filter(User.email == user_create.email).first()
                )
                if existing_user:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"User with email {user_create.email} already exists.",
                    )
            new_user = User(
                id=UtilsInterface.IdentifierService.generate_user_id(),
                email=user_create.email,
                email_verified=user_create.email_verified or False,
                full_name=user_create.full_name,
                given_name=user_create.given_name,
                family_name=user_create.family_name,
                picture_url=user_create.picture_url,
                oauth_provider=user_create.oauth_provider or "local",
                provider_user_id=user_create.provider_user_id,
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return ValidationInterface.UserRead.model_validate(new_user)

    def find_or_create_oauth_user(
        self,
        provider: str,
        provider_user_id: str,
        email: Optional[str],
        email_verified: Optional[bool],
        full_name: Optional[str],
        given_name: Optional[str],
        family_name: Optional[str],
        picture_url: Optional[str],
    ) -> ValidationInterface.UserRead:
        with SessionLocal() as db:
            user = (
                db.query(User)
                .filter(
                    User.oauth_provider == provider,
                    User.provider_user_id == provider_user_id,
                )
                .first()
            )
            if not user and email and email_verified:
                potential_user = db.query(User).filter(User.email == email).first()
                if potential_user and (
                    not potential_user.oauth_provider
                    or potential_user.oauth_provider == provider
                ):
                    user = potential_user
                    user.oauth_provider = provider
                    user.provider_user_id = provider_user_id
            if user:
                update_occurred = False
                if full_name is not None and user.full_name != full_name:
                    user.full_name = full_name
                    update_occurred = True
                if given_name is not None and user.given_name != given_name:
                    user.given_name = given_name
                    update_occurred = True
                if family_name is not None and user.family_name != family_name:
                    user.family_name = family_name
                    update_occurred = True
                if picture_url is not None and user.picture_url != picture_url:
                    user.picture_url = picture_url
                    update_occurred = True
                if email is not None and user.email != email:
                    user.email = email
                    user.email_verified = email_verified or False
                    update_occurred = True
                elif (
                    email_verified is not None and user.email_verified != email_verified
                ):
                    user.email_verified = email_verified
                    update_occurred = True
                if update_occurred:
                    user.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(user)
            else:
                user = User(
                    id=UtilsInterface.IdentifierService.generate_user_id(),
                    email=email,
                    email_verified=email_verified or False,
                    full_name=full_name,
                    given_name=given_name,
                    family_name=family_name,
                    picture_url=picture_url,
                    oauth_provider=provider,
                    provider_user_id=provider_user_id,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            return ValidationInterface.UserRead.model_validate(user)

    def get_user(self, user_id: str) -> ValidationInterface.UserRead:
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )
            return ValidationInterface.UserRead.model_validate(user)

    def get_user_by_email(self, email: str) -> Optional[ValidationInterface.UserRead]:
        with SessionLocal() as db:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                return None
            return ValidationInterface.UserRead.model_validate(user)

    def get_users(
        self, skip: int = 0, limit: int = 100
    ) -> List[ValidationInterface.UserRead]:
        with SessionLocal() as db:
            users = db.query(User).offset(skip).limit(limit).all()
            return [ValidationInterface.UserRead.model_validate(u) for u in users]

    def update_user(
        self, user_id: str, user_update: ValidationInterface.UserUpdate
    ) -> ValidationInterface.UserRead:
        with SessionLocal() as db:
            db_user = db.query(User).filter(User.id == user_id).first()
            if not db_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )
            update_data = user_update.model_dump(exclude_unset=True)
            updated = False
            for key, value in update_data.items():
                if hasattr(db_user, key) and getattr(db_user, key) != value:
                    setattr(db_user, key, value)
                    updated = True
            if updated:
                db_user.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(db_user)
            return ValidationInterface.UserRead.model_validate(db_user)

    def delete_user(self, user_id: str) -> None:
        """
        Hard delete with no physical asset cleanup.
        Use erase_user() for GDPR right-to-erasure.
        """
        with SessionLocal() as db:
            db_user = db.query(User).filter(User.id == user_id).first()
            if not db_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )
            db.delete(db_user)
            db.commit()

    def get_or_create_user(
        self, user_id: Optional[str] = None
    ) -> ValidationInterface.UserRead:
        if user_id:
            with SessionLocal() as db:
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    return ValidationInterface.UserRead.model_validate(user)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {user_id} not found",
                )
        minimal_user_data = ValidationInterface.UserCreate(oauth_provider="local")
        return self.create_user(minimal_user_data)

    def list_assistants_by_user(
        self, user_id: str
    ) -> List[ValidationInterface.AssistantRead]:
        with SessionLocal() as db:
            user = (
                db.query(User)
                .options(orm.joinedload(User.assistants))
                .filter(User.id == user_id)
                .first()
            )
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )
            return [
                ValidationInterface.AssistantRead.model_validate(a)
                for a in user.assistants
            ]
