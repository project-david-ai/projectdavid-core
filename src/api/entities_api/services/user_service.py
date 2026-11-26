import os
from datetime import datetime
from typing import List, Optional

from fastapi import HTTPException, status
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
from src.api.entities_api.services.erasure_state import (
    erasure_state,
    require_mutable_user,
)
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.vector_runtime.vector_store_manager import (
    VectorStoreManager,
)
from src.api.entities_api.utilities.samba_client import SambaClient

logging_utility = LoggingUtility()


class UserService:

    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # GDPR — Right to Erasure
    # ─────────────────────────────────────────────────────────────────────────

    def erase_user(self, user_id: str) -> None:
        """Delegate account deletion to the durable erasure coordinator."""
        from src.api.entities_api.services.account_erasure import erase_user

        erase_user(self, user_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Erasure helpers (private)
    # ─────────────────────────────────────────────────────────────────────────

    def _erase_physical_files(self, db: Session, user_id: str) -> None:
        """Keep database locators intact unless every physical deletion succeeds."""
        storage_rows = (
            db.query(FileStorage)
            .join(File, File.id == FileStorage.file_id)
            .filter(File.user_id == user_id)
            .all()
        )

        if not storage_rows:
            return
        if any(
            row.storage_system != "samba" or not row.storage_path
            for row in storage_rows
        ):
            raise HTTPException(409, "Account erasure requires storage reconciliation.")
        samba = None
        try:
            samba = SambaClient(
                os.getenv("SMBCLIENT_SERVER"),
                os.getenv("SMBCLIENT_SHARE"),
                os.getenv("SMBCLIENT_USERNAME"),
                os.getenv("SMBCLIENT_PASSWORD"),
            )
            for row in storage_rows:
                samba.delete_file(row.storage_path, missing_ok=True)
        except Exception:
            # Do not log exception strings: storage errors can contain private paths.
            raise HTTPException(
                503, "Physical file erasure is incomplete; retry required."
            ) from None
        finally:
            if samba is not None and samba.conn is not None:
                samba.conn.close()

    def _erase_vector_store_collections(self, db: Session, user_id: str) -> None:
        """Abort before deleting locators if Qdrant cleanup is unconfirmed."""
        stores = db.query(VectorStore).filter(VectorStore.user_id == user_id).all()
        if not stores:
            return
        vector_manager = None
        try:
            vector_manager = VectorStoreManager()
            for store in stores:
                vector_manager.delete_store(store.collection_name)
        except Exception:
            raise HTTPException(
                503, "Vector erasure is incomplete; retry required."
            ) from None
        finally:
            if vector_manager is not None:
                vector_manager.client.close()

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
                .with_for_update()
                .first()
            )
            matched_by_email = False
            if user is None and email and email_verified:
                user = (
                    db.query(User).filter(User.email == email).with_for_update().first()
                )
                matched_by_email = user is not None

            if user is not None:
                # The identity lookup already holds the user-row lock.
                # Read the receipt before changing any identity fields.
                if erasure_state(db, user.id, for_update=True) is not None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Account deletion is pending or complete.",
                    )

                # Preserve the existing verified-email linking policy.
                if matched_by_email and (
                    user.oauth_provider and user.oauth_provider != provider
                ):
                    user = None

            if user is not None:
                update_occurred = False
                if matched_by_email:
                    user.oauth_provider = provider
                    user.provider_user_id = provider_user_id
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
            db_user = require_mutable_user(db, user_id)
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
        """Reject legacy metadata-only deletion; never discard storage locators."""
        with SessionLocal() as db:
            require_mutable_user(db, user_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Direct user deletion is disabled. Account erasure is required.",
            )

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
