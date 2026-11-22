"""User-owned inference provider credential management."""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_orm import ProviderKey
from projectdavid_orm.projectdavid_orm.models import Credential
from sqlalchemy.exc import IntegrityError

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.services.credential_service import CredentialService

validator = ValidationInterface()


class ProviderKeyConfigurationError(RuntimeError):
    """Provider key security configuration is unavailable."""


class ProviderKeyNotFoundError(LookupError):
    """Provider key does not exist within the authenticated tenant."""


class ProviderKeyUnavailableError(RuntimeError):
    """Provider key cannot be used for inference."""


class ProviderKeyService:
    """Manage registered, user-owned inference provider credentials."""

    CREDENTIAL_KIND = "inference_provider"
    FINGERPRINT_ENV = "PROJECT_DAVID_PROVIDER_KEY_HMAC_KEY"

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        credential_service: CredentialService | None = None,
        fingerprint_key_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._credential_service = credential_service or CredentialService(
            session_factory=session_factory
        )
        self._fingerprint_key_provider = fingerprint_key_provider or (
            lambda: os.getenv(self.FINGERPRINT_ENV)
        )

    def _fingerprint_secret(self) -> bytes:
        configured = self._fingerprint_key_provider()

        if not configured:
            raise ProviderKeyConfigurationError(
                "Provider key fingerprinting is not configured"
            )

        try:
            secret = bytes.fromhex(configured.strip())
        except ValueError as exc:
            raise ProviderKeyConfigurationError(
                "Invalid provider key fingerprint configuration"
            ) from exc

        if len(secret) != 32:
            raise ProviderKeyConfigurationError(
                "Provider key fingerprint configuration must be 32 bytes"
            )

        return secret

    @staticmethod
    def _encode_component(value: bytes) -> bytes:
        """Length-prefix components to avoid ambiguous concatenations."""
        return len(value).to_bytes(4, "big") + value

    def _fingerprint(
        self,
        *,
        user_id: str,
        provider: str,
        api_key: str,
    ) -> str:
        payload = b"projectdavid:provider-key:v1\x00"

        for value in (user_id, provider, api_key):
            payload += self._encode_component(value.encode("utf-8"))

        return hmac.new(
            self._fingerprint_secret(),
            payload,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _read(row: ProviderKey) -> validator.ProviderKeyRead:
        return validator.ProviderKeyRead.model_validate(row)

    @staticmethod
    def _owned(
        db: Any,
        *,
        provider_key_id: str,
        user_id: str,
        lock: bool = False,
    ) -> ProviderKey:
        query = db.query(ProviderKey).filter(
            ProviderKey.id == provider_key_id,
            ProviderKey.owner_id == user_id,
        )

        if lock:
            query = query.with_for_update()

        row = query.first()

        if row is None:
            raise ProviderKeyNotFoundError("Provider key not found")

        return row

    def register(
        self,
        registration: validator.ProviderKeyCreate,
        *,
        user_id: str,
    ) -> tuple[validator.ProviderKeyRead, bool]:
        """Register or return the same active user/provider credential.

        The bool indicates whether a new registration was created.
        """

        if not user_id:
            raise ValueError("Authenticated user identity is required")

        provider = registration.provider
        api_key = registration.api_key.get_secret_value()

        fingerprint = self._fingerprint(
            user_id=user_id,
            provider=provider,
            api_key=api_key,
        )

        with self._session_factory() as db:
            existing = (
                db.query(ProviderKey)
                .filter(
                    ProviderKey.owner_id == user_id,
                    ProviderKey.provider == provider,
                    ProviderKey.fingerprint == fingerprint,
                    ProviderKey.revoked_at.is_(None),
                )
                .first()
            )

            if existing is not None:
                return self._read(existing), False

            try:
                credential = self._credential_service.create_in_session(
                    db=db,
                    user_id=user_id,
                    kind=self.CREDENTIAL_KIND,
                    payload={"api_key": api_key},
                )

                row = ProviderKey(
                    id=UtilsInterface.IdentifierService.generate_prefixed_id("pkey"),
                    owner_id=user_id,
                    credential_id=credential.id,
                    provider=provider,
                    fingerprint=fingerprint,
                    name=registration.name,
                )

                db.add(row)
                db.commit()
                db.refresh(row)

                return self._read(row), True

            except IntegrityError:
                db.rollback()

                # A concurrent registration may have won the unique
                # owner/provider/fingerprint constraint.
                existing = (
                    db.query(ProviderKey)
                    .filter(
                        ProviderKey.owner_id == user_id,
                        ProviderKey.provider == provider,
                        ProviderKey.fingerprint == fingerprint,
                        ProviderKey.revoked_at.is_(None),
                    )
                    .first()
                )

                if existing is None:
                    raise

                return self._read(existing), False

    def list_keys(
        self,
        *,
        user_id: str,
    ) -> list[validator.ProviderKeyRead]:
        with self._session_factory() as db:
            rows = (
                db.query(ProviderKey)
                .filter(ProviderKey.owner_id == user_id)
                .order_by(
                    ProviderKey.created_at.desc(),
                    ProviderKey.id,
                )
                .all()
            )

            return [self._read(row) for row in rows]

    def get_key(
        self,
        *,
        provider_key_id: str,
        user_id: str,
    ) -> validator.ProviderKeyRead:
        with self._session_factory() as db:
            row = self._owned(
                db,
                provider_key_id=provider_key_id,
                user_id=user_id,
            )

            return self._read(row)

    def update_key(
        self,
        *,
        provider_key_id: str,
        user_id: str,
        update: validator.ProviderKeyUpdate,
    ) -> validator.ProviderKeyRead:
        """Rename metadata; never replace credential material."""

        with self._session_factory() as db:
            row = self._owned(
                db,
                provider_key_id=provider_key_id,
                user_id=user_id,
                lock=True,
            )

            row.name = update.name

            db.commit()
            db.refresh(row)

            return self._read(row)

    def revoke(
        self,
        *,
        provider_key_id: str,
        user_id: str,
    ) -> None:
        """Revoke a credential, retaining only its historical identity."""

        with self._session_factory() as db:
            row = self._owned(
                db,
                provider_key_id=provider_key_id,
                user_id=user_id,
                lock=True,
            )

            if row.revoked_at is not None:
                return

            credential = (
                db.query(Credential)
                .filter(
                    Credential.id == row.credential_id,
                    Credential.owner_id == user_id,
                    Credential.kind == self.CREDENTIAL_KIND,
                )
                .with_for_update()
                .first()
            )

            if credential is None:
                raise ProviderKeyUnavailableError("Provider credential is unavailable")

            # Retain the required credential row for FK integrity,
            # but retire the stored ciphertext.
            credential.encrypted_payload = "revoked"

            # Both fields must change together to satisfy the
            # database's revocation CHECK constraint.
            row.fingerprint = None
            row.revoked_at = datetime.utcnow()

            db.commit()

    def resolve_for_inference(
        self,
        *,
        provider_key_id: str,
        user_id: str,
        provider: str,
    ) -> str:
        """Resolve an active credential for an authorized inference call.

        This method is internal only. Never expose it through an API
        endpoint returning plaintext.
        """

        with self._session_factory() as db:
            row = self._owned(
                db,
                provider_key_id=provider_key_id,
                user_id=user_id,
                lock=True,
            )

            if row.revoked_at is not None or row.provider != provider:
                raise ProviderKeyUnavailableError(
                    "Provider key is unavailable for this provider"
                )

            # The row lock serializes resolution against revocation.
            payload = self._credential_service.resolve_in_session(
                db=db,
                credential_id=row.credential_id,
                user_id=user_id,
                expected_kind=self.CREDENTIAL_KIND,
            )

            api_key = payload.get("api_key")

            if not isinstance(api_key, str) or not api_key:
                raise ProviderKeyUnavailableError(
                    "Provider credential payload is invalid"
                )

            return api_key


__all__ = [
    "ProviderKeyConfigurationError",
    "ProviderKeyNotFoundError",
    "ProviderKeyUnavailableError",
    "ProviderKeyService",
]
