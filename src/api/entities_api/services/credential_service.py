"""Tenant-owned reversible credential storage."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.models import Credential, McpServerRegistration
from pydantic import BaseModel, ConfigDict

from src.api.entities_api.db.database import SessionLocal


class CredentialConfigurationError(RuntimeError):
    """Credential encryption is not correctly configured."""


class CredentialResolutionError(RuntimeError):
    """A credential cannot be safely resolved."""


class CredentialReference(BaseModel):
    """Detached-safe, non-secret reference to a persisted credential."""

    id: str
    owner_id: str
    kind: str
    encryption_version: int

    model_config = ConfigDict(frozen=True)


class CredentialService:
    """Encrypt, persist, and resolve tenant-owned credentials.

    The service never logs or returns persisted ciphertext as a substitute for
    plaintext. Plaintext exists only at the write boundary and during explicit
    resolution for an outbound authenticated operation.
    """

    ENV_KEY = "PROJECT_DAVID_CREDENTIAL_KEY"
    ENCRYPTION_VERSION = 1

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        key_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._key_provider = key_provider or (lambda: os.getenv(self.ENV_KEY))

    def _cipher(self) -> Fernet:
        raw_key = self._key_provider()

        if raw_key is None or not raw_key.strip():
            raise CredentialConfigurationError(f"{self.ENV_KEY} is not configured")

        try:
            return Fernet(raw_key.strip().encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise CredentialConfigurationError(
                f"{self.ENV_KEY} is not a valid Fernet key"
            ) from exc

    @staticmethod
    def _validate_kind(kind: str) -> str:
        normalized = kind.strip()

        if not normalized:
            raise ValueError("Credential kind must not be blank")

        if len(normalized) > 32:
            raise ValueError("Credential kind must not exceed 32 characters")

        return normalized

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Credential payload must be a non-empty object")

        if not all(isinstance(key, str) and key for key in payload):
            raise ValueError("Credential payload keys must be non-empty strings")

        return payload

    def create_in_session(
        self,
        *,
        db: Any,
        user_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> CredentialReference:
        """Stage one encrypted credential in a caller-owned transaction."""

        if not user_id:
            raise ValueError("Credential owner is required")

        normalized_kind = self._validate_kind(kind)
        normalized_payload = self._validate_payload(payload)

        serialized = json.dumps(
            normalized_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        ciphertext = self._cipher().encrypt(serialized).decode("ascii")

        credential_id = UtilsInterface.IdentifierService.generate_prefixed_id("cred")

        row = Credential(
            id=credential_id,
            owner_id=user_id,
            kind=normalized_kind,
            encrypted_payload=ciphertext,
            encryption_version=self.ENCRYPTION_VERSION,
        )

        db.add(row)
        db.flush()

        return CredentialReference(
            id=credential_id,
            owner_id=user_id,
            kind=normalized_kind,
            encryption_version=self.ENCRYPTION_VERSION,
        )

    def create(
        self,
        *,
        user_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> CredentialReference:
        """Encrypt and persist one tenant-owned credential."""

        with self._session_factory() as db:
            reference = self.create_in_session(
                db=db,
                user_id=user_id,
                kind=kind,
                payload=payload,
            )
            db.commit()

        return reference

    def validate_runtime_configuration(self) -> None:
        """Validate the credential key only when encrypted MCP auth requires it."""

        with self._session_factory() as db:
            invalid_registration = (
                db.query(McpServerRegistration)
                .filter(
                    McpServerRegistration.auth_type != "none",
                    McpServerRegistration.credential_id.is_(None),
                )
                .first()
            )

            if invalid_registration is not None:
                raise CredentialConfigurationError(
                    "Authenticated MCP registration is missing "
                    "a credential reference"
                )

            authenticated_registration = (
                db.query(McpServerRegistration.id)
                .filter(McpServerRegistration.auth_type != "none")
                .first()
            )

        if authenticated_registration is not None:
            self._cipher()

    def resolve(
        self,
        *,
        credential_id: str,
        user_id: str,
        expected_kind: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one credential owned by the specified tenant."""

        with self._session_factory() as db:
            row = (
                db.query(Credential)
                .filter(
                    Credential.id == credential_id,
                    Credential.owner_id == user_id,
                )
                .first()
            )

            if row is None:
                raise CredentialResolutionError("Credential not found")

            kind = row.kind
            version = row.encryption_version
            ciphertext = row.encrypted_payload

        if expected_kind is not None:
            expected = self._validate_kind(expected_kind)

            if kind != expected:
                raise CredentialResolutionError("Credential kind mismatch")

        if version != self.ENCRYPTION_VERSION:
            raise CredentialResolutionError("Unsupported credential encryption version")

        try:
            plaintext = self._cipher().decrypt(ciphertext.encode("ascii"))
            payload = json.loads(plaintext.decode("utf-8"))
        except (
            InvalidToken,
            UnicodeDecodeError,
            UnicodeEncodeError,
            json.JSONDecodeError,
        ) as exc:
            raise CredentialResolutionError("Credential decryption failed") from exc

        if not isinstance(payload, dict) or not payload:
            raise CredentialResolutionError("Credential payload is invalid")

        return payload


__all__ = [
    "CredentialConfigurationError",
    "CredentialReference",
    "CredentialResolutionError",
    "CredentialService",
]
