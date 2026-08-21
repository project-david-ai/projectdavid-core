"""Idempotent Project Black Bird user and API-key provisioning."""

from dataclasses import dataclass

from fastapi import HTTPException, status
from projectdavid_common import UtilsInterface
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import ApiKey, User


@dataclass(frozen=True)
class ProvisionedUserApiKey:
    user: User
    api_key: ApiKey
    plain_key: str
    user_created: bool
    keys_revoked: int


class AdminProvisioningService:
    """Converges a verified external account to one active Project David key."""

    PROVIDER = "project-black-bird"
    KEY_PREFIX = "ea_"
    MAX_PREFIX_ATTEMPTS = 10

    def __init__(self, db: Session) -> None:
        self.db = db

    def ensure_user_api_key(
        self,
        *,
        email: str,
        full_name: str,
        external_reference: str,
        key_name: str,
    ) -> ProvisionedUserApiKey:
        normalized_email = email.strip().lower()
        clean_name = full_name.strip()
        clean_reference = external_reference.strip()
        clean_key_name = key_name.strip()
        if (
            "@" not in normalized_email
            or not clean_name
            or not clean_reference
            or not clean_key_name
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="email, full_name, external_reference and key_name are required",
            )

        # A concurrent first-time request may win the unique-email insert. Retry once
        # after rollback; the second attempt locks and reuses that committed user.
        for attempt in range(2):
            try:
                return self._ensure_once(
                    email=normalized_email,
                    full_name=clean_name,
                    external_reference=clean_reference,
                    key_name=clean_key_name,
                )
            except IntegrityError:
                self.db.rollback()
                if attempt == 1:
                    raise
        raise RuntimeError("Project David provisioning retry loop exhausted")

    def _ensure_once(
        self,
        *,
        email: str,
        full_name: str,
        external_reference: str,
        key_name: str,
    ) -> ProvisionedUserApiKey:
        try:
            by_reference = (
                self.db.query(User)
                .filter(
                    User.oauth_provider == self.PROVIDER,
                    User.provider_user_id == external_reference,
                )
                .with_for_update()
                .first()
            )
            by_email = (
                self.db.query(User)
                .filter(User.email == email)
                .with_for_update()
                .first()
            )
            if by_reference and by_email and by_reference.id != by_email.id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="External reference and verified email resolve to different users",
                )

            user = by_reference or by_email
            user_created = user is None
            if user is None:
                user = User(
                    id=UtilsInterface.IdentifierService.generate_user_id(),
                    email=email,
                    email_verified=True,
                    full_name=full_name,
                    oauth_provider=self.PROVIDER,
                    provider_user_id=external_reference,
                )
                self.db.add(user)
                self.db.flush()
            else:
                user.email = email
                user.email_verified = True
                user.full_name = full_name
                # Project David predates Black Bird and may already hold this verified
                # user. Claim only local/unclaimed identities; email remains the stable
                # reconciliation key for accounts owned by another identity provider.
                if user.oauth_provider in (None, "", "local", self.PROVIDER):
                    user.oauth_provider = self.PROVIDER
                    user.provider_user_id = external_reference

            stale_keys = (
                self.db.query(ApiKey)
                .filter(
                    ApiKey.user_id == user.id,
                    ApiKey.key_name == key_name,
                    ApiKey.is_active.is_(True),
                )
                .with_for_update()
                .all()
            )
            for stale_key in stale_keys:
                stale_key.is_active = False
                stale_key.last_used_at = None

            plain_key, prefix = self._new_unique_key()
            api_key = ApiKey(
                key_name=key_name,
                hashed_key=ApiKey.hash_key(plain_key),
                prefix=prefix,
                user_id=user.id,
                is_active=True,
            )
            self.db.add(api_key)
            self.db.commit()
            self.db.refresh(user)
            self.db.refresh(api_key)
            return ProvisionedUserApiKey(
                user=user,
                api_key=api_key,
                plain_key=plain_key,
                user_created=user_created,
                keys_revoked=len(stale_keys),
            )
        except Exception:
            self.db.rollback()
            raise

    def _new_unique_key(self) -> tuple[str, str]:
        for _ in range(self.MAX_PREFIX_ATTEMPTS):
            plain_key = ApiKey.generate_key(prefix=self.KEY_PREFIX)
            prefix = plain_key[:8]
            exists = self.db.query(ApiKey.id).filter(ApiKey.prefix == prefix).first()
            if not exists:
                return plain_key, prefix
        raise RuntimeError("Unable to generate a unique Project David API-key prefix")
