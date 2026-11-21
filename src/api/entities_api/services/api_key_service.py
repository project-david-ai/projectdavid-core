from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from projectdavid_common import UtilsInterface, ValidationInterface
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import ApiKey, User

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class ApiKeyService:
    """Service class for managing ordinary, user-visible API keys."""

    def __init__(self, db: Session):
        self.db = db

    def _generate_unique_key_and_prefix(
        self, desired_prefix: str = "ea_"
    ) -> Tuple[str, str]:
        """
        Generate an API key and a prefix not already present in the database.
        """
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            plain_key = ApiKey.generate_key(prefix=desired_prefix)
            prefix = plain_key[:8]

            existing_prefix = (
                self.db.query(ApiKey).filter(ApiKey.prefix == prefix).first()
            )

            if not existing_prefix:
                return (plain_key, prefix)

            attempts += 1
            logging_utility.warning(
                f"API key prefix collision detected for {prefix}. Retrying..."
            )

        raise RuntimeError(
            "Failed to generate a unique API key prefix after multiple attempts."
        )

    def create_key(
        self,
        user_id: str,
        key_name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        key_prefix: str = "ea_",
    ) -> Tuple[str, ApiKey]:
        """
        Create an ordinary API key for a user.

        User-facing key creation never creates protected service credentials.
        """
        user = self.db.query(User).filter(User.id == user_id).first()

        if not user:
            logging_utility.error(
                f"Attempted to create API key for non-existent user: {user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )

        try:
            plain_key_str, unique_prefix = self._generate_unique_key_and_prefix(
                desired_prefix=key_prefix
            )

            hashed_key = ApiKey.hash_key(plain_key_str)

            expires_at = None
            if expires_in_days is not None:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            db_api_key = ApiKey(
                key_name=key_name,
                hashed_key=hashed_key,
                prefix=unique_prefix,
                user_id=user.id,
                expires_at=expires_at,
                is_active=True,
                is_protected=False,
            )

            self.db.add(db_api_key)
            self.db.commit()
            self.db.refresh(db_api_key)

            logging_utility.info(
                f"API Key created successfully (Prefix: {db_api_key.prefix}) "
                f"for User ID: {user.id}"
            )

            return (plain_key_str, db_api_key)

        except Exception as e:
            self.db.rollback()
            logging_utility.error(
                f"Error creating API key for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while creating the API key.",
            ) from e

    def list_keys_for_user(
        self,
        user_id: str,
        include_inactive: bool = False,
    ) -> List[ApiKey]:
        """
        List ordinary API keys belonging to a user.

        Protected service keys are never included, regardless of
        whether inactive keys were requested.
        """
        user = self.db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )

        query = self.db.query(ApiKey).filter(
            ApiKey.user_id == user_id,
            ApiKey.is_protected.is_(False),
        )

        if not include_inactive:
            query = query.filter(ApiKey.is_active.is_(True))

        keys = query.order_by(ApiKey.created_at.desc()).all()

        # Defensive guard: never expose a protected object even if
        # an unexpected query implementation returns one.
        return [key for key in keys if not key.is_protected]

    def revoke_key(self, user_id: str, key_prefix: str) -> bool:
        """
        Revoke an ordinary API key.

        Protected keys behave as nonexistent through this interface.
        The guard applies to both active and already-inactive keys.
        """
        user = self.db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )

        api_key = (
            self.db.query(ApiKey)
            .filter(
                ApiKey.user_id == user_id,
                ApiKey.prefix == key_prefix,
                ApiKey.is_protected.is_(False),
            )
            .first()
        )

        if not api_key or api_key.is_protected:
            logging_utility.warning(
                f"Attempted to revoke unavailable key prefix "
                f"{key_prefix} for user {user_id}"
            )
            return False

        if not api_key.is_active:
            logging_utility.info(
                f"API Key with prefix {key_prefix} for user "
                f"{user_id} is already inactive."
            )
            return True

        try:
            api_key.is_active = False
            api_key.last_used_at = None
            self.db.commit()

            logging_utility.info(
                f"API Key with prefix {key_prefix} for user "
                f"{user_id} revoked successfully."
            )
            return True

        except Exception as e:
            self.db.rollback()
            logging_utility.error(
                f"Error revoking API key {key_prefix} " f"for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while revoking the API key.",
            ) from e

    def get_key_details_by_prefix(
        self,
        user_id: str,
        key_prefix: str,
    ) -> Optional[ApiKey]:
        """
        Retrieve details for an ordinary API key.

        Protected keys are never exposed by this method.
        """
        user = self.db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )

        api_key = (
            self.db.query(ApiKey)
            .filter(
                ApiKey.user_id == user_id,
                ApiKey.prefix == key_prefix,
                ApiKey.is_protected.is_(False),
            )
            .first()
        )

        if not api_key or api_key.is_protected:
            return None

        return api_key
