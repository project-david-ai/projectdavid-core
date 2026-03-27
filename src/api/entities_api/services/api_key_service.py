import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from projectdavid_common import UtilsInterface, ValidationInterface
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import ApiKey, User

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class ApiKeyService:
    """
    Service class for managing API Keys.
    """

    def __init__(self, db: Session):
        self.db = db

    def _generate_unique_key_and_prefix(
        self, desired_prefix: str = "ea_"
    ) -> Tuple[str, str]:
        """
        Generates a unique API key string and its corresponding prefix.
        Ensures the prefix doesn't already exist in the database.

        Returns:
            A tuple containing (plain_api_key, unique_prefix).
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
        Creates and stores an API key for a given user ID.

        Args:
            user_id: The ID of the user to create the key for.
            key_name: An optional user-friendly name for the key.
            expires_in_days: Optional number of days until the key expires.
            key_prefix: The desired prefix for the key (e.g., 'ea_').

        Returns:
            A tuple containing:
                - The generated plain text API key (show only once).
                - The created ApiKey database object.

        Raises:
            HTTPException(404): If the user_id is not found.
            RuntimeError: If a unique key prefix cannot be generated.
            Exception: For other database errors during creation.
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
            )
            self.db.add(db_api_key)
            self.db.commit()
            self.db.refresh(db_api_key)
            logging_utility.info(
                f"API Key created successfully (Prefix: {db_api_key.prefix}) for User ID: {user.id}"
            )
            return (plain_key_str, db_api_key)
        except Exception as e:
            self.db.rollback()
            logging_utility.error(
                f"Error creating API key for user {user_id}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while creating the API key.",
            ) from e

    def list_keys_for_user(
        self, user_id: str, include_inactive: bool = False
    ) -> List[ApiKey]:
        """
        Lists API keys associated with a user.

        Args:
            user_id: The ID of the user whose keys to list.
            include_inactive: Whether to include inactive/revoked keys.

        Returns:
            A list of ApiKey database objects.

        Raises:
            HTTPException(404): If the user_id is not found.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )
        query = self.db.query(ApiKey).filter(ApiKey.user_id == user_id)
        if not include_inactive:
            query = query.filter(ApiKey.is_active.is_(True))
        keys = query.order_by(ApiKey.created_at.desc()).all()
        return keys

    def revoke_key(self, user_id: str, key_prefix: str) -> bool:
        """
        Revokes an API key by setting its 'is_active' flag to False.

        Args:
            user_id: The ID of the user who owns the key.
            key_prefix: The unique prefix of the key to revoke.

        Returns:
            True if the key was found and revoked, False otherwise.

        Raises:
            HTTPException(404): If the user_id is not found.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )
        api_key = (
            self.db.query(ApiKey)
            .filter(ApiKey.user_id == user_id, ApiKey.prefix == key_prefix)
            .first()
        )
        if not api_key:
            logging_utility.warning(
                f"Attempted to revoke non-existent key prefix {key_prefix} for user {user_id}"
            )
            return False
        if not api_key.is_active:
            logging_utility.info(
                f"API Key with prefix {key_prefix} for user {user_id} is already inactive."
            )
            return True
        try:
            api_key.is_active = False
            api_key.last_used_at = None
            self.db.commit()
            logging_utility.info(
                f"API Key with prefix {key_prefix} for user {user_id} revoked successfully."
            )
            return True
        except Exception as e:
            self.db.rollback()
            logging_utility.error(
                f"Error revoking API key {key_prefix} for user {user_id}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while revoking the API key.",
            ) from e

    def get_key_details_by_prefix(
        self, user_id: str, key_prefix: str
    ) -> Optional[ApiKey]:
        """
        Retrieves details of a specific API key by its prefix for a given user.
        Does NOT return the hashed key itself, intended for display purposes.

        Args:
            user_id: The ID of the user who owns the key.
            key_prefix: The unique prefix of the key.

        Returns:
            The ApiKey object if found, otherwise None.

        Raises:
           HTTPException(404): If the user_id is not found.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found.",
            )
        api_key = (
            self.db.query(ApiKey)
            .filter(ApiKey.user_id == user_id, ApiKey.prefix == key_prefix)
            .first()
        )
        return api_key
