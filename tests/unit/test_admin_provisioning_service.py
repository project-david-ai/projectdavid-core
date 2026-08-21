from unittest.mock import MagicMock

from src.api.entities_api.models.models import ApiKey, User
from src.api.entities_api.services.admin_provisioning_service import (
    AdminProvisioningService,
)


def query_returning(*, first=None, all_items=None):
    query = MagicMock()
    query.filter.return_value = query
    query.with_for_update.return_value = query
    query.first.return_value = first
    query.all.return_value = all_items or []
    return query


class TestAdminProvisioningService:
    def test_reuses_user_and_rotates_named_key(self) -> None:
        user = User(
            id="user_existing",
            email="person@example.com",
            email_verified=True,
            full_name="Old Name",
            oauth_provider="project-black-bird",
            provider_user_id="usr_local",
        )
        stale = ApiKey(
            key_name="project-black-bird",
            hashed_key="old-hash",
            prefix="ea_old12",
            user_id=user.id,
            is_active=True,
        )
        db = MagicMock()
        db.query.side_effect = [
            query_returning(first=user),
            query_returning(first=user),
            query_returning(all_items=[stale]),
            query_returning(first=None),
        ]

        result = AdminProvisioningService(db).ensure_user_api_key(
            email="PERSON@example.com",
            full_name="Current Name",
            external_reference="usr_local",
            key_name="project-black-bird",
        )

        assert result.user is user
        assert result.user_created is False
        assert result.keys_revoked == 1
        assert stale.is_active is False
        assert result.api_key.is_active is True
        assert result.api_key.user_id == user.id
        assert result.api_key.key_name == "project-black-bird"
        assert result.plain_key.startswith("ea_")
        db.commit.assert_called_once()

    def test_creates_verified_mapped_user(self) -> None:
        db = MagicMock()
        db.query.side_effect = [
            query_returning(first=None),
            query_returning(first=None),
            query_returning(all_items=[]),
            query_returning(first=None),
        ]

        result = AdminProvisioningService(db).ensure_user_api_key(
            email="new@example.com",
            full_name="New Person",
            external_reference="usr_blackbird",
            key_name="project-black-bird",
        )

        assert result.user_created is True
        assert result.user.email == "new@example.com"
        assert result.user.email_verified is True
        assert result.user.oauth_provider == "project-black-bird"
        assert result.user.provider_user_id == "usr_blackbird"
        assert result.keys_revoked == 0
        db.commit.assert_called_once()
