from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.api.entities_api.models.models import ApiKey, User
from src.api.entities_api.services.admin_provisioning_service import (
    AdminProvisioningService,
)
from src.api.entities_api.services.api_key_service import ApiKeyService


def query_returning(*, first=None, all_items=None):
    query = MagicMock()
    query.filter.return_value = query
    query.order_by.return_value = query
    query.with_for_update.return_value = query
    query.first.return_value = first
    query.all.return_value = all_items if all_items is not None else []
    return query


def make_user():
    return User(
        id="user_test",
        email="test@example.com",
        email_verified=True,
        full_name="Test User",
        oauth_provider="project-black-bird",
        provider_user_id="external_test",
    )


def make_key(
    prefix,
    *,
    protected=False,
    active=True,
    key_name="test-key",
):
    return ApiKey(
        prefix=prefix,
        hashed_key="test-hash",
        key_name=key_name,
        user_id="user_test",
        is_active=active,
        is_protected=protected,
    )


def assert_protected_sql_filter(query):
    predicates = [
        str(predicate)
        for call in query.filter.call_args_list
        for predicate in call.args
    ]
    assert any("is_protected" in value for value in predicates)


def test_key_creation_is_unprotected():
    user = make_user()

    db = MagicMock()
    db.query.side_effect = [
        query_returning(first=user),
        query_returning(first=None),
    ]

    _, key = ApiKeyService(db).create_key(
        user_id=user.id,
        key_name="ordinary-key",
    )

    assert key.is_protected is False
    assert key.is_active is True
    db.commit.assert_called_once()


@pytest.mark.parametrize("include_inactive", [False, True])
def test_listing_never_exposes_protected_keys(include_inactive):
    user = make_user()
    ordinary = make_key("ea_normal")
    protected = make_key("ea_secret", protected=True)

    query = query_returning(
        all_items=[ordinary, protected],
    )

    db = MagicMock()
    db.query.side_effect = [
        query_returning(first=user),
        query,
    ]

    keys = ApiKeyService(db).list_keys_for_user(
        user_id=user.id,
        include_inactive=include_inactive,
    )

    assert keys == [ordinary]
    assert_protected_sql_filter(query)


def test_details_hide_protected_key():
    user = make_user()
    protected = make_key("ea_secret", protected=True)

    query = query_returning(first=protected)
    db = MagicMock()
    db.query.side_effect = [
        query_returning(first=user),
        query,
    ]

    result = ApiKeyService(db).get_key_details_by_prefix(
        user_id=user.id,
        key_prefix=protected.prefix,
    )

    assert result is None
    assert_protected_sql_filter(query)


@pytest.mark.parametrize("active", [True, False])
def test_protected_key_cannot_be_revoked(active):
    user = make_user()
    protected = make_key(
        "ea_secret",
        protected=True,
        active=active,
    )

    query = query_returning(first=protected)
    db = MagicMock()
    db.query.side_effect = [
        query_returning(first=user),
        query,
    ]

    result = ApiKeyService(db).revoke_key(
        user_id=user.id,
        key_prefix=protected.prefix,
    )

    assert result is False
    assert protected.is_active is active
    db.commit.assert_not_called()
    assert_protected_sql_filter(query)


def test_ordinary_key_can_still_be_revoked():
    user = make_user()
    ordinary = make_key("ea_normal")

    db = MagicMock()
    db.query.side_effect = [
        query_returning(first=user),
        query_returning(first=ordinary),
    ]

    result = ApiKeyService(db).revoke_key(
        user_id=user.id,
        key_prefix=ordinary.prefix,
    )

    assert result is True
    assert ordinary.is_active is False
    db.commit.assert_called_once()


def test_provisioning_preserves_protected_key():
    user = make_user()

    ordinary = make_key(
        "ea_normal",
        key_name="project-black-bird",
    )
    protected = make_key(
        "ea_secret",
        protected=True,
        key_name="project-black-bird",
    )

    query = query_returning(
        all_items=[ordinary, protected],
    )

    db = MagicMock()
    db.query.side_effect = [
        query_returning(first=user),
        query_returning(first=user),
        query,
        query_returning(first=None),
    ]

    result = AdminProvisioningService(db).ensure_user_api_key(
        email=user.email,
        full_name=user.full_name,
        external_reference=user.provider_user_id,
        key_name="project-black-bird",
    )

    assert ordinary.is_active is False
    assert protected.is_active is True
    assert result.keys_revoked == 1
    assert result.api_key.is_protected is False
    assert_protected_sql_filter(query)
    db.commit.assert_called_once()


def test_missing_user_still_returns_404():
    db = MagicMock()
    db.query.return_value = query_returning(first=None)

    with pytest.raises(HTTPException) as error:
        ApiKeyService(db).list_keys_for_user("missing-user")

    assert error.value.status_code == 404
