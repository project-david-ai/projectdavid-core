"""Identity mutation must not bypass durable account quarantine."""

import pytest
from fastapi import HTTPException
from projectdavid_common import ValidationInterface
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import ApiKey, File, FileStorage, User
from src.api.entities_api.services import account_erasure, user_service
from src.api.entities_api.services.erasure_state import PENDING, erasure_state, erasures


@pytest.fixture
def identity_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        for model in (User, ApiKey, File, FileStorage):
            model.__table__.create(engine)
        erasures.create(engine)
        factory = sessionmaker(engine)
        monkeypatch.setattr(user_service, "SessionLocal", factory)
        monkeypatch.setattr(account_erasure, "SessionLocal", factory)

        with factory() as db:
            db.add_all(
                [
                    User(
                        id="owner",
                        email="owner@example.com",
                        email_verified=True,
                        full_name="Original Owner",
                        oauth_provider="google",
                        provider_user_id="owner_subject",
                    ),
                    User(
                        id="unrelated",
                        email="unrelated@example.com",
                        email_verified=True,
                        full_name="Unrelated User",
                        oauth_provider="google",
                        provider_user_id="unrelated_subject",
                    ),
                ]
            )
            db.flush()
            db.add(
                File(
                    id="private_file",
                    user_id="owner",
                    bytes=4,
                    filename="private.txt",
                    purpose="assistants",
                )
            )
            db.flush()
            db.add(
                FileStorage(
                    file_id="private_file",
                    storage_system="samba",
                    storage_path="private/path",
                )
            )
            db.commit()
        yield factory
    finally:
        engine.dispose()


def oauth_login(provider, subject, email, full_name):
    return user_service.UserService().find_or_create_oauth_user(
        provider=provider,
        provider_user_id=subject,
        email=email,
        email_verified=True,
        full_name=full_name,
        given_name=None,
        family_name=None,
        picture_url=None,
    )


@pytest.mark.parametrize(
    "provider,subject,email",
    [
        ("google", "owner_subject", "changed@example.com"),
        ("google", "changed_subject", "owner@example.com"),
        ("microsoft", "different_subject", "owner@example.com"),
    ],
    ids=["provider-subject", "verified-email", "different-provider-email"],
)
def test_pending_oauth_identity_cannot_mutate_or_relink(
    identity_db, provider, subject, email
):
    assert account_erasure.begin_erasure("owner") == PENDING

    with pytest.raises(HTTPException) as error:
        oauth_login(provider, subject, email, "Modified Owner")

    assert error.value.status_code == 409

    with identity_db() as db:
        owner = db.get(User, "owner")
        assert owner.full_name == "Original Owner"
        assert owner.email == "owner@example.com"
        assert owner.oauth_provider == "google"
        assert owner.provider_user_id == "owner_subject"
        assert db.query(User).count() == 2
        assert erasure_state(db, "owner") == PENDING


def test_pending_user_cannot_be_updated(identity_db):
    assert account_erasure.begin_erasure("owner") == PENDING

    with pytest.raises(HTTPException) as error:
        user_service.UserService().update_user(
            "owner", ValidationInterface.UserUpdate(full_name="Modified Owner")
        )

    assert error.value.status_code == 409
    with identity_db() as db:
        assert db.get(User, "owner").full_name == "Original Owner"
        assert erasure_state(db, "owner") == PENDING


@pytest.mark.parametrize("pending", [False, True])
def test_legacy_delete_cannot_discard_storage_locators(identity_db, pending):
    if pending:
        assert account_erasure.begin_erasure("owner") == PENDING

    with pytest.raises(HTTPException) as error:
        user_service.UserService().delete_user("owner")

    assert error.value.status_code == 409
    with identity_db() as db:
        assert db.get(User, "owner") is not None
        assert db.get(File, "private_file") is not None
        assert db.query(FileStorage).one().storage_path == "private/path"
        assert erasure_state(db, "owner") == (PENDING if pending else None)


def test_legacy_delete_missing_user_returns_404(identity_db):
    with pytest.raises(HTTPException) as error:
        user_service.UserService().delete_user("unknown")
    assert error.value.status_code == 404


@pytest.mark.parametrize("subject", ["unrelated_subject", "replacement_subject"])
def test_unrelated_oauth_identity_can_still_update(identity_db, subject):
    assert account_erasure.begin_erasure("owner") == PENDING

    oauth_login("google", subject, "unrelated@example.com", "Updated User")

    with identity_db() as db:
        unrelated = db.get(User, "unrelated")
        assert unrelated.full_name == "Updated User"
        assert unrelated.provider_user_id == subject
        assert db.query(User).count() == 2
        assert erasure_state(db, "owner") == PENDING
        assert erasure_state(db, "unrelated") is None


def test_unrelated_user_can_still_be_updated(identity_db):
    assert account_erasure.begin_erasure("owner") == PENDING

    user_service.UserService().update_user(
        "unrelated", ValidationInterface.UserUpdate(full_name="Updated User")
    )

    with identity_db() as db:
        assert db.get(User, "unrelated").full_name == "Updated User"
        assert db.get(User, "owner").full_name == "Original Owner"
        assert erasure_state(db, "owner") == PENDING
