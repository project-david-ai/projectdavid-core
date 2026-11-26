"""Account quarantine and credential issuance regression tests."""

import pytest
from fastapi import HTTPException
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import ApiKey, User
from src.api.entities_api.services import account_erasure
from src.api.entities_api.services.admin_provisioning_service import (
    AdminProvisioningService,
)
from src.api.entities_api.services.api_key_service import ApiKeyService
from src.api.entities_api.services.erasure_state import (
    COMPLETE,
    PENDING,
    erasure_state,
    erasures,
    subject_hash,
)


@pytest.fixture
def quarantine_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")

    User.__table__.create(engine)
    ApiKey.__table__.create(engine)
    erasures.create(engine)

    factory = sessionmaker(engine)

    monkeypatch.setattr(
        account_erasure,
        "SessionLocal",
        factory,
    )

    with factory() as db:
        db.add_all(
            [
                User(id="owner"),
                User(id="unrelated"),
            ]
        )
        db.commit()

    try:
        yield factory
    finally:
        engine.dispose()


def test_quarantine_revokes_all_credentials(quarantine_db):
    with quarantine_db() as db:
        service = ApiKeyService(db)

        _, ordinary = service.create_key("owner")
        ordinary_prefix = ordinary.prefix

        _, protected = service.create_key("owner")
        protected_prefix = protected.prefix

        service.protect_existing_key(
            "owner",
            protected_prefix,
        )

        _, unrelated = service.create_key("unrelated")
        unrelated_prefix = unrelated.prefix

    assert account_erasure.begin_erasure("owner") == PENDING

    with quarantine_db() as db:
        assert erasure_state(db, "owner") == PENDING
        assert erasure_state(db, "unrelated") is None

        keys = {key.prefix: key for key in db.query(ApiKey).all()}

        assert not keys[ordinary_prefix].is_active
        assert not keys[protected_prefix].is_active
        assert keys[protected_prefix].is_protected
        assert keys[unrelated_prefix].is_active

        # Quarantine alone must not delete the user.
        assert db.get(User, "owner") is not None


def test_pending_account_cannot_receive_new_key(quarantine_db):
    account_erasure.begin_erasure("owner")

    with quarantine_db() as db:
        with pytest.raises(HTTPException) as error:
            ApiKeyService(db).create_key("owner")

        assert error.value.status_code == 409
        assert db.query(ApiKey).count() == 0


def test_pending_account_cannot_protect_key(quarantine_db):
    with quarantine_db() as db:
        _, key = ApiKeyService(db).create_key("owner")
        prefix = key.prefix

    account_erasure.begin_erasure("owner")

    with quarantine_db() as db:
        with pytest.raises(HTTPException) as error:
            ApiKeyService(db).protect_existing_key(
                "owner",
                prefix,
            )

        assert error.value.status_code == 409


def test_quarantine_is_idempotent(quarantine_db):
    assert account_erasure.begin_erasure("owner") == PENDING

    with quarantine_db() as db:
        first = db.execute(erasures.select()).mappings().one()

    assert account_erasure.begin_erasure("owner") == PENDING

    with quarantine_db() as db:
        second = db.execute(erasures.select()).mappings().one()

        assert dict(first) == dict(second)


def test_missing_user_does_not_imply_completion(quarantine_db):
    with pytest.raises(HTTPException) as error:
        account_erasure.begin_erasure("unknown")

    assert error.value.status_code == 404

    with quarantine_db() as db:
        assert erasure_state(db, "unknown") is None


def test_completed_receipt_confirms_previous_erasure(quarantine_db):
    with quarantine_db() as db:
        db.execute(
            erasures.insert().values(
                subject_hash=subject_hash("previously_erased"),
                state=COMPLETE,
                created_at=100,
                completed_at=200,
            )
        )
        db.commit()

    assert account_erasure.begin_erasure("previously_erased") == COMPLETE


def test_unrelated_user_can_still_receive_keys(quarantine_db):
    account_erasure.begin_erasure("owner")

    with quarantine_db() as db:
        plain_key, key = ApiKeyService(db).create_key("unrelated")

        assert plain_key
        assert key.is_active
        assert key.user_id == "unrelated"


# ---------------------------------------------------------------------
# Administrative provisioning quarantine
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "email,external_reference",
    [
        ("owner@example.com", "external_owner"),
        ("changed@example.com", "external_owner"),
        ("owner@example.com", "changed_reference"),
    ],
    ids=[
        "matching-identity",
        "matching-external-reference",
        "matching-email",
    ],
)
def test_pending_account_cannot_be_provisioned(
    quarantine_db,
    email,
    external_reference,
):
    # Establish an existing externally mapped identity.
    with quarantine_db() as db:
        owner = db.get(User, "owner")
        owner.email = "owner@example.com"
        owner.email_verified = True
        owner.full_name = "Original Owner"
        owner.oauth_provider = AdminProvisioningService.PROVIDER
        owner.provider_user_id = "external_owner"
        db.commit()

        # Establish existing credentials so that we can verify
        # provisioning does not rotate or reactivate them.
        service = ApiKeyService(db)

        _, ordinary = service.create_key(
            "owner",
            key_name="project-black-bird",
        )
        ordinary_prefix = ordinary.prefix

        _, protected = service.create_key(
            "owner",
            key_name="project-black-bird",
        )
        protected_prefix = protected.prefix

        service.protect_existing_key(
            "owner",
            protected_prefix,
        )

    assert account_erasure.begin_erasure("owner") == PENDING

    with quarantine_db() as db:
        service = AdminProvisioningService(db)

        with pytest.raises(HTTPException) as error:
            service.ensure_user_api_key(
                email=email,
                full_name="Modified Owner",
                external_reference=external_reference,
                key_name="project-black-bird",
            )

        assert error.value.status_code == 409

    # Use an independent session to verify the persisted state.
    with quarantine_db() as db:
        owner = db.get(User, "owner")

        assert owner is not None
        assert owner.full_name == "Original Owner"
        assert owner.email == "owner@example.com"
        assert owner.email_verified is True
        assert owner.oauth_provider == AdminProvisioningService.PROVIDER
        assert owner.provider_user_id == "external_owner"

        assert erasure_state(db, "owner") == PENDING
        assert db.query(User).count() == 2

        keys = {key.prefix: key for key in db.query(ApiKey).all()}

        # No new credentials may be issued.
        assert set(keys) == {
            ordinary_prefix,
            protected_prefix,
        }

        # No existing credential may be reactivated.
        assert not keys[ordinary_prefix].is_active
        assert not keys[protected_prefix].is_active

        # Credential classification must remain unchanged.
        assert not keys[ordinary_prefix].is_protected
        assert keys[protected_prefix].is_protected


def test_unrelated_account_can_still_be_provisioned(
    quarantine_db,
):
    assert account_erasure.begin_erasure("owner") == PENDING

    with quarantine_db() as db:
        unrelated = db.get(User, "unrelated")
        unrelated.email = "unrelated@example.com"
        unrelated.oauth_provider = AdminProvisioningService.PROVIDER
        unrelated.provider_user_id = "external_unrelated"
        db.commit()

        result = AdminProvisioningService(db).ensure_user_api_key(
            email="unrelated@example.com",
            full_name="Unrelated User",
            external_reference="external_unrelated",
            key_name="project-black-bird",
        )

        assert result.user.id == "unrelated"
        assert result.user_created is False
        assert result.api_key.is_active is True
        assert result.api_key.is_protected is False
        assert result.api_key.user_id == "unrelated"
        assert result.plain_key

        prefix = result.api_key.prefix

    with quarantine_db() as db:
        assert erasure_state(db, "owner") == PENDING
        assert erasure_state(db, "unrelated") is None
        assert db.query(User).count() == 2

        key = db.query(ApiKey).filter(ApiKey.prefix == prefix).one()

        assert key.user_id == "unrelated"
        assert key.is_active is True
        assert key.is_protected is False

        assert db.query(ApiKey).filter(ApiKey.user_id == "owner").count() == 0
