from __future__ import annotations

import json

import pytest
from cryptography.fernet import Fernet
from projectdavid_orm.projectdavid_orm.models import Credential, User
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
    CredentialResolutionError,
    CredentialService,
)


@pytest.fixture()
def credential_context():
    engine = create_engine("sqlite:///:memory:")

    User.__table__.create(bind=engine)
    Credential.__table__.create(bind=engine)

    factory = sessionmaker(bind=engine)

    with factory() as db:
        db.add(User(id="user_a"))
        db.flush()
        db.add(User(id="user_b"))
        db.commit()

    key = Fernet.generate_key().decode("ascii")

    service = CredentialService(
        session_factory=factory,
        key_provider=lambda: key,
    )

    try:
        yield service, factory, key
    finally:
        engine.dispose()


def test_credential_plaintext_is_not_persisted(
    credential_context,
):
    service, factory, _ = credential_context

    row = service.create(
        user_id="user_a",
        kind="bearer",
        payload={"token": "super-secret-token"},
    )

    with factory() as db:
        stored = db.query(Credential).filter(Credential.id == row.id).one()

        assert "super-secret-token" not in stored.encrypted_payload
        assert stored.owner_id == "user_a"
        assert stored.kind == "bearer"
        assert stored.encryption_version == 1


def test_credential_round_trip(
    credential_context,
):
    service, _, _ = credential_context

    row = service.create(
        user_id="user_a",
        kind="bearer",
        payload={"token": "abc123"},
    )

    payload = service.resolve(
        credential_id=row.id,
        user_id="user_a",
        expected_kind="bearer",
    )

    assert payload == {"token": "abc123"}


def test_cross_tenant_resolution_fails_closed(
    credential_context,
):
    service, _, _ = credential_context

    row = service.create(
        user_id="user_a",
        kind="bearer",
        payload={"token": "abc123"},
    )

    with pytest.raises(
        CredentialResolutionError,
        match="Credential not found",
    ):
        service.resolve(
            credential_id=row.id,
            user_id="user_b",
        )


def test_wrong_credential_kind_fails_closed(
    credential_context,
):
    service, _, _ = credential_context

    row = service.create(
        user_id="user_a",
        kind="bearer",
        payload={"token": "abc123"},
    )

    with pytest.raises(
        CredentialResolutionError,
        match="Credential kind mismatch",
    ):
        service.resolve(
            credential_id=row.id,
            user_id="user_a",
            expected_kind="oauth2",
        )


def test_tampered_ciphertext_fails_closed(
    credential_context,
):
    service, factory, _ = credential_context

    row = service.create(
        user_id="user_a",
        kind="bearer",
        payload={"token": "abc123"},
    )

    with factory() as db:
        stored = db.query(Credential).filter(Credential.id == row.id).one()

        stored.encrypted_payload = stored.encrypted_payload[:-4] + "AAAA"
        db.commit()

    with pytest.raises(
        CredentialResolutionError,
        match="Credential decryption failed",
    ):
        service.resolve(
            credential_id=row.id,
            user_id="user_a",
        )


def test_missing_root_key_fails_closed(
    credential_context,
):
    _, factory, _ = credential_context

    service = CredentialService(
        session_factory=factory,
        key_provider=lambda: None,
    )

    with pytest.raises(
        CredentialConfigurationError,
        match="PROJECT_DAVID_CREDENTIAL_KEY",
    ):
        service.create(
            user_id="user_a",
            kind="bearer",
            payload={"token": "abc123"},
        )


def test_payload_is_valid_json_before_encryption(
    credential_context,
):
    service, factory, key = credential_context

    row = service.create(
        user_id="user_a",
        kind="oauth2",
        payload={
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 1234567890,
        },
    )

    with factory() as db:
        stored = db.query(Credential).filter(Credential.id == row.id).one()

    decoded = Fernet(key.encode("ascii")).decrypt(
        stored.encrypted_payload.encode("ascii")
    )

    assert json.loads(decoded.decode("utf-8")) == {
        "access_token": "access",
        "expires_at": 1234567890,
        "refresh_token": "refresh",
    }
