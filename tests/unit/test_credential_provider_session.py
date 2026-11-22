"""Regression coverage for provider credentials and caller-owned resolution."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from projectdavid_orm import ProviderKey
from projectdavid_orm.projectdavid_orm.models import (
    Credential,
    McpServerRegistration,
    User,
)
from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
    CredentialResolutionError,
    CredentialService,
)
from src.api.entities_api.services.provider_key_service import ProviderKeyService


@pytest.fixture()
def credential_store():
    engine = create_engine("sqlite:///:memory:")
    User.__table__.create(bind=engine)
    Credential.__table__.create(bind=engine)
    McpServerRegistration.__table__.create(bind=engine)
    ProviderKey.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(User(id="provider_owner"))
        db.flush()
        db.add(User(id="other_owner"))
        db.commit()
    try:
        yield factory, Fernet.generate_key().decode("ascii")
    finally:
        engine.dispose()


def test_caller_owned_resolution_does_not_open_another_session(credential_store):
    factory, key = credential_store
    writer = CredentialService(session_factory=factory, key_provider=lambda: key)
    reference = writer.create(
        user_id="provider_owner",
        kind="inference_provider",
        payload={"api_key": "secret-value"},
    )

    def forbidden_session():
        pytest.fail("resolve_in_session must not open a second session")

    reader = CredentialService(
        session_factory=forbidden_session,
        key_provider=lambda: key,
    )
    with factory() as db:
        assert reader.resolve_in_session(
            db=db,
            credential_id=reference.id,
            user_id="provider_owner",
            expected_kind="inference_provider",
        ) == {"api_key": "secret-value"}

        with pytest.raises(CredentialResolutionError, match="Credential not found"):
            reader.resolve_in_session(
                db=db,
                credential_id=reference.id,
                user_id="other_owner",
                expected_kind="inference_provider",
            )


def test_provider_inference_resolution_reuses_locked_session(credential_store):
    factory, key = credential_store
    writer = CredentialService(session_factory=factory, key_provider=lambda: key)
    reference = writer.create(
        user_id="provider_owner",
        kind="inference_provider",
        payload={"api_key": "secret-value"},
    )
    with factory() as db:
        db.add(
            ProviderKey(
                id="pkey_owned",
                owner_id="provider_owner",
                credential_id=reference.id,
                provider="hyperbolic",
                fingerprint="a" * 64,
            )
        )
        db.commit()

    def forbidden_session():
        pytest.fail("ProviderKeyService must not open a nested credential session")

    reader = CredentialService(
        session_factory=forbidden_session,
        key_provider=lambda: key,
    )
    provider_service = ProviderKeyService(
        session_factory=factory,
        credential_service=reader,
    )
    assert (
        provider_service.resolve_for_inference(
            provider_key_id="pkey_owned",
            user_id="provider_owner",
            provider="hyperbolic",
        )
        == "secret-value"
    )


def test_startup_requires_key_for_live_inference_credential(credential_store):
    factory, key = credential_store
    writer = CredentialService(session_factory=factory, key_provider=lambda: key)
    writer.create(
        user_id="provider_owner",
        kind="inference_provider",
        payload={"api_key": "secret-value"},
    )
    without_key = CredentialService(session_factory=factory, key_provider=lambda: None)
    with pytest.raises(
        CredentialConfigurationError, match="PROJECT_DAVID_CREDENTIAL_KEY"
    ):
        without_key.validate_runtime_configuration()


def test_startup_does_not_require_key_for_retired_provider_credential(credential_store):
    factory, key = credential_store
    writer = CredentialService(session_factory=factory, key_provider=lambda: key)
    reference = writer.create(
        user_id="provider_owner",
        kind="inference_provider",
        payload={"api_key": "secret-value"},
    )
    with factory() as db:
        row = db.query(Credential).filter(Credential.id == reference.id).one()
        row.encrypted_payload = "revoked"
        db.commit()

    without_key = CredentialService(session_factory=factory, key_provider=lambda: None)
    without_key.validate_runtime_configuration()


def test_startup_without_encrypted_auth_does_not_require_key(credential_store):
    factory, _ = credential_store
    without_key = CredentialService(session_factory=factory, key_provider=lambda: None)
    without_key.validate_runtime_configuration()
