"""Regression for MySQL sibling cascade ordering and tenant boundaries."""

import pytest
from fastapi import HTTPException
from sqlalchemy import Column, ForeignKey, MetaData, String, Table, event, select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session

from src.api.entities_api.services.account_erasure import (
    _check_credential_boundary,
    _erase_owned_credentials,
)


@pytest.fixture
def graph():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    metadata = MetaData()
    credentials = Table(
        "credentials",
        metadata,
        Column("id", String, primary_key=True),
        Column("owner_id", String, nullable=False),
    )
    registrations = Table(
        "mcp_server_registrations",
        metadata,
        Column("id", String, primary_key=True),
        Column("owner_id", String, nullable=False),
        Column(
            "credential_id", String, ForeignKey("credentials.id", ondelete="RESTRICT")
        ),
    )
    tools = Table(
        "assistant_mcp_tools",
        metadata,
        Column("id", String, primary_key=True),
        Column(
            "registration_id",
            String,
            ForeignKey("mcp_server_registrations.id", ondelete="CASCADE"),
        ),
    )
    metadata.create_all(engine)
    try:
        with Session(engine) as db:
            db.execute(
                credentials.insert(),
                [
                    {"id": "target-cred", "owner_id": "target"},
                    {"id": "control-cred", "owner_id": "control"},
                ],
            )
            db.execute(
                registrations.insert(),
                [
                    {
                        "id": "target-reg",
                        "owner_id": "target",
                        "credential_id": "target-cred",
                    },
                    {
                        "id": "control-reg",
                        "owner_id": "control",
                        "credential_id": "control-cred",
                    },
                ],
            )
            db.execute(
                tools.insert(),
                [
                    {"id": "target-tool", "registration_id": "target-reg"},
                    {"id": "control-tool", "registration_id": "control-reg"},
                ],
            )
            db.commit()
            yield db, metadata.tables
    finally:
        engine.dispose()


def test_ordered_erasure_preserves_control_and_is_repeatable(graph):
    db, tables = graph
    for _ in range(2):
        _check_credential_boundary(db, tables, "target")
        _erase_owned_credentials(db, tables, "target")
        db.commit()
    for name, expected in (
        ("credentials", "control-cred"),
        ("mcp_server_registrations", "control-reg"),
        ("assistant_mcp_tools", "control-tool"),
    ):
        assert db.scalars(select(tables[name].c.id)).all() == [expected]


def test_cross_account_reference_blocks_before_deletion(graph):
    db, tables = graph
    registrations = tables["mcp_server_registrations"]
    db.execute(
        registrations.insert().values(
            id="foreign-reg", owner_id="control", credential_id="target-cred"
        )
    )
    db.commit()
    with pytest.raises(HTTPException) as error:
        _check_credential_boundary(db, tables, "target")
    assert error.value.status_code == 409
    assert set(db.scalars(select(tables["credentials"].c.id))) == {
        "target-cred",
        "control-cred",
    }
    assert set(db.scalars(select(registrations.c.id))) == {
        "target-reg",
        "control-reg",
        "foreign-reg",
    }


def test_target_registration_does_not_delete_foreign_credential(graph):
    db, tables = graph
    registrations = tables["mcp_server_registrations"]
    db.execute(
        registrations.update()
        .where(registrations.c.id == "target-reg")
        .values(credential_id="control-cred")
    )
    _check_credential_boundary(db, tables, "target")
    _erase_owned_credentials(db, tables, "target")
    db.commit()
    assert db.scalars(select(tables["credentials"].c.id)).all() == ["control-cred"]
