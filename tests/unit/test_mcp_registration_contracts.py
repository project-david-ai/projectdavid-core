from pathlib import Path

from projectdavid_common import ValidationInterface
from projectdavid_orm.projectdavid_orm.models import (
    AssistantMcpTool,
    McpServerRegistration,
)
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable


def _unique_column_sets(table):
    return {
        frozenset(constraint.columns.keys())
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def test_mcp_registration_is_idempotent_per_owner_and_endpoint_identity():
    assert frozenset({"owner_id", "identity_key"}) in _unique_column_sets(
        McpServerRegistration.__table__
    )


def test_assistant_mcp_attachment_has_both_idempotency_guards():
    unique_sets = _unique_column_sets(AssistantMcpTool.__table__)

    assert frozenset({"assistant_id", "registration_id", "remote_name"}) in unique_sets

    assert frozenset({"assistant_id", "provider_name"}) in unique_sets


def test_mcp_models_compile_for_mysql():
    registration_ddl = str(
        CreateTable(McpServerRegistration.__table__).compile(dialect=mysql.dialect())
    )
    attachment_ddl = str(
        CreateTable(AssistantMcpTool.__table__).compile(dialect=mysql.dialect())
    )

    assert "mcp_server_registrations" in registration_ddl
    assert "assistant_mcp_tools" in attachment_ddl


def test_validation_interface_exposes_mcp_contracts():
    assert ValidationInterface.McpServerRegistrationCreate is not None
    assert ValidationInterface.McpServerRegistrationRead is not None
    assert ValidationInterface.McpServerRegistrationUpdate is not None
    assert ValidationInterface.AssistantMcpToolsAttach is not None
    assert ValidationInterface.AssistantMcpToolsDetach is not None
    assert ValidationInterface.AssistantMcpToolRead is not None


def test_mcp5a_migration_extends_current_head():
    migration = Path(
        "migrations/versions/" "8c1e9d4a5b72_add_mcp_registration_tables.py"
    ).read_text(encoding="utf-8")

    assert 'revision: str = "8c1e9d4a5b72"' in migration
    assert 'down_revision: str | None = "371df74151b3"' in migration


def test_mcp5a_persistence_contains_no_credential_columns():
    registration_columns = set(McpServerRegistration.__table__.columns.keys())

    forbidden = {
        "credential",
        "credentials",
        "credential_ref",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "secret",
    }

    assert registration_columns.isdisjoint(forbidden)
