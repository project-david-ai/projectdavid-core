"""Research execution identities are not user-configurable inventory."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import JSON, Boolean, Column, Integer, MetaData, String, Table
from sqlalchemy.dialects import mysql
from sqlalchemy.engine.create import create_engine

from src.api.entities_api.services import assistants_service as service
from src.api.entities_api.utilities.assistant_manager import AssistantManager

OWNER = "user_inventory"
PLAYGROUND = service.playground_assistant_id(OWNER)


@pytest.fixture
def inventory(monkeypatch):
    # A projected schema suffices to execute the production summary query,
    # including both ownership branches and real SQL pagination.
    metadata = MetaData()
    a = Table(
        "assistants",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("name", String(128)),
        Column("description", String(255)),
        Column("model", String(64)),
        Column("created_at", Integer),
        Column("owner_id", String(64)),
        Column("deleted_at", Integer),
        Column("meta_data", JSON),
        *[
            Column(field, Boolean)
            for field in (
                "is_protected",
                "agent_mode",
                "web_access",
                "deep_research",
                "engineer",
                "decision_telemetry",
            )
        ],
    )
    assoc = Table(
        "assistant_users",
        metadata,
        Column("assistant_id", String(64)),
        Column("user_id", String(64)),
    )
    monkeypatch.setattr(
        service,
        "Assistant",
        SimpleNamespace(
            __table__=a,
            name=a.c.name,
            description=a.c.description,
            deep_research=a.c.deep_research,
            meta_data=a.c.meta_data,
            users=SimpleNamespace(property=SimpleNamespace(secondary=assoc)),
        ),
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    defaults = dict(
        name="User agent",
        description=None,
        model="test",
        created_at=1,
        owner_id=OWNER,
        deleted_at=None,
        meta_data=None,
        is_protected=False,
        agent_mode=False,
        web_access=False,
        deep_research=False,
        engineer=False,
        decision_telemetry=False,
    )

    def row(identifier, **changes):
        return dict(defaults, id=identifier) | changes

    with engine.begin() as db:
        db.execute(
            a.insert(),
            [
                row("asst_normal", created_at=5),
                row(
                    "asst_named_worker",
                    name="worker_12345678",
                    deep_research=True,
                    created_at=4,
                ),
                row(
                    "asst_named_description",
                    description="Temp research supervisor",
                    deep_research=True,
                    created_at=3,
                ),
                row(
                    "asst_false_flag",
                    meta_data={"is_research_ephemeral": "false"},
                    created_at=2,
                ),
                row("asst_legacy_visible", owner_id=None),
                row(
                    "asst_worker",
                    meta_data={"is_research_worker": "true"},
                    created_at=20,
                ),
                row(
                    "asst_old_worker",
                    meta_data={"research_worker_calling": "true"},
                    created_at=21,
                ),
                row(
                    "asst_supervisor",
                    meta_data={"is_research_ephemeral": "true"},
                    created_at=22,
                ),
                row(
                    "asst_old_supervisor",
                    name="worker_a1b2c3d4",
                    description="Temp research supervisor",
                    deep_research=True,
                    created_at=23,
                ),
                row(
                    "asst_legacy_worker",
                    owner_id=None,
                    meta_data={"is_research_worker": "true"},
                    created_at=24,
                ),
                row(PLAYGROUND, name="Playground", is_protected=True, created_at=6),
                row("asst_other", owner_id="other", created_at=30),
                row("asst_deleted", deleted_at=100, created_at=31),
            ],
        )
        db.execute(
            assoc.insert(),
            [
                dict(assistant_id="asst_legacy_visible", user_id=OWNER),
                dict(assistant_id="asst_legacy_worker", user_id=OWNER),
            ],
        )
        yield db
    engine.dispose()


def page(db, **filters):
    protected = filters.pop("protected", False)
    return service.AssistantService.list_summaries(
        db,
        OWNER,
        service.SummaryFilters(**filters),
        is_protected_key=protected,
    )


def test_research_exclusion_precedes_pagination(inventory):
    first = page(inventory, limit=2)
    assert [item.id for item in first.items] == ["asst_normal", "asst_named_worker"]
    assert first.has_more and first.next_cursor
    second = page(inventory, limit=2, cursor=first.next_cursor)
    assert [item.id for item in second.items] == [
        "asst_named_description",
        "asst_false_flag",
    ]
    third = page(inventory, limit=2, cursor=second.next_cursor)
    assert [item.id for item in third.items] == ["asst_legacy_visible"]
    assert not third.has_more and third.next_cursor is None


@pytest.mark.parametrize(
    "identifier",
    [
        "asst_worker",
        "asst_old_worker",
        "asst_supervisor",
        "asst_old_supervisor",
        "asst_legacy_worker",
    ],
)
def test_exact_id_cannot_reintroduce_internal_identity(inventory, identifier):
    result = page(
        inventory, assistant_id=identifier, protected=True, include_playground=True
    )
    assert result.items == [] and not result.has_more


def test_playground_opt_in_is_unchanged(inventory):
    assert PLAYGROUND not in [item.id for item in page(inventory).items]
    assert PLAYGROUND not in [
        item.id
        for item in page(inventory, protected=True, include_playground=False).items
    ]
    assert (
        PLAYGROUND
        == page(inventory, protected=True, include_playground=True).items[0].id
    )


def test_mysql_query_compiles(inventory):
    query = service.summary_query(OWNER, service.SummaryFilters(limit=2))
    sql = str(query.compile(dialect=mysql.dialect()))
    assert "JSON_EXTRACT" in sql and "REGEXP" in sql and "LIMIT" in sql


def test_research_factories_mark_new_identities():
    manager = AssistantManager()
    native = SimpleNamespace(create_assistant=AsyncMock())
    manager._native_exec_svc = native
    asyncio.run(manager.create_ephemeral_research_supervisor(OWNER))
    assert native.create_assistant.call_args.kwargs["meta_data"] == {
        "is_research_ephemeral": "true"
    }
    asyncio.run(manager.create_ephemeral_worker_assistant(OWNER))
    assert native.create_assistant.call_args.kwargs["meta_data"] == {
        "is_research_ephemeral": "true",
        "is_research_worker": "true",
    }
    asyncio.run(manager.create_primary_assistant(OWNER))
    assert "meta_data" not in native.create_assistant.call_args.kwargs
