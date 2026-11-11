import base64
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import Index, event, insert, text
from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import Run
from src.api.entities_api.services.run_activity_service import (
    list_agent_runs,
    retrieve_agent_run,
)


@pytest.fixture
def db():
    # Bypass the repository's global create_engine mock: exercise real SQL.
    engine = create_engine("sqlite://")
    Run.__table__.create(engine)
    existing_index = next(
        (
            i
            for i in Run.__table__.indexes
            if i.name == "ix_runs_user_assistant_created_id"
        ),
        None,
    )
    index = existing_index or Index(
        "ix_runs_user_assistant_created_id",
        Run.user_id,
        Run.assistant_id,
        Run.created_at,
        Run.id,
    )
    index.create(engine, checkfirst=True)
    with Session(engine) as session:
        yield session
    if existing_index is None:
        Run.__table__.indexes.remove(index)
    engine.dispose()


def add(db, id, *, user="user_a", agent="asst_a", created=100, **values):
    db.execute(
        insert(Run.__table__).values(
            id=id,
            user_id=user,
            assistant_id=agent,
            thread_id="thread_a",
            created_at=created,
            status="completed",
            object="thread.run",
            **values,
        )
    )
    db.commit()


def test_filters_before_limit_and_does_not_fetch_configuration(db):
    add(db, "run_ours", instructions="private configuration", tools=[{"a": 1}])
    add(db, "run_other_user", user="user_b", created=300)
    add(db, "run_other_agent", agent="asst_b", created=200)
    statements = []
    event.listen(
        db.bind, "before_cursor_execute", lambda c, cur, sql, *a: statements.append(sql)
    )
    page = list_agent_runs(db, "user_a", "asst_a", limit=1)
    assert [r.id for r in page.items] == ["run_ours"]
    assert not page.has_more
    sql = statements[-1].lower()
    assert "instructions" not in sql and "tools" not in sql and "meta_data" not in sql
    assert "runs.user_id =" in sql and "runs.assistant_id =" in sql
    assert "limit" in sql


def test_keyset_ties_and_null_timestamps_have_no_duplicates_or_gaps(db):
    for id in ["run_a", "run_b", "run_c"]:
        add(db, id)
    add(db, "run_old", created=50)
    add(db, "run_null_a", created=None)
    add(db, "run_null_b", created=None)
    seen = []
    cursor = None
    for _ in range(10):
        page = list_agent_runs(db, "user_a", "asst_a", limit=2, cursor=cursor)
        seen.extend(r.id for r in page.items)
        cursor = page.next_cursor
        if not page.has_more:
            break
    assert seen == ["run_c", "run_b", "run_a", "run_old", "run_null_b", "run_null_a"]
    assert cursor is None


def test_inserting_newer_run_does_not_repeat_previous_page(db):
    for id in ["run_a", "run_b", "run_c"]:
        add(db, id)
    first = list_agent_runs(db, "user_a", "asst_a", limit=2)
    add(db, "run_new", created=200)
    second = list_agent_runs(db, "user_a", "asst_a", limit=2, cursor=first.next_cursor)
    assert [r.id for r in second.items] == ["run_a"]
    assert list_agent_runs(db, "user_a", "asst_a").items[0].id == "run_new"


@pytest.mark.parametrize("user,agent", [("user_b", "asst_a"), ("user_a", "asst_b")])
def test_cursor_cannot_cross_scope(db, user, agent):
    add(db, "run_a")
    add(db, "run_b")
    cursor = list_agent_runs(db, "user_a", "asst_a", limit=1).next_cursor
    with pytest.raises(HTTPException) as error:
        list_agent_runs(db, user, agent, cursor=cursor)
    assert error.value.status_code == 422


@pytest.mark.parametrize("cursor", ["!", "A", "e30", "x" * 1025])
def test_rejects_invalid_cursors(db, cursor):
    with pytest.raises(HTTPException) as error:
        list_agent_runs(db, "user_a", "asst_a", cursor=cursor)
    assert error.value.status_code == 422


def test_forged_position_does_not_change_authorization(db):
    add(db, "run_a")
    add(db, "run_b", user="user_b")
    cursor = base64.urlsafe_b64encode(
        json.dumps(
            {
                "version": 1,
                "user": "user_a",
                "agent": "asst_a",
                "created": 999,
                "id": "run_z",
            }
        ).encode()
    ).decode()
    assert [
        r.id for r in list_agent_runs(db, "user_a", "asst_a", cursor=cursor).items
    ] == ["run_a"]


@pytest.mark.parametrize("user,agent", [("user_b", "asst_a"), ("user_a", "asst_b")])
def test_detail_requires_both_owner_and_agent(db, user, agent):
    add(db, "run_a")
    with pytest.raises(HTTPException) as error:
        retrieve_agent_run(db, user, agent, "run_a")
    assert error.value.status_code == 404


@pytest.mark.parametrize(
    "usage,expected",
    [
        (None, None),
        ({}, None),
        ({"total_tokens": 0}, 0),
        ({"total_tokens": 23}, 23),
        ({"total_tokens": True}, None),
        ({"total_tokens": -1}, None),
        ({"total_tokens": "12"}, None),
    ],
)
def test_does_not_turn_missing_usage_into_zero(db, usage, expected):
    add(db, "run_a", usage=usage)
    result = retrieve_agent_run(db, "user_a", "asst_a", "run_a")
    assert result.total_tokens == expected
    assert list_agent_runs(db, "user_a", "asst_a").items[0].total_tokens == expected
    assert "actions" not in result.model_dump()


def test_empty_history(db):
    assert list_agent_runs(db, "user_a", "asst_a").model_dump() == {
        "items": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_sqlite_plan_uses_composite_index_without_sort(db):
    plan = db.execute(
        text(
            "EXPLAIN QUERY PLAN SELECT id, created_at FROM runs "
            "WHERE user_id = 'user_a' AND assistant_id = 'asst_a' "
            "ORDER BY created_at DESC, id DESC LIMIT 21"
        )
    ).all()
    detail = " ".join(str(row) for row in plan)
    assert "ix_runs_user_assistant_created_id" in detail
    assert "TEMP B-TREE" not in detail
