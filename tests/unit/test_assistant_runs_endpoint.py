import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.api.entities_api.services.runs_service as runs_module
from src.api.entities_api.models.models import Run
from src.api.entities_api.services.runs_service import RunService


@pytest.fixture
def history(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Run.__table__.create(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(runs_module, "SessionLocal", sessions)

    with sessions() as db:
        for run_id, user_id, assistant_id, created_at in [
            ("run_a", "user_a", "asst_a", 100),
            ("run_b", "user_a", "asst_a", 100),
            ("run_c", "user_a", "asst_a", 200),
            ("run_foreign", "user_b", "asst_a", 400),
            ("run_other", "user_a", "asst_b", 300),
        ]:
            db.execute(
                insert(Run.__table__).values(
                    id=run_id,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    thread_id="thread_a",
                    created_at=created_at,
                    status="completed",
                    object="thread.run",
                    model="test-model",
                    instructions="Test instructions",
                )
            )
        db.commit()

    try:
        yield
    finally:
        engine.dispose()


@pytest.fixture
def api(history):
    # Load this router without importing the unrelated inference routers.
    path = (
        Path(__file__).resolve().parents[2]
        / "src/api/entities_api/routers/runs_router.py"
    )
    spec = importlib.util.spec_from_file_location("runs_router_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    app = FastAPI()
    app.include_router(module.router, prefix="/v1")
    app.dependency_overrides[module.get_api_key] = lambda: SimpleNamespace(
        user_id="user_a"
    )
    app.dependency_overrides[module.get_db] = lambda: None

    with TestClient(app) as client:
        yield client, app, module


@pytest.mark.parametrize(
    "order,expected",
    [
        ("desc", ["run_c", "run_b", "run_a"]),
        ("asc", ["run_a", "run_b", "run_c"]),
    ],
)
def test_service_lists_only_users_assistant_runs(history, order, expected):
    service = RunService()

    first, has_more = service.list_runs_by_assistant_id(
        "asst_a",
        user_id="user_a",
        limit=2,
        order=order,
    )

    assert [run.id for run in first] == expected[:2]
    assert has_more is True

    second, has_more = service.list_runs_by_assistant_id(
        "asst_a",
        user_id="user_a",
        limit=2,
        order=order,
        after=first[-1].id,
    )

    assert [run.id for run in second] == expected[2:]
    assert has_more is False


def test_endpoint_returns_scoped_paginated_envelope(api):
    client, _, _ = api
    url = "/v1/assistants/asst_a/runs"

    # Supplying another user_id must not override the authenticated user.
    response = client.get(
        url,
        params={"limit": 2, "user_id": "user_b"},
    )

    assert response.status_code == 200
    page = response.json()

    assert page["object"] == "list"
    assert [run["id"] for run in page["data"]] == [
        "run_c",
        "run_b",
    ]
    assert all(run["user_id"] == "user_a" for run in page["data"])
    assert page["first_id"] == "run_c"
    assert page["last_id"] == "run_b"
    assert page["has_more"] is True

    response = client.get(
        url,
        params={"limit": 2, "after": page["last_id"]},
    )

    assert response.status_code == 200
    page = response.json()

    assert [run["id"] for run in page["data"]] == ["run_a"]
    assert page["first_id"] == page["last_id"] == "run_a"
    assert page["has_more"] is False


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"order": "random"},
        {"after": ""},
        {"after": "run_foreign"},
        {"after": "run_other"},
        {"after": "missing"},
    ],
)
def test_endpoint_rejects_invalid_pagination(api, params):
    client, _, _ = api

    response = client.get(
        "/v1/assistants/asst_a/runs",
        params=params,
    )

    assert response.status_code == 422


def test_endpoint_empty_history(api):
    client, _, _ = api

    response = client.get("/v1/assistants/asst_empty/runs")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }


def test_endpoint_requires_authentication(api):
    client, app, module = api
    del app.dependency_overrides[module.get_api_key]

    response = client.get("/v1/assistants/asst_a/runs")

    assert response.status_code == 401


def test_endpoint_sanitizes_unexpected_errors(api, monkeypatch):
    client, _, _ = api

    def fail(*args, **kwargs):
        raise RuntimeError("private database details")

    monkeypatch.setattr(
        RunService,
        "list_runs_by_assistant_id",
        fail,
    )

    response = client.get("/v1/assistants/asst_a/runs")

    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected error occurred."}
