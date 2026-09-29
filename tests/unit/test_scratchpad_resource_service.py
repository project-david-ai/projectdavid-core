from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from projectdavid_common import ValidationInterface
from projectdavid_orm import Scratchpad as OrmScratchpad
from projectdavid_orm.projectdavid_orm.base import Base
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import Scratchpad, Thread, User
from src.api.entities_api.services.scratchpad_resource_service import (
    ScratchpadResourceService,
)

validator = ValidationInterface()


def _thread(
    thread_id: str,
    owner_id: str,
) -> Thread:
    return Thread(
        id=thread_id,
        created_at=1,
        meta_data=json.dumps({}),
        object="thread",
        tool_resources=json.dumps({}),
        owner_id=owner_id,
    )


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
    )

    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Thread.__table__,
            Scratchpad.__table__,
        ],
    )

    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    with factory() as db:
        db.add_all(
            [
                User(id="user_1"),
                User(id="user_2"),
            ]
        )

        db.flush()

        db.add_all(
            [
                _thread(
                    "thread_1",
                    "user_1",
                ),
                _thread(
                    "thread_2",
                    "user_2",
                ),
                _thread(
                    "thread_3",
                    "user_1",
                ),
            ]
        )

        db.commit()

    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture()
def service(
    session_factory: sessionmaker,
) -> ScratchpadResourceService:
    return ScratchpadResourceService(
        session_factory=session_factory,
    )


def test_core_model_shim_exports_canonical_scratchpad() -> None:
    assert Scratchpad is OrmScratchpad


def test_create_derives_owner_from_authenticated_user(
    service: ScratchpadResourceService,
    session_factory: sessionmaker,
) -> None:
    result = service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_1",
        ),
        user_id="user_1",
    )

    assert result.owner_id == "user_1"
    assert result.thread_id == "thread_1"
    assert result.meta_data == {}

    with session_factory() as db:
        row = db.get(
            Scratchpad,
            result.id,
        )

        assert row is not None
        assert row.owner_id == "user_1"
        assert row.thread_id == "thread_1"


def test_create_rejects_cross_tenant_thread_as_not_found(
    service: ScratchpadResourceService,
    session_factory: sessionmaker,
) -> None:
    with pytest.raises(
        HTTPException,
    ) as exc_info:
        service.create_scratchpad(
            validator.ScratchpadCreate(
                thread_id="thread_2",
            ),
            user_id="user_1",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread not found"

    with session_factory() as db:
        count = (
            db.query(Scratchpad)
            .filter(
                Scratchpad.thread_id == "thread_2",
            )
            .count()
        )

        assert count == 0


def test_duplicate_scratchpad_for_thread_returns_conflict(
    service: ScratchpadResourceService,
) -> None:
    create = validator.ScratchpadCreate(
        thread_id="thread_1",
    )

    service.create_scratchpad(
        create,
        user_id="user_1",
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        service.create_scratchpad(
            create,
            user_id="user_1",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Scratchpad already exists for this thread"


def test_retrieve_is_owner_scoped_and_cross_tenant_is_404(
    service: ScratchpadResourceService,
) -> None:
    created = service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_1",
        ),
        user_id="user_1",
    )

    retrieved = service.retrieve_scratchpad(
        created.id,
        user_id="user_1",
    )

    assert retrieved.id == created.id

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        service.retrieve_scratchpad(
            created.id,
            user_id="user_2",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Scratchpad not found"


def test_retrieve_by_thread_requires_owned_parent(
    service: ScratchpadResourceService,
) -> None:
    created = service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_1",
        ),
        user_id="user_1",
    )

    resolved = service.retrieve_scratchpad_by_thread(
        "thread_1",
        user_id="user_1",
    )

    assert resolved.id == created.id

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        service.retrieve_scratchpad_by_thread(
            "thread_1",
            user_id="user_2",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread not found"


def test_list_is_strictly_owner_scoped(
    service: ScratchpadResourceService,
) -> None:
    own = service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_1",
        ),
        user_id="user_1",
    )

    service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_2",
        ),
        user_id="user_2",
    )

    listing = service.list_scratchpads(
        user_id="user_1",
    )

    assert listing.object == "list"
    assert [row.id for row in listing.data] == [own.id]
    assert all(row.owner_id == "user_1" for row in listing.data)


def test_update_replaces_metadata_on_owned_scratchpad(
    service: ScratchpadResourceService,
) -> None:
    created = service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_1",
        ),
        user_id="user_1",
    )

    updated = service.update_scratchpad(
        created.id,
        validator.ScratchpadUpdate(
            meta_data={
                "phase": "sql-resource",
                "version": 1,
            }
        ),
        user_id="user_1",
    )

    assert updated.meta_data == {
        "phase": "sql-resource",
        "version": 1,
    }

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        service.update_scratchpad(
            created.id,
            validator.ScratchpadUpdate(
                meta_data={
                    "phase": "cross-tenant",
                }
            ),
            user_id="user_2",
        )

    assert exc_info.value.status_code == 404


def test_delete_is_owner_scoped_and_typed(
    service: ScratchpadResourceService,
    session_factory: sessionmaker,
) -> None:
    created = service.create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_3",
        ),
        user_id="user_1",
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        service.delete_scratchpad(
            created.id,
            user_id="user_2",
        )

    assert exc_info.value.status_code == 404

    deleted = service.delete_scratchpad(
        created.id,
        user_id="user_1",
    )

    assert deleted.id == created.id
    assert deleted.object == "scratchpad.deleted"
    assert deleted.deleted is True

    with session_factory() as db:
        assert (
            db.get(
                Scratchpad,
                created.id,
            )
            is None
        )
