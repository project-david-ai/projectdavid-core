from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from projectdavid_common import ValidationInterface

from src.api.entities_api.services.scratchpad_service import ScratchpadService

validator = ValidationInterface()


class FakeResourceService:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)

        self.resource = validator.ScratchpadRead(
            id="scratchpad_1",
            owner_id="user_1",
            thread_id="thread_1",
            created_at=now,
            updated_at=now,
            meta_data={},
        )

        self.ensure_calls = []
        self.deleted = []

    def create_scratchpad(
        self,
        scratchpad,
        *,
        user_id,
    ):
        return self.resource

    def retrieve_scratchpad(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        assert scratchpad_id == "scratchpad_1"
        assert user_id == "user_1"

        return self.resource

    def retrieve_scratchpad_by_thread(
        self,
        thread_id,
        *,
        user_id,
    ):
        assert thread_id == "thread_1"
        assert user_id == "user_1"

        return self.resource

    def ensure_scratchpad_for_thread(
        self,
        thread_id,
        *,
        user_id,
    ):
        self.ensure_calls.append(
            (
                thread_id,
                user_id,
            )
        )

        assert thread_id == "thread_1"
        assert user_id == "user_1"

        return self.resource

    def list_scratchpads(
        self,
        *,
        user_id,
    ):
        return validator.ScratchpadList(data=[self.resource])

    def update_scratchpad(
        self,
        scratchpad_id,
        scratchpad,
        *,
        user_id,
    ):
        return self.resource

    def delete_record(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        self.deleted.append(
            (
                scratchpad_id,
                user_id,
            )
        )

        return validator.ScratchpadDeleted(
            id=scratchpad_id,
        )


class FakeCache:
    def __init__(self) -> None:
        self.content = {}
        self.entries = []
        self.legacy = {}
        self.calls = []

    async def get_content(
        self,
        *,
        owner_id,
        scratchpad_id,
    ):
        self.calls.append(
            (
                "get_content",
                owner_id,
                scratchpad_id,
            )
        )

        return dict(
            self.content
            or {
                "content": "",
                "updated_at": None,
            }
        )

    async def set_content(
        self,
        *,
        owner_id,
        scratchpad_id,
        content,
    ):
        self.calls.append(
            (
                "set_content",
                owner_id,
                scratchpad_id,
                content,
            )
        )

        self.content = {
            "content": content,
            "updated_at": 100.0,
        }

        return dict(self.content)

    async def append_entry(
        self,
        *,
        owner_id,
        scratchpad_id,
        content,
    ):
        self.calls.append(
            (
                "append_entry",
                owner_id,
                scratchpad_id,
                content,
            )
        )

        row = {
            "content": content,
            "created_at": (200.0 + len(self.entries)),
        }

        self.entries.append(row)

        return dict(row)

    async def list_entries(
        self,
        *,
        owner_id,
        scratchpad_id,
    ):
        self.calls.append(
            (
                "list_entries",
                owner_id,
                scratchpad_id,
            )
        )

        return [dict(row) for row in self.entries]

    async def clear_content(
        self,
        *,
        owner_id,
        scratchpad_id,
    ):
        self.content = {}

    async def clear_entries(
        self,
        *,
        owner_id,
        scratchpad_id,
    ):
        self.entries = []

    async def delete_all_scratchpad_data(
        self,
        *,
        owner_id,
        scratchpad_id,
    ):
        self.calls.append(
            (
                "delete_all",
                owner_id,
                scratchpad_id,
            )
        )

        self.content = {}
        self.entries = []

    async def get_scratchpad(
        self,
        thread_id,
    ):
        self.calls.append(
            (
                "legacy_get",
                thread_id,
            )
        )

        return dict(
            self.legacy.get(
                thread_id,
                {
                    "content": "",
                    "last_updated": 0,
                },
            )
        )

    async def clear_scratchpad(
        self,
        thread_id,
    ):
        self.calls.append(
            (
                "legacy_clear",
                thread_id,
            )
        )

        self.legacy.pop(
            thread_id,
            None,
        )


def build_service():
    cache = FakeCache()
    resources = FakeResourceService()

    return (
        ScratchpadService(
            cache=cache,
            resource_service=resources,
        ),
        cache,
        resources,
    )


def test_content_and_entries_are_distinct_planes() -> None:
    service, cache, _ = build_service()

    async def exercise():
        await service.set_content(
            "scratchpad_1",
            "PLAN",
            user_id="user_1",
        )

        await service.append_entry(
            "scratchpad_1",
            "FINDING",
            user_id="user_1",
        )

        return await service.get_formatted_view(
            "scratchpad_1",
            user_id="user_1",
        )

    rendered = asyncio.run(exercise())

    assert cache.content["content"] == "PLAN"

    assert [row["content"] for row in cache.entries] == ["FINDING"]

    assert "WORKING BODY" in rendered
    assert "PLAN" in rendered
    assert "APPENDED ENTRIES" in rendered
    assert "[1] FINDING" in rendered


def test_append_note_uses_entry_plane() -> None:
    service, cache, resources = build_service()

    result = asyncio.run(
        service.append_note(
            "thread_1",
            "worker finding",
            user_id="user_1",
        )
    )

    assert result == "Note appended successfully."

    assert resources.ensure_calls == [
        (
            "thread_1",
            "user_1",
        )
    ]

    assert [row["content"] for row in cache.entries] == ["worker finding"]


def test_update_content_preserves_entries() -> None:
    service, cache, _ = build_service()

    cache.entries = [
        {
            "content": "worker finding",
            "created_at": 200.0,
        }
    ]

    asyncio.run(
        service.update_content(
            "thread_1",
            "new plan",
            user_id="user_1",
        )
    )

    assert cache.content["content"] == "new plan"

    assert [row["content"] for row in cache.entries] == ["worker finding"]


def test_clear_preserves_sql_resource() -> None:
    service, cache, resources = build_service()

    cache.content = {
        "content": "plan",
        "updated_at": 100.0,
    }

    cache.entries = [
        {
            "content": "finding",
            "created_at": 200.0,
        }
    ]

    result = asyncio.run(
        service.clear_scratchpad(
            "scratchpad_1",
            user_id="user_1",
        )
    )

    assert result.scope == "all"
    assert result.cleared is True

    assert cache.content == {}
    assert cache.entries == []
    assert resources.deleted == []


def test_delete_purges_redis_before_sql() -> None:
    service, cache, resources = build_service()

    order = []

    original_purge = cache.delete_all_scratchpad_data

    async def purge(**kwargs):
        order.append("redis")

        await original_purge(**kwargs)

    cache.delete_all_scratchpad_data = purge

    original_delete = resources.delete_record

    def delete(*args, **kwargs):
        order.append("sql")

        return original_delete(
            *args,
            **kwargs,
        )

    resources.delete_record = delete

    result = asyncio.run(
        service.delete_scratchpad(
            "scratchpad_1",
            user_id="user_1",
        )
    )

    assert result.deleted is True

    assert order == [
        "redis",
        "sql",
    ]


def test_legacy_thread_state_migrates_once() -> None:
    service, cache, _ = build_service()

    cache.legacy["thread_1"] = {
        "content": "legacy plan",
        "last_updated": 50.0,
    }

    rendered = asyncio.run(
        service.get_formatted_view_for_thread(
            "thread_1",
            user_id="user_1",
        )
    )

    assert cache.content["content"] == "legacy plan"

    assert "legacy plan" in rendered
    assert "thread_1" not in cache.legacy

    operations = [call[0] for call in cache.calls]

    assert "legacy_get" in operations
    assert "legacy_clear" in operations
