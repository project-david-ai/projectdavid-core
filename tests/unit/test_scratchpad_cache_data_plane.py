from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from src.api.entities_api.cache.scratchpad_cache import (
    REDIS_SCRATCHPAD_TTL,
    ScratchpadCache,
)


class FakeRedis:
    """Minimal thread-safe synchronous Redis double."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.expirations: dict[str, int] = {}
        self.lock = threading.Lock()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def get(
        self,
        key: str,
    ) -> str | None:
        with self.lock:
            self.calls.append(
                (
                    "get",
                    (key,),
                    {},
                )
            )
            return self.strings.get(key)

    def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
    ) -> bool:
        with self.lock:
            self.calls.append(
                (
                    "set",
                    (
                        key,
                        value,
                    ),
                    {
                        "ex": ex,
                    },
                )
            )

            self.strings[key] = value

            if ex is not None:
                self.expirations[key] = ex

            return True

    def rpush(
        self,
        key: str,
        value: str,
    ) -> int:
        with self.lock:
            self.calls.append(
                (
                    "rpush",
                    (
                        key,
                        value,
                    ),
                    {},
                )
            )

            values = self.lists.setdefault(
                key,
                [],
            )

            values.append(value)

            return len(values)

    def lrange(
        self,
        key: str,
        start: int,
        end: int,
    ) -> list[str]:
        with self.lock:
            self.calls.append(
                (
                    "lrange",
                    (
                        key,
                        start,
                        end,
                    ),
                    {},
                )
            )

            values = list(
                self.lists.get(
                    key,
                    [],
                )
            )

            if end == -1:
                return values[start:]

            return values[start : end + 1]

    def expire(
        self,
        key: str,
        ttl: int,
    ) -> bool:
        with self.lock:
            self.calls.append(
                (
                    "expire",
                    (
                        key,
                        ttl,
                    ),
                    {},
                )
            )

            self.expirations[key] = ttl

            return True

    def delete(
        self,
        *keys: str,
    ) -> int:
        deleted = 0

        with self.lock:
            self.calls.append(
                (
                    "delete",
                    keys,
                    {},
                )
            )

            for key in keys:
                if key in self.strings:
                    del self.strings[key]
                    deleted += 1

                if key in self.lists:
                    del self.lists[key]
                    deleted += 1

                self.expirations.pop(
                    key,
                    None,
                )

        return deleted


def test_first_class_keys_are_tenant_and_resource_scoped() -> None:
    assert (
        ScratchpadCache._content_key(
            "user_1",
            "scratchpad_1",
        )
        == "scratchpad:user_1:scratchpad_1:content"
    )

    assert (
        ScratchpadCache._entries_key(
            "user_1",
            "scratchpad_1",
        )
        == "scratchpad:user_1:scratchpad_1:entries"
    )


def test_missing_content_is_empty_and_unmaterialized() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    result = asyncio.run(
        cache.get_content(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
        )
    )

    assert result == {
        "content": "",
        "updated_at": None,
    }

    assert redis.strings == {}


def test_set_content_isolated_by_tenant_and_scratchpad() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    result = asyncio.run(
        cache.set_content(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="master plan",
        )
    )

    key = "scratchpad:user_1:" "scratchpad_1:content"

    stored = json.loads(redis.strings[key])

    assert stored["content"] == "master plan"
    assert result["content"] == "master plan"
    assert isinstance(
        result["updated_at"],
        float,
    )
    assert redis.expirations[key] == REDIS_SCRATCHPAD_TTL


def test_append_entry_uses_rpush_not_get_modify_set() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    asyncio.run(
        cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="finding-a",
        )
    )

    operations = [call[0] for call in redis.calls]

    assert "rpush" in operations
    assert "expire" in operations

    assert "get" not in operations
    assert "set" not in operations


def test_entries_preserve_append_order() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    async def exercise() -> list[dict[str, Any]]:
        await cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="worker-a",
        )

        await cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="worker-b",
        )

        return await cache.list_entries(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
        )

    entries = asyncio.run(exercise())

    assert [entry["content"] for entry in entries] == [
        "worker-a",
        "worker-b",
    ]

    assert all(
        isinstance(
            entry["created_at"],
            float,
        )
        for entry in entries
    )


def test_tenant_namespaces_do_not_collide() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    async def exercise() -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        await cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_same",
            content="tenant-one",
        )

        await cache.append_entry(
            owner_id="user_2",
            scratchpad_id="scratchpad_same",
            content="tenant-two",
        )

        one = await cache.list_entries(
            owner_id="user_1",
            scratchpad_id="scratchpad_same",
        )

        two = await cache.list_entries(
            owner_id="user_2",
            scratchpad_id="scratchpad_same",
        )

        return one, two

    one, two = asyncio.run(exercise())

    assert [entry["content"] for entry in one] == ["tenant-one"]

    assert [entry["content"] for entry in two] == ["tenant-two"]


def test_clear_content_preserves_entries() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    async def exercise() -> None:
        await cache.set_content(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="plan",
        )

        await cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="finding",
        )

        await cache.clear_content(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
        )

    asyncio.run(exercise())

    content_key = "scratchpad:user_1:" "scratchpad_1:content"

    entries_key = "scratchpad:user_1:" "scratchpad_1:entries"

    assert content_key not in redis.strings
    assert entries_key in redis.lists


def test_clear_entries_preserves_content() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    async def exercise() -> None:
        await cache.set_content(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="plan",
        )

        await cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="finding",
        )

        await cache.clear_entries(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
        )

    asyncio.run(exercise())

    content_key = "scratchpad:user_1:" "scratchpad_1:content"

    entries_key = "scratchpad:user_1:" "scratchpad_1:entries"

    assert content_key in redis.strings
    assert entries_key not in redis.lists


def test_delete_all_purges_both_state_planes() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    async def exercise() -> None:
        await cache.set_content(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="plan",
        )

        await cache.append_entry(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
            content="finding",
        )

        await cache.delete_all_scratchpad_data(
            owner_id="user_1",
            scratchpad_id="scratchpad_1",
        )

    asyncio.run(exercise())

    prefix = "scratchpad:user_1:" "scratchpad_1:"

    assert not any(key.startswith(prefix) for key in redis.strings)

    assert not any(key.startswith(prefix) for key in redis.lists)


def test_legacy_thread_key_contract_remains_operational() -> None:
    redis = FakeRedis()
    cache = ScratchpadCache(redis=redis)

    async def exercise() -> dict[str, Any]:
        await cache.overwrite_scratchpad(
            "thread_legacy",
            "plan",
        )

        await cache.append_to_scratchpad(
            "thread_legacy",
            "finding",
        )

        return await cache.get_scratchpad("thread_legacy")

    result = asyncio.run(exercise())

    assert result["content"] == ("plan\n\nfinding")

    assert "scratchpad:thread_legacy:notebook" in redis.strings
