from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from projectdavid_common import ValidationInterface


def _load_scratchpads_router_module():
    """
    Load the router module without executing routers/__init__.py.

    The working tree intentionally contains unrelated staged Deep Research
    module-map work. The aggregate router initializer imports the inference
    stack, so executing it would make this focused Scratchpad unit test
    depend on that unrelated in-progress rename.
    """

    package_name = "src.api.entities_api.routers"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)

        package.__path__ = [str(Path("src/api/entities_api/routers").resolve())]

        sys.modules[package_name] = package

    module_name = "src.api.entities_api.routers." "scratchpads_router"

    module_path = Path(
        "src/api/entities_api/routers/" "scratchpads_router.py"
    ).resolve()

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Scratchpads router module.")

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


scratchpads_router = _load_scratchpads_router_module()

append_scratchpad_entry = scratchpads_router.append_scratchpad_entry

clear_scratchpad = scratchpads_router.clear_scratchpad

create_scratchpad = scratchpads_router.create_scratchpad

delete_scratchpad = scratchpads_router.delete_scratchpad

get_scratchpad_content = scratchpads_router.get_scratchpad_content

router = scratchpads_router.router

set_scratchpad_content = scratchpads_router.set_scratchpad_content

validator = ValidationInterface()


class FakeScratchpadService:
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

        self.calls = []

    def create_scratchpad(
        self,
        scratchpad,
        *,
        user_id,
    ):
        self.calls.append(
            (
                "create",
                scratchpad.thread_id,
                user_id,
            )
        )

        return self.resource

    async def get_content(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        self.calls.append(
            (
                "get_content",
                scratchpad_id,
                user_id,
            )
        )

        return validator.ScratchpadContentRead(
            scratchpad_id=scratchpad_id,
            content="PLAN",
            updated_at=None,
        )

    async def set_content(
        self,
        scratchpad_id,
        content,
        *,
        user_id,
    ):
        self.calls.append(
            (
                "set_content",
                scratchpad_id,
                content,
                user_id,
            )
        )

        return validator.ScratchpadContentRead(
            scratchpad_id=scratchpad_id,
            content=content,
            updated_at=None,
        )

    async def append_entry(
        self,
        scratchpad_id,
        content,
        *,
        user_id,
    ):
        self.calls.append(
            (
                "append_entry",
                scratchpad_id,
                content,
                user_id,
            )
        )

        return validator.ScratchpadEntryRead(
            scratchpad_id=scratchpad_id,
            content=content,
            created_at=datetime.now(timezone.utc),
        )

    async def clear_scratchpad(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        self.calls.append(
            (
                "clear",
                scratchpad_id,
                user_id,
            )
        )

        return validator.ScratchpadStateCleared(
            id=scratchpad_id,
            scope="all",
        )

    async def delete_scratchpad(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        self.calls.append(
            (
                "delete",
                scratchpad_id,
                user_id,
            )
        )

        return validator.ScratchpadDeleted(
            id=scratchpad_id,
        )


def _auth():
    return SimpleNamespace(
        user_id="user_1",
    )


def test_public_route_surface_is_complete() -> None:
    routes = {
        (
            method,
            route.path,
        )
        for route in router.routes
        if isinstance(
            route,
            APIRoute,
        )
        for method in route.methods
    }

    expected = {
        ("POST", "/scratchpads"),
        ("GET", "/scratchpads"),
        (
            "GET",
            "/scratchpads/{scratchpad_id}",
        ),
        (
            "GET",
            "/threads/{thread_id}/scratchpad",
        ),
        (
            "PATCH",
            "/scratchpads/{scratchpad_id}",
        ),
        (
            "DELETE",
            "/scratchpads/{scratchpad_id}",
        ),
        (
            "GET",
            "/scratchpads/{scratchpad_id}/content",
        ),
        (
            "PUT",
            "/scratchpads/{scratchpad_id}/content",
        ),
        (
            "DELETE",
            "/scratchpads/{scratchpad_id}/content",
        ),
        (
            "POST",
            "/scratchpads/{scratchpad_id}/entries",
        ),
        (
            "GET",
            "/scratchpads/{scratchpad_id}/entries",
        ),
        (
            "DELETE",
            "/scratchpads/{scratchpad_id}/entries",
        ),
        (
            "POST",
            "/scratchpads/{scratchpad_id}/clear",
        ),
    }

    assert routes == expected


def test_create_derives_owner_from_auth_key() -> None:
    service = FakeScratchpadService()

    result = create_scratchpad(
        validator.ScratchpadCreate(
            thread_id="thread_1",
        ),
        service=service,
        auth_key=_auth(),
    )

    assert result.id == "scratchpad_1"

    assert service.calls == [
        (
            "create",
            "thread_1",
            "user_1",
        )
    ]


def test_content_endpoints_forward_authenticated_user() -> None:
    service = FakeScratchpadService()

    async def exercise():
        read = await get_scratchpad_content(
            "scratchpad_1",
            service=service,
            auth_key=_auth(),
        )

        written = await set_scratchpad_content(
            "scratchpad_1",
            validator.ScratchpadContentUpdate(
                content="NEW PLAN",
            ),
            service=service,
            auth_key=_auth(),
        )

        return read, written

    read, written = asyncio.run(exercise())

    assert read.content == "PLAN"
    assert written.content == "NEW PLAN"

    assert service.calls == [
        (
            "get_content",
            "scratchpad_1",
            "user_1",
        ),
        (
            "set_content",
            "scratchpad_1",
            "NEW PLAN",
            "user_1",
        ),
    ]


def test_append_entry_uses_authenticated_user() -> None:
    service = FakeScratchpadService()

    result = asyncio.run(
        append_scratchpad_entry(
            "scratchpad_1",
            validator.ScratchpadEntryCreate(
                content="finding",
            ),
            service=service,
            auth_key=_auth(),
        )
    )

    assert result.content == "finding"

    assert service.calls == [
        (
            "append_entry",
            "scratchpad_1",
            "finding",
            "user_1",
        )
    ]


def test_clear_and_delete_are_distinct() -> None:
    service = FakeScratchpadService()

    async def exercise():
        cleared = await clear_scratchpad(
            "scratchpad_1",
            service=service,
            auth_key=_auth(),
        )

        deleted = await delete_scratchpad(
            "scratchpad_1",
            service=service,
            auth_key=_auth(),
        )

        return cleared, deleted

    cleared, deleted = asyncio.run(exercise())

    assert cleared.scope == "all"
    assert cleared.cleared is True
    assert deleted.deleted is True

    assert service.calls == [
        (
            "clear",
            "scratchpad_1",
            "user_1",
        ),
        (
            "delete",
            "scratchpad_1",
            "user_1",
        ),
    ]


def test_http_exception_is_preserved() -> None:
    class FailingService(FakeScratchpadService):
        def create_scratchpad(
            self,
            scratchpad,
            *,
            user_id,
        ):
            raise HTTPException(
                status_code=409,
                detail="duplicate",
            )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        create_scratchpad(
            validator.ScratchpadCreate(
                thread_id="thread_1",
            ),
            service=FailingService(),
            auth_key=_auth(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "duplicate"


def test_response_models_are_fully_typed() -> None:
    expected = {
        ("POST", "/scratchpads"): ValidationInterface.ScratchpadRead,
        ("GET", "/scratchpads"): ValidationInterface.ScratchpadList,
        ("GET", "/scratchpads/{scratchpad_id}"): ValidationInterface.ScratchpadRead,
        ("GET", "/threads/{thread_id}/scratchpad"): ValidationInterface.ScratchpadRead,
        ("PATCH", "/scratchpads/{scratchpad_id}"): ValidationInterface.ScratchpadRead,
        (
            "DELETE",
            "/scratchpads/{scratchpad_id}",
        ): ValidationInterface.ScratchpadDeleted,
        (
            "GET",
            "/scratchpads/{scratchpad_id}/content",
        ): ValidationInterface.ScratchpadContentRead,
        (
            "PUT",
            "/scratchpads/{scratchpad_id}/content",
        ): ValidationInterface.ScratchpadContentRead,
        (
            "DELETE",
            "/scratchpads/{scratchpad_id}/content",
        ): ValidationInterface.ScratchpadStateCleared,
        (
            "POST",
            "/scratchpads/{scratchpad_id}/entries",
        ): ValidationInterface.ScratchpadEntryRead,
        (
            "GET",
            "/scratchpads/{scratchpad_id}/entries",
        ): ValidationInterface.ScratchpadEntryList,
        (
            "DELETE",
            "/scratchpads/{scratchpad_id}/entries",
        ): ValidationInterface.ScratchpadStateCleared,
        (
            "POST",
            "/scratchpads/{scratchpad_id}/clear",
        ): ValidationInterface.ScratchpadStateCleared,
    }

    found = {}

    for route in router.routes:
        if not isinstance(
            route,
            APIRoute,
        ):
            continue

        for method in route.methods:
            found[
                (
                    method,
                    route.path,
                )
            ] = route.response_model

    assert set(found) == set(expected)

    for key, model in expected.items():
        assert found[key] is model, (
            key,
            found[key],
            model,
        )
