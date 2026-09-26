"""Durable user-owned MCP registration and assistant attachment service."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException
from mcp.types import Tool
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_orm.projectdavid_orm.models import (
    AssistantMcpTool,
    McpServerRegistration,
)
from sqlalchemy.exc import IntegrityError

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Assistant
from src.api.entities_api.orchestration.mcp_remote_client import RemoteMcpClient
from src.api.entities_api.orchestration.mcp_tool_discovery import (
    McpDiscoveredTool,
    McpToolDiscoveryPage,
    adapt_mcp_list_tools_result,
    adapt_mcp_tools,
)
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.mcp_tool_config import (
    function_tool_name,
    remove_function_tools,
    upsert_function_tool,
)
from src.api.entities_api.utilities.cache_utils import get_sync_invalidator

validator = ValidationInterface()
logging_utility = LoggingUtility()

_MAX_DISCOVERY_PAGES = 100


class McpRegistrationService:
    """Own remote MCP registrations and assistant-specific tool provenance."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = SessionLocal,
        client_factory: Callable[..., RemoteMcpClient] = RemoteMcpClient,
        cache_invalidator_factory: Callable[[], Any] = get_sync_invalidator,
    ) -> None:
        self._session_factory = session_factory
        self._client_factory = client_factory
        self._cache_invalidator_factory = cache_invalidator_factory

    @staticmethod
    def _normalize_url(url: object) -> str:
        raw = str(url).strip()

        try:
            parsed = urlsplit(raw)
            port = parsed.port
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="MCP endpoint URL is invalid",
            ) from exc

        scheme = parsed.scheme.lower()

        if scheme not in {"http", "https"} or not parsed.hostname:
            raise HTTPException(
                status_code=400,
                detail=("Remote MCP endpoint must be an absolute " "HTTP or HTTPS URL"),
            )

        if parsed.username is not None or parsed.password is not None:
            raise HTTPException(
                status_code=400,
                detail="MCP endpoint URLs may not contain credentials",
            )

        host = parsed.hostname.lower()

        # urlsplit strips IPv6 brackets from hostname.
        if ":" in host:
            host = f"[{host}]"

        default_port = (scheme == "http" and port == 80) or (
            scheme == "https" and port == 443
        )

        if port is not None and not default_port:
            host = f"{host}:{port}"

        path = parsed.path or "/"

        # URL fragments are never sent to the remote server, so they must not
        # produce distinct registration identities.
        return urlunsplit(
            (
                scheme,
                host,
                path,
                parsed.query,
                "",
            )
        )

    @staticmethod
    def _identity_key(
        *,
        transport: str,
        normalized_url: str,
    ) -> str:
        payload = f"{transport}\0{normalized_url}".encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _registration_read(
        row: McpServerRegistration,
    ) -> validator.McpServerRegistrationRead:
        return validator.McpServerRegistrationRead.model_validate(row)

    @staticmethod
    def _attachment_read(
        row: AssistantMcpTool,
    ) -> validator.AssistantMcpToolRead:
        return validator.AssistantMcpToolRead.model_validate(row)

    @staticmethod
    def _assert_assistant_owner(
        assistant: Assistant,
        user_id: str,
    ) -> None:
        if assistant.owner_id is not None:
            if assistant.owner_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail=("You do not have permission to modify this assistant."),
                )
            return

        associated_ids = {user.id for user in assistant.users}

        if user_id not in associated_ids:
            raise HTTPException(
                status_code=403,
                detail=("You do not have permission to modify this assistant."),
            )

    @staticmethod
    def _owned_assistant(
        db: Any,
        *,
        assistant_id: str,
        user_id: str,
    ) -> Assistant:
        assistant = (
            db.query(Assistant)
            .filter(
                Assistant.id == assistant_id,
                Assistant.deleted_at.is_(None),
            )
            .first()
        )

        if assistant is None:
            raise HTTPException(
                status_code=404,
                detail="Assistant not found",
            )

        McpRegistrationService._assert_assistant_owner(
            assistant,
            user_id,
        )

        return assistant

    @staticmethod
    def _owned_registration(
        db: Any,
        *,
        server_id: str,
        user_id: str,
        require_enabled: bool = False,
    ) -> McpServerRegistration:
        registration = (
            db.query(McpServerRegistration)
            .filter(
                McpServerRegistration.id == server_id,
                McpServerRegistration.owner_id == user_id,
            )
            .first()
        )

        if registration is None:
            raise HTTPException(
                status_code=404,
                detail="MCP server registration not found",
            )

        if require_enabled and not registration.enabled:
            raise HTTPException(
                status_code=409,
                detail="MCP server registration is disabled",
            )

        return registration

    def _invalidate_assistant_cache(self, assistant_id: str) -> None:
        try:
            cache = self._cache_invalidator_factory()
            cache.invalidate_sync(assistant_id)
        except Exception as exc:  # noqa: BLE001
            logging_utility.error(
                "Failed to invalidate assistant cache after MCP update: %s",
                exc,
            )

    def register_server(
        self,
        registration: validator.McpServerRegistrationCreate,
        *,
        user_id: str,
    ) -> validator.McpServerRegistrationRead:
        """Create-or-return a user-owned remote MCP registration."""

        normalized_url = self._normalize_url(registration.url)

        identity_key = self._identity_key(
            transport=registration.transport,
            normalized_url=normalized_url,
        )

        with self._session_factory() as db:
            existing = (
                db.query(McpServerRegistration)
                .filter(
                    McpServerRegistration.owner_id == user_id,
                    McpServerRegistration.identity_key == identity_key,
                )
                .first()
            )

            if existing is not None:
                return self._registration_read(existing)

            row = McpServerRegistration(
                id=UtilsInterface.IdentifierService.generate_prefixed_id("mcpreg"),
                owner_id=user_id,
                name=registration.name,
                url=normalized_url,
                normalized_url=normalized_url,
                identity_key=identity_key,
                transport=registration.transport,
                timeout_seconds=registration.timeout_seconds,
                enabled=True,
            )

            db.add(row)

            try:
                db.commit()
            except IntegrityError:
                db.rollback()

                existing = (
                    db.query(McpServerRegistration)
                    .filter(
                        McpServerRegistration.owner_id == user_id,
                        McpServerRegistration.identity_key == identity_key,
                    )
                    .first()
                )

                if existing is None:
                    raise

                return self._registration_read(existing)

            db.refresh(row)
            return self._registration_read(row)

    def list_servers(
        self,
        *,
        user_id: str,
    ) -> list[validator.McpServerRegistrationRead]:
        """List registrations owned by one user."""

        with self._session_factory() as db:
            rows = (
                db.query(McpServerRegistration)
                .filter(McpServerRegistration.owner_id == user_id)
                .order_by(
                    McpServerRegistration.created_at,
                    McpServerRegistration.id,
                )
                .all()
            )

            return [self._registration_read(row) for row in rows]

    def get_server(
        self,
        *,
        server_id: str,
        user_id: str,
    ) -> validator.McpServerRegistrationRead:
        """Return one registration owned by the authenticated user."""

        with self._session_factory() as db:
            row = self._owned_registration(
                db,
                server_id=server_id,
                user_id=user_id,
            )

            return self._registration_read(row)

    def update_server(
        self,
        *,
        server_id: str,
        registration: validator.McpServerRegistrationUpdate,
        user_id: str,
    ) -> validator.McpServerRegistrationRead:
        """Update mutable registration configuration."""

        data = registration.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

        if not data:
            return self.get_server(
                server_id=server_id,
                user_id=user_id,
            )

        unexpected = set(data) - {
            "name",
            "timeout_seconds",
            "enabled",
        }

        if unexpected:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported MCP registration update fields: "
                    + ", ".join(sorted(unexpected))
                ),
            )

        assistant_ids: set[str] = set()

        with self._session_factory() as db:
            row = self._owned_registration(
                db,
                server_id=server_id,
                user_id=user_id,
            )

            attachments = (
                db.query(AssistantMcpTool)
                .filter(AssistantMcpTool.registration_id == row.id)
                .all()
            )

            assistant_ids = {attachment.assistant_id for attachment in attachments}

            for field in (
                "name",
                "timeout_seconds",
                "enabled",
            ):
                if field in data:
                    setattr(
                        row,
                        field,
                        data[field],
                    )

            db.commit()
            db.refresh(row)

            result = self._registration_read(row)

        # Persisted attachment provider aliases are deliberately
        # durable. Renaming a registration changes the namespace
        # used by future discovery/new attachments only.
        for assistant_id in assistant_ids:
            self._invalidate_assistant_cache(assistant_id)

        return result

    def delete_server(
        self,
        *,
        server_id: str,
        user_id: str,
    ) -> None:
        """Delete a registration and its assistant-facing capabilities."""

        assistant_ids: set[str] = set()

        with self._session_factory() as db:
            registration = self._owned_registration(
                db,
                server_id=server_id,
                user_id=user_id,
            )

            attachments = (
                db.query(AssistantMcpTool)
                .filter(AssistantMcpTool.registration_id == registration.id)
                .all()
            )

            aliases_by_assistant: dict[
                str,
                set[str],
            ] = {}

            for attachment in attachments:
                assistant_ids.add(attachment.assistant_id)

                aliases_by_assistant.setdefault(
                    attachment.assistant_id,
                    set(),
                ).add(attachment.provider_name)

            for (
                assistant_id,
                provider_names,
            ) in aliases_by_assistant.items():
                assistant = (
                    db.query(Assistant).filter(Assistant.id == assistant_id).first()
                )

                if assistant is not None:
                    assistant.tool_configs = remove_function_tools(
                        list(assistant.tool_configs or []),
                        provider_names,
                    )

            for attachment in attachments:
                db.delete(attachment)

            db.delete(registration)
            db.commit()

        for assistant_id in assistant_ids:
            self._invalidate_assistant_cache(assistant_id)

    async def discover_tools(
        self,
        *,
        user_id: str,
        server_id: str,
        cursor: str | None = None,
        reserved_provider_names: Iterable[str] = (),
    ) -> McpToolDiscoveryPage:
        """Discover one MCP tools/list page for an owned registration."""

        with self._session_factory() as db:
            registration = self._owned_registration(
                db,
                server_id=server_id,
                user_id=user_id,
                require_enabled=True,
            )

            url = registration.url
            timeout_seconds = registration.timeout_seconds
            stable_server_id = registration.id
            provider_namespace = registration.name

        try:
            async with self._client_factory(
                url,
                read_timeout_seconds=timeout_seconds,
            ) as client:
                result = await client.list_tools(cursor=cursor)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=("Remote MCP tool discovery failed: " f"{type(exc).__name__}"),
            ) from exc

        # The persistent registration id is deliberately used as MCP identity.
        # Display names are mutable and therefore cannot be canonical identity.
        return adapt_mcp_list_tools_result(
            stable_server_id,
            result,
            provider_namespace=provider_namespace,
            reserved_provider_names=reserved_provider_names,
        )

    async def _discover_all_tools(
        self,
        *,
        url: str,
        timeout_seconds: float,
        server_id: str,
        provider_namespace: str,
        reserved_provider_names: Iterable[str],
    ) -> tuple[McpDiscoveredTool, ...]:
        raw_tools: list[Tool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        try:
            async with self._client_factory(
                url,
                read_timeout_seconds=timeout_seconds,
            ) as client:
                for _ in range(_MAX_DISCOVERY_PAGES):
                    result = await client.list_tools(cursor=cursor)
                    raw_tools.extend(result.tools)

                    next_cursor = result.next_cursor

                    if next_cursor is None:
                        break

                    if next_cursor in seen_cursors:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "Remote MCP server returned a repeated "
                                "pagination cursor"
                            ),
                        )

                    seen_cursors.add(next_cursor)
                    cursor = next_cursor
                else:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Remote MCP tool discovery exceeded pagination "
                            "safety limit"
                        ),
                    )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=("Remote MCP tool discovery failed: " f"{type(exc).__name__}"),
            ) from exc

        return adapt_mcp_tools(
            server_id,
            raw_tools,
            provider_namespace=provider_namespace,
            reserved_provider_names=reserved_provider_names,
        )

    async def attach_tools(
        self,
        *,
        assistant_id: str,
        attachment: validator.AssistantMcpToolsAttach,
        user_id: str,
        _retry_on_integrity: bool = True,
    ) -> list[validator.AssistantMcpToolRead]:
        """Attach selected remote MCP tools and inject function definitions."""

        requested_names = tuple(attachment.tools)

        with self._session_factory() as db:
            assistant = self._owned_assistant(
                db,
                assistant_id=assistant_id,
                user_id=user_id,
            )

            registration = self._owned_registration(
                db,
                server_id=attachment.server_id,
                user_id=user_id,
                require_enabled=True,
            )

            current_tools = deepcopy(list(assistant.tool_configs or []))

            existing_rows = (
                db.query(AssistantMcpTool)
                .filter(AssistantMcpTool.assistant_id == assistant_id)
                .all()
            )

            selected_existing_aliases = {
                row.provider_name
                for row in existing_rows
                if (
                    row.registration_id == registration.id
                    and row.remote_name in requested_names
                )
            }

            reserved_names = {
                name
                for tool in current_tools
                if (name := function_tool_name(tool)) is not None
            }

            # Do not force an already-attached tool to hash itself merely
            # because its own provider alias is currently reserved.
            reserved_names -= selected_existing_aliases

            registration_url = registration.url
            timeout_seconds = registration.timeout_seconds
            stable_server_id = registration.id
            provider_namespace = registration.name

        discovered = await self._discover_all_tools(
            url=registration_url,
            timeout_seconds=timeout_seconds,
            server_id=stable_server_id,
            provider_namespace=provider_namespace,
            reserved_provider_names=reserved_names,
        )

        discovered_by_remote = {tool.remote_name: tool for tool in discovered}

        missing = [name for name in requested_names if name not in discovered_by_remote]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Requested MCP tools were not advertised by the server: "
                    + ", ".join(missing)
                ),
            )

        try:
            with self._session_factory() as db:
                assistant = self._owned_assistant(
                    db,
                    assistant_id=assistant_id,
                    user_id=user_id,
                )

                registration = self._owned_registration(
                    db,
                    server_id=attachment.server_id,
                    user_id=user_id,
                    require_enabled=True,
                )

                rows = (
                    db.query(AssistantMcpTool)
                    .filter(AssistantMcpTool.assistant_id == assistant_id)
                    .all()
                )

                by_identity = {
                    (
                        row.registration_id,
                        row.remote_name,
                    ): row
                    for row in rows
                }

                current_tools = deepcopy(list(assistant.tool_configs or []))

                current_names = {
                    name
                    for tool in current_tools
                    if (name := function_tool_name(tool)) is not None
                }

                for remote_name in requested_names:
                    discovered_tool = discovered_by_remote[remote_name]

                    key = (
                        registration.id,
                        remote_name,
                    )

                    row = by_identity.get(key)

                    if row is None:
                        provider_name = discovered_tool.provider_name

                        if provider_name in current_names:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    "Assistant tools changed concurrently; "
                                    "retry the MCP attachment"
                                ),
                            )

                        row = AssistantMcpTool(
                            id=(
                                UtilsInterface.IdentifierService.generate_prefixed_id(
                                    "mcptool"
                                )
                            ),
                            assistant_id=assistant_id,
                            registration_id=registration.id,
                            remote_name=remote_name,
                            canonical_id=discovered_tool.canonical_id,
                            provider_name=provider_name,
                            enabled=True,
                        )

                        db.add(row)
                        by_identity[key] = row
                    else:
                        provider_name = row.provider_name
                        row.canonical_id = discovered_tool.canonical_id
                        row.enabled = True

                    function_tool = discovered_tool.definition.to_function_tool()

                    # Existing attachments retain their durable provider alias,
                    # even if discovery metadata is refreshed.
                    function_tool["function"]["name"] = provider_name

                    current_tools = upsert_function_tool(
                        current_tools,
                        function_tool,
                    )

                    current_names.add(provider_name)

                assistant.tool_configs = current_tools

                db.commit()

                selected_rows = (
                    db.query(AssistantMcpTool)
                    .filter(
                        AssistantMcpTool.assistant_id == assistant_id,
                        AssistantMcpTool.registration_id == registration.id,
                        AssistantMcpTool.remote_name.in_(requested_names),
                    )
                    .all()
                )

                reads_by_remote = {
                    row.remote_name: self._attachment_read(row) for row in selected_rows
                }

                result = [reads_by_remote[name] for name in requested_names]

        except IntegrityError as exc:
            if _retry_on_integrity:
                return await self.attach_tools(
                    assistant_id=assistant_id,
                    attachment=attachment,
                    user_id=user_id,
                    _retry_on_integrity=False,
                )

            raise HTTPException(
                status_code=409,
                detail=("Concurrent MCP attachment conflict; retry the request"),
            ) from exc

        self._invalidate_assistant_cache(assistant_id)

        return result

    def detach_tools(
        self,
        *,
        assistant_id: str,
        attachment: validator.AssistantMcpToolsDetach,
        user_id: str,
    ) -> None:
        """Detach selected MCP tools; already-detached tools are a no-op."""

        requested_names = tuple(attachment.tools)

        with self._session_factory() as db:
            assistant = self._owned_assistant(
                db,
                assistant_id=assistant_id,
                user_id=user_id,
            )

            registration = self._owned_registration(
                db,
                server_id=attachment.server_id,
                user_id=user_id,
            )

            rows = (
                db.query(AssistantMcpTool)
                .filter(
                    AssistantMcpTool.assistant_id == assistant_id,
                    AssistantMcpTool.registration_id == registration.id,
                    AssistantMcpTool.remote_name.in_(requested_names),
                )
                .all()
            )

            provider_names = {row.provider_name for row in rows}

            if provider_names:
                assistant.tool_configs = remove_function_tools(
                    list(assistant.tool_configs or []),
                    provider_names,
                )

                for row in rows:
                    db.delete(row)

                db.commit()

        if provider_names:
            self._invalidate_assistant_cache(assistant_id)

    def list_assistant_tools(
        self,
        *,
        assistant_id: str,
        user_id: str,
    ) -> list[validator.AssistantMcpToolRead]:
        """List durable MCP attachments for an owned assistant."""

        with self._session_factory() as db:
            self._owned_assistant(
                db,
                assistant_id=assistant_id,
                user_id=user_id,
            )

            rows = (
                db.query(AssistantMcpTool)
                .filter(AssistantMcpTool.assistant_id == assistant_id)
                .order_by(
                    AssistantMcpTool.created_at,
                    AssistantMcpTool.id,
                )
                .all()
            )

            return [self._attachment_read(row) for row in rows]


__all__ = ["McpRegistrationService"]
