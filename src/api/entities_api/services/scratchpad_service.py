from __future__ import annotations

from datetime import datetime, timezone

from projectdavid_common import ValidationInterface

from src.api.entities_api.cache.scratchpad_cache import ScratchpadCache
from src.api.entities_api.services.scratchpad_resource_service import (
    ScratchpadResourceService,
)

validator = ValidationInterface()


class ScratchpadService:
    """
    First-class Scratchpad facade.

    SQL owns identity, ownership, Thread association,
    metadata and lifecycle.

    Redis owns mutable working content and the ordered
    append-only entry ledger.
    """

    def __init__(
        self,
        cache: ScratchpadCache,
        resource_service: ScratchpadResourceService | None = None,
    ) -> None:
        self.cache = cache
        self.resource_service = resource_service or ScratchpadResourceService()

    # ------------------------------------------------------------------
    # Resource plane
    # ------------------------------------------------------------------

    def create_scratchpad(
        self,
        scratchpad: validator.ScratchpadCreate,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        return self.resource_service.create_scratchpad(
            scratchpad,
            user_id=user_id,
        )

    def retrieve_scratchpad(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        return self.resource_service.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

    def retrieve_scratchpad_by_thread(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        return self.resource_service.retrieve_scratchpad_by_thread(
            thread_id,
            user_id=user_id,
        )

    def ensure_scratchpad_for_thread(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        return self.resource_service.ensure_scratchpad_for_thread(
            thread_id,
            user_id=user_id,
        )

    def list_scratchpads(
        self,
        *,
        user_id: str,
    ) -> validator.ScratchpadList:
        return self.resource_service.list_scratchpads(
            user_id=user_id,
        )

    def update_metadata(
        self,
        scratchpad_id: str,
        scratchpad: validator.ScratchpadUpdate,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        return self.resource_service.update_scratchpad(
            scratchpad_id,
            scratchpad,
            user_id=user_id,
        )

    # ------------------------------------------------------------------
    # Data plane
    # ------------------------------------------------------------------

    @staticmethod
    def _to_datetime(
        timestamp: float | None,
    ) -> datetime | None:
        if timestamp is None:
            return None

        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        )

    async def get_content(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadContentRead:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        data = await self.cache.get_content(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        return validator.ScratchpadContentRead(
            scratchpad_id=resource.id,
            content=data.get(
                "content",
                "",
            ),
            updated_at=self._to_datetime(data.get("updated_at")),
        )

    async def set_content(
        self,
        scratchpad_id: str,
        content: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadContentRead:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        payload = validator.ScratchpadContentUpdate(
            content=content,
        )

        data = await self.cache.set_content(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
            content=payload.content,
        )

        return validator.ScratchpadContentRead(
            scratchpad_id=resource.id,
            content=data["content"],
            updated_at=self._to_datetime(data.get("updated_at")),
        )

    async def append_entry(
        self,
        scratchpad_id: str,
        content: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadEntryRead:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        payload = validator.ScratchpadEntryCreate(
            content=content,
        )

        data = await self.cache.append_entry(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
            content=payload.content,
        )

        return validator.ScratchpadEntryRead(
            scratchpad_id=resource.id,
            content=data["content"],
            created_at=self._to_datetime(data["created_at"]),
        )

    async def list_entries(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadEntryList:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        rows = await self.cache.list_entries(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        return validator.ScratchpadEntryList(
            data=[
                validator.ScratchpadEntryRead(
                    scratchpad_id=resource.id,
                    content=row["content"],
                    created_at=self._to_datetime(row["created_at"]),
                )
                for row in rows
            ]
        )

    async def clear_content(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadStateCleared:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        await self.cache.clear_content(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        return validator.ScratchpadStateCleared(
            id=resource.id,
            scope="content",
        )

    async def clear_entries(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadStateCleared:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        await self.cache.clear_entries(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        return validator.ScratchpadStateCleared(
            id=resource.id,
            scope="entries",
        )

    async def clear_scratchpad(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadStateCleared:
        """
        Purge mutable Redis state while preserving the
        first-class SQL Scratchpad resource.
        """

        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        await self.cache.delete_all_scratchpad_data(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        return validator.ScratchpadStateCleared(
            id=resource.id,
            scope="all",
        )

    async def delete_scratchpad(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadDeleted:
        """
        Destroy Redis state and the SQL Scratchpad resource.
        """

        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        # Purge the data plane before deleting identity so
        # successful SQL deletion cannot orphan Redis state.
        await self.cache.delete_all_scratchpad_data(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        return self.resource_service.delete_record(
            resource.id,
            user_id=user_id,
        )

    # ------------------------------------------------------------------
    # LLM presentation
    # ------------------------------------------------------------------

    async def get_formatted_view(
        self,
        scratchpad_id: str,
        *,
        user_id: str,
    ) -> str:
        resource = self.retrieve_scratchpad(
            scratchpad_id,
            user_id=user_id,
        )

        content = await self.cache.get_content(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        entries = await self.cache.list_entries(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        body = content.get(
            "content",
            "",
        )

        if not body and not entries:
            return (
                "(The scratchpad is currently empty. "
                "Use `update_scratchpad` to create a plan.)"
            )

        lines = [
            "--- 📝 RESEARCH SCRATCHPAD ---",
            "",
            "WORKING BODY",
            "------------",
            body or "(empty)",
            "",
            "APPENDED ENTRIES",
            "----------------",
        ]

        if entries:
            lines.extend(
                f"[{index}] {entry['content']}"
                for index, entry in enumerate(
                    entries,
                    start=1,
                )
            )
        else:
            lines.append("(none)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Thread-based compatibility bridge
    # ------------------------------------------------------------------

    async def _resolve_thread_compatibility(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> validator.ScratchpadRead:
        """
        Resolve/create the first-class Scratchpad for an
        existing thread-based caller.

        If a legacy thread-keyed notebook still exists and
        canonical state has not yet been materialised, migrate
        its body once into the first-class content plane.
        """

        resource = self.ensure_scratchpad_for_thread(
            thread_id,
            user_id=user_id,
        )

        canonical_content = await self.cache.get_content(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        canonical_entries = await self.cache.list_entries(
            owner_id=resource.owner_id,
            scratchpad_id=resource.id,
        )

        canonical_exists = (
            canonical_content.get("updated_at") is not None
            or bool(canonical_content.get("content"))
            or bool(canonical_entries)
        )

        if canonical_exists:
            return resource

        legacy = await self.cache.get_scratchpad(thread_id)

        legacy_content = legacy.get(
            "content",
            "",
        )

        if legacy_content:
            await self.cache.set_content(
                owner_id=resource.owner_id,
                scratchpad_id=resource.id,
                content=legacy_content,
            )

            await self.cache.clear_scratchpad(thread_id)

        return resource

    async def get_formatted_view_for_thread(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> str:
        resource = await self._resolve_thread_compatibility(
            thread_id,
            user_id=user_id,
        )

        return await self.get_formatted_view(
            resource.id,
            user_id=user_id,
        )

    async def update_content(
        self,
        thread_id: str,
        content: str,
        *,
        user_id: str,
    ) -> str:
        resource = await self._resolve_thread_compatibility(
            thread_id,
            user_id=user_id,
        )

        await self.set_content(
            resource.id,
            content,
            user_id=user_id,
        )

        return "Scratchpad updated successfully."

    async def append_note(
        self,
        thread_id: str,
        note: str,
        *,
        user_id: str,
    ) -> str:
        resource = await self._resolve_thread_compatibility(
            thread_id,
            user_id=user_id,
        )

        await self.append_entry(
            resource.id,
            note,
            user_id=user_id,
        )

        return "Note appended successfully."

    async def clear(
        self,
        thread_id: str,
        *,
        user_id: str,
    ) -> str:
        resource = await self._resolve_thread_compatibility(
            thread_id,
            user_id=user_id,
        )

        await self.clear_scratchpad(
            resource.id,
            user_id=user_id,
        )

        return "Scratchpad cleared."


__all__ = ["ScratchpadService"]
