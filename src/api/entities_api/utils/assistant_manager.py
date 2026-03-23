import asyncio
import uuid
from typing import Any, Optional

from entities_api.platform_tools.tool_reigistry.junior_network_engineer import JUNIOR_ENGINEER_TOOLS
from entities_api.platform_tools.tool_reigistry.research_supervisor import SUPERVISOR_TOOLS
from entities_api.platform_tools.tool_reigistry.senior_network_engineer import SENIOR_ENGINEER_TOOLS

from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class AssistantManager:
    """
    Creates and destroys ephemeral assistants and threads used by the
    orchestration layer (deep research, senior/junior engineer flows).

    Previously used the Entity SDK (HTTP round-trip + ADMIN_API_KEY).
    Now routes through NativeExecutionService → AssistantService → DB directly.

    All mutating methods require a user_id so the ownership guard in
    AssistantService is satisfied. Callers should pass self._run_user_id
    resolved from the run record before any identity swap.
    """

    @property
    def _native_exec(self):
        if getattr(self, "_native_exec_svc", None) is None:
            from src.api.entities_api.services.native_execution_service import (
                NativeExecutionService,
            )

            self._native_exec_svc = NativeExecutionService()
        return self._native_exec_svc

    # ------------------------------------------------------------------
    # Primary (non-ephemeral) helpers
    # ------------------------------------------------------------------

    async def create_primary_assistant(
        self,
        user_id: str,
        name: str = "Test Assistant",
        model: str = "gpt-oss-120b",
    ):
        """Creates the main supervisor assistant."""
        return await self._native_exec.create_assistant(
            user_id=user_id,
            name=name,
            model=model,
            tools=SUPERVISOR_TOOLS,
            web_access=True,
        )

    async def create_thread(self, user_id: str):
        """Creates a thread owned by user_id."""
        return await self._native_exec.create_thread(user_id=user_id)

    # ------------------------------------------------------------------
    # Ephemeral assistants
    # ------------------------------------------------------------------

    async def create_ephemeral_research_supervisor(self, user_id: str):
        """Temporary research supervisor — owned by the run's user."""
        return await self._native_exec.create_assistant(
            user_id=user_id,
            name=f"worker_{uuid.uuid4().hex[:8]}",
            description="Temp research supervisor",
            tools=SUPERVISOR_TOOLS,
            deep_research=True,
        )

    async def create_ephemeral_senior_engineer(self, user_id: str):
        """Temporary senior network engineer — owned by the run's user."""
        return await self._native_exec.create_assistant(
            user_id=user_id,
            name=f"worker_{uuid.uuid4().hex[:8]}",
            description="Temp senior network engineer",
            tools=SENIOR_ENGINEER_TOOLS,
            deep_research=True,
        )

    async def create_ephemeral_worker_assistant(self, user_id: str):
        """
        Temporary research worker.
        Injects research_worker_calling=True into meta_data so
        QwenBaseWorker can identify its role on load.
        """
        return await self._native_exec.create_assistant(
            user_id=user_id,
            name=f"worker_{uuid.uuid4().hex[:8]}",
            description="Ephemeral research worker",
            tools=JUNIOR_ENGINEER_TOOLS,
            web_access=True,
            deep_research=False,
            meta_data={"is_research_worker": "true"},
        )

    async def create_ephemeral_thread(self, user_id: str):
        """Creates a throw-away thread owned by the run's user."""
        return await self._native_exec.create_thread(user_id=user_id)

    async def create_ephemeral_junior_engineer(self, user_id: str):
        """
        Spawns the Junior Engineer worker for Batfish analysis.
        web_access and deep_research are explicitly False to prevent
        the orchestrator injecting web tools or research instructions.
        """
        return await self._native_exec.create_assistant(
            user_id=user_id,
            name=f"junior_eng_{uuid.uuid4().hex[:8]}",
            description="Temp Junior Network Engineer for Batfish RCA",
            tools=JUNIOR_ENGINEER_TOOLS,
            web_access=False,
            deep_research=False,
            meta_data={"junior_engineer": "true"},
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def delete_assistant(
        self,
        assistant_id: str,
        user_id: str,
        permanent: bool = False,
    ) -> Any:
        """
        Deletes a temporary worker assistant.
        user_id must match the owner set at creation time so the
        ownership guard in AssistantService is satisfied.
        """
        return await self._native_exec.delete_assistant(
            assistant_id=assistant_id,
            user_id=user_id,
            permanent=permanent,
        )
