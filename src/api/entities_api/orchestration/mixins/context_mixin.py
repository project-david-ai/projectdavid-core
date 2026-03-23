# src/api/entities_api/orchestration/mixins/context_mixin.py
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from entities_api.constants.tools import PLATFORM_TOOL_MAP
from entities_api.orchestration.instructions.assembler import \
    assemble_instructions

from src.api.entities_api.orchestration.instructions.include_lists import (
    L2_INSTRUCTIONS, L3_INSTRUCTIONS, L3_WEB_USE_INSTRUCTIONS,
    L4_JUNIOR_ENGINEER_INSTRUCTIONS, L4_RESEARCH_INSTRUCTIONS,
    L4_SENIOR_ENGINEER_INSTRUCTIONS, LEVEL_4_SUPERVISOR_INSTRUCTIONS,
    NO_CORE_INSTRUCTIONS)
from src.api.entities_api.platform_tools.definitions.record_tool_decision import \
    record_tool_decision
from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class ContextMixin:
    """
    Owns: context window construction, system message assembly per agent role,
          message history retrieval and normalisation, tool resolution and
          deduplication.

    Requires on self (satisfied by _ProviderMixins / OrchestratorCore composition):
        self.get_assistant_cache()       — AssistantCacheMixin
        self._native_exec                — NativeExecMixin
        self.conversation_truncator      — set in BaseWorker.__init__
        self.message_cache               — lazy property, self-initialising

    Do NOT redefine get_assistant_cache() or _native_exec on any subclass —
    both are owned by their respective mixins and must resolve through the MRO.
    """

    _message_cache = None

    @property
    def message_cache(self):
        if not self._message_cache:
            from src.api.entities_api.cache.message_cache import \
                get_sync_message_cache

            self._message_cache = get_sync_message_cache()
        return self._message_cache

    # -----------------------------------------------------
    # PURE HELPERS (SYNC)
    # -----------------------------------------------------

    @staticmethod
    def _normalize_roles(msgs: List[Dict]) -> List[Dict]:
        out: List[Dict] = []

        for m in msgs:
            raw_role = str(m.get("role", "user")).lower()
            role = (
                raw_role
                if raw_role in ("user", "assistant", "system", "tool", "platform")
                else "user"
            )

            raw_content = m.get("content")

            # ── Multimodal content arrays must pass through untouched ──────
            # If content is already a list (Qwen base64 array from hydration)
            # do NOT stringify it — the LLM needs the list structure intact.
            if isinstance(raw_content, list):
                content = raw_content
            elif raw_content is not None:
                content = str(raw_content).strip()
            else:
                content = None

            normalized_msg = {"role": role, "content": content}
            has_tool_calls = False

            if "tool_calls" in m and m["tool_calls"]:
                normalized_msg["tool_calls"] = m["tool_calls"]
                has_tool_calls = True

            elif (
                role == "assistant"
                and content
                and isinstance(content, str)
                and content.strip().startswith("[{")
                and "function" in content
            ):
                try:
                    parsed_tools = json.loads(content)
                    if (
                        isinstance(parsed_tools, list)
                        and parsed_tools
                        and "function" in parsed_tools[0]
                    ):
                        normalized_msg["tool_calls"] = parsed_tools
                        has_tool_calls = True
                except (json.JSONDecodeError, TypeError):
                    pass

            if has_tool_calls:
                normalized_msg["content"] = ""

            if role == "tool":
                if m.get("tool_call_id") is not None:
                    normalized_msg["tool_call_id"] = m["tool_call_id"]
                if m.get("name"):
                    normalized_msg["name"] = m["name"]

            out.append(normalized_msg)

        return out

    @staticmethod
    def _resolve_and_prioritize_platform_tools(
        tools: Optional[List[Dict[str, Any]]],
        *,
        decision_telemetry: bool = False,
    ) -> List[Dict[str, Any]]:

        mandatory_platform_tools = []
        if decision_telemetry:
            mandatory_platform_tools.append(record_tool_decision)

        tools = tools or []
        resolved_platform_tools = []
        resolved_user_tools = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            tool_type = tool.get("type")

            if (
                tool_type in PLATFORM_TOOL_MAP
                and tool_type != "function"
                and "function" not in tool
            ):
                platform_def = PLATFORM_TOOL_MAP[tool_type]
                if isinstance(platform_def, (list, tuple)):
                    resolved_platform_tools.extend(platform_def)
                else:
                    resolved_platform_tools.append(platform_def)
            else:
                resolved_user_tools.append(tool)

        LOG.debug("RESOLVED TOOLS: %s", json.dumps(resolved_platform_tools, default=str))

        platform_tools_all = mandatory_platform_tools + resolved_platform_tools

        seen_names = set()
        deduped_platform_tools = []

        for tool in platform_tools_all:
            try:
                name = tool["function"]["name"]
            except (KeyError, TypeError):
                continue
            if name not in seen_names:
                seen_names.add(name)
                deduped_platform_tools.append(tool)

        return deduped_platform_tools + resolved_user_tools

    async def _build_system_message(
        self,
        assistant_id: str,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = False,
    ) -> Dict:
        cache = self.get_assistant_cache()
        cfg = await cache.retrieve(assistant_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        instruction_keys = list(
            dict.fromkeys(
                [
                    *(["TOOL_DECISION_PROTOCOL"] if decision_telemetry else []),
                    *(L3_INSTRUCTIONS if agent_mode else L2_INSTRUCTIONS),
                    *(L3_WEB_USE_INSTRUCTIONS if web_access else []),
                ]
            )
        )

        platform_instructions = assemble_instructions(include_keys=instruction_keys)
        raw_tools_list = list(cfg.get("tools") or [])

        if web_access:
            has_web_tool = any(
                isinstance(t, dict) and t.get("type") == "web_search" for t in raw_tools_list
            )
            if not has_web_tool:
                raw_tools_list.append({"type": "web_search"})

        final_tools = self._resolve_and_prioritize_platform_tools(
            tools=raw_tools_list,
            decision_telemetry=decision_telemetry,
        )

        content_blocks = [
            f"Today's date and time: {today}",
            "### ASSISTANT INSTRUCTIONS",
            cfg.get("instructions", ""),
            "### OPERATIONAL PROTOCOLS",
            platform_instructions,
            "### AVAILABLE TOOLS",
            f"tools:\n{json.dumps(final_tools)}",
        ]

        return {
            "role": "system",
            "content": "\n\n".join(block for block in content_blocks if block),
        }

    async def _build_research_supervisor_message(
        self,
        assistant_id: str,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = True,
        deep_research: bool = True,
    ) -> Dict:
        cache = self.get_assistant_cache()
        cfg = await cache.retrieve(assistant_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        instruction_keys = list(
            dict.fromkeys(
                [
                    *(["TOOL_DECISION_PROTOCOL"] if decision_telemetry else []),
                    *LEVEL_4_SUPERVISOR_INSTRUCTIONS,
                ]
            )
        )

        platform_instructions = assemble_instructions(include_keys=instruction_keys)
        raw_tools_list = list(cfg.get("tools") or [])

        if web_access:
            has_web_tool = any(
                isinstance(t, dict) and t.get("type") == "web_search" for t in raw_tools_list
            )
            if not has_web_tool:
                raw_tools_list.append({"type": "web_search"})

        final_tools = self._resolve_and_prioritize_platform_tools(
            tools=raw_tools_list,
            decision_telemetry=decision_telemetry,
        )

        content_blocks = [
            f"Today's date and time: {today}",
            "### ASSISTANT INSTRUCTIONS",
            cfg.get("instructions", ""),
            "### OPERATIONAL PROTOCOLS",
            platform_instructions,
            "### AVAILABLE TOOLS",
            f"tools:\n{json.dumps(final_tools)}",
        ]

        return {
            "role": "system",
            "content": "\n\n".join(block for block in content_blocks if block),
        }

    async def _build_research_worker_message(
        self,
        assistant_id: str,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = True,
        deep_research: bool = True,
    ) -> Dict:
        cache = self.get_assistant_cache()
        cfg = await cache.retrieve(assistant_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        instruction_keys = list(
            dict.fromkeys(
                [
                    *(["TOOL_DECISION_PROTOCOL"] if decision_telemetry else []),
                    *L4_RESEARCH_INSTRUCTIONS,
                ]
            )
        )

        platform_instructions = assemble_instructions(include_keys=instruction_keys)
        raw_tools_list = list(cfg.get("tools") or [])

        if web_access:
            has_web_tool = any(
                isinstance(t, dict) and t.get("type") == "web_search" for t in raw_tools_list
            )
            if not has_web_tool:
                raw_tools_list.append({"type": "web_search"})

        final_tools = self._resolve_and_prioritize_platform_tools(
            tools=raw_tools_list,
            decision_telemetry=decision_telemetry,
        )

        content_blocks = [
            f"Today's date and time: {today}",
            "### ASSISTANT INSTRUCTIONS",
            cfg.get("instructions", ""),
            "### OPERATIONAL PROTOCOLS",
            platform_instructions,
            "### AVAILABLE TOOLS",
            f"tools:\n{json.dumps(final_tools)}",
        ]

        return {
            "role": "system",
            "content": "\n\n".join(block for block in content_blocks if block),
        }

    async def _build_senior_engineer_message(
        self,
        assistant_id: str,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = True,
        deep_research: bool = True,
    ) -> Dict:
        cache = self.get_assistant_cache()
        cfg = await cache.retrieve(assistant_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        instruction_keys = list(
            dict.fromkeys(
                [
                    *(["TOOL_DECISION_PROTOCOL"] if decision_telemetry else []),
                    *L4_SENIOR_ENGINEER_INSTRUCTIONS,
                ]
            )
        )

        platform_instructions = assemble_instructions(include_keys=instruction_keys)
        raw_tools_list = list(cfg.get("tools") or [])

        if web_access:
            has_web_tool = any(
                isinstance(t, dict) and t.get("type") == "web_search" for t in raw_tools_list
            )
            if not has_web_tool:
                raw_tools_list.append({"type": "web_search"})

        final_tools = self._resolve_and_prioritize_platform_tools(
            tools=raw_tools_list,
            decision_telemetry=decision_telemetry,
        )

        content_blocks = [
            f"Today's date and time: {today}",
            "### ASSISTANT INSTRUCTIONS",
            cfg.get("instructions", ""),
            "### OPERATIONAL PROTOCOLS",
            platform_instructions,
            "### AVAILABLE TOOLS",
            f"tools:\n{json.dumps(final_tools)}",
        ]

        return {
            "role": "system",
            "content": "\n\n".join(block for block in content_blocks if block),
        }

    async def _build_junior_engineer_message(
        self,
        assistant_id: str,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = True,
        deep_research: bool = True,
    ) -> Dict:
        cache = self.get_assistant_cache()
        cfg = await cache.retrieve(assistant_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        instruction_keys = list(
            dict.fromkeys(
                [
                    *(["TOOL_DECISION_PROTOCOL"] if decision_telemetry else []),
                    *L4_JUNIOR_ENGINEER_INSTRUCTIONS,
                ]
            )
        )

        platform_instructions = assemble_instructions(include_keys=instruction_keys)
        raw_tools_list = list(cfg.get("tools") or [])

        final_tools = self._resolve_and_prioritize_platform_tools(
            tools=raw_tools_list,
            decision_telemetry=decision_telemetry,
        )

        content_blocks = [
            f"Today's date and time: {today}",
            "### ASSISTANT INSTRUCTIONS",
            cfg.get("instructions", ""),
            "### OPERATIONAL PROTOCOLS",
            platform_instructions,
            "### AVAILABLE TOOLS",
            f"tools:\n{json.dumps(final_tools)}",
        ]

        return {
            "role": "system",
            "content": "\n\n".join(block for block in content_blocks if block),
        }

    async def _build_native_function_calls_system_message(
        self,
        assistant_id: str,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = True,
    ) -> Dict:
        cache = self.get_assistant_cache()
        cfg = await cache.retrieve(assistant_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        instruction_keys = list(
            dict.fromkeys(
                [
                    *(["TOOL_DECISION_PROTOCOL"] if decision_telemetry else []),
                    *(L3_INSTRUCTIONS if agent_mode else NO_CORE_INSTRUCTIONS),
                    *(L3_WEB_USE_INSTRUCTIONS if web_access else []),
                ]
            )
        )

        platform_instructions = assemble_instructions(include_keys=instruction_keys)
        raw_tools_list = list(cfg.get("tools") or [])

        if web_access:
            has_web_tool = any(
                isinstance(t, dict) and t.get("type") == "web_search" for t in raw_tools_list
            )
            if not has_web_tool:
                raw_tools_list.append({"type": "web_search"})

        final_tools = self._resolve_and_prioritize_platform_tools(
            tools=raw_tools_list,
            decision_telemetry=decision_telemetry,
        )

        content_blocks = [
            f"Today's date and time: {today}",
            "### ASSISTANT INSTRUCTIONS",
            cfg.get("instructions", ""),
            "### OPERATIONAL PROTOCOLS",
            platform_instructions,
            "### AVAILABLE TOOLS",
            f"tools:\n{json.dumps(final_tools)}",
        ]

        return {
            "role": "system",
            "content": "\n\n".join(block for block in content_blocks if block),
        }

    # -----------------------------------------------------
    # ASYNC CONTEXT WINDOW
    # -----------------------------------------------------

    async def _set_up_context_window(
        self,
        assistant_id: str,
        thread_id: str,
        trunk: Optional[bool] = True,
        structured_tool_call: Optional[bool] = False,
        force_refresh: Optional[bool] = False,
        decision_telemetry: bool = False,
        agent_mode: bool = False,
        web_access: bool = False,
        deep_research: bool = False,
        engineer: bool = False,
        research_worker: bool = False,
        junior_engineer: bool = False,
    ) -> List[Dict]:

        # 1. Build the System Message
        if engineer:
            system_msg = await self._build_senior_engineer_message(
                assistant_id=assistant_id,
                decision_telemetry=decision_telemetry,
                agent_mode=agent_mode,
                web_access=web_access,
                deep_research=deep_research,
            )
        elif deep_research:
            system_msg = await self._build_research_supervisor_message(
                assistant_id=assistant_id,
                decision_telemetry=decision_telemetry,
                agent_mode=agent_mode,
                web_access=web_access,
                deep_research=True,
            )
        elif research_worker:
            system_msg = await self._build_research_worker_message(
                assistant_id=assistant_id,
                decision_telemetry=decision_telemetry,
                agent_mode=agent_mode,
                web_access=web_access,
                deep_research=False,
            )
        elif junior_engineer:
            system_msg = await self._build_junior_engineer_message(
                assistant_id=assistant_id,
                decision_telemetry=decision_telemetry,
                agent_mode=agent_mode,
                web_access=web_access,
                deep_research=False,
            )
        elif structured_tool_call:
            system_msg = await self._build_native_function_calls_system_message(
                assistant_id=assistant_id,
                decision_telemetry=decision_telemetry,
                web_access=web_access,
            )
        else:
            system_msg = await self._build_system_message(
                assistant_id=assistant_id,
                decision_telemetry=decision_telemetry,
                agent_mode=agent_mode,
                web_access=web_access,
            )

        # 2. Retrieve Message History
        if force_refresh:
            LOG.debug(f"[CTX-REFRESH] Force Refresh Active for {thread_id}")
            try:
                # Fetch lean messages (file_ids, no base64) and store in Redis
                lean_hist = await self._native_exec.get_raw_messages(thread_id)
            except Exception as e:
                LOG.warning(
                    "[CTX-REFRESH] get_raw_messages failed for %s: %s. " "Returning empty history.",
                    thread_id,
                    e,
                )
                lean_hist = []

            # Store lean in Redis — never store hydrated base64
            self.message_cache.set_history_sync(thread_id, lean_hist)
            msgs = lean_hist

        else:
            # Hot path — Redis returns lean messages with file_id refs
            msgs = self.message_cache.get_history_sync(thread_id)
            LOG.debug(f"[CTX-CACHE] Redis hit for {thread_id}")

        # 3. Hydrate images just-in-time — resolve file_ids → base64 right
        #    before LLM dispatch. Expired files are skipped gracefully.
        #    Plain text messages pass through untouched.
        msgs = await self._native_exec.hydrate_messages(msgs)

        # 4. Filter system messages, prepend fresh system msg, normalize
        msgs = [m for m in msgs if m.get("role") != "system"]
        full_context = [system_msg] + msgs
        normalized = self._normalize_roles(full_context)

        LOG.info(f"=== OUTBOUND CONTEXT (Size: {len(normalized)}) ===")

        if trunk:
            return self.conversation_truncator.truncate(normalized)

        return normalized

    # -----------------------------------------------------
    # REMAINING SYNC UTILITIES
    # -----------------------------------------------------

    def replace_system_message(
        self, context_window: list[dict], new_system_message: str | None = None
    ) -> list[dict]:
        filtered = [msg for msg in context_window if msg["role"] != "system"]
        if new_system_message is not None:
            filtered.insert(0, {"role": "system", "content": new_system_message})
        return filtered

    def prepare_native_tool_context(
        self, context_window: List[Dict]
    ) -> Tuple[List[Dict], Optional[List[Dict]]]:

        cleaned_ctx = []
        extracted_tools = None

        for msg in context_window:
            role = msg.get("role")
            content = msg.get("content") or ""
            new_msg = dict(msg)

            if role == "system" and "tools:\n[" in content:
                try:
                    parts = content.split("tools:\n", 1)
                    system_text = parts[0].strip()
                    tools_json_str = parts[1].strip()

                    if "\n" in tools_json_str:
                        json_part, instructions_part = tools_json_str.split("\n", 1)
                        extracted_tools = json.loads(json_part)
                        new_msg["content"] = f"{system_text}\n{instructions_part}".strip()
                    else:
                        extracted_tools = json.loads(tools_json_str)
                        new_msg["content"] = system_text or "You are a helpful assistant."
                except Exception as e:
                    LOG.error(f"[CTX-MIXIN] Failed tool extraction: {e}")

            if role == "tool" and not isinstance(content, str):
                new_msg["content"] = json.dumps(content)

            cleaned_ctx.append(new_msg)

        return cleaned_ctx, extracted_tools
