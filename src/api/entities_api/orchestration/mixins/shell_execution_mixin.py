from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import jwt
from entities_api.platform_tools.handlers.computer.shell_command_interface import \
    run_shell_commands_async
from projectdavid_common.utilities.tool_validator import ToolValidator
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class ShellExecutionMixin:
    """
    Executes POSIX-style shell commands inside the Project-David sandbox.

    Stream event types yielded
    ──────────────────────────
    shell_status          — bare JSON, no stream_type wrapper.
                            Routes to ShellStatusEvent → shell status panel.

    stream_type="shell"   — envelope for PTY chunks.
      inner type="shell_output"   — raw PTY text (xterm reads via WebSocket;
                                    suppressed from SSE in SDK _map_chunk_to_event)
      inner type="computer_file"  — file download event (passes through SDK)

    NOTE: The final aggregated content chunk that existed in previous versions
    has been REMOVED.  It is not needed because:
      • The LLM receives the full PTY text via submit_tool_output (not SSE).
      • xterm already has every line via its sandbox WebSocket listener.
      • Forwarding it via SSE caused it to appear in the main chat / code panel.
    """

    # ── Error formatting ───────────────────────────────────────────────────────

    @staticmethod
    def _format_level2_shell_error(error_content: str) -> str:
        return (
            f"Shell Execution Failed:\n{error_content}\n\n"
            "Instructions: Please analyse the error output above. If a command was "
            "'not found', verify the binary name or check if it needs to be installed. "
            "If a path is incorrect, list the directory contents first. Correct your "
            "commands and retry execution."
        )

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _generate_shell_auth_token(self, subject_id: str, room_id: str) -> str:
        secret = os.getenv("SANDBOX_AUTH_SECRET")
        if not secret:
            LOG.error("CRITICAL: SANDBOX_AUTH_SECRET is missing.")
            raise ValueError("Server configuration error: Sandbox secret missing.")

        payload = {
            "sub": subject_id,
            "room": room_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            "scopes": ["execution", "shell"],
        }
        return jwt.encode(payload, secret, algorithm="HS256")

    # ── Status helper ──────────────────────────────────────────────────────────
    # Yields bare JSON (no stream_type wrapper) so it arrives at the SDK as a
    # top-level chunk and maps directly to ShellStatusEvent.

    def _shell_status(self, activity: str, state: str, run_id: str) -> str:
        return json.dumps(
            {
                "type": "shell_status",
                "activity": activity,
                "state": state,
                "tool": "computer",
                "run_id": run_id,
            }
        )

    # ── Main handler ───────────────────────────────────────────────────────────

    async def handle_shell_action(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator handler for shell commands.

        Chunks yielded and their SSE fate
        ──────────────────────────────────
        _shell_status(...)
            → bare {"type":"shell_status",...}
            → SDK: ShellStatusEvent → Flask: "shell_status" → shell status panel ✓

        json.dumps({"stream_type":"shell","chunk":{"type":"shell_output",...}})
            → SDK step 1 unwraps JSON-in-content, step 2 unwraps shell envelope
            → c_type="shell_output", _from_shell_envelope=True → suppressed ✓
            (xterm reads via WebSocket — no SSE copy needed)

        json.dumps({"stream_type":"shell","chunk":{"type":"computer_file",...}})
            → SDK unwraps → c_type="computer_file" → ComputerGeneratedFileEvent
            → Flask: "computer_file" → shellFileAttachments ✓
        """

        tool_name = "computer"
        LOG.info("ShellExecutionMixin: started for run_id=%s", run_id)

        yield self._shell_status("Preparing shell executor...", "in_progress", run_id)

        # ── Validation ────────────────────────────────────────────────────────

        validator = ToolValidator()
        validator.schema_registry = {tool_name: ["commands"]}
        validation_error = validator.validate_args(tool_name, arguments_dict)

        if validation_error:
            LOG.warning("ShellExecution ▸ Validation Failed: %s", validation_error)
            yield self._shell_status(f"Validation failed: {validation_error}", "error", run_id)

            action = None
            try:
                action = await self._native_exec.create_action(
                    tool_name=tool_name,
                    run_id=run_id,
                    tool_call_id=tool_call_id,
                    function_args=arguments_dict,
                    decision=decision,
                )
            except Exception:
                pass

            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=f"{validation_error}\nPlease correct the arguments and retry.",
                action=action,
                is_error=True,
            )
            return

        # ── Create action ─────────────────────────────────────────────────────

        action = None
        try:
            action = await self._native_exec.create_action(
                tool_name=tool_name,
                run_id=run_id,
                tool_call_id=tool_call_id,
                function_args=arguments_dict,
                decision=decision,
            )
        except Exception as e:
            LOG.error("ShellExecution ▸ Action creation failed: %s", e)
            yield self._shell_status(f"Failed to register action: {e}", "error", run_id)
            return

        commands: List[str] = arguments_dict.get("commands", [])
        if not commands:
            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content="No commands provided.",
                action=action,
            )
            return

        # ── Execute & stream ──────────────────────────────────────────────────

        yield self._shell_status("Executing in sandbox shell...", "in_progress", run_id)

        text_chunks: List[str] = []
        harvested_files: List[dict] = []
        execution_had_error: bool = False

        try:
            auth_token = self._generate_shell_auth_token(
                subject_id=f"run_{run_id}",
                room_id=thread_id,
            )

            async for chunk in run_shell_commands_async(
                commands, thread_id=thread_id, token=auth_token
            ):
                # ── computer_file chunks ──────────────────────────────────────
                if chunk.startswith('{"type": "computer_file"'):
                    try:
                        file_meta = json.loads(chunk)
                        if file_meta.get("type") == "computer_file":
                            harvested_files.append(file_meta)
                            continue
                    except json.JSONDecodeError:
                        pass

                # ── Plain PTY text ────────────────────────────────────────────
                # Accumulated for LLM tool output (submit_tool_output below).
                # Forwarded via SSE as stream_type="shell" so the SDK can
                # identify and suppress it — xterm reads via WebSocket instead.
                text_chunks.append(chunk)

                yield json.dumps(
                    {
                        "stream_type": "shell",
                        "chunk": {"type": "shell_output", "content": chunk},
                    }
                )

                # Filter out known benign sandbox startup warnings
                chunk_lower = chunk.lower()
                eval_chunk = chunk_lower.replace("bash: /root/.bashrc: permission denied", "")
                eval_chunk = eval_chunk.replace("warning: an existing sandbox was detected", "")

                if any(
                    marker in eval_chunk
                    for marker in [
                        "not found",
                        "permission denied",
                        "error:",
                        "no such file",
                    ]
                ):
                    execution_had_error = True

        except Exception as e:
            execution_had_error = True
            LOG.error("ShellExecution ▸ Exception during execution: %s", e)
            yield self._shell_status(
                "Shell execution interrupted — attempting recovery...",
                "error",
                run_id,
            )

        # ── Process harvested files ───────────────────────────────────────────

        LOG.info("[FILE_DEBUG] Shell harvest queue: %d file(s)", len(harvested_files))

        for file_meta in harvested_files:
            file_id = file_meta.get("file_id")
            filename = file_meta.get("filename")

            if not file_id or not filename:
                continue

            mime_type = (
                file_meta.get("mime_type")
                or mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            )

            try:
                file_url = file_meta.get("url")
                try:
                    file_url = await asyncio.to_thread(
                        self.project_david_client.files.get_signed_url,
                        file_id=file_id,
                        use_real_filename=True,
                    )
                except Exception as url_err:
                    LOG.warning(
                        "[FILE_DEBUG] Could not refresh signed URL for %s: %s — "
                        "using sandbox URL as fallback.",
                        filename,
                        url_err,
                    )

                yield json.dumps(
                    {
                        "stream_type": "shell",
                        "run_id": run_id,
                        "chunk": {
                            "type": "computer_file",
                            "filename": filename,
                            "file_id": file_id,
                            "url": file_url,
                            "mime_type": mime_type,
                            "base64": None,
                        },
                    }
                )

            except Exception as e:
                LOG.error("[FILE_DEBUG] Error generating URL for %s: %s", filename, e)
                yield self._shell_status(
                    f"Could not generate download URL for {filename}",
                    "error",
                    run_id,
                )

        # ── Finalise ──────────────────────────────────────────────────────────

        raw_output = "".join(text_chunks).strip()

        if execution_had_error:
            llm_content = self._format_level2_shell_error(raw_output or "Unknown shell failure.")
        else:
            llm_content = raw_output or "[Shell commands executed successfully.]"

        # NOTE: No final "content" SSE chunk is emitted here.
        # Previous versions yielded {"stream_type":"shell","chunk":{"type":"content",...}}
        # which leaked through the SDK as a ContentEvent and appeared as plain
        # text in the main chat.  The LLM receives llm_content via
        # submit_tool_output below — no SSE copy is needed.

        final_state = "completed" if not execution_had_error else "error"
        yield self._shell_status(
            (
                "Shell execution complete."
                if not execution_had_error
                else "Shell execution finished with errors."
            ),
            final_state,
            run_id,
        )

        # ── Submit tool output to LLM ─────────────────────────────────────────

        try:
            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=llm_content,
                action=action,
                is_error=execution_had_error,
            )
        except Exception as e:
            LOG.error("ShellExecution ▸ Tool output submission failed: %s", e)
            yield self._shell_status(f"Tool output submission failed: {e}", "error", run_id)
            return

        # ── Update action status ───────────────────────────────────────────────

        try:
            await self._native_exec.update_action_status(
                action.id,
                (
                    StatusEnum.completed.value
                    if not execution_had_error
                    else StatusEnum.failed.value
                ),
            )
        except Exception as e:
            LOG.error("ShellExecution ▸ Action status update failed: %s", e)
