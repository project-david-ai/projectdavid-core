# src/api/entities_api/orchestration/mixins/code_interpreter_mixin.py
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import jwt
from projectdavid_common import ToolValidator
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class CodeInterpreterMixin:
    """
    Mixin that handles the `code_interpreter` tool.
    """

    @staticmethod
    def _format_level2_code_error(error_content: str) -> str:
        return (
            f"❌ CODE EXECUTION FAILED:\n{error_content}\n\n"
            "## IMMEDIATE RECOVERY INSTRUCTIONS:\n"
            "1. Analyze the Traceback above.\n"
            "2. If 'ModuleNotFoundError': Use standard libraries (e.g., 'python-docx' not 'docx').\n"
            "3. If 'FileNotFound': Check your paths. Did you generate the data first?\n"
            "4. **DO NOT APOLOGIZE.** Immediately generate a new tool call with the corrected code."
        )

    def _generate_sandbox_token(self, subject_id: str) -> str:
        secret = os.getenv("SANDBOX_AUTH_SECRET")
        if not secret:
            LOG.error("CRITICAL: SANDBOX_AUTH_SECRET is missing in environment variables.")
            raise ValueError("Server configuration error: Sandbox secret missing.")

        payload = {
            "sub": subject_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            "scopes": ["execution"],
        }
        return jwt.encode(payload, secret, algorithm="HS256")

    def _code_status(self, activity: str, state: str, run_id: str) -> str:
        """Emits a code interpreter status event conforming to the EVENT_CONTRACT."""
        return json.dumps(
            {
                "type": "code_status",
                "activity": activity,
                "state": state,
                "tool": "code_interpreter",
                "run_id": run_id,
            }
        )

    async def handle_code_interpreter_action(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: dict,
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:

        # 1. Notify start
        yield self._code_status("Preparing code interpreter...", "in_progress", run_id)

        # --- VALIDATION ---
        validator = ToolValidator()
        validator.schema_registry = {"code_interpreter": ["code"]}
        validation_error = validator.validate_args("code_interpreter", arguments_dict)

        if validation_error:
            LOG.warning(f"CodeInterpreter ▸ Validation Failed: {validation_error}")

            # Surface as a recoverable status message — not a raw error chunk
            yield self._code_status(f"Validation failed: {validation_error}", "error", run_id)

            # Native execution for failed validation (creates action, marks failed, submits output)
            await self._native_exec.submit_failed_tool_execution(
                tool_name="code_interpreter",
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                error_message=f"{validation_error}\nPlease correct arguments.",
                function_args=arguments_dict,
                decision=decision,
            )
            return

        # 2. Create Action Natively
        action = None
        try:
            action = await self._native_exec.create_action(
                tool_name="code_interpreter",
                run_id=run_id,
                tool_call_id=tool_call_id,
                function_args=arguments_dict,
                decision=decision,
            )
        except Exception as e:
            LOG.error(f"CodeInterpreter ▸ Action creation failed: {e}")
            yield self._code_status(f"Failed to register action: {e}", "error", run_id)
            return

        code: str = arguments_dict.get("code", "")

        # ⚡ HOT CODE REPLAY — streams the code itself, not errors
        if code:
            yield json.dumps(
                {
                    "stream_type": "code_execution",
                    "chunk": {"type": "hot_code", "content": "```python\n"},
                }
            )
            for line in code.splitlines(keepends=True):
                yield json.dumps(
                    {
                        "stream_type": "code_execution",
                        "chunk": {"type": "hot_code", "content": line},
                    }
                )
            yield json.dumps(
                {
                    "stream_type": "code_execution",
                    "chunk": {"type": "hot_code", "content": "\n```\n"},
                }
            )

        uploaded_files: List[dict] = []
        hot_code_buffer: List[str] = []
        decoder = json.JSONDecoder()
        stream_buffer = ""
        execution_had_error = False

        # 3. Stream Execution Output
        yield self._code_status("Executing code in sandbox...", "in_progress", run_id)

        try:
            auth_token = self._generate_sandbox_token(subject_id=f"run_{run_id}")
            sync_iter = iter(self.code_execution_client.stream_output(code, token=auth_token))

            def safe_next(it):
                try:
                    return next(it)
                except (StopIteration, Exception):
                    return None

            while True:
                chunk_str = await asyncio.to_thread(safe_next, sync_iter)
                if chunk_str is None:
                    break

                stream_buffer += chunk_str

                while stream_buffer:
                    stream_buffer = stream_buffer.lstrip()
                    if not stream_buffer:
                        break
                    try:
                        wrapper, idx = decoder.raw_decode(stream_buffer)
                        stream_buffer = stream_buffer[idx:]
                    except json.JSONDecodeError:
                        break

                    payload = (
                        wrapper["chunk"]
                        if isinstance(wrapper, dict) and "chunk" in wrapper
                        else wrapper if isinstance(wrapper, dict) else None
                    )

                    if not isinstance(payload, dict):
                        continue

                    ctype = payload.get("type")
                    content = payload.get("content")

                    if ctype in ("hot_code_output", "stdout", "stderr", "console"):
                        if content is not None:
                            clean_content = str(content).replace("\\n", "\n")
                            hot_code_buffer.append(clean_content)

                            is_error_line = (
                                ctype == "stderr"
                                or "Traceback" in clean_content
                                or "Error:" in clean_content
                            )

                            if is_error_line:
                                execution_had_error = True
                                # Log internally; do NOT forward raw error to consumer
                                LOG.warning(
                                    f"CodeInterpreter ▸ Sandbox stderr: {clean_content[:200]}"
                                )
                            else:
                                # Only forward clean stdout/output to consumer
                                yield json.dumps(
                                    {
                                        "stream_type": "code_execution",
                                        "chunk": {
                                            "type": "hot_code_output",
                                            "content": clean_content,
                                        },
                                    }
                                )

                    elif ctype == "status":
                        if content == "complete":
                            raw_files = payload.get("uploaded_files")
                            LOG.info(
                                f"[FILE_DEBUG] Sandbox 'complete' status received. Raw Files Payload: {raw_files}"
                            )
                            if isinstance(raw_files, list) and len(raw_files) > 0:
                                uploaded_files.extend(raw_files)
                            else:
                                LOG.warning(
                                    "[FILE_DEBUG] No files found in sandbox completion payload."
                                )

                        # Forward sandbox status events as activity messages
                        status_val = payload.get("status", content or "unknown")
                        yield self._code_status(
                            f"Sandbox status: {status_val}", "in_progress", run_id
                        )

                    elif ctype == "error":
                        # Recoverable sandbox error — capture for LLM, surface as activity
                        execution_had_error = True
                        LOG.error(f"CodeInterpreter ▸ Sandbox error chunk: {content}")
                        hot_code_buffer.append(f"[Code Exec Error] {content}")
                        yield self._code_status(
                            "Code execution encountered an error — attempting recovery...",
                            "error",
                            run_id,
                        )

        except Exception as stream_err:
            execution_had_error = True
            LOG.error(f"CodeInterpreter ▸ Stream error: {stream_err}")
            yield self._code_status(
                "Sandbox stream interrupted — attempting recovery...",
                "error",
                run_id,
            )

        # ------------------------------------------------------------------
        # PHANTOM FILE DETECTION (Self-Healing Logic)
        # ------------------------------------------------------------------
        code_intent_save = any(
            k in code for k in [".save(", ".to_csv(", ".to_excel(", "open(", ".write("]
        )
        is_reading_only = "r'" in code or 'r"' in code or "read_csv" in code

        if (
            code_intent_save
            and not is_reading_only
            and not uploaded_files
            and not execution_had_error
        ):
            LOG.warning(
                "CodeInterpreter ▸ Phantom File Detected. Code implies save, but sandbox returned nothing."
            )
            error_msg = (
                "SYSTEM ERROR: Your code appears to save a file (detected '.save' or '.write'), "
                "but the sandbox execution completed without returning any generated files.\n\n"
                "Possible Fixes:\n"
                "1. Did you save to the current directory? (Use `doc.save('filename.docx')`, NOT `/tmp/...`)\n"
                "2. Did the script crash silently? Add `print('Finished saving')` to debug.\n"
                "3. Retry execution ensuring the file is saved to the root path."
            )
            execution_had_error = True
            hot_code_buffer.append(error_msg)

            # Surface as a recoverable activity — not a raw error chunk
            yield self._code_status(
                "File generation expected but not returned — requesting retry...",
                "error",
                run_id,
            )

        # 4. Final Summary Calculation
        raw_output = "\n".join(hot_code_buffer).strip()

        if execution_had_error:
            llm_content = self._format_level2_code_error(raw_output or "Unknown execution failure.")
            user_content = raw_output or "❌ Code execution failed."
        else:
            llm_content = raw_output or "[Code executed successfully.]"
            user_content = llm_content

        # 5. Process Files
        LOG.info(
            f"[FILE_DEBUG] Entering File Processing Loop. Total queue size: {len(uploaded_files)}"
        )

        for file_meta in uploaded_files:
            file_id = file_meta.get("id")
            filename = file_meta.get("filename")

            if not file_id or not filename:
                continue

            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            try:
                # Still utilizing the SDK for file storage/signed URLs
                file_url = file_meta.get("url")
                if not file_url:
                    file_url = await asyncio.to_thread(
                        self.project_david_client.files.get_signed_url,
                        file_id=file_id,
                        use_real_filename=True,
                    )

                yield json.dumps(
                    {
                        "stream_type": "code_execution",
                        "run_id": run_id,
                        "chunk": {
                            "type": "code_interpreter_file",
                            "filename": filename,
                            "file_id": file_id,
                            "url": file_url,
                            "mime_type": mime_type,
                            "base64": None,
                        },
                    }
                )

            except Exception as e:
                LOG.error(f"[FILE_DEBUG] Error generating URL: {e}", exc_info=True)
                yield self._code_status(
                    f"Could not generate download URL for {filename}",
                    "error",
                    run_id,
                )

        # 6. Close out current stream — send clean user_content, never raw llm_content
        yield json.dumps(
            {
                "stream_type": "code_execution",
                "chunk": {"type": "content", "content": user_content},
            }
        )

        final_state = "completed" if not execution_had_error else "error"
        yield self._code_status(
            (
                "Code execution complete."
                if not execution_had_error
                else "Code execution finished with errors."
            ),
            final_state,
            run_id,
        )

        # 7. Submit Result Natively (LLM receives llm_content with recovery instructions)
        try:
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=llm_content,
                action_id=action.id if action else None,
                is_error=execution_had_error,
            )
            # Update Action State locally
            if action:
                status_val = (
                    StatusEnum.failed.value if execution_had_error else StatusEnum.completed.value
                )
                await self._native_exec.update_action_status(action.id, status_val)

        except Exception as e:
            LOG.error(f"CodeInterpreter ▸ Submission failure: {e}")
            yield self._code_status(f"Tool output submission failed: {e}", "error", run_id)

    def process_hot_code_buffer(
        self,
        buffer: str,
        start_index: int,
        cursor: int,
        redis_client: Any,
        stream_key: str,
    ) -> tuple[int, int, Optional[str]]:
        if start_index == -1:
            match = re.search(r"[\"\']code[\"\']\s*:\s*[\"\']", buffer)
            if match:
                start_index = match.end()
            else:
                return start_index, cursor, None

        full_code_value = buffer[start_index:]
        unsent_buffer = full_code_value[cursor:]
        if not unsent_buffer:
            return start_index, cursor, None

        has_newline = "\\n" in unsent_buffer
        is_long_enough = len(unsent_buffer) > 15
        is_closed = unsent_buffer.endswith('"') or unsent_buffer.endswith("'")
        safe_cut = not unsent_buffer.endswith("\\")

        if (has_newline or is_long_enough or is_closed) and safe_cut:
            cursor += len(unsent_buffer)
            clean_segment = (
                unsent_buffer.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
            )
            if len(clean_segment) == 1 and clean_segment in ('"', "}"):
                return start_index, cursor, None
            payload_dict = {"type": "hot_code", "content": clean_segment}
            return start_index, cursor, json.dumps(payload_dict)

        return start_index, cursor, None
