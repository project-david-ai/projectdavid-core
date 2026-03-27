# src/api/entities_api/orchestration/mixins/file_search_mixin.py
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

# Server-side native DB manager
from projectdavid.clients.vector_store_manager import VectorStoreManager
from projectdavid_common import ToolValidator
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class FileSearchMixin:
    """
    Drive the **file_search** tool-call asynchronously.
    """

    @staticmethod
    def _format_level2_search_error(error_content: str, query: str) -> str:
        return (
            f"File Search Result for '{query}': {error_content}\n\n"
            "Instructions: If no results were found, please try again with broader keywords, "
            "remove specific filters, or check if the document names provided in context exist. "
            "If this was a system error, you may attempt to retry once."
        )

    @staticmethod
    def _format_tool_resources_error(query: str) -> str:
        return (
            f"File Search Result for '{query}': Configuration Error — "
            "No vector stores are associated with this assistant.\n\n"
            "This means the assistant's 'tool_resources' field either missing, empty, "
            "or does not contain a valid 'file_search.vector_store_ids' list.\n\n"
            "Instructions: Inform the user that file search is currently unavailable because "
            "no document store has been linked to this assistant. They should upload files and "
            "attach a vector store to the assistant before retrying this operation."
        )

    async def _resolve_vector_store_ids(self, assistant_id: str) -> list[str]:
        try:
            config = await self.assistant_cache.retrieve(assistant_id)
            tool_resources: dict = config.get("tool_resources") or {}
            ids: list = tool_resources.get("file_search", {}).get(
                "vector_store_ids", []
            )
            return [vid for vid in ids if vid]
        except Exception as exc:
            LOG.error(
                "FileSearch ▸ Failed to resolve tool_resources for assistant %s: %s",
                assistant_id,
                exc,
            )
            return []

    async def handle_file_search(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> None:
        ts_start = asyncio.get_event_loop().time()

        # ---[L2] SHARED INPUT VALIDATION ---
        validator = ToolValidator()
        validator.schema_registry = {"file_search": ["query_text"]}

        validation_error = validator.validate_args("file_search", arguments_dict)

        if validation_error:
            LOG.warning(f"FileSearch ▸ Validation Failed: {validation_error}")
            error_msg = f"{validation_error}\nPlease provide a valid 'query_text' string for the search."

            await self._native_exec.submit_failed_tool_execution(
                tool_name="file_search",
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                error_message=error_msg,
                function_args=arguments_dict,
                decision=decision,
            )
            return

        query_text: str = arguments_dict.get("query_text", "")

        # ---[L3] RESOLVE VECTOR STORES FROM tool_resources ---
        vector_store_ids = await self._resolve_vector_store_ids(assistant_id)

        if not vector_store_ids:
            LOG.error(
                "FileSearch ▸ No vector_store_ids found in tool_resources for assistant %s.",
                assistant_id,
            )
            await self._native_exec.submit_failed_tool_execution(
                tool_name="file_search",
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                error_message=self._format_tool_resources_error(query_text),
                function_args=arguments_dict,
                decision=decision,
            )
            return

        # 1. Create Action Record Natively
        try:
            action = await self._native_exec.create_action(
                tool_name="file_search",
                run_id=run_id,
                tool_call_id=tool_call_id,
                function_args=arguments_dict,
                decision=decision,
            )
        except Exception as e:
            LOG.error(f"FileSearch ▸ Action creation failed for run {run_id}: {e}")
            return

        try:
            # 2. Fan out — search every vector store concurrently
            LOG.info(
                "FileSearch ▸ Searching %d vector store(s) natively for run %s: %s",
                len(vector_store_ids),
                run_id,
                vector_store_ids,
            )

            qdrant_host = os.getenv("VECTOR_STORE_HOST", "qdrant")
            vector_manager = VectorStoreManager(vector_store_host=qdrant_host)

            async def _search_one(
                vid: str,
            ) -> tuple[str, list | None, Exception | None]:
                try:
                    try:
                        top_k_val = int(arguments_dict.get("top_k", 5))
                    except (ValueError, TypeError):
                        top_k_val = 5

                    # A: Fetch metadata natively directly from the database!
                    store_info = await self._native_exec.get_vector_store(vid)

                    if not store_info:
                        raise Exception(
                            f"Vector store '{vid}' not found in the database."
                        )

                    # B: Local Embedding (Reusing SDK's loaded model to save RAM)
                    file_processor = self.project_david_client.vectors.file_processor
                    if store_info.vector_size == 1024:
                        vec_array = await asyncio.to_thread(
                            file_processor.encode_clip_text, query_text
                        )
                        vector_field = "caption_vector"
                    else:
                        vec_array = await asyncio.to_thread(
                            file_processor.encode_text, query_text
                        )
                        vector_field = None

                    query_vector = vec_array.tolist()

                    # C: Direct Qdrant query
                    raw_hits = await asyncio.to_thread(
                        vector_manager.query_store,
                        store_name=store_info.collection_name,
                        query_vector=query_vector,
                        top_k=top_k_val,
                        filters=arguments_dict.get("filters"),
                        vector_field=vector_field,
                    )

                    for h in raw_hits:
                        h["store_id"] = vid

                    return vid, raw_hits, None
                except Exception as exc:
                    return vid, None, exc

            outcomes = await asyncio.gather(
                *[_search_one(vid) for vid in vector_store_ids]
            )

            # 3. Aggregate results
            aggregated: list = []
            store_errors: list[str] = []

            for vid, results, exc in outcomes:
                if exc is not None:
                    LOG.error("FileSearch ▸ Store %s raised an exception: %s", vid, exc)
                    store_errors.append(f"Store '{vid}': {exc}")
                elif results:
                    aggregated.extend(results)

            # 4. Determine final outcome
            has_results = len(aggregated) > 0
            is_soft_failure = False

            if not has_results and not store_errors:
                is_soft_failure = True
                final_content = self._format_level2_search_error(
                    "No relevant document snippets found across all attached vector stores.",
                    query_text,
                )
                LOG.warning(
                    "FileSearch ▸ No results... Triggering self-correction turn."
                )

            elif not has_results and store_errors:
                is_soft_failure = True
                final_content = self._format_level2_search_error(
                    f"All vector store searches failed:\n{chr(10).join(store_errors)}",
                    query_text,
                )
                LOG.error("FileSearch ▸ All %d store(s) failed.", len(vector_store_ids))

            else:
                if store_errors:
                    LOG.warning(
                        "FileSearch ▸ Partial failure. Errors: %s", store_errors
                    )
                aggregated.sort(key=lambda x: x.get("score", 0), reverse=True)
                final_content = json.dumps(aggregated, indent=2)

            # 5. Native submit tool output
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=final_content,
                action_id=action.id,
                is_error=is_soft_failure,
            )

            # 6. Mark Native Action Status
            status_val = (
                StatusEnum.failed.value
                if is_soft_failure
                else StatusEnum.completed.value
            )
            await self._native_exec.update_action_status(action.id, status_val)

            LOG.info(
                "[%s] file_search completed in %.2fs (action=%s, stores=%d, results=%d)",
                run_id,
                asyncio.get_event_loop().time() - ts_start,
                action.id,
                len(vector_store_ids),
                len(aggregated),
            )

        except Exception as exc:
            LOG.error(
                "[%s] file_search HARD FAILURE action=%s exc=%s", run_id, action.id, exc
            )
            error_hint = self._format_level2_search_error(str(exc), query_text)

            try:
                await self._native_exec.update_action_status(
                    action.id, StatusEnum.failed.value
                )
                await self._native_exec.submit_tool_output(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_call_id=tool_call_id,
                    content=error_hint,
                    action_id=action.id,
                    is_error=True,
                )
            except Exception as inner:
                LOG.exception(
                    "[%s] Critical failure during error surfacing: %s", run_id, inner
                )
