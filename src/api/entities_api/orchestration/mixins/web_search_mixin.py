from __future__ import annotations

import asyncio
import json
import os
import re
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from projectdavid_common import ToolValidator
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

LOG = LoggingUtility()

# Hard cap on scroll_web_page calls per URL per research session.
# After this many scrolls on a single URL, the tool returns a blocking
# instruction instead of executing, forcing the worker to pivot.
SCROLL_LIMIT_PER_URL = 3

# Internal SearxNG container URL — direct, no SDK round-trip needed.
SEARXNG_BASE_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARXNG_TIMEOUT = int(os.getenv("SEARXNG_TIMEOUT", "15"))


def _status(run_id: str, tool: str, message: str, status: str = "running") -> str:
    """
    Emit a status event as raw JSON conforming to the stream EVENT_CONTRACT.

    Shape:
        {
            "type":    "web_status",
            "run_id":  "<uuid>",
            "tool":    "<tool_name>",
            "status":  "running" | "success" | "error" | "warning",
            "message": "<human readable>"
        }

    The SDK deserializes and routes this — no StatusEvent dataclass needed here.
    """
    return json.dumps(
        {
            "type": "web_status",
            "run_id": run_id,
            "tool": tool,
            "status": status,
            "message": message,
        }
    )


# ---------------------------------------------------------------------------
# SearxNG Direct Client
# ---------------------------------------------------------------------------


class SearxNGResult:
    """Lightweight wrapper around a single SearxNG result object."""

    def __init__(self, data: Dict[str, Any]):
        self.title: str = data.get("title", "No Title")
        self.url: str = data.get("url", "")
        self.snippet: str = data.get("content", "")
        self.engine: str = data.get("engine", "unknown")

        # FIX: Safely handle null/None or string scores from SearxNG
        raw_score = data.get("score")
        try:
            self.score: float = float(raw_score) if raw_score is not None else 0.0
        except (ValueError, TypeError):
            self.score = 0.0


class SearxNGClient:
    """
    Thin async HTTP client for the internal SearxNG container.

    Calls http://searxng:8080 directly — no SDK round-trip, no browser
    overhead. The browserless container is reserved for page *reading*,
    not discovery.
    """

    def __init__(
        self,
        base_url: str = SEARXNG_BASE_URL,
        timeout: int = SEARXNG_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = timeout

    async def search(
        self,
        query: str,
        count: int = 10,
        engines: Optional[List[str]] = None,
        categories: str = "general",
        language: str = "en",
    ) -> List[SearxNGResult]:
        """
        Execute a search and return a sorted list of SearxNGResult objects.

        Args:
            query:      The search query string.
            count:      Maximum results to return (applied client-side).
            engines:    Optional explicit engine list e.g. ['duckduckgo', 'bing'].
                        None → SearxNG uses its configured defaults.
            categories: SearxNG category — 'general', 'news', 'it', etc.
            language:   BCP-47 language code for results.

        Raises:
            RuntimeError: On HTTP error or connection failure.
        """
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
        }
        if engines:
            params["engines"] = ",".join(engines)

        LOG.info(f"🔎 SearxNG query: '{query}' | engines={engines or 'default'}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/search", params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            LOG.error(f"SearxNG HTTP error: {exc}")
            raise RuntimeError(f"SearxNG returned {exc.response.status_code}")
        except httpx.RequestError as exc:
            LOG.error(f"SearxNG connection error: {exc}")
            raise RuntimeError(f"Could not reach SearxNG at {self.base_url}: {exc}")

        raw = data.get("results", [])
        results = sorted(
            [SearxNGResult(r) for r in raw],
            key=lambda r: r.score,
            reverse=True,
        )
        LOG.info(f"✅ SearxNG returned {len(results)} results for '{query}'")
        return results[:count]

    async def format_for_agent(
        self,
        query: str,
        count: int = 10,
        engines: Optional[List[str]] = None,
        query_tier: int = 1,
    ) -> str:
        """
        Search and return a structured, agent-ready string with tier-aware
        guidance. Drop-in replacement for the old DDG markdown scrape output.

        Args:
            query:       Search query.
            count:       Max results.
            engines:     Optional engine override list.
            query_tier:  Tier (1/2/3) from _classify_query_tier — controls
                         the mandatory source count in the agent instructions.
        """
        try:
            results = await self.search(query=query, count=count, engines=engines)
        except RuntimeError as exc:
            return f"❌ SearxNG search failed: {exc}"

        if not results:
            return f"❌ No results found for '{query}'. Try a broader query or different keywords."

        min_sources = {1: 2, 2: 3, 3: 5}.get(query_tier, 2)

        lines: List[str] = [
            f"🔍 SEARCH RESULTS for '{query}' "
            f"({len(results)} found | Tier {query_tier} → requires {min_sources} sources):\n"
        ]

        for i, r in enumerate(results, 1):
            authority_tag = ""
            if any(d in r.url for d in [".gov", ".edu", ".org"]):
                authority_tag = " [HIGH AUTHORITY]"
            elif "wikipedia.org" in r.url:
                authority_tag = " [ENCYCLOPEDIA]"

            lines.append(
                f"{i}. **{r.title}**{authority_tag}  [via {r.engine}]\n"
                f"   URL: {r.url}\n"
                f"   {r.snippet}\n"
            )

        lines.append(
            f"\n⚠️ MANDATORY NEXT ACTION:\n"
            f"You MUST call read_web_page on at least {min_sources} of the URLs above.\n"
            f"Current progress: 0/{min_sources} sources read.\n"
            f"SERP snippets alone are NOT sufficient — always read the source pages.\n"
            f"Recommended: start with the top {min(min_sources + 1, len(results))} results."
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class WebSearchMixin:
    """
    Drive the **Level 3 Agentic Web Tools** (read, scroll, search, discovery).

    Tool routing:
      - perform_web_search  → SearxNG          (direct internal HTTP)
      - read_web_page       → NativeExecutionService → browserless
      - scroll_web_page     → NativeExecutionService → browserless / Redis
      - search_web_page     → NativeExecutionService → Redis cache

    Features:
    - Lazy-loaded session state (crash-proof).
    - Query Tiering (adjusts research depth based on complexity).
    - Navigation Guidance (detects pagination).
    - SERP Quality Analysis.
    - Scroll-limit enforcement (prevents doom-scrolling).
    - Search-first gate (blocks scroll when search_web_page hasn't been run).

    Emission style:
    - All stream events are emitted as raw JSON strings conforming to the
      EVENT_CONTRACT. The SDK deserializes and routes them — this mixin has
      no dependency on StatusEvent dataclasses.
    """

    # ------------------------------------------------------------------
    # 1. BULLETPROOF STATE MANAGEMENT (Lazy Load)
    # ------------------------------------------------------------------

    @property
    def _research_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Safely retrieves the session store.
        If 'OrchestratorCore' didn't initialize it, we do it ourselves here.
        This prevents 'AttributeError' in complex Mixin hierarchies.
        """
        if not hasattr(self, "_internal_sessions_storage"):
            self._internal_sessions_storage = defaultdict(
                lambda: {
                    "sources_read": 0,
                    "urls_visited": set(),
                    # Tracks how many times scroll_web_page has been called
                    # per URL so we can enforce SCROLL_LIMIT_PER_URL.
                    "url_scroll_counts": defaultdict(int),
                    # Tracks which URLs have had search_web_page run against
                    # them, used by the search-first gate.
                    "urls_searched": set(),
                    "query_tier": None,
                    "search_performed": False,
                    "initial_query": None,
                }
            )
        return self._internal_sessions_storage

    # ------------------------------------------------------------------
    # 2. HELPER METHODS & LOGIC
    # ------------------------------------------------------------------

    def _classify_query_tier(self, query: str) -> int:
        """
        Classifies query complexity to determine required source count.

        Returns:
            1: Simple factual query       (requires 2 sources)
            2: Moderate multi-faceted     (requires 3 sources)
            3: Complex analytical query   (requires 5 sources)
        """
        query_lower = query.lower()

        tier_3_keywords = [
            "compare",
            "difference",
            "why",
            "analyze",
            "trend",
            "versus",
            "vs",
            "impact of",
            "how do",
            "what are the effects",
        ]
        if any(kw in query_lower for kw in tier_3_keywords):
            return 3

        tier_2_keywords = [
            "what are",
            "list",
            "benefits",
            "advantages",
            "disadvantages",
            "types of",
            "examples of",
            "methods",
            "ways to",
        ]
        if any(kw in query_lower for kw in tier_2_keywords):
            return 2

        return 1

    def _get_or_create_session(
        self, run_id: str, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get or initialize research session for this run safely."""
        session = self._research_sessions[run_id]

        if query and session["query_tier"] is None:
            session["query_tier"] = self._classify_query_tier(query)
            session["initial_query"] = query
            LOG.info(
                f"[{run_id}] Query classified as Tier {session['query_tier']}: {query}"
            )

        return session

    async def _validate_research_completeness(
        self,
        run_id: str,
        session: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        Validates if research meets quality standards.
        Useful for internal checks even if not blocking the orchestrator.
        """
        if session is None:
            if run_id not in self._research_sessions:
                return True, "No active research session"
            session = self._research_sessions[run_id]

        query_tier = session.get("query_tier", 1)
        sources_read = session.get("sources_read", 0)
        search_performed = session.get("search_performed", False)

        min_sources = {1: 2, 2: 3, 3: 5}
        required = min_sources.get(query_tier, 2)

        if not search_performed:
            return True, "No web search performed, skipping validation"

        if sources_read < required:
            shortage = required - sources_read
            return False, (
                f"⚠️ RESEARCH INCOMPLETE: Tier {query_tier} query requires {required} sources.\n"
                f"Currently read: {sources_read} source(s).\n"
                f"ACTION REQUIRED: Read {shortage} more source(s) before providing final answer."
            )

        return True, f"Research complete: {sources_read}/{required} sources read."

    def _clear_session(self, run_id: str) -> None:
        """Clear research session after completion."""
        if run_id in self._research_sessions:
            del self._research_sessions[run_id]

    @staticmethod
    def _is_tool_response_usable(content: Any) -> bool:
        """Return True only for non-empty string payloads."""
        return isinstance(content, str) and bool(content.strip())

    @staticmethod
    def _looks_like_explicit_tool_error(content: str) -> bool:
        """
        Detect only explicit tool-generated failure payloads, not arbitrary page text.
        """
        normalized = content.lstrip()
        return normalized.startswith("❌ ")

    # ------------------------------------------------------------------
    # SCROLL GUARD
    # ------------------------------------------------------------------

    def _check_scroll_allowed(
        self, run_id: str, target_url: str, page_num: int
    ) -> Optional[str]:
        """
        Enforces two scroll safety rules. Returns a blocking error string
        if the scroll should be intercepted, or None if it is permitted.

        Rule 1 — Search-first gate:
            scroll_web_page on page > 0 is blocked if search_web_page has
            never been called on this URL. Page 0 is always allowed.

        Rule 2 — Hard scroll limit:
            Once SCROLL_LIMIT_PER_URL successful scrolls have been issued for a URL,
            further calls are blocked and the worker is instructed to pivot.
        """
        session = self._get_or_create_session(run_id)

        # Rule 1: search-first gate (only applies beyond page 0)
        if page_num > 0 and target_url not in session["urls_searched"]:
            return (
                f"🛑 SCROLL BLOCKED — search_web_page not yet run on this URL.\n"
                f"MANDATORY NEXT STEP: Call search_web_page('{target_url}', '<your query>') first.\n"
                f"search_web_page scans ALL pages instantly and is always faster than scrolling.\n"
                f"Only return to scroll_web_page if search confirms the relevant section "
                f"and you need surrounding narrative context."
            )

        # Rule 2: hard scroll limit
        current_count = session["url_scroll_counts"][target_url]
        if current_count >= SCROLL_LIMIT_PER_URL:
            return (
                f"🛑 SCROLL LIMIT REACHED — scroll_web_page succeeded {current_count} times "
                f"on this URL (limit: {SCROLL_LIMIT_PER_URL}).\n"
                f"MANDATORY ACTION: Stop scrolling '{target_url}'.\n"
                f"You are doom-scrolling. This URL is exhausted. You MUST:\n"
                f"  1. If search_web_page has not been run: call it now with your target query.\n"
                f"  2. If search_web_page already returned nothing: append ⚠️ to the Scratchpad "
                f"and call perform_web_search to find a different source.\n"
                f"DO NOT call scroll_web_page on this URL again."
            )

        return None

    @staticmethod
    def _format_web_tool_error(tool_name: str, error_content: str, inputs: Dict) -> str:
        """Translates failures into actionable instructions."""
        if not error_content:
            error_content = "Unknown Error (Empty Response)"

        error_lower = error_content.lower()

        if "validation error" in error_lower or "missing arguments" in error_lower:
            return (
                f"❌ SCHEMA ERROR: Invalid arguments for '{tool_name}'.\n"
                f"ERROR DETAILS: {error_content}\n"
                "SYSTEM INSTRUCTION: Check the tool definition and retry with valid arguments."
            )

        if "403" in error_content or "access denied" in error_lower:
            return (
                f"❌ Web Tool '{tool_name}' Failed: Access Denied (Bot Protection).\n"
                "SYSTEM INSTRUCTION: This URL is blocking bots. STOP trying this specific URL. "
                "Pick a different URL from your search results."
            )

        if "out of bounds" in error_lower or "invalid page" in error_lower:
            return (
                f"⚠️ Web Tool '{tool_name}' Note: End of Document Reached.\n"
                "SYSTEM INSTRUCTION: No more pages. Synthesize collected info."
            )

        return (
            f"❌ Web Tool '{tool_name}' Error: {error_content}\n"
            "SYSTEM INSTRUCTION: Review arguments. If error persists, stop using this tool."
        )

    def _inject_navigation_guidance(self, content: str, url: str) -> str:
        """Injects hints if content looks like a paginated document."""
        if not content:
            return ""

        page_match = re.search(
            r"Page\s+(\d+)\s*(?:of|/)\s*(\d+)", content, re.IGNORECASE
        )
        if not page_match:
            return content

        curr, total = int(page_match.group(1)), int(page_match.group(2))
        next_p = curr + 1

        if curr >= total - 1:
            return content + "\n\n--- ✅ END OF DOCUMENT ---"

        return (
            content + f"\n\n--- 🧭 NAVIGATION (Page {curr}/{total}) ---\n"
            f"1. [STOP] if you have the answer.\n"
            f"2. [SEARCH] 'search_web_page' for keywords.\n"
            f"3. [SCROLL] 'scroll_web_page' to page {next_p}."
        )

    # ------------------------------------------------------------------
    # 3. CORE EXECUTION LOGIC
    # ------------------------------------------------------------------

    async def _execute_web_tool_logic(
        self,
        tool_name: str,
        required_keys: list[str],
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str],
        decision: Optional[Dict],
    ) -> AsyncGenerator[str, None]:
        """
        Shared core logic for Read, Scroll, Search, and SERP with progress tracking.

        Routing (all paths are now SDK-free):
          perform_web_search  → SearxNGClient              (direct internal HTTP)
          read_web_page       → NativeExecutionService.read_url
          scroll_web_page     → NativeExecutionService.scroll_url
          search_web_page     → NativeExecutionService.search_url

        Yields raw JSON strings conforming to the stream EVENT_CONTRACT.
        """

        LOG.critical(
            "🔧 [TOOL EXECUTION] tool=%s | assistant_id=%s | run_id=%s",
            tool_name,
            assistant_id,
            run_id,
        )

        ts_start = asyncio.get_event_loop().time()

        # --- [1] STATUS: VALIDATING ---
        yield _status(run_id, tool_name, "Validating parameters...")

        # --- VALIDATION ---
        validator = ToolValidator()
        validator.schema_registry = {tool_name: required_keys}
        validation_error = validator.validate_args(tool_name, arguments_dict)

        if validation_error:
            LOG.warning(f"{tool_name} ▸ Validation Failed: {validation_error}")
            yield _status(run_id, tool_name, "Validation failed.", status="error")

            error_feedback = self._format_web_tool_error(
                tool_name, f"Validation Error: {validation_error}", arguments_dict
            )

            try:
                await self._native_exec.submit_failed_tool_execution(
                    tool_name=tool_name,
                    run_id=run_id,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_call_id=tool_call_id,
                    error_message=error_feedback,
                    function_args=arguments_dict,
                    decision=decision,
                )
            except Exception as exc:
                LOG.error(f"Failed to submit validation error: {exc}")

            return

        # --- [2] STATUS: CREATING ACTION ---
        yield _status(run_id, tool_name, "Initializing tool action...")

        action = None
        try:
            action = await self._native_exec.create_action(
                tool_name=tool_name,
                run_id=run_id,
                tool_call_id=tool_call_id,
                function_args=arguments_dict,
                decision=decision,
            )
        except Exception as exc:
            LOG.error(f"{tool_name} ▸ Action creation failed: {exc}")
            yield _status(run_id, tool_name, "Internal system error.", status="error")
            return

        # --- [3] EXECUTION ---
        try:
            final_content = ""
            is_soft_failure = False

            # ----------------------------------------------------------------
            # read_web_page — NativeExecutionService → browserless
            # ----------------------------------------------------------------
            if tool_name == "read_web_page":
                target_url = arguments_dict["url"]
                yield _status(run_id, tool_name, f"Reading: {target_url}...")

                session = self._get_or_create_session(run_id)

                raw_content = await self._native_exec.read_url(
                    url=target_url,
                    force_refresh=arguments_dict.get("force_refresh", False),
                )

                if not self._is_tool_response_usable(raw_content):
                    is_soft_failure = True
                    final_content = "Tool execution returned no readable page content."
                else:
                    if target_url not in session["urls_visited"]:
                        session["urls_visited"].add(target_url)
                        session["sources_read"] += 1
                        LOG.info(
                            f"[{run_id}] Progress: {session['sources_read']} sources read "
                            f"(Tier {session.get('query_tier', 'N/A')} query)"
                        )

                    tier = session.get("query_tier", 1)
                    min_sources = {1: 2, 2: 3, 3: 5}.get(tier, 2)
                    progress_note = (
                        f"\n\n📊 RESEARCH PROGRESS: {session['sources_read']}/{min_sources} "
                        f"sources read (Tier {tier} query)\n"
                        f"⚡ NEXT STEP: Call search_web_page('{target_url}', '<your query>') "
                        f"to extract facts. Do NOT scroll."
                    )

                    final_content = (
                        self._inject_navigation_guidance(raw_content, target_url)
                        + progress_note
                    )

            # ----------------------------------------------------------------
            # scroll_web_page — NativeExecutionService → browserless / Redis
            # ----------------------------------------------------------------
            elif tool_name == "scroll_web_page":
                target_url = arguments_dict["url"]
                page_num = arguments_dict["page"]

                # --- SCROLL GUARD ---
                scroll_block_msg = self._check_scroll_allowed(
                    run_id, target_url, page_num
                )
                if scroll_block_msg:
                    LOG.warning(
                        f"[{run_id}] scroll_web_page INTERCEPTED for '{target_url}' "
                        f"page={page_num}: scroll guard triggered."
                    )
                    yield _status(
                        run_id,
                        tool_name,
                        f"Scroll blocked by guard (page {page_num}).",
                        status="warning",
                    )
                    await self._native_exec.submit_tool_output(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        tool_call_id=tool_call_id,
                        content=scroll_block_msg,
                        action_id=action.id,
                        is_error=True,
                    )
                    await self._native_exec.update_action_status(
                        action_id=action.id,
                        status=StatusEnum.failed.value,
                    )
                    return

                session = self._get_or_create_session(run_id)
                attempted_scroll = session["url_scroll_counts"][target_url] + 1

                LOG.info(
                    f"[{run_id}] scroll_web_page '{target_url}' page={page_num} "
                    f"(attempt {attempted_scroll}/{SCROLL_LIMIT_PER_URL})"
                )

                yield _status(
                    run_id,
                    tool_name,
                    f"Scrolling to page {page_num} "
                    f"(attempt {attempted_scroll}/{SCROLL_LIMIT_PER_URL})...",
                )

                raw_content = await self._native_exec.scroll_url(
                    url=target_url,
                    page=page_num,
                )

                if not self._is_tool_response_usable(raw_content):
                    is_soft_failure = True
                    final_content = f"Tool execution returned no scroll content for page {page_num}."
                else:
                    session["url_scroll_counts"][target_url] = attempted_scroll
                    scroll_count = session["url_scroll_counts"][target_url]

                    LOG.info(
                        f"[{run_id}] scroll_web_page '{target_url}' page={page_num} "
                        f"accepted (scroll {scroll_count}/{SCROLL_LIMIT_PER_URL})"
                    )

                    remaining = SCROLL_LIMIT_PER_URL - scroll_count
                    scroll_budget_note = (
                        f"\n\n📜 SCROLL BUDGET: {scroll_count}/{SCROLL_LIMIT_PER_URL} successful "
                        f"scrolls used on this URL ({remaining} remaining).\n"
                        + (
                            "⚠️ BUDGET EXHAUSTED after this scroll. "
                            "Run search_web_page or get a new URL."
                            if remaining == 0
                            else "Prefer search_web_page over further scrolling unless reading a narrative."
                        )
                    )

                    final_content = (
                        self._inject_navigation_guidance(raw_content, target_url)
                        + scroll_budget_note
                    )

            # ----------------------------------------------------------------
            # search_web_page — NativeExecutionService → Redis cache
            # ----------------------------------------------------------------
            elif tool_name == "search_web_page":
                target_url = arguments_dict["url"]
                query_val = arguments_dict["query"]

                yield _status(run_id, tool_name, f"Searching page for '{query_val}'...")

                raw_content = await self._native_exec.search_url(
                    url=target_url,
                    query=query_val,
                )

                if not self._is_tool_response_usable(raw_content):
                    is_soft_failure = True
                    final_content = f"Tool execution returned no search results for query '{query_val}'."
                else:
                    session = self._get_or_create_session(run_id)
                    session["urls_searched"].add(target_url)
                    final_content = raw_content

            # ----------------------------------------------------------------
            # perform_web_search — SearxNG direct (internal HTTP, no SDK hop)
            # ----------------------------------------------------------------
            elif tool_name == "perform_web_search":
                query_val = arguments_dict["query"]
                yield _status(run_id, tool_name, f"Querying SearxNG: '{query_val}'...")

                session = self._get_or_create_session(run_id, query_val)
                session["search_performed"] = True
                query_tier = session.get("query_tier", 1)

                searxng = SearxNGClient()
                yield _status(run_id, tool_name, "Parsing search results...")

                raw_content = await searxng.format_for_agent(
                    query=query_val,
                    count=10,
                    query_tier=query_tier,
                )

                if not self._is_tool_response_usable(raw_content):
                    is_soft_failure = True
                    final_content = "Search returned no usable response."
                else:
                    final_content = raw_content

            else:
                raise ValueError(f"Unknown web tool: {tool_name}")

            # --- [4] RESULT ANALYSIS ---
            if self._looks_like_explicit_tool_error(final_content):
                is_soft_failure = True

            if is_soft_failure:
                yield _status(
                    run_id, tool_name, "Encountered external error...", status="warning"
                )
                final_content = self._format_web_tool_error(
                    tool_name, final_content, arguments_dict
                )
            else:
                yield _status(
                    run_id,
                    tool_name,
                    "Content retrieved successfully.",
                    status="success",
                )

            # --- [5] SUBMIT OUTPUT ---
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=final_content,
                action_id=action.id if action else None,
                is_error=is_soft_failure,
            )

            # --- [6] UPDATE ACTION ---
            if action:
                await self._native_exec.update_action_status(
                    action_id=action.id,
                    status=(
                        StatusEnum.completed.value
                        if not is_soft_failure
                        else StatusEnum.failed.value
                    ),
                )

            LOG.info(
                "[%s] %s completed in %.2fs",
                run_id,
                tool_name,
                asyncio.get_event_loop().time() - ts_start,
            )

        except Exception as exc:
            # --- [7] HARD FAILURE ---
            LOG.error(f"[{run_id}] {tool_name} HARD FAILURE: {exc}")
            yield _status(
                run_id, tool_name, f"Critical failure: {str(exc)}", status="error"
            )

            error_hint = self._format_web_tool_error(
                tool_name, str(exc), arguments_dict
            )

            try:
                if action:
                    await self._native_exec.update_action_status(
                        action_id=action.id,
                        status=StatusEnum.failed.value,
                    )
                await self._native_exec.submit_tool_output(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_call_id=tool_call_id,
                    content=error_hint,
                    action_id=action.id if action else None,
                    is_error=True,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 4. PUBLIC HANDLERS
    # ------------------------------------------------------------------

    async def handle_perform_web_search(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Handler for 'perform_web_search' — routes to SearxNG directly."""
        async for event in self._execute_web_tool_logic(
            tool_name="perform_web_search",
            required_keys=["query"],
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            arguments_dict=arguments_dict,
            tool_call_id=tool_call_id,
            decision=decision,
        ):
            yield event

    async def handle_read_web_page(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Handler for 'read_web_page' — routes via NativeExecutionService → browserless."""
        async for event in self._execute_web_tool_logic(
            tool_name="read_web_page",
            required_keys=["url"],
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            arguments_dict=arguments_dict,
            tool_call_id=tool_call_id,
            decision=decision,
        ):
            yield event

    async def handle_scroll_web_page(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Handler for 'scroll_web_page' — routes via NativeExecutionService → browserless / Redis."""
        async for event in self._execute_web_tool_logic(
            tool_name="scroll_web_page",
            required_keys=["url", "page"],
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            arguments_dict=arguments_dict,
            tool_call_id=tool_call_id,
            decision=decision,
        ):
            yield event

    async def handle_search_web_page(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Handler for 'search_web_page' — routes via NativeExecutionService → Redis cache."""
        async for event in self._execute_web_tool_logic(
            tool_name="search_web_page",
            required_keys=["url", "query"],
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            arguments_dict=arguments_dict,
            tool_call_id=tool_call_id,
            decision=decision,
        ):
            yield event
