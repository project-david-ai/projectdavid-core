"""
Function-Call Filter Mixin
—————————
• One-time 4-byte peek: if the very first non-whitespace bytes of the
  stream are "<fc", flip a flag and suppress every subsequent
  function-call payload without further regex work.

• Otherwise fall back to a lightweight per-chunk filter:
    1) swallow provider-labelled {"type":"function_call"} JSON
    2) remove any chunk that simultaneously contains <fc> and </fc>
"""

from __future__ import annotations

import json
import re
from typing import Optional

from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()


class FunctionCallFilterMixin:
    _FC_OPEN = re.compile("<\\s*fc\\s*>", re.IGNORECASE)
    _FC_CLOSE = re.compile("</\\s*fc\\s*>", re.IGNORECASE)

    def _peek_for_fc(self, raw: str) -> None:
        """Run exactly once on the first outbound payload."""
        if getattr(self, "_fc_peek_done", False):
            return
        self._fc_peek_done = True
        lead4 = raw.lstrip()[:4].lower()
        self._fc_force_block = lead4.startswith("<fc")
        if self._fc_force_block:
            LOG.debug(
                "PeekGate: leading <fc> detected — global function-call suppression ON"
            )

    def _filter_out_fc(self, payload_str: str) -> Optional[str]:
        """
        Return None to hide payload from the client;
        return payload_str (unchanged) to forward it.
        """
        self._peek_for_fc(payload_str)
        if getattr(self, "_fc_force_block", False):
            if "<fc" in payload_str or '"function_call"' in payload_str:
                LOG.debug("Suppressed payload (global block active).")
                return None
        try:
            if json.loads(payload_str).get("type") == "function_call":
                LOG.debug("Suppressed provider-labelled function_call chunk.")
                return None
        except Exception:
            pass
        if self._FC_OPEN.search(payload_str) and self._FC_CLOSE.search(payload_str):
            LOG.debug("Suppressed inline <fc> … </fc> payload.")
            return None
        return payload_str
