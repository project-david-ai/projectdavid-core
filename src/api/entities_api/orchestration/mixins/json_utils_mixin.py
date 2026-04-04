"""
Regex-heavy helpers for:
• quote / comma sanitisation                       → ensure_valid_json
• generic schema guards                            → is_valid_function_call_response
• vector-store mongo-style query validator         → is_complex_vector_search
• multi-line / fence-stripped <fc>{…}</fc> search → extract_function_calls_within_body_of_text
• misc parsing helpers                             → extract_function_candidates, …
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from projectdavid_common.constants.ai_model_map import MODEL_MAP
from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()


class JsonUtilsMixin:
    REASONING_PATTERN = re.compile("(<think>|</think>)")

    @staticmethod
    def parse_code_interpreter_partial(text: str) -> Optional[Dict[str, str]]:
        """
        Detects an in-progress code-interpreter tool call.
        """
        pattern = re.compile(
            r'\n\s*\{\s*["\']name["\']\s*:\s*["\']code_interpreter["\']\s*,\s*\n\s*["\']arguments["\']\s*:\s*\{\s*["\']code["\']\s*:\s*\n\s*(?P<code>.*)',
            re.VERBOSE | re.DOTALL,
        )
        m = pattern.search(text)
        if not m:
            return None
        code_fragment = m.group("code").strip()
        fc_json = {"name": "code_interpreter", "arguments": {"code": code_fragment}}
        fc_block = f"<fc>{json.dumps(fc_json, ensure_ascii=False)}</fc>"
        return {"code": code_fragment, "fc_block": fc_block}

    @staticmethod
    def convert_smart_quotes(text: str) -> str:
        return (
            text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        )

    @staticmethod
    def is_valid_function_call_response(json_data: dict) -> bool:
        """
        Generalized validation.
        FIX: Explicitly allows 'arguments' to be a dict OR a string.
        """
        try:
            if not isinstance(json_data, dict):
                return False
            if {"name", "arguments"} - json_data.keys():
                return False
            if not isinstance(json_data["name"], str) or not json_data["name"].strip():
                return False

            # --- CRITICAL FIX START ---
            args = json_data["arguments"]
            # Allow stringified JSON arguments (OpenAI/Llama style)
            if not isinstance(args, (dict, str)):
                return False
            # --- CRITICAL FIX END ---

            if isinstance(args, dict):
                for key, value in args.items():
                    if not isinstance(key, str):
                        return False
            return True
        except (TypeError, KeyError):
            return False

    def is_complex_vector_search(self, data: dict) -> bool:
        """Recursively validate operators with $ prefix"""
        for key, value in data.items():
            if key.startswith("$"):
                if isinstance(value, dict) and (
                    not self.is_complex_vector_search(value)
                ):
                    return False
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and (
                            not self.is_complex_vector_search(item)
                        ):
                            return False
            elif isinstance(value, dict):
                if not self.is_complex_vector_search(value):
                    return False
            elif isinstance(value, list):
                pass
        return True

    def ensure_valid_json(self, text: str):
        """
        Ensures the input text represents a valid JSON dictionary.
        """
        if not isinstance(text, str) or not text.strip():
            LOG.error("Received empty or non-string content for JSON validation.")
            return False

        processed_text = text.strip()

        # 1. Attempt Direct Parse
        try:
            intermediate_parse = json.loads(processed_text)
            if isinstance(intermediate_parse, dict):
                return intermediate_parse
            elif isinstance(intermediate_parse, str):
                # Recursion check: if it parsed to a string, try parsing that string
                try:
                    inner = json.loads(intermediate_parse)
                    if isinstance(inner, dict):
                        return inner
                except (
                    Exception
                ):  # <--- FIXED: Changed from bare 'except:' to 'except Exception:'
                    pass
                # LOG.warning("Initial parse resulted in string...")
                processed_text = intermediate_parse
            else:
                return False
        except json.JSONDecodeError:
            pass
        except Exception as e:
            LOG.error(f"Unexpected error during initial JSON parse: {e}")
            return False

        # 2. Heuristic Fixes
        fixed_text = processed_text
        try:
            if "'" in fixed_text and '"' not in fixed_text.replace("\\'", ""):
                fixed_text = fixed_text.replace("'", '"')
            fixed_text = re.sub(r",(\s*[}\]])", r"\1", fixed_text)
            parsed_json = json.loads(fixed_text)
            if isinstance(parsed_json, dict):
                return parsed_json
            return False
        except Exception:
            return False

    def extract_function_calls_within_body_of_text(self, text: str):
        """
        Extracts tool invocation patterns from unstructured text.
        """
        text = re.sub("```(?:json)?(.*?)```", "\\1", text, flags=re.DOTALL)
        text = self.convert_smart_quotes(text)

        tool_matches = []

        # Optimization: Try full text parse first
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = self.ensure_valid_json(stripped)
            if parsed and self.is_valid_function_call_response(parsed):
                tool_matches.append(parsed)
                return tool_matches

        return tool_matches

    def extract_function_candidates(self, text):
        pattern = r"""
            \{
            \s*
            (["'])name\1\s*:\s*
            (["'])(.*?)\2\s*,\s*
            (["'])arguments\4\s*:\s*
            (\{.*?\})
            \s*\}
        """
        candidates = []
        try:
            matches = re.finditer(pattern, text, re.DOTALL | re.VERBOSE)
            for match in matches:
                candidate = match.group(0)
                if '"name"' in candidate and '"arguments"' in candidate:
                    candidates.append(candidate)
        except Exception:
            pass
        return candidates

    @staticmethod
    def parse_nested_function_call_json(text):
        # Implementation kept for compatibility
        return None

    def _get_model_map(self, value: str) -> str | None:
        return MODEL_MAP.get(value)

    @staticmethod
    def validate_tool_args(
        tool_name: str, args: Dict, schema_registry: Dict[str, list]
    ) -> Optional[str]:
        """
        Level 2 Enhancement: Checks if the LLM actually provided what's required.
        Returns a string error message if validation fails, else None.
        """
        required_fields = schema_registry.get(tool_name, [])
        missing = [f for f in required_fields if f not in args or args[f] is None]

        if missing:
            return (
                f"Missing required arguments for '{tool_name}': {', '.join(missing)}."
            )

        # Optional: Add type checking here (e.g. ensure 'count' is an int)
        return None
