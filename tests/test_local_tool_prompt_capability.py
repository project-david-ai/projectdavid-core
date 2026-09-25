import importlib.util
import unittest
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/api/entities_api/orchestration/mixins/local_tool_capability.py"
)

spec = importlib.util.spec_from_file_location(
    "q_local_tool_capability_test_target",
    SOURCE,
)

capability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capability)


MODEL = "vllm/qwen3-8b-awq-4bit"


def metadata(
    model=MODEL,
    variant="qwen3-8b-awq-4bit",
    enabled=False,
    status="unratified",
):
    return {
        capability.CAPABILITY_KEY: {
            "model": model,
            "variant_id": variant,
            "tools_enabled": enabled,
            "tool_calling_status": status,
        }
    }


class LocalToolPromptCapabilityTests(unittest.TestCase):
    def test_absent_marker_preserves_direct_core_behavior(self):
        self.assertIsNone(capability.resolve_q_tool_mode({}, MODEL))

    def test_unratified_model_disables_tools(self):
        self.assertIs(
            capability.resolve_q_tool_mode(metadata(), MODEL),
            False,
        )

    def test_verified_exact_variant_enables_tools(self):
        self.assertIs(
            capability.resolve_q_tool_mode(
                metadata(enabled=True, status="verified"),
                MODEL,
            ),
            True,
        )

    def test_cross_variant_grant_fails_closed(self):
        self.assertIs(
            capability.resolve_q_tool_mode(
                metadata(enabled=True, status="verified"),
                "vllm/deepseek-r1-7b-awq-4bit",
            ),
            False,
        )

    def test_invalid_marker_fails_closed(self):
        self.assertIs(
            capability.resolve_q_tool_mode(
                {capability.CAPABILITY_KEY: "verified"},
                MODEL,
            ),
            False,
        )

    def test_missing_run_metadata_fails_closed(self):
        self.assertIs(
            capability.resolve_q_tool_mode(None, MODEL),
            False,
        )

    def test_unverified_prompt_omits_generated_tool_schema(self):
        result = capability.build_tool_free_system_message(
            {
                "instructions": "Answer concisely.",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "computer"},
                    }
                ],
                "web_access": True,
            }
        )

        content = result["content"]

        self.assertEqual(result["role"], "system")
        self.assertIn("Answer concisely.", content)
        self.assertNotIn("computer", content)
        self.assertNotIn("### AVAILABLE TOOLS", content)
        self.assertNotIn("### OPERATIONAL PROTOCOLS", content)

    def test_context_and_dispatch_use_same_tool_mode(self):
        orchestration = SOURCE.parents[1]

        context = (orchestration / "mixins/context_mixin.py").read_text(
            encoding="utf-8"
        )

        worker = (orchestration / "workers/base_workers/vllm_raw_worker.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("if tools_enabled is False:", context)
        self.assertIn(
            "build_tool_free_system_message(config)",
            context,
        )
        self.assertIn("tools_enabled=tool_mode", worker)
        self.assertIn(
            "tools=[] if tool_mode is False else None",
            worker,
        )


if __name__ == "__main__":
    unittest.main()
