"""
Automated Model Benchmark: Reasoning & Tool Use
-----------------------------------------------
1. Loops through specified models.
2. Updates a persistent Markdown report.
3. BEHAVIOR: One table per Provider.
4. GUARANTEES ID UNIQUENESS: Merges new results into existing rows based on Endpoint ID.
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from entities_api.orchestration.instructions.assembler import assemble_instructions

# Import definitions
from models import TOGETHER_AI_MODELS
from projectdavid import Entity

# ------------------------------------------------------------------
# 0. Setup & Config
# ------------------------------------------------------------------
root_dir = Path(__file__).resolve().parents[2]
load_dotenv()

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Global Config
BASE_URL = os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY")
USER_ID = os.getenv("ENTITIES_USER_ID")
REPORT_FILE = root_dir / "model_compatibility_report.md"

tool_instructions = assemble_instructions(
    include_keys=[
        # "TOOL_USAGE_PROTOCOL",
        # "FUNCTION_CALL_FORMATTING",
        # "FUNCTION_CALL_WRAPPING",
    ]
)


def get_api_key_for_provider(provider: str) -> str:
    """Dynamic key resolver."""
    p = provider.lower()
    if "together" in p:
        return os.getenv("TOGETHER_API_KEY", "")
    elif "hyperbolic" in p:
        return os.getenv("HYPERBOLIC_API_KEY", "")
    elif "openai" in p:
        return os.getenv("OPENAI_API_KEY", "")
    return ""


# ------------------------------------------------------------------
# 1. Tool Logic (Mock)
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments) -> str:
    """Mock flight-time lookup tool."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return json.dumps({"status": "error", "message": "Invalid JSON"})

    return json.dumps(
        {
            "status": "success",
            "duration": "4h 30m",
            "departure_time": "10:00 AM PST",
            "arrival_time": "06:30 PM EST",
        }
    )


# ------------------------------------------------------------------
# 2. Report Manager (Simple ID-Based Deduplication)
# ------------------------------------------------------------------
class ReportManager:
    @staticmethod
    def update_report(new_results: List[Dict]):
        """
        Updates the MD file.
        - Reads ALL existing rows.
        - Deduplicates based on Endpoint ID.
        - Groups by Provider.
        """
        file_path = Path(REPORT_FILE)

        # Structure: { provider_name: { endpoint_id: row_string } }
        # Using a nested dict ensures that ID is the unique key per provider.
        # (Technically ID should be unique globally, but grouping by provider is cleaner for display)
        rows_by_provider = defaultdict(dict)

        # 1. Read Existing File
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        clean_line = line.strip()
                        # Only parse data rows
                        if clean_line.startswith("| **"):
                            parts = [p.strip() for p in clean_line.split("|")]
                            if len(parts) >= 4:
                                # Extract Provider (Column 2)
                                provider = parts[2].strip()
                                # Extract ID from `backticks` (Column 3)
                                match = re.search(r"`(.*?)`", parts[3])

                                if match:
                                    eid = match.group(1).strip()
                                    # Store it. If duplicate ID exists in file, this overwrites it (last one wins).
                                    rows_by_provider[provider][eid] = clean_line
            except Exception as e:
                print(f"{RED}[!] Error parsing report: {e}{RESET}")

        # 2. Merge New Results
        for res in new_results:
            inf_icon = "✅" if res["inference_ok"] else "❌"
            reas_icon = "🧠" if res["reasoning_detected"] else "—"
            tool_icon = "✅" if res["tool_call_ok"] else "❌"

            if not res["inference_ok"]:
                tool_icon = "—"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            note = res["error_msg"] if res["error_msg"] else "OK"

            new_row = (
                f"| **{res['name']}** | {res['provider']} | `{res['id']}` | "
                f"{inf_icon} | {reas_icon} | {tool_icon} | {timestamp} | {note} |"
            )

            # Upsert: This guarantees we update the existing entry if the ID matches
            rows_by_provider[res["provider"]][res["id"]] = new_row

        # 3. Write Report
        table_header_row = "| Model Name | Provider | Endpoint ID | Inference | Reasoning | Tools | Last Run | Notes |"
        table_align_row = "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |"

        main_title = [
            "# 🧪 Model Compatibility Report",
            f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Write File Title
                f.write("\n".join(main_title))

                # Write Tables per Provider (Sorted Alphabetically)
                for provider in sorted(rows_by_provider.keys()):
                    f.write(f"\n## 🟢 Provider: {provider}\n")
                    f.write(f"{table_header_row}\n{table_align_row}\n")

                    # Sort rows by Name (Column 1) for readability
                    # We sort the values (the full row strings)
                    rows = sorted(
                        rows_by_provider[provider].values(),
                        key=lambda x: x.split("|")[1].strip().lower(),
                    )
                    f.write("\n".join(rows) + "\n")

            print(f"{GREEN}[✓] Report updated.{RESET}")
        except Exception as e:
            print(f"{RED}[!] Write failed: {e}{RESET}")


# ------------------------------------------------------------------
# 3. Test Logic
# ------------------------------------------------------------------
class ModelTester:
    def __init__(self, model_label: str, config: Dict):
        self.model_label = model_label
        self.model_id = config["id"]
        self.provider = config["provider"]
        self.client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)
        self.assistant_id: Optional[str] = None
        self.user_id = USER_ID
        self.provider_api_key = get_api_key_for_provider(self.provider)

    def setup(self) -> Tuple[bool, str]:
        if not self.provider_api_key:
            return False, f"Missing API Key for {self.provider}"

        try:
            print(f"{YELLOW}[*] [{self.model_label}] Creating Assistant...{RESET}")
            assistant = self.client.assistants.create_assistant(
                name=f"Bench_{self.model_label}",
                instructions=tool_instructions,
                model=self.model_id,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_flight_times",
                            "description": "Get flight times",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "departure": {"type": "string"},
                                    "arrival": {"type": "string"},
                                },
                                "required": ["departure", "arrival"],
                            },
                        },
                    }
                ],
            )
            self.assistant_id = assistant.id
            return True, None
        except Exception as e:
            return False, str(e)

    def run_benchmark(self) -> Dict:
        result = {
            "name": self.model_label,
            "id": self.model_id,
            "provider": self.provider,
            "inference_ok": False,
            "reasoning_detected": False,
            "tool_call_ok": False,
            "error_msg": "",
        }

        # Stage 1: Inference
        print(f"\n{YELLOW}--- Stage 1: Inference ({self.model_label}) ---{RESET}")
        try:
            thread = self.client.threads.create_thread(participant_ids=[self.user_id])
            self.client.messages.create_message(
                thread_id=thread.id,
                role="user",
                content="Calculate Fibonacci recursively. Explain why iterative is better.",
                assistant_id=self.assistant_id,
            )
            run = self.client.runs.create_run(
                assistant_id=self.assistant_id, thread_id=thread.id
            )
            flags = self._stream_and_analyze(thread.id, run.id, timeout=180.0)

            if flags["has_error"]:
                result["error_msg"] = "Stream Error"
            else:
                result["inference_ok"] = flags["has_content"] or flags["has_reasoning"]
                result["reasoning_detected"] = flags["has_reasoning"]
        except Exception as e:
            result["error_msg"] = f"Inference Exception: {str(e)[:50]}"
            return result

        # Stage 2: Tools (Only if Inference OK)
        if result["inference_ok"]:
            print(f"\n{YELLOW}--- Stage 2: Tool Calls ({self.model_label}) ---{RESET}")
            try:
                thread = self.client.threads.create_thread(
                    participant_ids=[self.user_id]
                )
                self.client.messages.create_message(
                    thread_id=thread.id,
                    role="user",
                    content="Fetch flight times between LAX and JFK.",
                    assistant_id=self.assistant_id,
                )
                run = self.client.runs.create_run(
                    assistant_id=self.assistant_id, thread_id=thread.id
                )
                flags = self._stream_and_analyze(thread.id, run.id, timeout=60.0)

                if flags["has_tool_call"]:
                    print(f"\n{YELLOW}[*] Executing Tool Logic...{RESET}")
                    handled = self.client.runs.poll_and_execute_action(
                        run_id=run.id,
                        thread_id=thread.id,
                        assistant_id=self.assistant_id,
                        tool_executor=get_flight_times,
                        actions_client=self.client.actions,
                        messages_client=self.client.messages,
                        timeout=180.0,
                    )
                    if handled:
                        print(f"{YELLOW}[*] Streaming Final Response...{RESET}")
                        final_flags = self._stream_and_analyze(
                            thread.id, run.id, timeout=30.0
                        )
                        if final_flags["has_content"]:
                            result["tool_call_ok"] = True
                else:
                    print(f"{RED}[!] Tool not triggered.{RESET}")
            except Exception as e:
                result["error_msg"] = f"Tool Exception: {str(e)[:50]}"

        return result

    def _stream_and_analyze(self, thread_id, run_id, timeout) -> Dict:
        flags = {
            "has_reasoning": False,
            "has_content": False,
            "has_tool_call": False,
            "has_error": False,
        }
        sync_stream = self.client.synchronous_inference_stream
        sync_stream.setup(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
            message_id="",
            run_id=run_id,
            api_key=self.provider_api_key,
        )
        current_mode = None
        try:
            for chunk in sync_stream.stream_chunks(
                provider=self.provider,
                model=self.model_id,
                timeout_per_chunk=timeout,
                suppress_fc=False,
            ):
                ctype = chunk.get("type")
                content = chunk.get("content", "")

                if ctype == "reasoning":
                    flags["has_reasoning"] = True
                elif ctype == "content":
                    flags["has_content"] = True
                elif ctype in ["tool_name", "function_call", "call_arguments"]:
                    flags["has_tool_call"] = True
                elif ctype == "error":
                    flags["has_error"] = True

                if ctype == "reasoning":
                    if current_mode != "reasoning":
                        print(f"\n{CYAN}🤔 [THOUGHT PROCESS]{RESET}\n", end="")
                        current_mode = "reasoning"
                    print(f"{CYAN}{content}{RESET}", end="", flush=True)
                elif ctype == "content":
                    if current_mode != "content":
                        if current_mode == "reasoning":
                            print("\n")
                        print(f"\n{GREEN}🤖 [ANSWER]{RESET}\n", end="")
                        current_mode = "content"
                    print(f"{GREEN}{content}{RESET}", end="", flush=True)
                elif ctype == "tool_name":
                    print(f"\n{YELLOW}🛠  [TOOL DETECTED]: {content}{RESET}", end="")
                    current_mode = "tool"
                elif ctype == "call_arguments":
                    print(f"{YELLOW}{content}{RESET}", end="", flush=True)
            print()
        except Exception as e:
            print(f"\n{RED}[!] Stream Error: {e}{RESET}")
            flags["has_error"] = True
        return flags


# ------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------
def main(models_to_run):
    if not USER_ID:
        print(f"{RED}CRITICAL: Missing USER_ID in .env{RESET}")
        return

    print(f"Starting Benchmark for {len(models_to_run)} models...\n")

    # Initialize report (Ensure title/headers exist immediately)
    ReportManager.update_report([])

    for label, raw_config in models_to_run.items():
        if isinstance(raw_config, str):
            config = {"id": raw_config, "provider": "hyperbolic"}
        else:
            config = raw_config

        tester = ModelTester(label, config)

        ok, err = tester.setup()
        if not ok:
            print(f"{RED}Failed setup for {label}: {err}{RESET}")
            ReportManager.update_report(
                [
                    {
                        "name": label,
                        "provider": config["provider"],
                        "id": config["id"],
                        "inference_ok": False,
                        "reasoning_detected": False,
                        "tool_call_ok": False,
                        "error_msg": f"Setup Failed: {err}",
                    }
                ]
            )
            continue

        res = tester.run_benchmark()
        ReportManager.update_report([res])
        time.sleep(2)


if __name__ == "__main__":
    main(models_to_run=TOGETHER_AI_MODELS)
