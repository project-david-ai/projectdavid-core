"""
Automated Model Benchmark: Reasoning, Tool Use & L3 Batching
------------------------------------------------------------
1. Loops through specified models.
2. Updates a persistent Markdown report.
3. BEHAVIOR: One table per Provider.
4. GUARANTEES ID UNIQUENESS: Merges new results based on Endpoint ID.
5. TESTS: Inference -> Single Tool -> Parallel Batch Tools.
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config_benchmark import config
from dotenv import load_dotenv

# --- SDK Event Imports ---
from projectdavid import PlanEvent  # [NEW] For L3 Strategy Visibility
from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    StatusEvent,
    ToolCallRequestEvent,
)

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
MAGENTA = "\033[95m"
BLUE = "\033[94m"  # Planning Color
RESET = "\033[0m"

# --- Load External Config ---
CONFIG_FILE_NAME = "benchmark_config.json"
CONFIG_PATH = root_dir / CONFIG_FILE_NAME

# --- Global Constants ---
BASE_URL = config.get("base_url") or os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = config.get("entities_api_key") or os.getenv("ENTITIES_API_KEY")
USER_ID = config.get("entities_user_id") or os.getenv("ENTITIES_USER_ID")
report_name = config.get("report_file_name", "model_compatibility_report.md")
REPORT_FILE = root_dir / report_name


def get_api_key_for_provider(provider: str) -> str:
    """Dynamic key resolver."""
    p = provider.lower()
    if "together" in p:
        return config.get("together_api_key") or os.getenv("TOGETHER_API_KEY", "")
    elif "hyperbolic" in p:
        return config.get("hyperbolic_api_key") or os.getenv("HYPERBOLIC_API_KEY", "")
    elif "openai" in p:
        return config.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    return ""


# ------------------------------------------------------------------
# 1. Tool Logic (Registry)
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments: dict) -> str:
    """Mock flight-time lookup tool."""
    return json.dumps(
        {
            "status": "success",
            "duration": "4h 30m",
            "departure_time": "10:00 AM PST",
            "arrival_time": "06:30 PM EST",
            "route": f"{arguments.get('departure', 'UNK')} -> {arguments.get('arrival', 'UNK')}",
        }
    )


def get_weather(tool_name: str, arguments: dict) -> str:
    """Mock weather tool for Batch testing."""
    loc = arguments.get("location", "Unknown")
    return json.dumps(
        {"status": "success", "location": loc, "temp": "15C", "condition": "Cloudy"}
    )


# Registry for Dynamic Dispatch
TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
    "get_weather": get_weather,
}


# ------------------------------------------------------------------
# 2. Report Manager
# ------------------------------------------------------------------
class ReportManager:
    @staticmethod
    def update_report(new_results: List[Dict]):
        """
        Updates the MD file with Parallel Call column.
        """
        file_path = Path(REPORT_FILE)
        rows_by_provider = defaultdict(dict)

        # 1. Read Existing File
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        clean_line = line.strip()
                        if clean_line.startswith("| **"):
                            parts = [p.strip() for p in clean_line.split("|")]
                            # Ensure we capture enough columns for the new format
                            if len(parts) >= 4:
                                provider = parts[2].strip()
                                match = re.search(r"`(.*?)`", parts[3])
                                if match:
                                    eid = match.group(1).strip()
                                    rows_by_provider[provider][eid] = clean_line
            except Exception as e:
                print(f"{RED}[!] Error parsing report: {e}{RESET}")

        # 2. Merge New Results
        for res in new_results:
            inf_icon = "✅" if res["inference_ok"] else "❌"
            reas_icon = "🧠" if res["reasoning_detected"] else "—"
            tool_icon = "✅" if res["tool_call_ok"] else "❌"

            # 🚀 = Batch Success (2+ tools), — = Not tested/Failed
            batch_icon = "🚀" if res.get("batch_ok") else "—"
            if not res["tool_call_ok"]:
                batch_icon = "—"

            telem_icon = "📡" if res["call_telemetry"] else "—"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            note = res["error_msg"] if res["error_msg"] else "OK"

            new_row = (
                f"| **{res['name']}** | {res['provider']} | `{res['id']}` | "
                f"{inf_icon} | {reas_icon} | {tool_icon} | {batch_icon} | {telem_icon} | {timestamp} | {note} |"
            )
            rows_by_provider[res["provider"]][res["id"]] = new_row

        # 3. Write Report
        table_header_row = "| Model Name | Provider | Endpoint ID | Inference | Reasoning | Tools | Parallel | Telemetry | Last Run | Notes |"
        table_align_row = "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |"

        main_title = [
            "# 🧪 Model Compatibility Report",
            f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(main_title))
                for provider in sorted(rows_by_provider.keys()):
                    f.write(f"\n## 🟢 Provider: {provider}\n")
                    f.write(f"{table_header_row}\n{table_align_row}\n")
                    rows = sorted(
                        rows_by_provider[provider].values(),
                        key=lambda x: x.split("|")[1].strip().lower(),
                    )
                    f.write("\n".join(rows) + "\n")
            print(f"{GREEN}[✓] Report updated: {file_path.name}{RESET}")
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

        # Bind clients for recursive SDK handling
        if hasattr(self.client, "synchronous_inference_stream"):
            self.client.synchronous_inference_stream.bind_clients(
                self.client.runs,
                self.client.actions,
                self.client.messages,
                self.client.assistants,
            )

    def _classify_error(self, error_text: str) -> str:
        text = str(error_text).lower()
        if "404" in text or "not found" in text:
            return "💀 DEAD / UNAVAILABLE"
        if "401" in text or "unauthorized" in text:
            return "⛔ AUTH ERROR"
        if "429" in text:
            return "⏳ RATE LIMIT"
        if "500" in text:
            return "🔥 SERVER ERROR"
        clean_err = str(error_text).replace("\n", " ")
        return f"Error: {clean_err[:30]}..."

    def setup(self) -> Tuple[bool, str]:
        if not self.provider_api_key:
            return False, f"Missing API Key for {self.provider}"

        try:
            print(f"{YELLOW}[*] [{self.model_label}] Creating Assistant...{RESET}")
            assistant = self.client.assistants.create_assistant(
                name=f"Bench_{self.model_label}",
                instructions="You are a helpful AI assistant. Use tools when needed.",
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
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    },
                ],
            )
            self.assistant_id = assistant.id
            return True, None
        except Exception as e:
            return False, self._classify_error(str(e))

    def run_benchmark(self) -> Dict:
        result = {
            "name": self.model_label,
            "id": self.model_id,
            "provider": self.provider,
            "inference_ok": False,
            "reasoning_detected": False,
            "tool_call_ok": False,
            "batch_ok": False,
            "call_telemetry": False,
            "error_msg": "",
        }

        # --- Stage 1: Inference ---
        print(f"\n{YELLOW}--- Stage 1: Inference ({self.model_label}) ---{RESET}")
        try:
            thread = self.client.threads.create_thread()
            message = self.client.messages.create_message(
                thread_id=thread.id,
                role="user",
                content="Calculate Fibonacci recursively. Explain why iterative is better.",
                assistant_id=self.assistant_id,
            )
            run = self.client.runs.create_run(
                assistant_id=self.assistant_id, thread_id=thread.id
            )
            stats = self._stream_and_analyze(
                thread.id, message.id, run.id, timeout=180.0
            )

            result["inference_ok"] = stats["inference_ok"]
            result["reasoning_detected"] = stats["reasoning_detected"]
            if stats["error"]:
                result["error_msg"] = self._classify_error(stats["detailed_error"])
                return result

        except Exception as e:
            result["error_msg"] = self._classify_error(str(e))
            return result

        # --- Stage 2: Single Tool ---
        if result["inference_ok"]:
            print(f"\n{YELLOW}--- Stage 2: Single Tool ({self.model_label}) ---{RESET}")
            try:
                thread = self.client.threads.create_thread()
                message = self.client.messages.create_message(
                    thread_id=thread.id,
                    role="user",
                    content="Fetch flight times between LAX and JFK.",
                    assistant_id=self.assistant_id,
                )
                run = self.client.runs.create_run(
                    assistant_id=self.assistant_id, thread_id=thread.id
                )
                stats = self._stream_and_analyze(
                    thread.id, message.id, run.id, timeout=120.0
                )

                if stats["decision_detected"]:
                    result["call_telemetry"] = True

                if stats["tool_executed_count"] > 0 and stats["inference_ok"]:
                    result["tool_call_ok"] = True
                    print(f"{GREEN}[✓] Single Tool Verified.{RESET}")
                else:
                    if stats["error"]:
                        result["error_msg"] = self._classify_error(
                            stats["detailed_error"]
                        )
                    elif stats["tool_executed_count"] == 0:
                        print(f"{RED}[!] Tool call not triggered.{RESET}")

            except Exception as e:
                result["error_msg"] = self._classify_error(str(e))

        # --- Stage 3: Batch/Parallel Tools (Level 3) ---
        if result["tool_call_ok"]:
            print(
                f"\n{MAGENTA}--- Stage 3: Parallel Batch ({self.model_label}) ---{RESET}"
            )
            try:
                thread = self.client.threads.create_thread()
                # Prompt requires TWO tools
                message = self.client.messages.create_message(
                    thread_id=thread.id,
                    role="user",
                    content="I need two things: check the weather in London, AND find flight times from NYC to Paris.",
                    assistant_id=self.assistant_id,
                )
                run = self.client.runs.create_run(
                    assistant_id=self.assistant_id, thread_id=thread.id
                )
                stats = self._stream_and_analyze(
                    thread.id, message.id, run.id, timeout=120.0
                )

                # Check if we executed at least 2 distinct tools or 2 calls total
                if stats["tool_executed_count"] >= 2 and stats["inference_ok"]:
                    result["batch_ok"] = True
                    print(
                        f"{GREEN}[✓] Batch Verified: {stats['tool_executed_count']} tools executed.{RESET}"
                    )
                else:
                    print(
                        f"{RED}[!] Batch Failed. Tools executed: {stats['tool_executed_count']}{RESET}"
                    )

            except Exception as e:
                result["error_msg"] = f"Batch Error: {str(e)[:20]}"

        return result

    def _stream_and_analyze(self, thread_id, message_id, run_id, timeout) -> Dict:
        """Unified Event Loop for all stages."""
        stats = {
            "inference_ok": False,
            "reasoning_detected": False,
            "tool_executed_count": 0,
            "decision_detected": False,
            "error": False,
            "detailed_error": None,
        }

        stream = self.client.synchronous_inference_stream
        stream.setup(
            thread_id=thread_id,
            assistant_id=self.assistant_id,
            message_id=message_id,
            run_id=run_id,
            api_key=self.provider_api_key,
        )

        current_mode = None
        seen_tools = set()

        try:
            for event in stream.stream_events(
                provider=self.provider, model=self.model_id, timeout_per_chunk=timeout
            ):

                if isinstance(event, ReasoningEvent):
                    if current_mode != "reasoning":
                        print(f"\n{CYAN}🤔 [THOUGHT PROCESS]{RESET}\n", end="")
                        current_mode = "reasoning"
                        stats["reasoning_detected"] = True
                    print(f"{CYAN}{event.content}{RESET}", end="", flush=True)

                elif isinstance(event, PlanEvent):
                    # Level 3 Visualization
                    print(
                        f"\n{BLUE}🗺️  [PLAN]: {event.content}{RESET}",
                        end="",
                        flush=True,
                    )

                elif isinstance(event, ContentEvent):
                    if current_mode != "content":
                        if current_mode == "reasoning":
                            print("\n")
                        print(f"\n{GREEN}🤖 [ANSWER]{RESET}\n", end="")
                        current_mode = "content"
                        stats["inference_ok"] = True
                    print(f"{GREEN}{event.content}{RESET}", end="", flush=True)

                elif isinstance(event, DecisionEvent):
                    stats["decision_detected"] = True
                    print(
                        f"\n{MAGENTA}⚡ [DECISION]: {event.to_dict()}{RESET}",
                        end="",
                        flush=True,
                    )

                elif isinstance(event, ToolCallRequestEvent):
                    print(f"\n{YELLOW}🛠  [TOOL DETECTED]: {event.tool_name}{RESET}")

                    # Dynamic Dispatch
                    handler = TOOL_REGISTRY.get(event.tool_name)
                    if handler:
                        # SDK automatically handles output submission
                        if event.execute(handler):
                            stats["tool_executed_count"] += 1
                            seen_tools.add(event.tool_name)
                    else:
                        print(f"{RED} [!] No handler for {event.tool_name}{RESET}")

                elif isinstance(event, StatusEvent) and event.status == "failed":
                    stats["error"] = True
                    stats["detailed_error"] = "Stream Status: Failed"

            print()

        except Exception as e:
            print(f"\n{RED}[!] Stream Error: {e}{RESET}")
            stats["error"] = True
            stats["detailed_error"] = str(e)

        return stats


# ------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------
def main(models_to_run):
    if not USER_ID:
        print(f"{RED}CRITICAL: Missing USER_ID in {CONFIG_FILE_NAME} or .env{RESET}")
        return

    print(f"Starting Benchmark for {len(models_to_run)} models...\n")
    ReportManager.update_report([])

    for label, raw_config in models_to_run.items():
        config = (
            raw_config
            if isinstance(raw_config, dict)
            else {"id": raw_config, "provider": "hyperbolic"}
        )
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
                        "batch_ok": False,
                        "call_telemetry": False,
                        "error_msg": err,
                    }
                ]
            )
            continue

        res = tester.run_benchmark()
        ReportManager.update_report([res])
        time.sleep(2)


if __name__ == "__main__":
    main(models_to_run=config.get("models_to_run"))
