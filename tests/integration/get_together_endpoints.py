import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from together import Together

# ------------------------------------------------------------------
# 1. Setup & Config
# ------------------------------------------------------------------

# Resolve the project root directory (2 levels up)
root_dir = Path(__file__).resolve().parents[2]
load_dotenv()

# Define the Report File Path in the Root Directory
REPORT_FILE = root_dir / "together_status_report.md"

API_KEY = os.getenv("TOGETHER_API_KEY")

# ------------------------------------------------------------------
# 2. Model Dictionary
# ------------------------------------------------------------------
TOGETHER_AI_MODELS = {
    # --- Qwen ---
    "Qwen2.5-14B-Instruct": {
        "id": "together-ai/Qwen/Qwen2.5-14B-Instruct",
        "provider": "together-ai",
    },
    "Qwen2.5-72B-Instruct": {
        "id": "together-ai/Qwen/Qwen2.5-72B-Instruct",
        "provider": "together-ai",
    },
    "Qwen2.5-72B-Instruct-Turbo": {
        "id": "together-ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        "provider": "together-ai",
    },
    "Qwen2.5-7B-Instruct-Turbo": {
        "id": "together-ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        "provider": "together-ai",
    },
    "Qwen2.5-VL-72B-Instruct": {
        "id": "together-ai/Qwen/Qwen2.5-VL-72B-Instruct",
        "provider": "together-ai",
    },
    "Qwen3-235B-A22B-Instruct-2507-tput": {
        "id": "together-ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "provider": "together-ai",
    },
    "Qwen3-235B-A22B-Thinking-2507": {
        "id": "together-ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        "provider": "together-ai",
    },
    "Qwen3-235B-A22B-fp8-tput": {
        "id": "together-ai/Qwen/Qwen3-235B-A22B-fp8-tput",
        "provider": "together-ai",
    },
    "Qwen3-Coder-480B-A35B-Instruct-FP8": {
        "id": "together-ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "provider": "together-ai",
    },
    "Qwen3-Next-80B-A3B-Instruct": {
        "id": "together-ai/Qwen/Qwen3-Next-80B-A3B-Instruct",
        "provider": "together-ai",
    },
    "Qwen3-Next-80B-A3B-Thinking": {
        "id": "together-ai/Qwen/Qwen3-Next-80B-A3B-Thinking",
        "provider": "together-ai",
    },
    "Qwen3-VL-32B-Instruct": {
        "id": "together-ai/Qwen/Qwen3-VL-32B-Instruct",
        "provider": "together-ai",
    },
    "Qwen3-VL-8B-Instruct": {
        "id": "together-ai/Qwen/Qwen3-VL-8B-Instruct",
        "provider": "together-ai",
    },
    # --- ServiceNow-AI ---
    "Apriel-1.5-15b-Thinker": {
        "id": "together-ai/ServiceNow-AI/Apriel-1.5-15b-Thinker",
        "provider": "together-ai",
    },
    "Apriel-1.6-15b-Thinker": {
        "id": "together-ai/ServiceNow-AI/Apriel-1.6-15b-Thinker",
        "provider": "together-ai",
    },
    # --- arcee-ai ---
    "trinity-mini": {
        "id": "together-ai/arcee-ai/trinity-mini",
        "provider": "together-ai",
    },
    # --- arize-ai ---
    "qwen-2-1.5b-instruct": {
        "id": "together-ai/arize-ai/qwen-2-1.5b-instruct",
        "provider": "together-ai",
    },
    # --- deepcogito ---
    "cogito-v2-1-671b": {
        "id": "together-ai/deepcogito/cogito-v2-1-671b",
        "provider": "together-ai",
    },
    "cogito-v2-preview-llama-109B-MoE": {
        "id": "together-ai/deepcogito/cogito-v2-preview-llama-109B-MoE",
        "provider": "together-ai",
    },
    "cogito-v2-preview-llama-405B": {
        "id": "together-ai/deepcogito/cogito-v2-preview-llama-405B",
        "provider": "together-ai",
    },
    "cogito-v2-preview-llama-70B": {
        "id": "together-ai/deepcogito/cogito-v2-preview-llama-70B",
        "provider": "together-ai",
    },
    # --- deepseek-ai ---
    "DeepSeek-R1": {
        "id": "together-ai/deepseek-ai/DeepSeek-R1",
        "provider": "together-ai",
    },
    "DeepSeek-R1-0528-tput": {
        "id": "together-ai/deepseek-ai/DeepSeek-R1-0528-tput",
        "provider": "together-ai",
    },
    "DeepSeek-R1-Distill-Llama-70B": {
        "id": "together-ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "provider": "together-ai",
    },
    "DeepSeek-V3.1": {
        "id": "together-ai/deepseek-ai/DeepSeek-V3.1",
        "provider": "together-ai",
    },
    # --- essentialai ---
    "rnj-1-instruct": {
        "id": "together-ai/essentialai/rnj-1-instruct",
        "provider": "together-ai",
    },
    # --- google ---
    "gemma-2b-it-Ishan": {
        "id": "together-ai/google/gemma-2b-it-Ishan",
        "provider": "together-ai",
    },
    "gemma-3n-E4B-it": {
        "id": "together-ai/google/gemma-3n-E4B-it",
        "provider": "together-ai",
    },
    # --- marin-community ---
    "marin-8b-instruct": {
        "id": "together-ai/marin-community/marin-8b-instruct",
        "provider": "together-ai",
    },
    # --- meta-llama ---
    "Llama-3-70b-hf": {
        "id": "together-ai/meta-llama/Llama-3-70b-hf",
        "provider": "together-ai",
    },
    "Llama-3.1-405B-Instruct": {
        "id": "together-ai/meta-llama/Llama-3.1-405B-Instruct",
        "provider": "together-ai",
    },
    "Llama-3.2-1B-Instruct": {
        "id": "together-ai/meta-llama/Llama-3.2-1B-Instruct",
        "provider": "together-ai",
    },
    "Llama-3.2-3B-Instruct-Turbo": {
        "id": "together-ai/meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "provider": "together-ai",
    },
    "Llama-3.3-70B-Instruct-Turbo": {
        "id": "together-ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "provider": "together-ai",
    },
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "id": "together-ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "provider": "together-ai",
    },
    "Llama-4-Scout-17B-16E-Instruct": {
        "id": "together-ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "provider": "together-ai",
    },
    "Meta-Llama-3-8B-Instruct": {
        "id": "together-ai/meta-llama/Meta-Llama-3-8B-Instruct",
        "provider": "together-ai",
    },
    "Meta-Llama-3-8B-Instruct-Lite": {
        "id": "together-ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "provider": "together-ai",
    },
    "Meta-Llama-3.1-405B-Instruct-Turbo": {
        "id": "together-ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "provider": "together-ai",
    },
    "Meta-Llama-3.1-70B-Instruct-Reference": {
        "id": "together-ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
        "provider": "together-ai",
    },
    "Meta-Llama-3.1-70B-Instruct-Turbo": {
        "id": "together-ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "provider": "together-ai",
    },
    "Meta-Llama-3.1-8B-Instruct-Reference": {
        "id": "together-ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        "provider": "together-ai",
    },
    "Meta-Llama-3.1-8B-Instruct-Turbo": {
        "id": "together-ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "provider": "together-ai",
    },
    # --- mistralai ---
    "Ministral-3-14B-Instruct-2512": {
        "id": "together-ai/mistralai/Ministral-3-14B-Instruct-2512",
        "provider": "together-ai",
    },
    "Mistral-7B-Instruct-v0.2": {
        "id": "together-ai/mistralai/Mistral-7B-Instruct-v0.2",
        "provider": "together-ai",
    },
    "Mistral-7B-Instruct-v0.3": {
        "id": "together-ai/mistralai/Mistral-7B-Instruct-v0.3",
        "provider": "together-ai",
    },
    "Mistral-Small-24B-Instruct-2501": {
        "id": "together-ai/mistralai/Mistral-Small-24B-Instruct-2501",
        "provider": "together-ai",
    },
    "Mixtral-8x7B-Instruct-v0.1": {
        "id": "together-ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "provider": "together-ai",
    },
    # --- moonshotai ---
    "Kimi-K2-Instruct-0905": {
        "id": "together-ai/moonshotai/Kimi-K2-Instruct-0905",
        "provider": "together-ai",
    },
    "Kimi-K2-Thinking": {
        "id": "together-ai/moonshotai/Kimi-K2-Thinking",
        "provider": "together-ai",
    },
    # --- nvidia ---
    "NVIDIA-Nemotron-Nano-9B-v2": {
        "id": "together-ai/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "provider": "together-ai",
    },
    # --- openai ---
    "gpt-oss-120b": {
        "id": "together-ai/openai/gpt-oss-120b",
        "provider": "together-ai",
    },
    "gpt-oss-20b": {
        "id": "together-ai/openai/gpt-oss-20b",
        "provider": "together-ai",
    },
    # --- scb10x ---
    "scb10x-typhoon-2-1-gemma3-12b": {
        "id": "together-ai/scb10x/scb10x-typhoon-2-1-gemma3-12b",
        "provider": "together-ai",
    },
    # --- togethercomputer ---
    "MoA-1": {
        "id": "together-ai/togethercomputer/MoA-1",
        "provider": "together-ai",
    },
    "MoA-1-Turbo": {
        "id": "together-ai/togethercomputer/MoA-1-Turbo",
        "provider": "together-ai",
    },
    "Refuel-Llm-V2": {
        "id": "together-ai/togethercomputer/Refuel-Llm-V2",
        "provider": "together-ai",
    },
    "Refuel-Llm-V2-Small": {
        "id": "together-ai/togethercomputer/Refuel-Llm-V2-Small",
        "provider": "together-ai",
    },
    # --- zai-org ---
    "GLM-4.5-Air-FP8": {
        "id": "together-ai/zai-org/GLM-4.5-Air-FP8",
        "provider": "together-ai",
    },
    "GLM-4.6": {
        "id": "together-ai/zai-org/GLM-4.6",
        "provider": "together-ai",
    },
    "GLM-4.7": {
        "id": "together-ai/zai-org/GLM-4.7",
        "provider": "together-ai",
    },
}


# ------------------------------------------------------------------
# 3. Report Management
# ------------------------------------------------------------------
def update_markdown(result: Dict):
    """
    Reads the existing MD file at REPORT_FILE, updates entries based on Endpoint ID,
    and writes back to disk.
    """
    # Define Layout
    header = "| Provider | Model Name | Endpoint ID | Status | Last Run |"
    divider = "| :--- | :--- | :--- | :--- | :--- |"

    # Storage for rows: {endpoint_id: line_string}
    existing_rows = {}

    # 1. Read existing
    if REPORT_FILE.exists():
        try:
            with open(REPORT_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if (
                        line.strip().startswith("|")
                        and "Endpoint ID" not in line
                        and ":---" not in line
                    ):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 6:
                            # parts[0] is empty due to leading |
                            # parts[3] is Endpoint ID
                            row_id = parts[3].strip("`")
                            existing_rows[row_id] = line.strip()
        except Exception as e:
            print(f"⚠️ Could not read existing report: {e}")

    # 2. Format the new result row
    status_icon = result["status_icon"]
    status_msg = result["status_msg"]
    full_status = f"{status_icon} {status_msg}"

    row_str = (
        f"| {result['provider']} | {result['name']} | `{result['full_id']}` | "
        f"{full_status} | {result['timestamp']} |"
    )

    # Update dictionary (upsert)
    existing_rows[result["full_id"]] = row_str

    # 3. Write Back
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("# 📡 Together AI Endpoint Status\n\n")
            f.write(header + "\n")
            f.write(divider + "\n")

            # Sort by Provider, then Model Name
            sorted_rows = sorted(
                existing_rows.values(), key=lambda x: x.split("|")[1].strip()
            )

            for row in sorted_rows:
                f.write(row + "\n")
    except Exception as e:
        print(f"❌ Failed to write report to {REPORT_FILE}: {e}")


# ------------------------------------------------------------------
# 4. Testing Logic
# ------------------------------------------------------------------
def endpoints():
    if not API_KEY:
        print("❌ Error: TOGETHER_API_KEY not found in environment.")
        return

    client = Together(api_key=API_KEY)

    print(f"📂 Report File: {REPORT_FILE}")
    print(f"🚀 Starting Test for {len(TOGETHER_AI_MODELS)} endpoints...\n")

    for friendly_name, config in TOGETHER_AI_MODELS.items():
        full_id = config["id"]
        provider = config["provider"]

        # LOGIC: Strip "together-ai/" prefix to get actual API endpoint string
        actual_api_model = full_id.replace("together-ai/", "")

        print(f"Testing: {friendly_name}...", end=" ", flush=True)

        result_payload = {
            "name": friendly_name,
            "provider": provider,
            "full_id": full_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status_icon": "❓",
            "status_msg": "Unknown",
        }

        try:
            # Perform a minimal generation request
            stream = client.chat.completions.create(
                model=actual_api_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                stream=True,
            )

            # Consume first chunk to verify life
            received_content = False
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    received_content = True
                    break  # One chunk is enough

            if received_content:
                print("✅ OK")
                result_payload["status_icon"] = "✅"
                result_payload["status_msg"] = "Online"
            else:
                print("⚠️ No Content")
                result_payload["status_icon"] = "⚠️"
                result_payload["status_msg"] = "Empty Response"

        except Exception as e:
            err_str = str(e).lower()
            if (
                "404" in err_str
                or "not found" in err_str
                or "model_not_available" in err_str
                or "does not exist" in err_str
            ):
                print("💀 Dead")
                result_payload["status_icon"] = "💀"
                result_payload["status_msg"] = "Dead / 404"
            elif "rate limit" in err_str or "429" in err_str:
                print("⏳ Rate Limit")
                result_payload["status_icon"] = "⏳"
                result_payload["status_msg"] = "Rate Limited"
            else:
                print(f"❌ Error: {str(e)[:20]}...")
                result_payload["status_icon"] = "❌"
                result_payload["status_msg"] = f"Error: {str(e)[:30]}"

        # Update the file immediately after each test
        update_markdown(result_payload)

        # Sleep briefly
        time.sleep(1.0)


if __name__ == "__main__":
    endpoints()
