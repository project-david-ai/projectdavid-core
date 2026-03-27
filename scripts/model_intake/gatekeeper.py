import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import dotenv

dotenv.load_dotenv()

# --- CONFIGURATION ---
TARGET_FILE = "models.py"
MANIFEST_FILE = "model_manifest.json"
REPORT_FILE = "HEALTH_REPORT.md"
CONCURRENCY = 8  # Higher concurrency since threads are lightweight
TIMEOUT_SECONDS = 15


# Load .env manually if dotenv lib is not allowed,
# otherwise we assume env vars are set in the shell or we parse .env crudely here.
def load_env_manual():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val.strip('"').strip("'")


load_env_manual()

PROVIDERS = {
    "hyperbolic": {
        "url": "https://api.hyperbolic.xyz/v1/models",
        "chat_url": "https://api.hyperbolic.xyz/v1/chat/completions",
        "api_key": os.getenv("HYPERBOLIC_API_KEY"),
        "exclude": ["StableDiffusion", "FLUX", "TTS", "SDXL", "diffusion"],
    },
    "together-ai": {
        "url": "https://api.together.xyz/v1/models",
        "chat_url": "https://api.together.xyz/v1/chat/completions",
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "exclude": ["image", "video"],
    },
}

TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_flight_times",
        "description": "Get flight info",
        "parameters": {
            "type": "object",
            "properties": {"dep": {"type": "string"}, "arr": {"type": "string"}},
            "required": ["dep", "arr"],
        },
    },
}

# --- 1. NETWORK UTILS (Standard Lib) ---


def raw_request(url, method="GET", api_key=None, payload=None):
    """
    Executes a raw HTTP request using urllib (No 'requests' lib needed).
    """
    if not api_key:
        return None, "Missing API Key"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Entities-Gatekeeper/1.0",
    }

    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Connection Error: {e.reason}"
    except Exception as e:
        return None, str(e)


# --- 2. FETCHERS ---


def fetch_model_list(provider_name):
    print(f"   ⬇️  Fetching {provider_name} list...")
    config = PROVIDERS[provider_name]
    data, err = raw_request(config["url"], "GET", config["api_key"])

    if err:
        print(f"   ❌ {provider_name} failed: {err}")
        return []

    # Unify response formats (Together returns list directly, Hyperbolic wraps in 'data')
    raw_list = data.get("data", data) if isinstance(data, dict) else data

    candidates = []
    for m in raw_list:
        mid = m.get("id")
        mtype = m.get("type", "chat")  # Default to chat if unspecified

        # Filter Logic
        if any(x in mid for x in config["exclude"]):
            continue
        if mtype not in ["chat", "language", "code", None]:
            continue

        candidates.append(
            {
                "id": mid,
                "provider": provider_name,
                "full_id": f"{provider_name}/{mid}",
                "name": mid.split("/")[-1],
            }
        )
    return candidates


# --- 3. VERIFICATION WORKER ---


def verify_model(model_info):
    """
    Performs the 'Dual Detector' check via a live POST request.
    """
    config = PROVIDERS[model_info["provider"]]

    payload = {
        "model": model_info["id"],
        "messages": [
            {"role": "user", "content": "What is the flight time from NYC to LON?"}
        ],
        "tools": [TEST_TOOL],
        "tool_choice": "auto",
        "max_tokens": 300,
        "temperature": 0.1,
    }

    start_t = time.time()
    response_data, error = raw_request(
        config["chat_url"], "POST", config["api_key"], payload
    )
    latency = round(time.time() - start_t, 2)

    result = {
        **model_info,
        "status": "dead",
        "protocol": "none",
        "latency": latency,
        "error": error,
    }

    if error:
        return result

    # Analysis
    try:
        choice = response_data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        result["status"] = "alive"

        # Protocol Detection
        if tool_calls:
            result["protocol"] = "native"
        elif re.search(r"<fc>|<tool_code>|<tool_call>", content):
            result["protocol"] = "hermes"
        else:
            result["protocol"] = "chat-only"

    except (KeyError, IndexError):
        result["error"] = "Malformed Response"

    # Console Feedback
    icon = "✅" if result["status"] == "alive" else "💀"
    print(
        f"{icon} [{model_info['provider'][:4].upper()}] {model_info['name'][:30]:<30} | {result['protocol'].upper():<10} | {latency}s"
    )

    return result


# --- 4. FILE OPERATIONS ---

# --- 4. FILE OPERATIONS (Fixed for Windows) ---


def generate_files(results):
    # 1. Manifest (JSON)
    # Added encoding="utf-8" here
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Markdown Report
    alive = [r for r in results if r["status"] == "alive"]

    # Added encoding="utf-8" here to support Emojis on Windows
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Model Health Report\n**Generated:** {datetime.now()}\n\n")
        f.write(f"**Total:** {len(results)} | **Alive:** {len(alive)}\n\n")
        f.write(
            "| Provider | Model | Proto | Latency | Status |\n|---|---|---|---|---|\n"
        )
        for r in sorted(results, key=lambda x: (x["provider"], x["status"])):
            icon = "🟢" if r["status"] == "alive" else "🔴"
            f.write(
                f"| {r['provider']} | {r['name']} | {r['protocol']} | {r['latency']}s | {icon} |\n"
            )

    print(f"\n📄 Manifest & Report generated.")
    return alive


def inject_python(valid_models):
    if not os.path.exists(TARGET_FILE):
        print(f"⚠️ {TARGET_FILE} not found. Skipping injection.")
        return

    # Helper to build dict string
    def build_dict_str(models):
        grouped = defaultdict(list)
        for m in models:
            # Org grouping
            org = m["id"].split("/")[0] if "/" in m["id"] else "Other"
            grouped[org].append(m)

        lines = ["{"]
        for org in sorted(grouped.keys()):
            lines.append(f"    # --- {org} ---")
            for m in sorted(grouped[org], key=lambda x: x["name"]):
                lines.append(f"    \"{m['name']}\": {{")
                lines.append(f"        \"id\": \"{m['full_id']}\",")
                lines.append(f"        \"provider\": \"{m['provider']}\",")
                lines.append(f"        \"protocol\": \"{m['protocol']}\",")
                lines.append(f"    }},")
            lines.append("")
        lines.append("}")
        return "\n".join(lines)

    # Filter sets
    hyp_str = build_dict_str([m for m in valid_models if m["provider"] == "hyperbolic"])
    tog_str = build_dict_str(
        [m for m in valid_models if m["provider"] == "together-ai"]
    )

    # Added encoding="utf-8" for safe reading
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex Replacements
    content = re.sub(
        r"(# --- HYPERBOLIC_MODELS_START ---)(.*?)(# --- HYPERBOLIC_MODELS_END ---)",
        f"\\1\nHYPERBOLIC_MODELS = {hyp_str}\n\\3",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"(# --- TOGETHER_AI_MODELS_START ---)(.*?)(# --- TOGETHER_AI_MODELS_END ---)",
        f"\\1\nTOGETHER_AI_MODELS = {tog_str}\n\\3",
        content,
        flags=re.DOTALL,
    )

    # Added encoding="utf-8" for safe writing
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"💉 Successfully injected healthy models into {TARGET_FILE}")


# --- MAIN ---


def main():
    print("🚀 Starting Zero-Dependency Gatekeeper...")

    # 1. Fetch Lists
    candidates = []
    candidates.extend(fetch_model_list("hyperbolic"))
    candidates.extend(fetch_model_list("together-ai"))

    print(
        f"📋 Verification started for {len(candidates)} models (Threads={CONCURRENCY})..."
    )

    # 2. Threaded Verification
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_model = {executor.submit(verify_model, m): m for m in candidates}
        for future in as_completed(future_to_model):
            results.append(future.result())

    # 3. Output
    alive_models = generate_files(results)
    inject_python(alive_models)


if __name__ == "__main__":
    main()
