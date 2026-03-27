import os
import re
from pathlib import Path
from typing import Set

from projectdavid_common.constants.ai_model_map import TOGETHER_AI_MODELS

# ------------------------------------------------------------------
# 1. Config & Paths
# ------------------------------------------------------------------
root_dir = Path(__file__).resolve().parents[2]
REPORT_FILE = root_dir / "together_status_report.md"
OUTPUT_FILE = root_dir / "together_candidate_endpoints.py"  # <--- New Output File


# ------------------------------------------------------------------
# 3. Logic to Parse, Merge, and Save
# ------------------------------------------------------------------
def generate_and_save_candidates():
    print(f"Reading Report from: {REPORT_FILE}")

    candidate_endpoints = {}
    known_dead_ids: Set[str] = set()

    # --- Step A: Parse Report for Dead IDs ---
    if REPORT_FILE.exists():
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if (
                    stripped.startswith("|")
                    and "Endpoint ID" not in stripped
                    and ":---" not in stripped
                ):
                    parts = [p.strip() for p in stripped.split("|")]
                    if len(parts) >= 6:
                        raw_id = parts[3].replace("`", "").strip()
                        status_field = parts[4].lower()

                        # Identify Dead
                        if (
                            "💀" in status_field
                            or "dead" in status_field
                            or "404" in status_field
                        ):
                            known_dead_ids.add(raw_id)
                        else:
                            # It is Alive -> Add to candidates
                            mapped_value = raw_id.replace("together-ai/", "")
                            candidate_endpoints[raw_id] = mapped_value

    # --- Step B: Merge with Original (excluding Dead) ---
    for key, val in TOGETHER_AI_MODELS.items():
        if key not in known_dead_ids:
            # If it wasn't in the report at all, or was in the report as ALIVE, keep it.
            candidate_endpoints[key] = val

    # --- Step C: Write to Python File ---
    print(f"Writing clean dictionary to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write("Auto-generated Candidate Endpoints for Together AI.\n")
        f.write(f'Generated on: {os.environ.get("DATE", "Today")}\n')
        f.write("Source: Integration Test Report & Manual Config\n")
        f.write('"""\n\n')
        f.write("together_candidate_endpoints = {\n")

        sorted_keys = sorted(candidate_endpoints.keys())
        current_prefix = ""

        for key in sorted_keys:
            # Grouping Logic for comments
            prefix_match = re.search(r"together-ai/([^/]+)/", key)
            if prefix_match:
                prefix = prefix_match.group(1)
                if prefix != current_prefix:
                    f.write(f"\n    # --- {prefix.capitalize()} ---\n")
                current_prefix = prefix

            value = candidate_endpoints[key]
            f.write(f'    "{key}": "{value}",\n')

        f.write("}\n")

    print("✅ Success! File saved.")
    print(f"   - Total Candidates: {len(candidate_endpoints)}")
    print(f"   - Dead/Excluded:    {len(known_dead_ids)}")


if __name__ == "__main__":
    generate_and_save_candidates()
