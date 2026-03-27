# -------------------------------------------------------------
#  model_table_generator.py
# -------------------------------------------------------------
import argparse
import sys
from pathlib import Path

from projectdavid_common.constants.ai_model_map import MODEL_MAP


# --------------------------------------------------------------------
# Helper: build the markdown table
# --------------------------------------------------------------------
def build_markdown_table(model_map: dict) -> str:
    """
    Returns a markdown string where:
        - inference_provider = first segment of the original key
        - route              = the original key itself
        - tool_calling       = boolean status (✅/❌)
    """
    rows = []
    for full_key, metadata in model_map.items():
        provider = full_key.split("/", 1)[0] if "/" in full_key else full_key
        entities_route = full_key

        # Defensive check: handle both dict metadata and simple string IDs
        if isinstance(metadata, dict):
            tool_calling = metadata.get("tool_calling", False)
        else:
            # If metadata is a string, we default tool_calling to False
            tool_calling = False

        rows.append((provider, entities_route, tool_calling))

    # Sort by provider then route
    rows.sort(key=lambda x: (x[0].lower(), x[1].lower()))

    # Build the Markdown string
    header = "| inference_provider | route | tool_calling |\n|---|---|---|"
    lines = [header]

    for prov, route, tc in rows:
        tc_display = "✅" if tc else "❌"
        lines.append(f"| {prov} | {route} | {tc_display} |")

    return "\n".join(lines)


# --------------------------------------------------------------------
# CLI handling
# --------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown table of model identifiers."
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="model_map.md",
        help="Name of the Markdown file to write (default: model_map.md).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve the docs folder relative to this file
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    docs_dir = project_root / "docs"

    # Make sure the docs folder exists
    docs_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_file
    if Path(output_name).parent == Path("."):
        output_path = docs_dir / output_name
    else:
        output_path = Path(output_name).expanduser().resolve()

    # Generate markdown using the map
    markdown = build_markdown_table(MODEL_MAP)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"✅ Markdown table written to '{output_path}'")
    except OSError as exc:
        print(f"❌ Failed to write file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
