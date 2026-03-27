import logging
import re
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Define the files relative to the script's location or use absolute paths
SCRIPT_DIR = Path(__file__).parent
# Using different output names to avoid confusion if you run multiple versions
FILES = [
    (SCRIPT_DIR / "../api_requirements.txt", SCRIPT_DIR / "../api_reqs_cleaned.txt"),
    (
        SCRIPT_DIR / "../sandbox_requirements.txt",
        SCRIPT_DIR / "../sandbox_reqs_cleaned.txt",
    ),
]

# Regex to find hash annotations (case-insensitive hex)
HASH_PATTERN = re.compile(r"\s*--hash=sha256:[a-fA-F0-9]{64}")


def strip_hashes_and_backslashes(line: str) -> str:
    """
    Removes requirement hashes (--hash=...) and trailing backslashes
    used for line continuation. Also strips trailing whitespace.
    """
    # Remove hash pattern first
    line_no_hash = HASH_PATTERN.sub("", line)

    # Strip trailing whitespace
    line_stripped = line_no_hash.rstrip()

    # Remove trailing backslash if present, and strip again
    if line_stripped.endswith("\\"):
        line_final = line_stripped[:-1].rstrip()
    else:
        line_final = line_stripped

    return line_final


def process_and_clean_file(source_path: Path, output_path: Path):
    """
    Reads a file:
    1. Skips lines that are entirely blank (or only whitespace).
    2. For non-blank lines, removes hashes and trailing backslashes.
    3. Skips lines that become blank *after* removing hashes/backslashes.
    4. Writes the cleaned lines to the output file.
    """
    if not source_path.is_file():
        logging.warning(f"⚠️ Source file not found: {source_path}")
        return

    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
        logging.info(f"Read {len(lines)} lines from {source_path.name}")
    except Exception as e:
        logging.error(f"❌ Error reading {source_path}: {e}")
        return

    cleaned_lines = []
    for i, line in enumerate(lines):
        # Check if the original line is just whitespace
        if not line.strip():
            # logging.debug(f"Line {i+1}: Skipping blank line.")
            continue  # Skip original blank lines

        # Process the line (remove hash, backslash, trailing whitespace)
        processed_line = strip_hashes_and_backslashes(line)

        # Check if the line became blank *after* processing
        if not processed_line.strip():
            # This could happen if a line ONLY contained a hash and whitespace
            # logging.debug(f"Line {i+1}: Skipping line that became blank after processing: '{line}'")
            continue  # Skip lines that are now blank

        # If we reach here, the line had content and still has content
        cleaned_lines.append(processed_line)
        # logging.debug(f"Line {i+1}: Keeping processed line: '{processed_line}'")

    output_content = "\n".join(cleaned_lines)
    if output_content:  # Add trailing newline only if there's content
        output_content += "\n"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding="utf-8")
        logging.info(
            f"✅ Wrote {len(cleaned_lines)} cleaned lines to {output_path.name}"
        )
    except Exception as e:
        logging.error(f"❌ Error writing {output_path}: {e}")


if __name__ == "__main__":
    logging.info(
        "--- Processing requirements files (Cleaned: No Blanks, No Hashes) ---"
    )
    for src, dst in FILES:
        process_and_clean_file(src, dst)
    logging.info("--- Done ---")
