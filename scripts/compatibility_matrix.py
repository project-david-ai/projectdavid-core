"""
compatibility_matrix.py

Deduces SDK <-> Server compatibility using the relay model:

Each SDK version is compatible with the server versions that existed
during its active lifetime — from the server current when it was released,
up to (but not including) the server version that the next SDK release
was built against.

USAGE:
    python compatibility_matrix.py \
        --core  C:/Users/franc/PycharmProjects/projectdavid-core \
        --sdk   C:/Users/franc/PycharmProjects/entitites_sdk \
        --out   compatibility.json \
        --md    COMPATIBILITY.md

OUTPUT:
    compatibility.json   — machine-readable matrix
    COMPATIBILITY.md     — human-readable two-column table
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ─── DATA STRUCTURES ────────────────────────────────────────────────────────


@dataclass
class ReleasePin:
    tag: str
    timestamp: datetime
    common_pin: str | None = None
    orm_pin: str | None = None
    raw_deps: list[str] = field(default_factory=list)


# ─── GIT HELPERS ────────────────────────────────────────────────────────────


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git error in {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def get_tags(repo: Path) -> list[str]:
    raw = git(repo, "tag", "--sort=version:refname")
    return [t for t in raw.splitlines() if t.strip()]


def get_tag_timestamp(repo: Path, tag: str) -> datetime:
    raw = git(repo, "log", "-1", "--format=%ct", tag)
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (ValueError, OSError):
        return datetime.min.replace(tzinfo=timezone.utc)


def get_file_at_tag(repo: Path, tag: str, filepath: str) -> str | None:
    try:
        return git(repo, "show", f"{tag}:{filepath}")
    except RuntimeError:
        return None


PYPROJECT_CANDIDATES = [
    "pyproject.toml",
    "src/api/entities_api/pyproject.toml",
    "src/projectdavid/pyproject.toml",
    "src/pyproject.toml",
    "entities_api/pyproject.toml",
]


def get_pyproject_at_tag(repo: Path, tag: str) -> str | None:
    for candidate in PYPROJECT_CANDIDATES:
        content = get_file_at_tag(repo, tag, candidate)
        if content:
            return content
    return None


# ─── PYPROJECT PARSING ──────────────────────────────────────────────────────


def extract_pins(pyproject_content: str) -> tuple[str | None, str | None, list[str]]:
    common_pin = None
    orm_pin = None
    raw_deps = []

    dep_block_match = re.search(
        r"dependencies\s*=\s*\[(.*?)\]",
        pyproject_content,
        re.DOTALL,
    )
    if not dep_block_match:
        return None, None, []

    block = dep_block_match.group(1)
    deps = re.findall(r'"([^"]+)"', block)
    raw_deps = deps

    for dep in deps:
        normalized = dep.replace("_", "-").lower()
        if normalized.startswith("projectdavid-common"):
            common_pin = dep
        elif normalized.startswith("projectdavid-orm"):
            orm_pin = dep

    return common_pin, orm_pin, raw_deps


# ─── RELEASE EXTRACTION ─────────────────────────────────────────────────────


def extract_releases(repo: Path, label: str) -> list[ReleasePin]:
    tags = get_tags(repo)
    releases = []
    print(f"\n[{label}] Found {len(tags)} tags — extracting...")

    for tag in tags:
        content = get_pyproject_at_tag(repo, tag)
        ts = get_tag_timestamp(repo, tag)

        if content:
            common_pin, orm_pin, raw_deps = extract_pins(content)
        else:
            common_pin, orm_pin, raw_deps = None, None, []

        releases.append(
            ReleasePin(
                tag=tag,
                timestamp=ts,
                common_pin=common_pin,
                orm_pin=orm_pin,
                raw_deps=raw_deps,
            )
        )

    print(f"  [{label}] {len(releases)} releases parsed")
    return releases


# ─── RELAY MODEL ────────────────────────────────────────────────────────────


def version_sort_key(release: ReleasePin) -> tuple[int, ...]:
    """Sort by version number — handles retroactive tagging correctly."""
    tag = release.tag.lstrip("v")
    parts = []
    for p in re.split(r"[\.\-]", tag):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def nearest_server_at_or_before(
    ts: datetime,
    server_timeline: list[ReleasePin],
) -> ReleasePin | None:
    """Return the most recent server release at or before the given timestamp."""
    result = None
    for s in server_timeline:
        if s.timestamp <= ts:
            result = s
    return result


def server_index(tag: str, server_timeline: list[ReleasePin]) -> int:
    for i, s in enumerate(server_timeline):
        if s.tag == tag:
            return i
    return -1


def build_matrix(
    sdk_releases: list[ReleasePin],
    server_releases: list[ReleasePin],
) -> list[dict]:
    """
    Relay model: each SDK version is compatible with server versions
    that existed during its active lifetime — from the server current
    when it was released, up to the server that the next SDK targets.
    """
    matrix = []

    # Exclude pre-release server tags (alpha/beta/rc) from the compatibility timeline
    server_timeline = sorted(
        [
            s
            for s in server_releases
            if not any(x in s.tag.lower() for x in ["alpha", "beta", "rc"])
        ],
        key=version_sort_key,
    )
    sdk_sorted = sorted(sdk_releases, key=version_sort_key)

    for i, sdk in enumerate(sdk_sorted):
        lower = nearest_server_at_or_before(sdk.timestamp, server_timeline)

        if i + 1 < len(sdk_sorted):
            next_sdk = sdk_sorted[i + 1]
            upper = nearest_server_at_or_before(next_sdk.timestamp, server_timeline)
        else:
            upper = server_timeline[-1] if server_timeline else None

        if not lower:
            compatible_versions = []
            compatible_range = "unknown"
        elif lower.tag == (upper.tag if upper else None):
            compatible_versions = [lower.tag]
            compatible_range = lower.tag
        else:
            lo_idx = server_index(lower.tag, server_timeline)
            hi_idx = server_index(
                upper.tag if upper else server_timeline[-1].tag, server_timeline
            )
            if lo_idx > hi_idx:
                lo_idx, hi_idx = hi_idx, lo_idx
            compatible_versions = [s.tag for s in server_timeline[lo_idx : hi_idx + 1]]
            compatible_range = _summarize_range(compatible_versions) or "unknown"

        # Skip pre-release SDK tags from output
        if sdk.tag.startswith("test-") or "alpha" in sdk.tag or "beta" in sdk.tag:
            continue

        matrix.append(
            {
                "sdk_version": sdk.tag,
                "compatible_server_versions": compatible_versions,
                "compatible_server_range": compatible_range,
            }
        )

    # Restore original tag order
    tag_order = {s.tag: i for i, s in enumerate(sdk_releases)}
    matrix.sort(key=lambda r: tag_order.get(r["sdk_version"], 9999))

    return matrix


def _summarize_range(versions: list[str]) -> str | None:
    if not versions:
        return None
    if len(versions) == 1:
        return versions[0]
    return f">={versions[0]},<={versions[-1]}"


# ─── OUTPUT ──────────────────────────────────────────────────────────────────


def write_json(matrix: list[dict], path: Path) -> None:
    path.write_text(json.dumps(matrix, indent=2))
    print(f"\n✅ JSON written → {path}")


def write_markdown(matrix: list[dict], path: Path) -> None:
    lines = [
        "# Project David — SDK ↔ Server Compatibility Matrix",
        "",
        "Auto-generated by `compatibility_matrix.py`.",
        "Compatibility deduced from chronological co-release history.",
        "",
        "| SDK Version | Compatible Server Versions |",
        "|-------------|---------------------------|",
    ]

    for row in matrix:
        servers = row["compatible_server_range"] or "unknown"
        lines.append(f"| `{row['sdk_version']}` | `{servers}` |")

    lines += [
        "",
        "## Notes",
        "",
        "- Compatibility is deduced from chronological co-release history, not tested.",
        "- Each SDK version is compatible with server versions that existed during its active lifetime.",
        "- Rows marked `unknown` are early tags that predate parseable dependency metadata.",
        "- When in doubt, always use the latest SDK with the latest server.",
        "",
    ]

    path.write_text("\n".join(lines))
    print(f"✅ Markdown written → {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SDK <-> Server compatibility matrix"
    )
    parser.add_argument("--core", required=True, help="Path to projectdavid-core repo")
    parser.add_argument("--sdk", required=True, help="Path to projectdavid SDK repo")
    parser.add_argument("--out", default="compatibility.json", help="Output JSON path")
    parser.add_argument("--md", default="COMPATIBILITY.md", help="Output markdown path")
    args = parser.parse_args()

    core_path = Path(args.core)
    sdk_path = Path(args.sdk)
    out_path = Path(args.out)
    md_path = Path(args.md)

    for p, label in [(core_path, "--core"), (sdk_path, "--sdk")]:
        if not (p / ".git").exists():
            print(f"ERROR: {label} path '{p}' is not a git repo")
            sys.exit(1)

    print("=" * 60)
    print("Project David — Compatibility Matrix Builder")
    print("=" * 60)

    server_releases = extract_releases(core_path, "SERVER")
    sdk_releases = extract_releases(sdk_path, "SDK")

    print(
        f"\n[MATRIX] Building: {len(sdk_releases)} SDK × {len(server_releases)} server releases"
    )

    matrix = build_matrix(sdk_releases, server_releases)

    write_json(matrix, out_path)
    write_markdown(matrix, md_path)

    unmatched = [r for r in matrix if not r["compatible_server_versions"]]
    print(
        f"\nSummary: {len(matrix)} SDK releases, {len(unmatched)} with no server match"
    )


if __name__ == "__main__":
    main()
