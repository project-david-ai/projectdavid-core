"""Classify package metadata changes without unnecessary Docker rebuilds."""

from __future__ import annotations

import copy
import os
import subprocess
import tomllib
from pathlib import Path

COMPONENTS = (
    "api",
    "sandbox",
    "training",
    "inference",
    "router",
    "shared",
)

ROOT_TOML = "pyproject.toml"
SANDBOX_TOML = "src/api/sandbox/pyproject.toml"


def meaningful_toml_change(
    before: bytes | None,
    after: bytes | None,
) -> bool:
    """
    Return False when the only semantic change is project.version.

    Missing, malformed, or unreadable TOML is treated as a
    build-relevant change.
    """
    if before is None or after is None:
        return True

    try:
        old = tomllib.loads(before.decode("utf-8-sig"))
        new = tomllib.loads(after.decode("utf-8-sig"))
    except (UnicodeError, tomllib.TOMLDecodeError):
        return True

    old = copy.deepcopy(old)
    new = copy.deepcopy(new)

    if isinstance(old.get("project"), dict):
        old["project"].pop("version", None)

    if isinstance(new.get("project"), dict):
        new["project"].pop("version", None)

    return old != new


def git_file(
    revision: str,
    filename: str,
) -> bytes | None:
    """
    Read file contents from a committed Git revision.

    Never inspect the potentially dirty working tree.
    """
    result = subprocess.run(
        ["git", "show", f"{revision}:{filename}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if result.returncode != 0:
        return None

    return result.stdout


def effective_filters(
    raw: dict[str, bool],
    *,
    base: str,
    head: str,
    root_toml_changed: bool,
    sandbox_toml_changed: bool,
) -> dict[str, bool]:
    """
    Combine GitHub path-filter results with semantic TOML detection.

    Preserve all existing source and Docker change flags.
    """
    flags = {name: bool(raw.get(name, False)) for name in COMPONENTS}

    if root_toml_changed:
        flags["api"] |= meaningful_toml_change(
            git_file(base, ROOT_TOML),
            git_file(head, ROOT_TOML),
        )

    if sandbox_toml_changed:
        flags["sandbox"] |= meaningful_toml_change(
            git_file(base, SANDBOX_TOML),
            git_file(head, SANDBOX_TOML),
        )

    return flags


def main() -> None:
    base = os.getenv("COMPARE_BASE", "origin/main")

    if not base or set(base) == {"0"}:
        base = "origin/main"

    head = os.getenv("GITHUB_SHA", "HEAD")

    raw = {name: os.getenv(f"RAW_{name.upper()}") == "true" for name in COMPONENTS}

    flags = effective_filters(
        raw,
        base=base,
        head=head,
        root_toml_changed=(os.getenv("ROOT_TOML_CHANGED") == "true"),
        sandbox_toml_changed=(os.getenv("SANDBOX_TOML_CHANGED") == "true"),
    )

    lines = [
        f"{name}={'true' if changed else 'false'}" for name, changed in flags.items()
    ]

    print("\n".join(lines))

    if output_path := os.getenv("GITHUB_OUTPUT"):
        with Path(output_path).open(
            "a",
            encoding="utf-8",
        ) as output:
            output.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
