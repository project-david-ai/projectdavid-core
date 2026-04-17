"""
stack_insights.py

Deterministic Docker Hub pull analytics for projectdavid-core.

Uses GitHub Releases API for exact tag publication timestamps — no inference,
no proximity matching. Gap = tag_last_pulled - release published_at.
"""

import json
import re
import subprocess
from datetime import datetime, timezone

import requests

REPO = "project-david-ai/projectdavid-core"
IMAGES = [
    "projectdavid-core-api",
    "projectdavid-core-sandbox",
    "projectdavid-core-inference-worker",
    "projectdavid-core-training-worker",
    "projectdavid-core-training-api",
    "projectdavid-core-router",
]
ERA1_END = datetime(2026, 3, 16, tzinfo=timezone.utc)
ERA2_END = datetime(2026, 4, 1, tzinfo=timezone.utc)
PULLS_PER_IMAGE_PER_RUN = 17


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# CI noise — all runs across all branches
# ---------------------------------------------------------------------------
result = subprocess.run(
    [
        "gh",
        "run",
        "list",
        "--repo",
        REPO,
        "--limit",
        "300",
        "--json",
        "updatedAt,conclusion",
    ],
    capture_output=True,
    text=True,
)
all_runs = json.loads(result.stdout)
if isinstance(all_runs, dict) and "value" in all_runs:
    all_runs = all_runs["value"]

era1 = sum(
    1 for r in all_runs if r.get("updatedAt") and parse_dt(r["updatedAt"]) < ERA1_END
)
era2 = sum(
    1
    for r in all_runs
    if r.get("updatedAt") and ERA1_END <= parse_dt(r["updatedAt"]) < ERA2_END
)
era3 = sum(
    1 for r in all_runs if r.get("updatedAt") and parse_dt(r["updatedAt"]) >= ERA2_END
)
ci_noise = (era1 * 2 + era2 * 4 + era3 * 6) * PULLS_PER_IMAGE_PER_RUN

# ---------------------------------------------------------------------------
# GitHub Releases — exact publication timestamp per tag
# Deduplicate keeping latest published_at per tag name
# ---------------------------------------------------------------------------
result = subprocess.run(
    [
        "gh",
        "api",
        "repos/project-david-ai/projectdavid-core/releases",
        "--paginate",
        "--jq",
        ".[] | {tag_name, published_at}",
    ],
    capture_output=True,
    text=True,
)

release_map = {}
for line in result.stdout.strip().split("\n"):
    line = line.strip()
    if not line:
        continue
    try:
        entry = json.loads(line)
        tag = entry["tag_name"].lstrip("v")
        pub = parse_dt(entry["published_at"])
        # Keep latest published_at if duplicates exist
        if tag not in release_map or pub > release_map[tag]:
            release_map[tag] = pub
    except Exception:
        continue

# ---------------------------------------------------------------------------
# Docker Hub pull counts
# ---------------------------------------------------------------------------
print(f"\n{'Image':<45} {'Pulls':>8}")
print("-" * 55)
total_pulls = 0
for image in IMAGES:
    r = requests.get(
        f"https://hub.docker.com/v2/repositories/thanosprime/{image}/",
        timeout=10,
    ).json()
    pulls = r["pull_count"]
    total_pulls += pulls
    print(f"{image:<45} {pulls:>8,}")

# ---------------------------------------------------------------------------
# Semver tags from Docker Hub
# ---------------------------------------------------------------------------
url = "https://hub.docker.com/v2/repositories/thanosprime/projectdavid-core-api/tags/?page_size=100"
all_tags = []
while url:
    resp = requests.get(url, timeout=10).json()
    all_tags += resp.get("results", [])
    url = resp.get("next")

semver_tags = [t for t in all_tags if re.match(r"^\d+\.\d+\.\d+$", t["name"])]
semver_tags.sort(key=lambda t: list(map(int, t["name"].split("."))), reverse=True)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print(f"\n{'Tag':<10} {'Released':<14} {'LastPulled':<14} {'GapHours':>10} {'Likely'}")
print("-" * 60)

real_user_tags = 0
active_tags = 0
now = datetime.now(timezone.utc)

for tag in semver_tags:
    pulled = parse_dt(tag["tag_last_pulled"])
    released = release_map.get(tag["name"])
    gap = round((pulled - released).total_seconds() / 3600, 1) if released else None
    likely = "REAL USER" if gap is not None and gap > 2 else "CI/YOU"

    if likely == "REAL USER":
        real_user_tags += 1
    if (now - pulled).days < 7:
        active_tags += 1

    rel_str = released.strftime("%m-%d %H:%M") if released else "n/a"
    print(
        f"{tag['name']:<10} {rel_str:<14} {pulled.strftime('%m-%d %H:%M'):<14} {str(gap):>10} {likely}"
    )

print(f"\n=== SUMMARY ===")
print(f"Total pulls (all images):      {total_pulls:,}")
print(f"CI run breakdown:              Era1={era1} Era2={era2} Era3={era3}")
print(f"Estimated CI noise:            {ci_noise:,}")
print(f"Estimated real user pulls:     {total_pulls - ci_noise:,}")
print(f"Confirmed real user tags:      {real_user_tags}")
print(f"Active tags (pulled <7d):      {active_tags}")
print(f"Snapshot time:                 {now.strftime('%Y-%m-%d %H:%M')} UTC")
