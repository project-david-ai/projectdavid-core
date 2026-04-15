import json
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

# --- CI run era counts ---
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
runs = json.loads(result.stdout)
if isinstance(runs, dict) and "value" in runs:
    runs = runs["value"]

era1 = sum(
    1
    for r in runs
    if r.get("updatedAt")
    and datetime.fromisoformat(r["updatedAt"].replace("Z", "+00:00")) < ERA1_END
)
era2 = sum(
    1
    for r in runs
    if r.get("updatedAt")
    and ERA1_END
    <= datetime.fromisoformat(r["updatedAt"].replace("Z", "+00:00"))
    < ERA2_END
)
era3 = sum(
    1
    for r in runs
    if r.get("updatedAt")
    and datetime.fromisoformat(r["updatedAt"].replace("Z", "+00:00")) >= ERA2_END
)
ci_noise = (era1 * 2 + era2 * 4 + era3 * 6) * PULLS_PER_IMAGE_PER_RUN

# --- Docker Hub pull counts ---
print(f"\n{'Image':<45} {'Pulls':>8}")
print("-" * 55)
total_pulls = 0
for image in IMAGES:
    r = requests.get(
        f"https://hub.docker.com/v2/repositories/thanosprime/{image}/"
    ).json()
    pulls = r["pull_count"]
    total_pulls += pulls
    print(f"{image:<45} {pulls:>8,}")

# --- Last pulled tag analysis ---
result = subprocess.run(
    [
        "gh",
        "run",
        "list",
        "--repo",
        REPO,
        "--branch",
        "main",
        "--workflow",
        "Lint, Test, Build, and Publish Docker Images",
        "--limit",
        "50",
        "--json",
        "displayTitle,updatedAt,conclusion",
    ],
    capture_output=True,
    text=True,
)
ci_runs = [
    r
    for r in json.loads(result.stdout)
    if r.get("conclusion") == "success"
    and (
        r.get("displayTitle", "").startswith("fix")
        or r.get("displayTitle", "").startswith("feat")
    )
]

url = "https://hub.docker.com/v2/repositories/thanosprime/projectdavid-core-api/tags/?page_size=100"
all_tags = []
while url:
    resp = requests.get(url).json()
    all_tags += resp.get("results", [])
    url = resp.get("next")

import re

semver_tags = [t for t in all_tags if re.match(r"^\d+\.\d+\.\d+$", t["name"])]
semver_tags.sort(key=lambda t: list(map(int, t["name"].split("."))), reverse=True)

print(
    f"\n{'Tag':<10} {'CIFinished':<14} {'LastPulled':<14} {'GapHours':>10} {'Likely'}"
)
print("-" * 60)

real_user_tags = 0
active_tags = 0
now = datetime.now(timezone.utc)

for tag in semver_tags:
    pulled = datetime.fromisoformat(tag["tag_last_pulled"].replace("Z", "+00:00"))
    closest = (
        min(
            ci_runs,
            key=lambda r: abs(
                (
                    datetime.fromisoformat(r["updatedAt"].replace("Z", "+00:00"))
                    - pulled
                ).total_seconds()
            ),
        )
        if ci_runs
        else None
    )
    ci_finished = (
        datetime.fromisoformat(closest["updatedAt"].replace("Z", "+00:00"))
        if closest
        else None
    )
    gap = (
        round((pulled - ci_finished).total_seconds() / 3600, 1) if ci_finished else None
    )
    likely = "REAL USER" if gap is not None and gap > 2 else "CI/YOU"
    if likely == "REAL USER":
        real_user_tags += 1
    if (now - pulled).days < 7:
        active_tags += 1
    ci_str = ci_finished.strftime("%m-%d %H:%M") if ci_finished else "n/a"
    print(
        f"{tag['name']:<10} {ci_str:<14} {pulled.strftime('%m-%d %H:%M'):<14} {str(gap):>10} {likely}"
    )

print(f"\n=== SUMMARY ===")
print(f"Total pulls (all images):      {total_pulls:,}")
print(f"CI run breakdown:              Era1={era1} Era2={era2} Era3={era3}")
print(f"Estimated CI noise:            {ci_noise:,}")
print(f"Estimated real user pulls:     {total_pulls - ci_noise:,}")
print(f"Confirmed real user tags:      {real_user_tags}")
print(f"Active tags (pulled <7d):      {active_tags}")
print(f"Snapshot time:                 {now.strftime('%Y-%m-%d %H:%M')} UTC")
