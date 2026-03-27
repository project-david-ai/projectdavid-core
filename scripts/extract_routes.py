#!/usr/bin/env python

import os
import sys

from tabulate import tabulate

# Setup path to allow module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# =================================================================
# FIX: Load environment variables before importing FastAPI routers
# =================================================================
try:
    from dotenv import load_dotenv

    env_path = os.path.join(project_root, ".env")

    # Load the .env file
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"⚙️ Loaded environment variables from: {env_path}")
    else:
        print(f"⚠️ Warning: No .env file found at {env_path}")
except ImportError:
    print("⚠️ python-dotenv is not installed. Run: pip install python-dotenv")

# Fallback: If the DB connection string is STILL empty, inject a dummy one.
# (Note: Change 'DATABASE_URL' if your database.py expects a different name like 'DB_URL')
db_var_names = ["DATABASE_URL", "DB_URL", "SQLALCHEMY_DATABASE_URI"]
if not any(os.environ.get(var) for var in db_var_names):
    print(
        "⚠️ No Database URL found in environment. Injecting dummy SQLite URL to prevent crash."
    )
    for var in db_var_names:
        os.environ[var] = "sqlite:///:memory:"
# =================================================================

try:
    from entities_api.routers import api_router  # Central router only
    from fastapi.routing import APIRoute
except ImportError as e:
    print(f"❌ Failed to import FastAPI components: {e}")
    sys.exit(1)


def extract_routes(router, prefix=""):
    """Recursively extract routes from a FastAPI router."""
    routes_data = []

    for route in router.routes:
        if isinstance(route, APIRoute):
            methods = ",".join(sorted(route.methods))
            path = prefix + route.path
            name = route.name
            summary = route.summary or ""
            tags = ", ".join(route.tags or [])
            routes_data.append([methods, path, name, tags, summary])

        elif hasattr(route, "include_router"):
            # It's a nested router
            sub_prefix = route.prefix or ""
            nested_routes = extract_routes(route.include_router, prefix + sub_prefix)
            routes_data.extend(nested_routes)

    return routes_data


def save_markdown_table(rows, headers, output_path):
    """Save the route table as a markdown file."""
    markdown = tabulate(rows, headers=headers, tablefmt="github")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 📘 API Endpoint Table\n\n")
        f.write(markdown)
        f.write("\n")
    print(f"\n✅ Saved markdown table to: {output_path}")


def main():
    print("📡 Extracting FastAPI route metadata...\n")

    routes = extract_routes(api_router)
    routes.sort(key=lambda r: r[1])  # Sort by path

    headers = ["Method", "Path", "Name", "Tags", "Summary"]
    print(tabulate(routes, headers=headers, tablefmt="fancy_grid", stralign="left"))

    # Save Markdown version
    markdown_output_path = os.path.join(project_root, "docs", "routes.md")
    os.makedirs(os.path.dirname(markdown_output_path), exist_ok=True)
    save_markdown_table(routes, headers, markdown_output_path)


if __name__ == "__main__":
    main()
