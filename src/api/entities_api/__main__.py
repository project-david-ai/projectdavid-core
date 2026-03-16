# src/api/entities_api/__main__.py
#
# Entry point for:
#   python -m entities_api <command>
#   entities-api <command>          (via [project.scripts] in pyproject.toml)
#
# Available commands
#   bootstrap-admin   – create the initial admin user + API key
#   docker-manager    – manage the Docker Compose stack
#
from __future__ import annotations

import typer
from entities_api.cli.bootstrap_admin import app as bootstrap_admin_app
from entities_api.cli.docker_manager import app as docker_manager_app

root = typer.Typer(
    name="platform-api",
    help="Entities API management CLI.",
    add_completion=False,
)

root.add_typer(bootstrap_admin_app, name="bootstrap-admin")
root.add_typer(docker_manager_app, name="docker-manager")


def main() -> None:
    root()


if __name__ == "__main__":
    main()
