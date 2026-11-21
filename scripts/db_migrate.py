"""Safe, host-side Alembic operator for Project David Core.

Run from the Core repository: python -m scripts.db_migrate <command> --env local

This tool never guesses database credentials, upgrades packages, generates
migrations, stamps revisions, or runs a downgrade. Database writes require
interactive approval of the exact target and migration head.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, make_url

ROOT = Path(__file__).resolve().parent.parent
MIGRATION_ENV = ROOT / ".env.migrations"
ALEMBIC_INI = ROOT / "alembic.ini"


class PreflightError(RuntimeError):
    """Refuse to proceed when a migration precondition is not satisfied."""


@dataclass(frozen=True)
class Snapshot:
    url: URL
    orm_version: str
    orm_location: Path
    database: str
    current: str | None
    head: str
    pending: tuple[tuple[str, str, str], ...]


def local_url(path: Path = MIGRATION_ENV) -> URL:
    """Read the host-only URL; refuse nonlocal or ambiguous targets."""
    if not path.is_file():
        raise PreflightError(f"Host configuration not found: {path}")
    raw = dotenv_values(path).get("SPECIAL_DB_URL")
    if not raw:
        raise PreflightError("SPECIAL_DB_URL missing from .env.migrations")
    try:
        url = make_url(raw)
    except Exception as exc:
        raise PreflightError("SPECIAL_DB_URL is not a valid SQLAlchemy URL") from exc
    if url.drivername != "mysql+pymysql":
        raise PreflightError("Host migrations require mysql+pymysql")
    if url.host not in ("localhost", "127.0.0.1"):
        raise PreflightError("Host migrations require a loopback MySQL endpoint")
    if url.port not in (3307, 3308):
        raise PreflightError("Unexpected host MySQL port (expected 3307 or 3308)")
    if not all((url.username, url.password, url.database)):
        raise PreflightError("Migration URL needs a username, password and database")
    return url


def installed_orm(expected: str | None) -> tuple[str, Path]:
    """Reject accidental editable/source-tree ORM shadowing."""
    try:
        distribution = metadata.distribution("projectdavid-orm")
        installed = distribution.version
        package = importlib.import_module("projectdavid_orm")
    except (metadata.PackageNotFoundError, ImportError) as exc:
        raise PreflightError("Install the released projectdavid-orm wheel") from exc

    location = Path(package.__file__ or "").resolve()
    if "site-packages" not in {part.lower() for part in location.parts} and (
        "dist-packages" not in {part.lower() for part in location.parts}
    ):
        raise PreflightError(f"ORM import is shadowed by local source: {location}")

    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            if json.loads(direct_url).get("dir_info", {}).get("editable"):
                raise PreflightError(
                    "Editable ORM installation is not a release artifact"
                )
        except json.JSONDecodeError as exc:
            raise PreflightError("Unrecognized ORM distribution provenance") from exc

    if expected is not None and installed != expected:
        raise PreflightError(
            f"ORM version mismatch: installed {installed}, expected {expected}"
        )
    return installed, location


def revision_script() -> tuple[ScriptDirectory, str]:
    """Validate Alembic's on-disk revision graph without connecting to MySQL."""
    if not ALEMBIC_INI.is_file():
        raise PreflightError(f"Missing {ALEMBIC_INI}")
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    heads = scripts.get_heads()
    if len(heads) != 1:
        raise PreflightError(f"Exactly one Alembic head required; found {heads}")
    return scripts, heads[0]


def database_revision(url: URL) -> tuple[str, str | None]:
    """Read database identity and Alembic revision without modifying either."""
    engine = create_engine(url, connect_args={"connect_timeout": 5}, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            actual = connection.execute(text("SELECT DATABASE() ")).scalar_one()
            if actual != url.database:
                raise PreflightError(
                    f"Connected to unexpected database {actual!r}; "
                    f"expected {url.database!r}"
                )
            inspector = inspect(connection)
            existing_tables = inspector.get_table_names()
            if "alembic_version" not in existing_tables:
                if existing_tables:
                    raise PreflightError(
                        "Existing database has no alembic_version table; "
                        "reconcile manually"
                    )
                return actual, None
            versions = connection.execute(
                text("SELECT version_num FROM alembic_version")
            )
            rows = [row[0] for row in versions]
            if len(rows) > 1:
                raise PreflightError("Multiple revisions recorded in alembic_version")
            if not rows and len(existing_tables) > 1:
                raise PreflightError(
                    "Existing database has no recorded Alembic revision; "
                    "reconcile manually"
                )
            return actual, rows[0] if rows else None
    finally:
        engine.dispose()


def collect_snapshot(expected_orm: str | None) -> Snapshot:
    url = local_url()
    version, location = installed_orm(expected_orm)
    scripts, head = revision_script()
    database, current = database_revision(url)
    try:
        revisions = list(scripts.iterate_revisions(head, current or "base"))
    except Exception as exc:
        raise PreflightError(
            f"Database revision {current!r} is unknown or not an ancestor of {head}"
        ) from exc
    pending = tuple(
        (revision.revision, revision.doc or "(no description)", str(revision.path))
        for revision in reversed(revisions)
    )
    return Snapshot(url, version, location, database, current, head, pending)


def print_snapshot(snapshot: Snapshot) -> None:
    url = snapshot.url
    print(f"Target: {snapshot.database}@{url.host}:{url.port} (local MySQL)")
    print(f"ORM: projectdavid-orm=={snapshot.orm_version}")
    print(f"ORM import: {snapshot.orm_location}")
    print(f"Database revision: {snapshot.current or '(unversioned)'}")
    print(f"Alembic head: {snapshot.head}")
    print(f"Pending revisions: {len(snapshot.pending)}")


def print_plan(snapshot: Snapshot) -> None:
    print_snapshot(snapshot)
    if snapshot.pending:
        print("\nPending revisions, in application order:")
        for revision, description, file_path in snapshot.pending:
            title = description.strip().splitlines()[0]
            print(f"  {revision}  {title}")
            print(f"    {file_path}")
        print("Review the listed migration source files before applying.")
    else:
        print("Already at head; no migration to apply.")


def alembic_command(command: str, url: URL) -> int:
    """Use Core's existing Alembic environment with process-local credentials."""
    env = os.environ.copy()
    env["DATABASE_URL"] = url.render_as_string(hide_password=False)
    result = subprocess.run(
        (
            [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), command, "head"]
            if command == "upgrade"
            else [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), command]
        ),
        cwd=ROOT,
        env=env,
        check=False,
    )
    return result.returncode


def require_approval(snapshot: Snapshot) -> bool:
    target = f"{snapshot.database}@{snapshot.url.host}:{snapshot.url.port}"
    phrase = f"APPLY {target} {snapshot.head}"
    print("\nWARNING: MySQL DDL may commit immediately and may not roll back.")
    print("Ensure a suitable database backup exists before continuing.")
    print(f"To proceed, type exactly: {phrase}")
    try:
        return input("> ").strip() == phrase
    except (EOFError, KeyboardInterrupt):
        return False


def verify_columns(url: URL, expected: list[str]) -> None:
    if not expected:
        return
    # Use the same released ORM metadata that Alembic uses.
    from projectdavid_orm.projectdavid_orm import models

    metadata_tables = models.User.metadata.tables
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            for item in expected:
                if item.count(".") != 1:
                    raise PreflightError(f"Expected table.column: {item!r}")
                table, name = item.split(".")
                if table not in metadata_tables or name not in metadata_tables[table].c:
                    raise PreflightError(
                        f"Column absent from released ORM metadata: {item}"
                    )
                if not inspector.has_table(table):
                    raise PreflightError(f"Physical table missing: {table}")
                actual = next(
                    (
                        column
                        for column in inspector.get_columns(table)
                        if column["name"] == name
                    ),
                    None,
                )
                if actual is None:
                    raise PreflightError(f"Physical column missing: {item}")
                expected_nullable = metadata_tables[table].c[name].nullable
                if bool(actual["nullable"]) != bool(expected_nullable):
                    raise PreflightError(f"Nullability mismatch: {item}")
                print(f"PASS: {item} exists; nullable={bool(actual['nullable'])}")
    finally:
        engine.dispose()


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        description="Safe Project David Core Alembic runner (host-local only)"
    )
    commands = main_parser.add_subparsers(dest="command", required=True)
    for command in ("status", "plan", "apply", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--env", required=True, choices=("local",))
        sub.add_argument(
            "--orm-version",
            required=command == "apply",
            help="Exact released projectdavid-orm version (required to apply)",
        )
        if command == "verify":
            sub.add_argument(
                "--expect-column",
                action="append",
                default=[],
                metavar="TABLE.COLUMN",
                help="Check physical column and nullable metadata (repeatable)",
            )
    return main_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        snapshot = collect_snapshot(args.orm_version)
        if args.command == "status":
            print_snapshot(snapshot)
        elif args.command == "plan":
            print_plan(snapshot)
        elif args.command == "apply":
            print_plan(snapshot)
            if not snapshot.pending:
                return 0
            if not require_approval(snapshot):
                raise PreflightError("Approval not provided. No migration applied")
            if alembic_command("upgrade", snapshot.url):
                raise PreflightError("Alembic upgrade failed; inspect database state")
            after = collect_snapshot(args.orm_version)
            if after.current != snapshot.head:
                raise PreflightError("Database did not reach the expected head")
            print(f"PASS: Database now at {after.head}")
        elif args.command == "verify":
            print_snapshot(snapshot)
            if snapshot.pending:
                raise PreflightError(
                    "Database is behind Alembic head; apply before verify"
                )
            verify_columns(snapshot.url, args.expect_column)
            print("Running alembic check (classify any historical drift manually)...")
            if alembic_command("check", snapshot.url):
                raise PreflightError(
                    "Alembic drift check failed; classify, do not auto-fix"
                )
            print("PASS: Alembic revision and drift check")
        return 0
    except PreflightError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Do not echo the exception: underlying DB exceptions can expose URLs.
        print(
            f"STOP: {type(exc).__name__} during {args.command}; see logs safely",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
