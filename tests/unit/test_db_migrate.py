"""No MySQL, Docker, network, or migration writes are used by these tests."""

from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import create_engine, make_url

from scripts import db_migrate


def test_local_url_rejects_docker_hostname(tmp_path):
    env_file = tmp_path / ".env.migrations"
    env_file.write_text(
        "SPECIAL_DB_URL=mysql+pymysql://user:secret@db:3306/entities_db\n",
        encoding="utf-8",
    )
    with pytest.raises(db_migrate.PreflightError, match="loopback"):
        db_migrate.local_url(env_file)


def test_local_url_rejects_unexpected_host_port(tmp_path):
    env_file = tmp_path / ".env.migrations"
    env_file.write_text(
        "SPECIAL_DB_URL=mysql+pymysql://user:secret@localhost:3306/entities_db\n",
        encoding="utf-8",
    )
    with pytest.raises(db_migrate.PreflightError, match="port"):
        db_migrate.local_url(env_file)


def test_local_url_never_requires_application_env(tmp_path):
    env_file = tmp_path / ".env.migrations"
    env_file.write_text(
        "SPECIAL_DB_URL=mysql+pymysql://user:secret@127.0.0.1:3307/entities_db\n",
        encoding="utf-8",
    )
    url = db_migrate.local_url(env_file)
    assert url.database == "entities_db"
    assert url.port == 3307


def test_apply_cannot_run_without_exact_orm_version():
    with pytest.raises(SystemExit) as result:
        db_migrate.parser().parse_args(["apply", "--env", "local"])
    assert result.value.code == 2


def test_mistyped_confirmation_never_calls_alembic(monkeypatch):
    snapshot = db_migrate.Snapshot(
        url=make_url("mysql+pymysql://u:secret@localhost:3307/entities_db"),
        orm_version="1.15.0",
        orm_location=Path("C:/env/Lib/site-packages/projectdavid_orm/__init__.py"),
        database="entities_db",
        current="old",
        head="new",
        pending=(("new", "Add protected keys", "migrations/new.py"),),
    )
    monkeypatch.setattr(db_migrate, "collect_snapshot", lambda expected: snapshot)
    monkeypatch.setattr("builtins.input", lambda prompt: "NOT APPROVED")

    def forbidden(*args, **kwargs):
        raise AssertionError("Alembic must not run")

    monkeypatch.setattr(db_migrate, "alembic_command", forbidden)
    assert db_migrate.main(["apply", "--env", "local", "--orm-version", "1.15.0"]) == 2


def test_apply_at_head_is_a_noop(monkeypatch):
    snapshot = db_migrate.Snapshot(
        url=make_url("mysql+pymysql://u:secret@localhost:3307/entities_db"),
        orm_version="1.15.0",
        orm_location=Path("/tmp/site-packages/projectdavid_orm/__init__.py"),
        database="entities_db",
        current="head",
        head="head",
        pending=(),
    )
    monkeypatch.setattr(db_migrate, "collect_snapshot", lambda expected: snapshot)
    monkeypatch.setattr(
        db_migrate,
        "alembic_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No DDL")),
    )
    assert db_migrate.main(["apply", "--env", "local", "--orm-version", "1.15.0"]) == 0


def test_verify_rejects_pending_migrations(monkeypatch):
    snapshot = db_migrate.Snapshot(
        url=make_url("mysql+pymysql://u:secret@localhost:3307/entities_db"),
        orm_version="1.15.0",
        orm_location=Path("/tmp/site-packages/projectdavid_orm/__init__.py"),
        database="entities_db",
        current="old",
        head="new",
        pending=(("new", "new schema", "migration.py"),),
    )
    monkeypatch.setattr(db_migrate, "collect_snapshot", lambda expected: snapshot)
    monkeypatch.setattr(
        db_migrate,
        "alembic_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No check")),
    )
    assert db_migrate.main(["verify", "--env", "local"]) == 2


def test_repository_graph_has_single_head():
    """Existing Core CI runs pytest tests/, so this guards PRs too."""
    _scripts, head = db_migrate.revision_script()
    assert isinstance(head, str)
    assert head


def test_exact_approval_runs_upgrade_and_checks_head(monkeypatch):
    url = make_url("mysql+pymysql://u:secret@localhost:3307/entities_db")
    first = db_migrate.Snapshot(
        url=url,
        orm_version="1.15.0",
        orm_location=Path("/tmp/site-packages/projectdavid_orm/__init__.py"),
        database="entities_db",
        current="old",
        head="new",
        pending=(("new", "New migration", "migrations/new.py"),),
    )
    last = db_migrate.Snapshot(
        url=url,
        orm_version="1.15.0",
        orm_location=first.orm_location,
        database="entities_db",
        current="new",
        head="new",
        pending=(),
    )
    snapshots = iter([first, last])
    monkeypatch.setattr(
        db_migrate, "collect_snapshot", lambda expected: next(snapshots)
    )
    monkeypatch.setattr(
        "builtins.input", lambda prompt: "APPLY entities_db@localhost:3307 new"
    )
    commands = []

    def fake_alembic(command, target_url):
        commands.append(command)
        assert target_url == url
        return 0

    monkeypatch.setattr(db_migrate, "alembic_command", fake_alembic)
    assert db_migrate.main(["apply", "--env", "local", "--orm-version", "1.15.0"]) == 0
    assert commands == ["upgrade"]


def test_verify_reports_drift_without_migration_writes(monkeypatch):
    snapshot = db_migrate.Snapshot(
        url=make_url("mysql+pymysql://u:secret@localhost:3307/entities_db"),
        orm_version="1.15.0",
        orm_location=Path("/tmp/site-packages/projectdavid_orm/__init__.py"),
        database="entities_db",
        current="head",
        head="head",
        pending=(),
    )
    monkeypatch.setattr(db_migrate, "collect_snapshot", lambda expected: snapshot)
    commands = []

    def fake_alembic(command, _url):
        commands.append(command)
        return 1

    monkeypatch.setattr(db_migrate, "alembic_command", fake_alembic)
    assert db_migrate.main(["verify", "--env", "local"]) == 2
    assert commands == ["check"]


def test_existing_unversioned_database_fails_closed(tmp_path, monkeypatch):
    database_file = tmp_path / "existing.sqlite"
    url = make_url(f"sqlite:///{database_file}")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE api_keys (id INTEGER PRIMARY KEY)"))
    finally:
        engine.dispose()

    # Simulate the MySQL DATABASE() function in this isolated SQLite test.
    def fake_engine(*args, **kwargs):
        instance = create_engine(url)

        @event.listens_for(instance, "connect")
        def on_connect(connection, _record):
            connection.create_function("DATABASE", 0, lambda: url.database)

        return instance

    monkeypatch.setattr(db_migrate, "create_engine", fake_engine)
    with pytest.raises(db_migrate.PreflightError, match="no alembic_version"):
        db_migrate.database_revision(url)
