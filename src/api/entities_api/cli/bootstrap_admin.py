# src/api/entities_api/cli/bootstrap_admin.py
#
# Run via:
#   python -m entities_api bootstrap-admin
#   python -m entities_api bootstrap-admin --email admin@example.com --db-url mysql+pymysql://...
#


from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Optional

import typer
from sqlalchemy import create_engine
from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Project imports — these will resolve correctly when the package is installed
# or when the project root is on PYTHONPATH.
# ---------------------------------------------------------------------------
try:
    from entities_api.models.models import ApiKey, User
    from projectdavid_common import UtilsInterface
    from projectdavid_common.utilities.logging_service import LoggingUtility
except ImportError as exc:
    typer.echo(
        f"[error] Could not import project modules: {exc}\n"
        "Ensure the package is installed (`pip install -e .`) or run from the project root.",
        err=True,
    )
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_NAME = "Default Admin"
DEFAULT_ADMIN_KEY_NAME = "Admin Bootstrap Key"
ADMIN_API_KEY_ENV_VAR = "ADMIN_API_KEY"
DB_URL_ENV_VAR = "DATABASE_URL"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
log = LoggingUtility()
identifier_service = UtilsInterface.IdentifierService()

app = typer.Typer(
    name="bootstrap-admin",
    help="Bootstrap the initial admin user and API key for the Entities API.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _build_engine(db_url: str):
    """Return a SQLAlchemy engine, or exit with a clear message on failure."""
    try:
        engine = create_engine(db_url, echo=False, pool_pre_ping=True)
        with engine.connect():
            pass
        log.info(f"Connected to database: {engine.url.database}")
        return engine
    except sqlalchemy_exc.OperationalError as exc:
        typer.echo(
            f"[error] Could not connect to database.\n"
            f"  URL prefix : {db_url[:20]}…\n"
            f"  Detail     : {exc}",
            err=True,
        )
        raise SystemExit(1)
    except Exception as exc:
        typer.echo(f"[error] Unexpected database error: {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Domain logic  (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # store as naive UTC


def _ensure_admin_user(db: Session, email: str, name: str) -> tuple[User, bool]:
    """
    Return (user, created).

    If the user already exists it is returned as-is (is_admin is enforced).
    If not, a new admin user is created and added to the session.
    The caller is responsible for committing.
    """
    user = db.query(User).filter(User.email == email).first()

    if user:
        created = False
        log.info(f"Existing user found: {email} (id={user.id})")
        if not user.is_admin:
            log.warning(f"User {email} exists but is not admin — elevating.")
            user.is_admin = True
            user.updated_at = _now()
    else:
        created = True
        user = User(
            id=identifier_service.generate_user_id(),
            email=email,
            full_name=name,
            email_verified=True,
            oauth_provider="local",
            is_admin=True,
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(user)
        log.info(f"New admin user queued for creation: {email} (id={user.id})")

    return user, created


def _ensure_api_key(db: Session, user: User, key_name: str) -> tuple[str | None, str]:
    """
    Return (plain_text_key, prefix).

    plain_text_key is None if the user already has a key (we cannot recover the
    original plain-text value).  The caller is responsible for committing.
    """
    existing = db.query(ApiKey).filter(ApiKey.user_id == user.id).first()
    if existing:
        log.info(f"API key already exists for user {user.id} (prefix={existing.prefix})")
        return None, existing.prefix

    plain_key = ApiKey.generate_key(prefix="ad_")
    prefix = plain_key[:8]
    record = ApiKey(
        user_id=user.id,
        key_name=key_name,
        hashed_key=ApiKey.hash_key(plain_key),
        prefix=prefix,
        is_active=True,
        created_at=_now(),
    )
    db.add(record)
    log.info(f"New API key queued for creation (prefix={prefix})")
    return plain_key, prefix


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_new_key(user: User, prefix: str, plain_key: str) -> None:
    width = 64
    typer.echo("\n" + "=" * width)
    typer.echo("  ✓  Admin API Key Generated")
    typer.echo("=" * width)
    typer.echo(f"  Email   : {user.email}")
    typer.echo(f"  User ID : {user.id}")
    typer.echo(f"  Prefix  : {prefix}")
    typer.echo("-" * width)
    typer.echo(f"  API KEY : {plain_key}")
    typer.echo("-" * width)
    typer.echo("  This key will NOT be shown again.")
    typer.echo(f"  Set it in your environment:")
    typer.echo(f"    export {ADMIN_API_KEY_ENV_VAR}={plain_key}")
    typer.echo("=" * width + "\n")


def _print_existing_key(user: User, prefix: str) -> None:
    typer.echo(
        f"\n[info] Admin user '{user.email}' already has an API key (prefix={prefix}).\n"
        "       No new key was generated.\n"
        "       If you have lost the key, delete the existing ApiKey row and re-run.\n"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


@app.command()
def bootstrap_admin(
    email: str = typer.Option(
        DEFAULT_ADMIN_EMAIL,
        "--email",
        envvar="ADMIN_EMAIL",
        help="Email address for the admin user.",
    ),
    name: str = typer.Option(
        DEFAULT_ADMIN_NAME,
        "--name",
        help="Full name for the admin user.",
    ),
    key_name: str = typer.Option(
        DEFAULT_ADMIN_KEY_NAME,
        "--key-name",
        help="Label for the generated API key.",
    ),
    db_url: str = typer.Option(
        ...,
        "--db-url",
        envvar=DB_URL_ENV_VAR,
        help="SQLAlchemy database URL.  Required if SPECIAL_DB_URL is not set.",
    ),
) -> None:
    """
    Bootstrap an admin user and API key.

    Safe to re-run: existing users and keys are detected and left untouched.
    The generated API key is printed once to stdout — store it immediately.
    """
    engine = _build_engine(db_url)

    with Session(engine) as db:
        try:
            # --- All mutations in a single atomic transaction ---
            user, user_created = _ensure_admin_user(db, email, name)
            plain_key, prefix = _ensure_api_key(db, user, key_name)
            db.commit()
            # Refresh while session is still open so attributes are accessible
            # for the print functions below — avoids DetachedInstanceError.
            db.refresh(user)
            log.info("Bootstrap transaction committed successfully.")
        except Exception as exc:
            db.rollback()
            log.error(f"Bootstrap failed, transaction rolled back: {exc}", exc_info=True)
            typer.echo(f"[error] Bootstrap failed: {exc}", err=True)
            raise SystemExit(1)

        # --- Output inside the session block so user attributes remain accessible ---
        if plain_key:
            _print_new_key(user, prefix, plain_key)
        else:
            _print_existing_key(user, prefix)


# ---------------------------------------------------------------------------
# Allow `python -m entities_api.cli.bootstrap_admin` as a fallback
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
