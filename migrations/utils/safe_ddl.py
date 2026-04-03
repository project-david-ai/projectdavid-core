"""
migrations/utils/safe_ddl.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fault-tolerant, idempotent helpers for Alembic migrations.

All operations are safe to run against both fresh databases (Scenario B)
and existing databases with live data (Scenario A). Repeated runs never
raise errors — operations are skipped with a warning log instead.

Usage
-----
    from migrations.utils.safe_ddl import (
        add_column_if_missing,
        drop_column_if_exists,
        safe_alter_column,
        create_index_if_not_exists,
        drop_index_if_exists,
        drop_fk_if_exists,
        drop_table_if_exists,
        has_table,
        has_column,
    )

Logging
-------
Every operation emits a structured log line:

    [Alembic-safeDDL] ✅ Added column: users.profile_url
    [Alembic-safeDDL] ⚠️  Skipped – column already exists: users.profile_url
"""

from __future__ import annotations

import logging
from typing import Any, List

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)
_PREFIX = "[Alembic-safeDDL]"


# ---------------------------------------------------------------------------
# Internal inspection helpers
# ---------------------------------------------------------------------------


def _bind():
    """Return the current migration connection."""
    return op.get_bind()


def has_table(table_name: str) -> bool:
    """Return True if *table_name* exists in the current database."""
    result = (
        _bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() "
                "AND table_name = :t"
            ),
            {"t": table_name},
        )
        .scalar()
    )
    return bool(result)


def has_column(table_name: str, column_name: str) -> bool:
    """Return True if *column_name* exists in *table_name*."""
    result = (
        _bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = DATABASE() "
                "AND table_name = :t "
                "AND column_name = :c"
            ),
            {"t": table_name, "c": column_name},
        )
        .scalar()
    )
    return bool(result)


def _has_index(table_name: str, index_name: str) -> bool:
    """Return True if *index_name* exists on *table_name*."""
    result = (
        _bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM information_schema.statistics "
                "WHERE table_schema = DATABASE() "
                "AND table_name = :t "
                "AND index_name = :i"
            ),
            {"t": table_name, "i": index_name},
        )
        .scalar()
    )
    return bool(result)


def _has_fk(table_name: str, constraint_name: str) -> bool:
    """Return True if a foreign key constraint exists on *table_name*."""
    result = (
        _bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS "
                "WHERE table_schema = DATABASE() "
                "AND table_name = :t "
                "AND constraint_name = :c "
                "AND constraint_type = 'FOREIGN KEY'"
            ),
            {"t": table_name, "c": constraint_name},
        )
        .scalar()
    )
    return bool(result)


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


def add_column_if_missing(table_name: str, column: sa.Column) -> None:
    """Add *column* to *table_name* only if it does not already exist."""
    if has_column(table_name, column.name):
        logger.info(
            "%s ⚠️  Skipped – column already exists: %s.%s",
            _PREFIX,
            table_name,
            column.name,
        )
        return
    op.add_column(table_name, column)
    logger.info(
        "%s ✅ Added column: %s.%s",
        _PREFIX,
        table_name,
        column.name,
    )


def drop_column_if_exists(table_name: str, column_name: str) -> None:
    """Drop *column_name* from *table_name* only if it exists."""
    if not has_column(table_name, column_name):
        logger.info(
            "%s ⚠️  Skipped – column does not exist: %s.%s",
            _PREFIX,
            table_name,
            column_name,
        )
        return
    op.drop_column(table_name, column_name)
    logger.info(
        "%s ✅ Dropped column: %s.%s",
        _PREFIX,
        table_name,
        column_name,
    )


def safe_alter_column(table_name: str, column_name: str, **kwargs) -> None:
    """
    Alter *column_name* on *table_name* only if the column exists.

    All keyword arguments are forwarded to ``op.alter_column``.
    """
    if not has_column(table_name, column_name):
        logger.info(
            "%s ⚠️  Skipped alter – column does not exist: %s.%s",
            _PREFIX,
            table_name,
            column_name,
        )
        return
    op.alter_column(table_name, column_name, **kwargs)
    logger.info(
        "%s ✅ Altered column: %s.%s",
        _PREFIX,
        table_name,
        column_name,
    )


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def create_index_if_not_exists(
    index_name: str,
    table_name: str,
    columns: List[str],
    unique: bool = False,
) -> None:
    """
    Create an index only if it does not already exist.

    Accepts the same positional arguments as ``op.create_index``.
    ``index_name`` may be a plain string or the result of ``op.f()``.
    """
    # op.f() returns an object whose str() is the resolved name
    resolved_name = str(index_name)
    if _has_index(table_name, resolved_name):
        logger.info(
            "%s ⚠️  Skipped – index already exists: %s on %s",
            _PREFIX,
            resolved_name,
            table_name,
        )
        return
    op.create_index(index_name, table_name, columns, unique=unique)
    logger.info(
        "%s ✅ Created index: %s on %s",
        _PREFIX,
        resolved_name,
        table_name,
    )


def drop_index_if_exists(index_name: str, table_name: str) -> None:
    """
    Drop *index_name* on *table_name* only if it exists.

    NOTE: MySQL 8.0 does not support DROP INDEX IF EXISTS (MariaDB only).
    This helper queries information_schema.statistics instead.
    """
    if not _has_index(table_name, index_name):
        logger.info(
            "%s ⚠️  Skipped – index does not exist: %s on %s",
            _PREFIX,
            index_name,
            table_name,
        )
        return
    _bind().execute(sa.text(f"DROP INDEX `{index_name}` ON `{table_name}`"))
    logger.info(
        "%s ✅ Dropped index: %s on %s",
        _PREFIX,
        index_name,
        table_name,
    )


# ---------------------------------------------------------------------------
# Foreign key helpers
# ---------------------------------------------------------------------------


def drop_fk_if_exists(table_name: str, constraint_name: str) -> None:
    """
    Drop a foreign key constraint only if it exists.

    Must be called before dropping any index that backs the FK —
    MySQL raises OperationalError 1553 otherwise.
    """
    if not _has_fk(table_name, constraint_name):
        logger.info(
            "%s ⚠️  Skipped – FK does not exist: %s on %s",
            _PREFIX,
            constraint_name,
            table_name,
        )
        return
    _bind().execute(
        sa.text(f"ALTER TABLE `{table_name}` DROP FOREIGN KEY `{constraint_name}`")
    )
    logger.info(
        "%s ✅ Dropped FK: %s on %s",
        _PREFIX,
        constraint_name,
        table_name,
    )


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def drop_table_if_exists(table_name: str) -> None:
    """
    Drop *table_name* only if it exists.

    IMPORTANT: Drop all foreign keys and indexes on the table before
    calling this — use drop_fk_if_exists and drop_index_if_exists first.
    """
    if not has_table(table_name):
        logger.info(
            "%s ⚠️  Skipped – table does not exist: %s",
            _PREFIX,
            table_name,
        )
        return
    _bind().execute(sa.text(f"DROP TABLE IF EXISTS `{table_name}`"))
    logger.info(
        "%s ✅ Dropped table: %s",
        _PREFIX,
        table_name,
    )
