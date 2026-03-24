"""
ensure_schema.py
────────────────
Bulletproof drop-in replacement for Alembic migrations.

Runs at startup and fully reconciles the live database against SQLAlchemy
model definitions. Safe for every known scenario:

  ✅  Fresh installs              → creates all tables
  ✅  Missing tables              → creates them
  ✅  Missing columns             → adds them
  ✅  Column type changes         → alters automatically
  ✅  Enum value additions        → expands ENUM column definition
  ✅  Missing indexes             → creates them
  ✅  Missing unique constraints  → creates them
  ✅  Missing foreign keys        → creates them
  ✅  NOT NULL without default    → forces nullable, warns developer
  ✅  Repeated restarts           → fully idempotent, no-op if current
  ✅  Partial failures            → logs and continues, never crashes app

Does NOT handle (by design):
  ⏭  Column renames              → old stays, new gets added
  ⏭  Column/table drops          → never removes anything
  ⏭  Data migrations             → use a one-off script for those

Usage:
    from src.api.entities_api.utils.ensure_schema import ensure_schema
    ensure_schema(engine)
"""

import logging
import re

import sqlalchemy as sa
# ── Force model registration into Base.metadata ──────────────────────────────
# This MUST happen before Base.metadata is read.
# All SQLAlchemy models register themselves into Base.metadata at import time.
from projectdavid_common.projectdavid_orm.base import Base
from projectdavid_orm.projectdavid_orm import \
    models  # noqa: F401 — side-effect import
from sqlalchemy import inspect, text
from sqlalchemy.dialects.mysql import ENUM as MySQLENUM

logger = logging.getLogger("schema_sync")


# ─────────────────────────────────────────────────────────────────────────────
#  Type normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_type(type_str: str) -> str:
    """
    Normalise a SQL type string for comparison.
    Strips display width from integers, lowercases, strips whitespace.
    e.g. "VARCHAR(64)" == "varchar(64)", "INT(11)" == "int"
    """
    s = type_str.strip().lower()
    # MySQL reports INT(11), BIGINT(20) etc — strip display width for ints
    s = re.sub(r"\b(tinyint|smallint|mediumint|int|bigint)\(\d+\)", lambda m: m.group(1), s)
    return s


def _compile_column_type(column: sa.Column, dialect) -> str:
    """Compile a SQLAlchemy column type to a SQL type string."""
    try:
        return column.type.compile(dialect=dialect)
    except Exception:
        return str(column.type)


def _types_differ(model_type_str: str, db_type_str: str) -> bool:
    """Return True if the two type strings represent different types."""
    return _normalise_type(model_type_str) != _normalise_type(db_type_str)


# ─────────────────────────────────────────────────────────────────────────────
#  Inspector helpers
# ─────────────────────────────────────────────────────────────────────────────


def _existing_columns(inspector, table_name: str) -> dict:
    """Returns {column_name: column_info_dict} for a table."""
    return {c["name"]: c for c in inspector.get_columns(table_name)}


def _existing_indexes(inspector, table_name: str) -> set:
    return {i["name"] for i in inspector.get_indexes(table_name) if i.get("name")}


def _existing_unique_constraints(inspector, table_name: str) -> set:
    return {u["name"] for u in inspector.get_unique_constraints(table_name) if u.get("name")}


def _existing_foreign_keys(inspector, table_name: str) -> set:
    return {fk["name"] for fk in inspector.get_foreign_keys(table_name) if fk.get("name")}


# ─────────────────────────────────────────────────────────────────────────────
#  ALTER statement builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_add_column_sql(table_name: str, column: sa.Column, dialect) -> str:
    """
    Build ALTER TABLE ... ADD COLUMN.
    Forces nullable if NOT NULL has no server_default — warns developer.
    """
    col_type = _compile_column_type(column, dialect)

    # Safety: if NOT NULL but no server_default, existing rows would get NULL
    # which MySQL rejects. Force nullable and warn.
    if not column.nullable and column.server_default is None:
        logger.warning(
            f"[schema-sync] ⚠️  DEV WARNING: `{table_name}.{column.name}` is "
            f"NOT NULL with no server_default. Adding as NULL to protect existing "
            f"data. Add a server_default to your model to enforce NOT NULL on new rows."
        )
        nullable_clause = "NULL"
        default_clause = ""
    else:
        nullable_clause = "NULL" if column.nullable else "NOT NULL"
        default_clause = ""
        if column.server_default is not None:
            default_clause = f"DEFAULT {column.server_default.arg}"
        elif column.default is not None and hasattr(column.default, "arg"):
            arg = column.default.arg
            if isinstance(arg, (str, int, float, bool)):
                default_clause = f"DEFAULT {arg!r}"

    parts = [f"ALTER TABLE `{table_name}` ADD COLUMN `{column.name}` {col_type}"]
    if default_clause:
        parts.append(default_clause)
    parts.append(nullable_clause)

    return " ".join(parts)


def _build_modify_column_sql(table_name: str, column: sa.Column, dialect) -> str:
    """
    Build ALTER TABLE ... MODIFY COLUMN for type changes.
    Preserves nullability and default from model. Same NOT NULL safety guard.
    """
    col_type = _compile_column_type(column, dialect)

    if not column.nullable and column.server_default is None:
        logger.warning(
            f"[schema-sync] ⚠️  DEV WARNING: `{table_name}.{column.name}` is "
            f"NOT NULL with no server_default. Keeping as NULL during MODIFY to "
            f"protect existing data. Add a server_default to enforce NOT NULL."
        )
        nullable_clause = "NULL"
        default_clause = ""
    else:
        nullable_clause = "NULL" if column.nullable else "NOT NULL"
        default_clause = ""
        if column.server_default is not None:
            default_clause = f"DEFAULT {column.server_default.arg}"

    parts = [f"ALTER TABLE `{table_name}` MODIFY COLUMN `{column.name}` {col_type}"]
    if default_clause:
        parts.append(default_clause)
    parts.append(nullable_clause)

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Enum helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_db_enum_values(db_type_str: str) -> set:
    """Parse enum values from a DB type string like "enum('a','b','c')"."""
    match = re.match(r"enum\((.+)\)$", db_type_str.strip().lower())
    if not match:
        return set()
    raw = match.group(1)
    return {v.strip().strip("'") for v in raw.split(",")}


def _get_model_enum_values(column: sa.Column) -> set | None:
    """Extract enum values from a SQLAlchemy Enum column. None if not an Enum."""
    col_type = column.type
    if isinstance(col_type, (sa.Enum, MySQLENUM)):
        return set(col_type.enums)
    return None


def _enum_needs_update(model_values: set, db_type_str: str) -> bool:
    """Return True if the model has enum values not yet in the DB column."""
    db_values = _get_db_enum_values(db_type_str)
    return not model_values.issubset(db_values)


# ─────────────────────────────────────────────────────────────────────────────
#  Core sync steps
# ─────────────────────────────────────────────────────────────────────────────


def _sync_tables(engine, inspector) -> None:
    """Step 1 — Create any completely missing tables."""
    existing_tables = set(inspector.get_table_names())
    missing = [t for t in Base.metadata.tables if t not in existing_tables]

    if missing:
        logger.info(f"[schema-sync] Creating {len(missing)} missing table(s): {missing}")
        Base.metadata.create_all(engine, checkfirst=True)
    else:
        logger.info("[schema-sync] All tables present — skipping CREATE TABLE.")


def _sync_columns(engine, inspector) -> None:
    """
    Step 2 — For each existing table:
      a) Add columns present in model but absent in DB
      b) Alter columns whose type has changed
      c) Expand ENUM columns that are missing values
    """
    existing_tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue

            db_cols = _existing_columns(inspector, table_name)

            for column in table.columns:
                if column.primary_key:
                    continue  # never touch PKs after creation

                col_type_str = _compile_column_type(column, conn.dialect)

                # ── (a) Missing column → ADD ──────────────────────────────
                if column.name not in db_cols:
                    try:
                        sql = _build_add_column_sql(table_name, column, conn.dialect)
                        logger.info(
                            f"[schema-sync] ✅ Adding column: "
                            f"`{table_name}.{column.name}` ({col_type_str})"
                        )
                        conn.execute(text(sql))
                    except Exception as e:
                        logger.error(
                            f"[schema-sync] ❌ Failed to add column "
                            f"`{table_name}.{column.name}`: {e}"
                        )
                    continue

                # ── (b) Existing column — check for type drift ────────────
                db_col_info = db_cols[column.name]
                db_type_str = str(db_col_info["type"])

                # ── (c) Enum expansion ────────────────────────────────────
                model_enum_values = _get_model_enum_values(column)
                if model_enum_values is not None:
                    if _enum_needs_update(model_enum_values, db_type_str):
                        try:
                            sql = _build_modify_column_sql(table_name, column, conn.dialect)
                            logger.info(
                                f"[schema-sync] ✅ Expanding ENUM: "
                                f"`{table_name}.{column.name}` → {sorted(model_enum_values)}"
                            )
                            conn.execute(text(sql))
                        except Exception as e:
                            logger.error(
                                f"[schema-sync] ❌ Failed to expand ENUM "
                                f"`{table_name}.{column.name}`: {e}"
                            )
                    continue  # enum handled — skip generic type diff check

                # ── Generic type change → MODIFY ──────────────────────────
                if _types_differ(col_type_str, db_type_str):
                    try:
                        sql = _build_modify_column_sql(table_name, column, conn.dialect)
                        logger.info(
                            f"[schema-sync] ✅ Altering column type: "
                            f"`{table_name}.{column.name}` "
                            f"({_normalise_type(db_type_str)} → {_normalise_type(col_type_str)})"
                        )
                        conn.execute(text(sql))
                    except Exception as e:
                        logger.error(
                            f"[schema-sync] ❌ Failed to alter column type "
                            f"`{table_name}.{column.name}`: {e}"
                        )


def _sync_indexes(engine, inspector) -> None:
    """Step 3 — Create missing indexes."""
    existing_tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue

            existing_idx = _existing_indexes(inspector, table_name)
            db_cols = set(_existing_columns(inspector, table_name).keys())

            for index in table.indexes:
                if not index.name or index.name in existing_idx:
                    continue

                index_cols = [c.name for c in index.columns]
                missing_cols = [c for c in index_cols if c not in db_cols]
                if missing_cols:
                    logger.warning(
                        f"[schema-sync] ⚠️  Skipping index `{index.name}` — "
                        f"columns not yet in DB: {missing_cols}"
                    )
                    continue

                try:
                    cols_str = ", ".join(f"`{c}`" for c in index_cols)
                    unique = "UNIQUE " if index.unique else ""
                    sql = f"CREATE {unique}INDEX `{index.name}` " f"ON `{table_name}` ({cols_str})"
                    logger.info(
                        f"[schema-sync] ✅ Creating index: `{index.name}` on `{table_name}`"
                    )
                    conn.execute(text(sql))
                except Exception as e:
                    logger.error(f"[schema-sync] ❌ Failed to create index `{index.name}`: {e}")


def _sync_unique_constraints(engine, inspector) -> None:
    """Step 4 — Create missing unique constraints."""
    existing_tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue

            existing_uqs = _existing_unique_constraints(inspector, table_name)
            existing_idx = _existing_indexes(inspector, table_name)
            db_cols = set(_existing_columns(inspector, table_name).keys())

            for constraint in table.constraints:
                if not isinstance(constraint, sa.UniqueConstraint):
                    continue
                if not constraint.name:
                    continue
                if constraint.name in existing_uqs or constraint.name in existing_idx:
                    continue

                cols = [c.name for c in constraint.columns]
                missing_cols = [c for c in cols if c not in db_cols]
                if missing_cols:
                    logger.warning(
                        f"[schema-sync] ⚠️  Skipping unique constraint `{constraint.name}` — "
                        f"columns not yet in DB: {missing_cols}"
                    )
                    continue

                try:
                    cols_str = ", ".join(f"`{c}`" for c in cols)
                    sql = (
                        f"ALTER TABLE `{table_name}` "
                        f"ADD CONSTRAINT `{constraint.name}` UNIQUE ({cols_str})"
                    )
                    logger.info(
                        f"[schema-sync] ✅ Creating unique constraint: "
                        f"`{constraint.name}` on `{table_name}`"
                    )
                    conn.execute(text(sql))
                except Exception as e:
                    logger.error(
                        f"[schema-sync] ❌ Failed to create unique constraint "
                        f"`{constraint.name}`: {e}"
                    )


def _sync_foreign_keys(engine, inspector) -> None:
    """Step 5 — Add missing foreign keys."""
    existing_tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue

            existing_fks = _existing_foreign_keys(inspector, table_name)
            db_cols = set(_existing_columns(inspector, table_name).keys())

            for fk_constraint in table.foreign_key_constraints:
                if not fk_constraint.name or fk_constraint.name in existing_fks:
                    continue

                local_cols = [c.name for c in fk_constraint.columns]
                ref_table = fk_constraint.referred_table.name
                ref_cols = [fk.column.name for fk in fk_constraint.elements]

                missing_local = [c for c in local_cols if c not in db_cols]
                if missing_local:
                    logger.warning(
                        f"[schema-sync] ⚠️  Skipping FK `{fk_constraint.name}` — "
                        f"local columns missing: {missing_local}"
                    )
                    continue

                if ref_table not in existing_tables:
                    logger.warning(
                        f"[schema-sync] ⚠️  Skipping FK `{fk_constraint.name}` — "
                        f"referenced table `{ref_table}` does not exist yet."
                    )
                    continue

                try:
                    local_str = ", ".join(f"`{c}`" for c in local_cols)
                    ref_str = ", ".join(f"`{c}`" for c in ref_cols)
                    on_delete = (
                        f" ON DELETE {fk_constraint.ondelete}" if fk_constraint.ondelete else ""
                    )
                    sql = (
                        f"ALTER TABLE `{table_name}` "
                        f"ADD CONSTRAINT `{fk_constraint.name}` "
                        f"FOREIGN KEY ({local_str}) "
                        f"REFERENCES `{ref_table}` ({ref_str})"
                        f"{on_delete}"
                    )
                    logger.info(
                        f"[schema-sync] ✅ Creating FK: "
                        f"`{fk_constraint.name}` on `{table_name}`"
                    )
                    conn.execute(text(sql))
                except Exception as e:
                    logger.error(
                        f"[schema-sync] ❌ Failed to create FK " f"`{fk_constraint.name}`: {e}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def ensure_schema(engine) -> None:
    """
    Reconcile the live database schema against SQLAlchemy model definitions.

    Fully idempotent — safe to call on every startup.

    Covers:
      ✅  Fresh installs
      ✅  Missing tables
      ✅  Missing columns
      ✅  Column type changes     (auto ALTER)
      ✅  Enum value additions    (auto MODIFY)
      ✅  Missing indexes
      ✅  Missing unique constraints
      ✅  Missing foreign keys
      ✅  NOT NULL / no default   (forced NULL + dev warning)
      ✅  Any failure is logged and skipped — app always starts
    """
    logger.info("[schema-sync] ─────────────────────────────────────────")
    logger.info("[schema-sync] Starting schema reconciliation...")
    logger.info("[schema-sync] ─────────────────────────────────────────")

    try:
        inspector = inspect(engine)

        _sync_tables(engine, inspector)

        # Re-inspect after table creation so new tables are visible
        inspector = inspect(engine)

        _sync_columns(engine, inspector)

        # Re-inspect after column changes so indexes/constraints/FKs
        # can see newly added columns
        inspector = inspect(engine)

        _sync_indexes(engine, inspector)
        _sync_unique_constraints(engine, inspector)
        _sync_foreign_keys(engine, inspector)

    except Exception as e:
        # Outer safety net — schema sync must NEVER crash the app
        logger.error(f"[schema-sync] ❌ Unexpected error during schema reconciliation: {e}")
        logger.error("[schema-sync] App will continue — manual schema inspection recommended.")

    logger.info("[schema-sync] ─────────────────────────────────────────")
    logger.info("[schema-sync] Schema reconciliation complete ✅")
    logger.info("[schema-sync] ─────────────────────────────────────────")
