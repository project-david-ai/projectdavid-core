# Alembic Migration Runbook

This repository uses Alembic in an architecture where SQLAlchemy models live in the separately released `projectdavid-orm` package while migrations live in `projectdavid-core`.

That separation makes migration work easy to get wrong.

> **Alembic must be reconciled against the released ORM package that Core will actually consume — not merely against a local ORM worktree.**

## 1. Release order

For any schema change that originates in `projectdavid-orm`, use this order:

```text
projectdavid-orm
        ↓
Core Alembic migration / reconciliation
        ↓
projectdavid-common
        ↓
projectdavid-core
```

Do not reorder these steps.

- Do not release Common before the required ORM version exists.
- Do not release Core before the required Common and ORM versions exist.
- Do not reconcile Alembic against an unreleased ORM implementation and assume production will behave the same way.
- If Core imports ORM directly, Core must declare the ORM dependency directly.

## 2. Why this order exists

Core owns the Alembic migration history, but Alembic loads its metadata from the installed ORM package.

`migrations/env.py` imports ORM metadata approximately as follows:

```python
from projectdavid_orm import Base
from projectdavid_orm.projectdavid_orm import models
```

Therefore the schema Alembic sees is determined by the **installed `projectdavid-orm` package** in the environment running Alembic.

A local ORM worktree can be correct while the released package is stale. That creates the failure mode this runbook is designed to prevent:

```text
local code says schema A
released package says schema B
Alembic migration was generated for A
production installs B
```

## 3. Preconditions

Before touching Alembic:

1. Finish the ORM model change.
2. Run the ORM tests.
3. Merge the ORM change to `main`.
4. Release the ORM package.
5. Confirm the exact released version is installable from PyPI.
6. Confirm the published package actually contains the expected models, tables, columns, constraints, and relationships.

Do not proceed merely because CI says the ORM repository built successfully. The released artifact is authoritative.

## 4. Verify the published ORM

Install the released ORM into an isolated temporary location.

```powershell
$target = Join-Path $env:TEMP "projectdavid-orm-release-check"

Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue

python -m pip install `
    --no-deps `
    --target $target `
    "projectdavid-orm==<VERSION>"
```

Then import from that isolated location and inspect the expected model objects.

For a new table, verify at minimum:

- model import succeeds;
- `__tablename__` is correct;
- expected columns exist;
- expected foreign keys exist;
- expected unique constraints exist;
- expected relationships exist where relevant.

Example:

```python
from projectdavid_orm.projectdavid_orm.models import McpServerRegistration

assert McpServerRegistration.__tablename__ == "mcp_server_registrations"
```

If the released wheel does not expose the new schema, stop. Fix and release the ORM package before continuing.

## 5. Create the Core migration

Create the migration in `projectdavid-core`.

The migration must represent only the intended schema change.

Prefer explicit, conservative DDL:

- create tables explicitly;
- create indexes explicitly;
- create foreign keys explicitly;
- create unique constraints explicitly;
- make downgrade order the exact reverse of upgrade order;
- use safe existence helpers where repository migration conventions require them;
- avoid unrelated schema cleanup.

A migration is not a place to repair every historical difference Alembic happens to notice.

One migration should have one schema purpose.

## 6. Verify the migration chain

Before applying anything, inspect the migration graph.

Confirm:

```text
new revision
    ↓
correct down_revision
    ↓
existing Core Alembic head
```

Then verify Alembic sees the intended migration as the current head:

```powershell
alembic heads
alembic history
```

There should be no accidental second head unless a branch migration is explicitly intended.

## 7. Use the real database engine

Do not validate a MySQL production migration against SQLite.

Start the repository's real MySQL service only:

```powershell
docker compose up -d --no-build db
```

The rule is:

> **Start the database without triggering unrelated application image builds.**

The local compose database may expose a different host endpoint than the container-internal `DATABASE_URL`.

Example:

```text
container: db:3306
host:      127.0.0.1:3307
```

When running Alembic from Windows, override only the connection endpoint required by the host process.

Do not rewrite the repository `.env` merely to perform the migration test. Use a process-local or in-memory override.

## 8. Run Alembic against released dependencies

The Alembic process must resolve the released package versions that production Core is expected to consume.

At minimum this normally means:

```text
projectdavid-orm == released migration target
projectdavid-common == currently compatible released version
```

Do not allow a local editable ORM checkout or source `PYTHONPATH` entry to shadow the published package during this gate.

The question being tested is:

> **Can released Core dependencies describe and migrate the database correctly?**

Not:

> Does my development checkout happen to work?

## 9. Inspect the current database revision

Before upgrading:

```powershell
alembic current
```

Record the revision.

Do not assume the local database is already at the immediate predecessor of the new migration. A real development database may be several revisions behind; that is useful because it tests the actual migration chain.

## 10. Upgrade to head

Run:

```powershell
alembic upgrade head
```

Observe every migration applied.

A valid run may look conceptually like:

```text
old_revision
    →
intermediate_revision
    →
migration_predecessor
    →
new_revision
```

Then verify:

```powershell
alembic current
```

The database must report the new migration as head.

If `upgrade head` fails, stop and diagnose the failing migration. Do not continue into package releases.

## 11. Run `alembic check`

After the database reaches head:

```powershell
alembic check
```

This is the reconciliation gate.

### Clean result

Alembic reports no new upgrade operations. Migration reconciliation is complete.

### Additional deltas appear

Do **not** immediately generate another migration.

First classify every delta.

## 12. Classify Alembic drift before acting

Alembic may report historical or dialect-level differences unrelated to the schema change being released.

Examples:

- MySQL `LONGTEXT` versus SQLAlchemy `Text(length=4294967295)`;
- comments that exist in migration history but not ORM metadata;
- type renderings that are semantically equivalent;
- pre-existing drift that predates the current change.

For each unexpected delta, compare:

```text
previous released ORM metadata
current released ORM metadata
actual database schema
```

Ask one question:

> **Did the ORM release being migrated actually introduce this difference?**

If the metadata definition is identical in both the previous and current released ORM versions, the delta is pre-existing drift. It does not belong in the current migration.

## 13. Prove pre-existing drift

When `alembic check` reports an unrelated difference:

1. install the previous released ORM package in isolation;
2. inspect the relevant model or column metadata;
3. install the new released ORM package in isolation;
4. inspect the same metadata;
5. compare them directly.

Example:

```text
ORM 1.9.1:
messages.content = Text(length=4294967295)

ORM 1.10.0:
messages.content = Text(length=4294967295)

Alembic check:
database LONGTEXT → metadata Text(length=4294967295)
```

Conclusion:

```text
The metadata did not change in the release.
Therefore this is pre-existing dialect/schema drift.
It is not part of the new migration.
```

Record the result and leave it out of the release.

This prevents migration pollution.

## 14. Do not create opportunistic cleanup migrations

A migration release is not permission to absorb unrelated Alembic noise.

Do not create a cleanup migration simply because `alembic check` reports something.

Create an additional migration only when one of these is true:

- the current ORM release intentionally changed that schema object;
- the existing database cannot reach the ORM's required schema without it;
- the difference is known to be materially incorrect;
- the cleanup has been explicitly scoped and reviewed as its own migration.

Otherwise, classify it as known drift and move on.

## 15. Migration gate definition

The Core migration gate is closed only when all of the following are true:

```text
[PASS] Target ORM version is released.
[PASS] Published ORM artifact contains the intended schema.
[PASS] Migration revision chain is correct.
[PASS] Real MySQL database starts without unrelated builds.
[PASS] Alembic uses the released ORM package.
[PASS] Database upgrades successfully to head.
[PASS] Database reports the expected head revision.
[PASS] `alembic check` has been run.
[PASS] Any remaining deltas are classified.
[PASS] Unrelated pre-existing drift is excluded from the migration.
```

Only then continue the release train.

## 16. Release Common

If `projectdavid-common` depends on the new ORM schema or contracts:

1. bump its ORM minimum;
2. test against the released ORM version;
3. release Common;
4. verify the new Common package is installable from PyPI.

Example:

```text
projectdavid-orm>=1.10.0
```

Do not merely update source code and assume Core will resolve it correctly later. Verify the artifact.

## 17. Update Core dependencies

Core must declare every package it imports directly.

If Core imports `projectdavid_orm`, then Core must have its own ORM dependency even if Common also depends on ORM.

Example:

```toml
dependencies = [
    "projectdavid-common>=0.74.0",
    "projectdavid-orm>=1.10.0",
]
```

For the API runtime requirement surface, update the corresponding package requirements as well:

```text
projectdavid-orm>=1.10.0
projectdavid_common>=0.74.0
```

Do not rely on transitive dependency luck.

## 18. Validate Core against released packages

Before releasing Core:

1. install the exact released Common and ORM versions in isolation;
2. prove Core can import them;
3. prove the new ORM models are visible;
4. run the maintained Core unit suite in a dependency-complete environment;
5. run pre-commit;
6. verify only intended release files changed.

The release candidate must consume the same package chain expected in CI and production.

## 19. Final release sequence

The complete sequence is:

```text
1. Implement ORM schema.
2. Test ORM.
3. Release ORM.
4. Verify published ORM artifact.
5. Create/reconcile Core Alembic migration.
6. Start real MySQL only.
7. Run Alembic with released ORM.
8. Upgrade database to head.
9. Run alembic check.
10. Classify all remaining drift.
11. Reject unrelated migration noise.
12. Release Common with new ORM minimum.
13. Verify published Common artifact.
14. Update Core direct Common + ORM minimums.
15. Test Core against released packages.
16. Release Core.
```

That sequence is authoritative.

# Failure rules

## Alembic cannot import a new model

The installed ORM package is stale or being shadowed.

Check:

```text
import path
installed package version
PYTHONPATH
editable installs
temporary source paths
```

Do not modify the migration until the dependency source is known.

## `alembic upgrade head` cannot connect to `db:3306`

Alembic is being executed from the host while the URL contains the Docker-network hostname.

Use the host-mapped endpoint for that process, for example:

```text
127.0.0.1:3307
```

Do not permanently mutate application configuration to solve a migration test.

## MySQL takes too long to become healthy

Inspect logs before declaring failure.

A database image upgrade may legitimately perform internal data-dictionary work before becoming ready.

Do not repeatedly recreate or destroy the database unless there is evidence the instance is corrupt.

## `alembic check` reports unrelated operations

Do not generate a migration automatically.

Compare previous and current released ORM metadata first.

If both releases define the object identically, classify the difference as pre-existing drift.

## Local source passes but released package fails

The released artifact wins.

Fix and re-release the package.

Never make Core migration assumptions from unreleased source code.

# Architectural principle

The database migration is part of the **released dependency graph**, not merely part of the source tree.

In this architecture:

```text
ORM metadata
    +
released package version
    +
Core migration history
    +
real database state
```

must all agree.

Alembic is the reconciliation mechanism between those four facts.

Treating any one of them in isolation is what makes migration work brittle. Treating them sequentially makes it deterministic.
