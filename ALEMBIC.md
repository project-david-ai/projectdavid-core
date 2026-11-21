# Alembic Migration Runbook

- **Owner:** `projectdavid-core`
- **Model source:** the **released** `projectdavid-orm` package
- **Database:** MySQL
- **Host-side operator:** `python -m scripts.db_migrate`

This runbook replaces repeated, hand-written PowerShell/Python connection scripts with a small, **local-only** migration runner. It does not eliminate migration review or authorize unattended DDL.

## 1. The architecture and release order

```text
projectdavid-orm model change
  -> ORM tests -> merge -> publish released wheel
  -> Core Alembic migration, tested against that released wheel
  -> projectdavid-common release, only if its ORM contract must change
  -> Core direct dependencies / application changes -> Core CI and release
```

Core owns Alembic revision history (`migrations/versions/`). ORM owns SQLAlchemy definitions. Core's `migrations/env.py` imports installed ORM metadata from `projectdavid_orm.projectdavid_orm.models`. Therefore **the ORM package installed in the Alembic process is authoritative**, not an ORM source checkout beside Core.

**Do not deploy a Core runtime using a new ORM column against a database where that column is absent.** Coordinate each environment's physical schema upgrade before switching its runtime to models that require the new schema. Publishing packages, pushing images, and upgrading databases are distinct operations.

If Core directly imports `projectdavid_orm`, Core must declare an explicit dependency on it. Common must be updated and released **only if it actually needs** the new ORM version or contract. Verify any updated Common and Core dependency constraints against published packages; don't rely on editable local worktrees or transitive installations.

## 2. What the runner does and does not do

The runner is `scripts/db_migrate.py`, invoked from the **Core repository root**:

```powershell
python -m scripts.db_migrate status --env local
python -m scripts.db_migrate plan --env local
python -m scripts.db_migrate apply --env local --orm-version 1.15.0
python -m scripts.db_migrate verify --env local --orm-version 1.15.0 --expect-column api_keys.is_protected
```

Replace `1.15.0` with the exact published ORM version required **for the migration being performed**. Replace the sample expected column with the column relevant to that migration, or omit `--expect-column` for migrations not involving a new column.

| Command | Behavior | Writes to database? |
| --- | --- | --- |
| `status` | Verify target, installed ORM provenance, live revision and code head | **No** |
| `plan` | List pending migrations, in execution order, and their source paths | **No** |
| `apply` | Repeat preflight, require exact released ORM version and an **interactive typed approval**, execute `alembic upgrade head`, verify head | **Yes, only after approval** |
| `verify` | Confirm head, optional physical column/nullability, run `alembic check` | **No schema writes** |

The runner **does not** install or update ORM, invent migration files, autogenerate unrelated schema changes, edit credentials, stamp revisions, downgrade, start Docker, publish packages, push images, or run production DDL. Its `plan` shows revision files to review; it is **not** a SQL dry run.

The supported target is currently `--env local` **only**. Production and staging migration execution require a separate, explicitly reviewed procedure. Never repurpose `.env.migrations` as a production deployment file.

## 3. One-time prerequisites on Windows

1. Open PowerShell in the root of your `projectdavid-core` checkout.
2. Activate the environment that contains Alembic, SQLAlchemy, PyMySQL and the **released** ORM wheel.
3. Use an existing MySQL container; **do not rebuild the stack** for migration work.
4. Ensure `.env.migrations` exists **locally** with `SPECIAL_DB_URL` pointing to a host-mapped loopback endpoint.
5. Keep `.env.migrations` out of source control, logs and container environments.
6. Verify a usable database backup exists before any real migration.

Example setup:

```powershell
conda activate entities_api
git rev-parse --show-toplevel
python -m pip show projectdavid-orm
docker ps --format "table {{.Names}}\t{{.Status}}"
docker port my_mysql_cosmic_catalyst
```

A host URL may use `localhost:3307`, while the container uses `db:3306`. **These are not interchangeable.** The runner reads `SPECIAL_DB_URL` from `.env.migrations`, checks for `mysql+pymysql`, loopback hostname, port **3307 or 3308**, and a nonempty database/user/password. It never falls back to the container's `DATABASE_URL` when called with `--env local`.

Core's existing Alembic environment reads `DATABASE_URL`, not `SPECIAL_DB_URL`. The runner passes the selected URL **only to its Alembic subprocess environment**; it doesn't edit `.env` or `migrations/env.py`.

### Released ORM package verification

Before any migration, ensure the target ORM wheel was published and imported from the active environment's `site-packages` (not a local editable checkout):

```powershell
python -m pip install --upgrade "projectdavid-orm==1.15.0"
python -c "from importlib.metadata import version; from projectdavid_orm import ApiKey; print(version('projectdavid-orm')); print('is_protected' in ApiKey.__table__.columns)"
```

For future releases, update the version and model assertion. `--orm-version` verifies the installed version **exactly** during `apply`. The runner refuses obvious editable and source-tree shadowing; confirm the target version exists as an actual published PyPI artifact as a separate release gate. It never upgrades packages for you.

`pip check` is useful for detecting broader environment conflicts; treat unrelated conflicts as their own maintenance task, not an excuse to alter an unrelated migration.

## 4. Creating a migration

1. Finish and publish the intended ORM change first.
2. Create **one narrowly scoped Alembic revision** under `migrations/versions/`.
3. Set `down_revision` to the actual existing head; verify `python -m alembic heads` reports exactly one head.
4. Prefer explicit, reviewed DDL. Do not accept a blanket autogenerate diff as a migration.
5. Include a downgrade, with a guard against destructive rollback if the new schema may contain live data.
6. Format, lint, stage and review only the migration file.

Example for the already-used protected-key migration:

```text
b14a6c9d2e70  (previous head)
      |
      v
c48bd2a910ef  (add api_keys.is_protected)
```

The intended physical column is `is_protected BOOLEAN NOT NULL DEFAULT 0`; existing keys begin unprotected. No unrelated tables or indexes belong in that revision.

```powershell
python -m alembic heads
ruff check migrations/versions/c48bd2a910ef_add_api_key_is_protected.py
pre-commit run --files migrations/versions/c48bd2a910ef_add_api_key_is_protected.py
git diff --cached --check
git diff --cached -- migrations/versions/c48bd2a910ef_add_api_key_is_protected.py
```

**Do not stage unrelated local files**, including untracked SQL work products.

## 5. Standard local migration procedure

Run each command from Core root after the ORM wheel and migration revision are in place.

### Inspect the target and revision

```powershell
python -m scripts.db_migrate status --env local --orm-version 1.15.0
```

Check the printed **database name, loopback host/port, ORM version and import location, database revision, Alembic head and number of pending migrations**. No password is printed.

If the target is unexpected, stop. The runner will not silently fall back to `db:3306` or connect to an arbitrary remote database.

### Review pending work

```powershell
python -m scripts.db_migrate plan --env local --orm-version 1.15.0
```

Read the migration source files listed by the plan, in order. Confirm the operations are intentional and the database has a valid backup. `plan` deliberately does not use `alembic revision --autogenerate` or emit unreviewed SQL.

### Apply — explicit human approval

```powershell
python -m scripts.db_migrate apply --env local --orm-version 1.15.0
```

For a non-empty plan, the runner prints a target-specific approval phrase resembling:

```text
APPLY entities_db@localhost:3307 c48bd2a910ef
```

Type the **exact phrase actually shown**. A mismatch, `Ctrl+C`, or non-interactive execution refuses the upgrade. If already at head, `apply` is a no-op. On successful upgrade it reads the database revision again and requires it to match the target head.

**MySQL DDL is often non-transactional.** Failure may leave a partly changed schema. Do not blindly retry or run `alembic stamp`: inspect the physical state and reconcile manually.

### Verify — independent database inspection

```powershell
python -m scripts.db_migrate verify --env local --orm-version 1.15.0 --expect-column api_keys.is_protected
```

`verify` requires the live database at head. Each `--expect-column table.column` additionally checks that the physical column exists and its nullability matches the **installed released ORM metadata**. Repeat `--expect-column` for more columns.

`verify` also invokes `alembic check`. If it reports differences, the tool stops and asks for **drift classification**; it does not change anything. A migration's physical success and an unrelated historical Alembic drift warning are different findings, and must be recorded separately.

## 6. Drift classification — never migrate unrelated noise

If `alembic check` reports changes after the intended schema is present:

1. Identify each reported table, column, type, comment, constraint or index.
2. Compare **previous released ORM metadata**, **target released ORM metadata**, and **actual MySQL schema**.
3. If metadata did not change between releases, classify the difference as **pre-existing drift**, not part of this migration.
4. Record the classification; do not automatically create cleanup migrations.
5. Create a separate migration only when the change is intentional, required for compatibility, materially incorrect, or explicitly approved as a separate scope.

Typical noise includes MySQL `LONGTEXT` versus SQLAlchemy length-qualified `Text`, dialect type rendering and historical comments. **A failing `alembic check` is not authorization for a blanket autogenerate.**



### Known local schema drift — ORM 1.15.0

After applying revision `c48bd2a910ef`, the local MySQL
database reached the correct Alembic head and the physical
`api_keys.is_protected` column was verified.

However, `alembic check` reported:

- Four assistant indexes:
  `ix_asst_owner_active_created`,
  `ix_asst_owner_active_mode`,
  `ix_asst_owner_active_model`,
  `ix_asst_owner_active_name`
- `messages.content`: LONGTEXT versus Text(length=4294967295)
- `messages.reasoning`: LONGTEXT versus Text(length=4294967295)
- `training_jobs.cancelled_at`: column comment difference

The complete Assistant, Message and TrainingJob model
definitions were compared between released ORM versions
1.14.0 and 1.15.0 and found to be unchanged.

Classification: pre-existing drift, not introduced by the
protected API-key migration.

These differences are NOT automatically considered harmless.
They remain outside this migration's scope and require separate
review if reconciliation becomes necessary.

Do not generate migrations to remove these differences as part
of the protected API-key work package.




## 7. Failures and recovery

| Symptom | Diagnosis / next action |
| --- | --- |
| `projectdavid-orm` version mismatch | Install the intended **released** version into the correct Python environment; don't silently upgrade |
| ORM path outside `site-packages` or editable install | Local source is shadowing the released wheel; fix environment before migrating |
| `getaddrinfo` for host `db` | Host is using a container-only hostname; use host-side `.env.migrations` with `localhost:3307` |
| MySQL `1045 Access denied` | Connection works but username/password/auth grant does not; compare safely with running container configuration and repair host-only config, without printing secrets or resetting credentials blindly |
| Unknown database revision or multiple heads | Reconcile history before any DDL; **never stamp automatically** |
| Some revisions pending | Review the whole chain; do not assume database is only one revision behind |
| `alembic upgrade` fails | Stop, inspect applied DDL and `alembic_version`; MySQL may have partially committed |
| `alembic check` detects extra operations | Classify current release changes versus pre-existing drift; do not autogenerate fixes |
| Version at head but expected column missing | Physical schema is inconsistent; stop and reconcile manually |
| CI fails migration graph test | Repair duplicate/disconnected Alembic heads before promotion |

Never copy a database password or full URL into chat, PRs, shell transcripts or CI logs. A password mismatch does **not** imply that a MySQL account should be recreated or reset.

## 8. CI validation

`tests/unit/test_db_migrate.py` contains **offline-only** checks for safe URL selection, approval refusal, no-op behavior and revision graph integrity. Core's existing CI already runs `pytest tests/` on Python 3.11 and 3.12, so these tests need **no new Actions workflow or Docker service**.

```powershell
pytest -q tests/unit/test_db_migrate.py
pre-commit run --files scripts/db_migrate.py tests/unit/test_db_migrate.py ALEMBIC.md
```

CI **does not** load `.env.migrations`, use a real MySQL connection or execute `apply`. Interactive application remains an operator-controlled step.

## 9. Release and environment-promotion gates

A migration work package is complete for its **local test database** only when all are true:

```text
[ ] Target ORM artifact is published and version is verified.
[ ] Installed runtime resolves the published wheel, not editable source.
[ ] Alembic graph has one correct head and down_revision chain.
[ ] Host connection resolves to the intended local MySQL database.
[ ] Migration source is reviewed; database backup is verified.
[ ] Migration applied successfully against real MySQL.
[ ] Live alembic_version equals intended head.
[ ] Target physical schema matches ORM metadata.
[ ] alembic check has run; any historical drift is classified.
[ ] Only intended files are committed to Core dev and pass CI.
```

Then, and **only** when required by the dependency graph, release Common, update Core's direct dependencies and runtime requirement surfaces, test with released packages, and promote Core via the existing `dev -> CI -> Auto-PR -> manual merge` workflow.

**Each deployed database is a separate migration target.** A local migration does not modify staging or Azure production. Before activating new-model application code in any environment, schedule and verify that environment's schema migration. Production changes require their own backup, maintenance and approval procedure; this local-only runner does not perform them.

## 10. Guiding principle

```text
released ORM metadata
+ Core Alembic revision history
+ verified database identity and state
+ explicit operator approval
= controlled schema migration
```

Keep the operations deterministic, small and reviewable. Automate discovery and verification; retain human control over irreversible schema changes.
