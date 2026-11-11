# Assistant summaries and the portal Agents page

## Contract

`GET /v1/assistants/summaries` authenticates with the caller's Core API key. It selects
active assistants whose `owner_id` is the caller, plus ownerless assistants associated
with the caller. An association to a canonically owned assistant does not grant access.
The legacy `GET /v1/assistants` retains its array of full `AssistantRead` records and
uses the same ownerless fallback. `GET /v1/assistants/{assistant_id}` and the SDK detail
method retain their existing contracts. The static summary route is registered first.

| Query parameter | Meaning |
| --- | --- |
| `limit` | 1–100, default 50 |
| `cursor` | Opaque URL-safe cursor from the previous page, at most 512 characters |
| `assistant_id` | Exact ID, up to 64 characters |
| `name_prefix` | Prefix of the name, up to 128 characters |
| `model` | Full model identifier, up to 64 characters |
| `mode` | `standard` or `autonomous`, based on `agent_mode` |

Filters combine with AND. Name matching follows the database's name collation. The
prefix starts at the first character, with no substring or fuzzy matching. `%`, `_`
and `/` in the supplied prefix are literal characters; only the implicit trailing
wildcard is added. ID and model use database equality (no wildcard matching).
Reuse the cursor with the same filters, and omit it to restart the collection.

The response is `{items, next_cursor, has_more}`. Order is `created_at DESC, id DESC`,
with a strict keyset boundary. One query returns at most `limit + 1` rows. Its two
disjoint ownership branches each contribute at most that many candidates to the
database's final merge. No total count or ORM relationship fetch occurs. The SQL
projection excludes instructions, tool schemas, tool resources, and raw metadata.
Paging is not a historical snapshot; refresh to include newly created records.

The portal exposes `GET /api/agents` and `GET /api/agents/{assistant_id}`. Both use the
signed-in user's encrypted managed key. List uses the fixed additive Core endpoint;
detail uses the SDK's authorized retrieval. There is no old-list fallback. Portal
detail shows instructions, execution settings, and tool type/name, excluding raw tool
schemas and metadata. Configuration is separate from run activity; no run history is
inferred. The web UI has a bookmarkable `/agents/{assistant_id}` route.

## Activate on the existing Windows Docker Desktop runtime

The narrow overlay is for the installed Core 1.61 image at Alembic revision
`869efe5119ee`; it layers the assistant modules and one additive migration over the
existing image. It does not rebuild the other Core components. From PowerShell:

```powershell
$projects = 'C:\Users\franc\PycharmProjects'
$core = Join-Path $projects 'projectdavid-core'
$runtime = Join-Path $projects 'reference-projectdavid-platform'
$frontend = Join-Path $projects 'projectdavid-portal-frontend'

docker --context desktop-linux build -t projectdavid-core-api:1.61-agent-summaries-local -f "$core\docker\api\Dockerfile.agent-summaries" "$core"
docker --context desktop-linux compose -f "$runtime\docker-compose.yml" -f "$core\docker\api\compose.agent-summaries.yml" up -d --no-deps --pull never api
docker --context desktop-linux exec fastapi_cosmic_catalyst alembic current
```

Expect `eaf2f16ac073 (head)` after startup. Core's entrypoint applies migrations. On
large databases, schedule the index build after reviewing DDL duration and locking.
If the proxy still holds the previous API container IP, reload its existing config
with `docker --context desktop-linux exec nginx_proxy nginx -s reload`; this was not
needed in this activation. Keep the override in future Core Compose invocations to
retain the local API image. Do not run `down -v` or modify database credentials.

If the portal dev containers already run, Uvicorn and Vite reload the mounted source;
refresh [the Agents page](http://localhost:5173/agents). If necessary, restart only
the portal backend. If the dev stack is stopped, start it normally:

```powershell
Set-Location $frontend
docker --context desktop-linux compose -f docker-compose.dev.yml up -d --build
```

For an already-running production preview, rebuild only the two portal services:

```powershell
Set-Location $frontend
docker --context desktop-linux compose -f docker-compose.prod.yml up -d --build backend frontend
```

Keep the existing secrets, SQLite volume and port settings. No new portal migration
or dependency is required for Agents. Do not run dev and production concurrently
under the same Compose project. Production Agents URL: http://127.0.0.1:8081/agents.

## Checks run on Windows Docker Desktop, 10 November 2026

- **Core:** 23 tests passed, including the new summary tests and the existing
  assistant-cache tests. Tests use isolated SQLite fixtures; coverage includes
  canonical and fallback authorization, deletion, equal timestamps, malformed cursors,
  literal prefix metacharacters, combined filters, static routing and projected SQL.
- **Portal backend:** 21 tests passed, including managed-key isolation for two users,
  sanitized Core status handling, filter forwarding and list/detail projection.
- **Frontend:** typecheck, lint, five tests and production build passed. The new tests
  cover aborting and ignoring obsolete responses/errors. Interactive Agents
  back/forward and clipboard behavior were not manually exercised in this pass.
- **Live:** the new Core route responded through its existing proxy. A test assistant
  `portal-agents-it-c9e9c2ba` created externally with the SDK was visible in the portal
  list and detail before and after rotating the exported credential. Core rejected
  the revoked credential; the replacement credential could still retrieve the
  same ID. The existing managed key, account and session were preserved.
- **Live MySQL EXPLAIN:** the default owned branch chose
  `ix_asst_owner_active_created` (`range`), the exact model branch chose
  `ix_asst_owner_active_model` (`ref`), and the legacy branch used `user_assistants`
  primary key `(user_id, assistant_id)` followed by an `eq_ref` assistant lookup.
  The mode and prefix plans considered their new indexes and chose the ordered owner
  index on this small dataset. The final filesort sorts bounded branch candidates;
  the legacy branch may sort the caller's associations. Selectivity, rows examined,
  and index choice on larger data still need measurement. This is not a scale benchmark.

To repeat the live rotation check against the running **dev** portal, first seed a
test session with `python -m scripts.integration` inside the portal backend (creates
identifiable test users). Then run `python -m scripts.integration_agents` there. The
test only revokes the exported keys it created. The Core explain utility is read-only:

```powershell
docker --context desktop-linux cp "$core\scripts\explain_assistant_summaries.py" fastapi_cosmic_catalyst:/tmp/explain_assistant_summaries.py
docker --context desktop-linux exec -e PYTHONPATH=/app:/app/src/api fastapi_cosmic_catalyst python /tmp/explain_assistant_summaries.py --user-id <test-core-user-id>
```
