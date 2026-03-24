## [1.26.3](https://github.com/project-david-ai/projectdavid-core/compare/v1.26.2...v1.26.3) (2026-03-24)


### Bug Fixes

* resolve MySQL 1824 error by implementing idempotent baseline ([e436231](https://github.com/project-david-ai/projectdavid-core/commit/e4362316686b0abcf99346592f11b76aac813484))

## [1.26.2](https://github.com/project-david-ai/projectdavid-core/compare/v1.26.1...v1.26.2) (2026-03-24)


### Bug Fixes

* resolve FK ordering failure in add_fine_tuning_tables ([7de8ed1](https://github.com/project-david-ai/projectdavid-core/commit/7de8ed13b7d4aacc4b3b05e6e14fa4a74e3581ab))

## [1.23.1] - 2026-03-24

### Fixed

- **Critical: First-run stack failure on clean installs**

  On a fresh installation with no existing database, the API container
  would crash-loop during database migration with:
```
  sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError)
  (1824, "Failed to open the referenced table 'users'")
```

  **Root cause:** Migration `005820173bc4` (add_fine_tuning_tables) was
  on a branch that did not pass through the baseline schema migration
  that creates the `users` table. MySQL InnoDB enforces foreign key
  constraints at CREATE TABLE time, so the `datasets`, `training_jobs`,
  and `fine_tuned_models` tables could not be created.

  **Fix:** Added `depends_on = "9351530d20ab"` to the migration,
  forcing Alembic to resolve the merge point (which includes the
  baseline schema) before executing fine-tuning table creation.

  **Affected:** Clean installs only. Existing deployments with a
  populated database volume are not affected — the migration guard
  (`has_table`) skips table creation if the tables already exist.

### Upgrade
```bash
  pip install --upgrade projectdavid-platform
  pdavid --mode up --pull
```

  If your stack failed to start on first run, the volumes will be
  empty. A clean restart is sufficient — no data recovery needed:
```bash
  pdavid --mode down_only --clear-volumes
  pdavid --mode up --pull
```


## [1.26.1](https://github.com/project-david-ai/projectdavid-core/compare/v1.26.0...v1.26.1) (2026-03-24)


### Bug Fixes

* resolve FK ordering failure in add_fine_tuning_tables ([7030c4f](https://github.com/project-david-ai/projectdavid-core/commit/7030c4fae6a8e6e1d4e6699cdd4949ca0488a5ef))

# [1.26.0](https://github.com/project-david-ai/projectdavid-core/compare/v1.25.0...v1.26.0) (2026-03-23)


### Features

* Phase 2 — replace MySQL VRAM ledger with Ray resource scheduling ([733c7eb](https://github.com/project-david-ai/projectdavid-core/commit/733c7ebd567622544b9931dd3694350b33372839))
* Phase 2 — replace MySQL VRAM ledger with Ray resource scheduling ([21acbd6](https://github.com/project-david-ai/projectdavid-core/commit/21acbd6143aad210a185ca91c22de67764d02cfd))
* Phase 3 — Ray task execution and DeploymentSupervisor actor ([1fe6e61](https://github.com/project-david-ai/projectdavid-core/commit/1fe6e61b019a258ab0c92b860263cc67fa1e2282))
* Phase 3 — Ray task execution and DeploymentSupervisor actor ([adf9719](https://github.com/project-david-ai/projectdavid-core/commit/adf97192523b4dbf70b753befe3655085512dfd4))
* Release ([3130181](https://github.com/project-david-ai/projectdavid-core/commit/3130181ecc701542c1977b2927e1437a9663be4f))
* ReleaseX ([40fed5a](https://github.com/project-david-ai/projectdavid-core/commit/40fed5a272fa995af8031601cd21d2869a700faf))
* wire tensor_parallel_size into activation endpoints ([86b0990](https://github.com/project-david-ai/projectdavid-core/commit/86b0990077b125a8cc627342b3de14dbb0ce84b9))
* wire tensor_parallel_size into activation endpoints ([d8ad4cc](https://github.com/project-david-ai/projectdavid-core/commit/d8ad4cc48278eb46040475fd528883caddac8a04))

# [1.25.0](https://github.com/project-david-ai/projectdavid-core/compare/v1.24.1...v1.25.0) (2026-03-22)


### Bug Fixes

* change samba hostname in docker-compose.yml from -->samba_server-->samba ([2b2d9ce](https://github.com/project-david-ai/projectdavid-core/commit/2b2d9ce4df2a8bf7d61ae0eb308ce90d21623662))
* conftest.py errors ([dc8dc35](https://github.com/project-david-ai/projectdavid-core/commit/dc8dc35b8fbe0917186d40c1564fa8b1a6b7378a))
* conftest.py errors ([44e8c9a](https://github.com/project-david-ai/projectdavid-core/commit/44e8c9ae49bf69930d8b52430fd4820e6bec6f90))
* RUF errors. ([a010658](https://github.com/project-david-ai/projectdavid-core/commit/a010658a0d7d2295b8364e8dad0477479be11f67))


### Features

* add `force` flag to `generate_dev_docker_compose` for overwrite control and update generated compose files and secrets ([baf943a](https://github.com/project-david-ai/projectdavid-core/commit/baf943a57273e6abcf93258577f7ea64d84e4d4c))
* add cluster management, node telemetry, and VRAM-aware scheduling ([be8294b](https://github.com/project-david-ai/projectdavid-core/commit/be8294b41b5353a64af54f2f161fc296b24f08df))
* add datasets router, JWT-based authentication, and multi-stage build for Training API ([d429a9f](https://github.com/project-david-ai/projectdavid-core/commit/d429a9fc574af720abceadb689abf421ca1934fb))
* add fine-tuned model registry API, service logic, and integration tests ([1a923b2](https://github.com/project-david-ai/projectdavid-core/commit/1a923b2f28c40c4178d63d7271ad3ba619d4a722))
* add fine-tuning pipeline with Dataset, TrainingJob, and FineTunedModel tables ([56a4632](https://github.com/project-david-ai/projectdavid-core/commit/56a46321c867cd6f69aaeb8061f857c8085a12e2))
* add fine-tuning support with new database tables and models for datasets, training jobs, and fine-tuned models ([7d6ccc0](https://github.com/project-david-ai/projectdavid-core/commit/7d6ccc0fe358d55b8869009c62d96afb14053845))
* add fine-tuning table schemas, update Alembic migration, and extend Docker Compose for training services ([7cd4021](https://github.com/project-david-ai/projectdavid-core/commit/7cd4021b18bba4d8cddf94162cd3b98d3ff99259))
* add self-hosted fine-tuning container with Axolotl and Unsloth support, including Dockerfile, training scripts, and entrypoint ([abafa7a](https://github.com/project-david-ai/projectdavid-core/commit/abafa7a146c6fa2668d16e847062a97d7b0144ab))
* add training jobs router, service logic, and integration tests ([dc96d3c](https://github.com/project-david-ai/projectdavid-core/commit/dc96d3cde48b8326ee1cad4ff9506d2f97f5525a))
* data drive vllm cluster ([72caf8d](https://github.com/project-david-ai/projectdavid-core/commit/72caf8d6aeb4105e41eeb1e724d508560e35df51))
* document fine-tuning pipeline architecture, endpoints, and implementation details ([53c99dc](https://github.com/project-david-ai/projectdavid-core/commit/53c99dcd3762498e96fd68caf8b7ad11cde19388))
* extend Docker Compose with new services, default handling, and improved generation logic ([6dfc5aa](https://github.com/project-david-ai/projectdavid-core/commit/6dfc5aac5e695a53f7201dd4db6fa3cd92c5638c))
* implement dataset service with CRUD operations, Samba integration, and dataset preparation logic ([2c75b2d](https://github.com/project-david-ai/projectdavid-core/commit/2c75b2dca9288071c116368caf11b08777529532))
* implement dataset service with CRUD operations, Samba integration, and dataset preparation logic ([2d59c10](https://github.com/project-david-ai/projectdavid-core/commit/2d59c103c6244711afc008f11dd1ea98e42dfd53))
* implement dataset service with CRUD operations, Samba integration, and dataset preparation logic ([c7fc414](https://github.com/project-david-ai/projectdavid-core/commit/c7fc414d5e42c38b188086012f3889fffaa85560))
* improve dataset preprocessing with Samba integration and Nginx fixes ([edf3bcf](https://github.com/project-david-ai/projectdavid-core/commit/edf3bcfe1c5c00ed06be042d48ae5b92bb828d15))
* improve dataset preprocessing with Samba integration and Nginx fixes ([7738cf9](https://github.com/project-david-ai/projectdavid-core/commit/7738cf99b5291bb2f2c31d15755a296de54d2888))
* integrate training worker, Redis-based job queue, and fine-tuning status cache with expanded dependency handling ([85dcb1b](https://github.com/project-david-ai/projectdavid-core/commit/85dcb1b164166904f38c1daa68bb007b0436c7af))
* introduce Training Service API with FastAPI, updated Dockerfile, and dependencies for fine-tuning REST operations ([1b20898](https://github.com/project-david-ai/projectdavid-core/commit/1b20898e3614155522527bd4acabbf028f9ba345))
* introduce Training Service API with FastAPI, updated Dockerfile, and dependencies for fine-tuning REST operations ([aa11327](https://github.com/project-david-ai/projectdavid-core/commit/aa11327a4a53ad5a44fce1c415215194be0c8823))
* protect training and model files during purge, add fine-tuning integration tests ([05fef00](https://github.com/project-david-ai/projectdavid-core/commit/05fef00f401d35a34f2e2c2d4180a7d6b5dea286))
* rebuild the worker.py ([13aae1b](https://github.com/project-david-ai/projectdavid-core/commit/13aae1b740a1b3ad632218fe7a6ed8a8c4febc26))
* refactor Docker Compose generation, streamline training services configuration, and enhance Docker Manager with Ollama support ([7abef74](https://github.com/project-david-ai/projectdavid-core/commit/7abef74bc835c87ee3ed4a76f204de770da5b500))
* refactor Docker Compose generation, streamline training services configuration, and enhance Docker Manager with Ollama support ([d8f1546](https://github.com/project-david-ai/projectdavid-core/commit/d8f15462983513a811dd80010408b3533cf59dfb))
* refactor fine-tuning table management, update schema migration, and cleanup unused model definitions ([c351136](https://github.com/project-david-ai/projectdavid-core/commit/c3511363dc6614fb06e277ff888eed8d7047eb85))
* Release. ([c2e8b41](https://github.com/project-david-ai/projectdavid-core/commit/c2e8b41fd1d895bd55518c6cc6f4e8c81b1314a7))
* Sovereign_Forge.md ([ba86924](https://github.com/project-david-ai/projectdavid-core/commit/ba869248f2c5c19e40569023e13b2485da3926e6))
* synch generate_dev_docker_compose ([7cb299a](https://github.com/project-david-ai/projectdavid-core/commit/7cb299ad3c1cba783b06772a857804dae1700f64))
* unify authentication to support Bearer tokens and X-API-Key ([d5ddcef](https://github.com/project-david-ai/projectdavid-core/commit/d5ddcefe688c7a97f0a97397974f046276023c00))
* vLLM hot swap ([fed312d](https://github.com/project-david-ai/projectdavid-core/commit/fed312d6c625d4ce218f7d8c954789747eebdf39))

## [1.24.1](https://github.com/project-david-ai/projectdavid-core/compare/v1.24.0...v1.24.1) (2026-03-16)


### Bug Fixes

* update build cache configuration and remove redundant Docker prune step in CI workflow ([28f01a0](https://github.com/project-david-ai/projectdavid-core/commit/28f01a05b143ed9c7d1a17a877d615a64dfca935))

# [1.24.0](https://github.com/project-david-ai/projectdavid-core/compare/v1.23.0...v1.24.0) (2026-03-16)


### Bug Fixes

* remove fine-tuning pipeline and associated tables (Dataset, TrainingJob, FineTunedModel) ([6771b9c](https://github.com/project-david-ai/projectdavid-core/commit/6771b9c656c254ff74178544346064eaedffd723))


### Features

* add fine-tuning pipeline with Dataset, TrainingJob, and FineTunedModel tables ([3d4b652](https://github.com/project-david-ai/projectdavid-core/commit/3d4b6529ba4123652876dd6bde383ef60f959b4e))

# [1.23.0](https://github.com/project-david-ai/projectdavid-core/compare/v1.22.1...v1.23.0) (2026-03-16)


### Bug Fixes

* update README commands, improve token counting with safe text handling, and add guard for empty inputs ([46a645f](https://github.com/project-david-ai/projectdavid-core/commit/46a645f67ac843e2f3fec7c2bb1c0df090abfa62))


### Features

* implement admin check utility and enhance access control logic ([062896d](https://github.com/project-david-ai/projectdavid-core/commit/062896da533eba4931c72e6a07056e95e1d2c944))

## [1.22.1](https://github.com/project-david-ai/platform/compare/v1.22.0...v1.22.1) (2026-03-15)


### Bug Fixes

* enable Docker build cache export and remove redundant prune step in CI workflow ([b07c3bf](https://github.com/project-david-ai/platform/commit/b07c3bfb1ee7711750e968d23b93178ff915d1fd))

# [1.22.0](https://github.com/project-david-ai/platform/compare/v1.21.1...v1.22.0) (2026-03-15)


### Features

* implement admin check utility and enhance access control logic ([57ac673](https://github.com/project-david-ai/platform/commit/57ac6739e9dfdc8365eeccc7b852fd5d823c1d6a))

## [1.21.1](https://github.com/project-david-ai/platform/compare/v1.21.0...v1.21.1) (2026-03-15)


### Bug Fixes

* rename `SPECIAL_DB_URL` to `DATABASE_URL` for consistency ([a86e353](https://github.com/project-david-ai/platform/commit/a86e3533b0b8d7599a8693616954c669f135e8d0))

# [1.21.0](https://github.com/project-david-ai/platform/compare/v1.20.0...v1.21.0) (2026-03-15)


### Features

* Resolve issue in n 741d86dd5ac8. ([e2b6a36](https://github.com/project-david-ai/platform/commit/e2b6a367fb6bbe77ca7171d906548d5ae5aaab0f))

# [1.20.0](https://github.com/project-david-ai/platform/compare/v1.19.8...v1.20.0) (2026-03-14)


### Bug Fixes

* add guards for missing tables in FK creation and enhance safe-DDL logic for constraints ([c4a569d](https://github.com/project-david-ai/platform/commit/c4a569dc9b2a390fcef1c2705141cc81648d0033))
* remove redundant blank lines in `bootstrap_admin.py` ([73d1fd9](https://github.com/project-david-ai/platform/commit/73d1fd9f582499b94da3df1f7aff840172b4673a))
* replace `docker_manager` with simplified `generate_docker_compose` for dev-friendly configurations ([e4a6d4f](https://github.com/project-david-ai/platform/commit/e4a6d4fb7f286f9bf32d2935648d90bec6074af1))


### Features

* add initial NGINX configuration for FastAPI upstream, WebSocket support, and Jaeger UI proxy ([c697385](https://github.com/project-david-ai/platform/commit/c697385d2f0d83029c3fd0234ca6b3ea1e58a645))

# [1.20.0](https://github.com/project-david-ai/platform/compare/v1.19.8...v1.20.0) (2026-03-14)


### Bug Fixes

* add guards for missing tables in FK creation and enhance safe-DDL logic for constraints ([c4a569d](https://github.com/project-david-ai/platform/commit/c4a569dc9b2a390fcef1c2705141cc81648d0033))
* replace `docker_manager` with simplified `generate_docker_compose` for dev-friendly configurations ([e4a6d4f](https://github.com/project-david-ai/platform/commit/e4a6d4fb7f286f9bf32d2935648d90bec6074af1))


### Features

* add initial NGINX configuration for FastAPI upstream, WebSocket support, and Jaeger UI proxy ([c697385](https://github.com/project-david-ai/platform/commit/c697385d2f0d83029c3fd0234ca6b3ea1e58a645))

## [1.19.8](https://github.com/project-david-ai/platform/compare/v1.19.7...v1.19.8) (2026-03-14)


### Bug Fixes

* enhance migrations with deferred FK creation, safe column alterations, and improved table guards ([85aed8e](https://github.com/project-david-ai/platform/commit/85aed8e4e2059c1a74855c51ce669e605dffd4dc))

## [1.19.7](https://github.com/project-david-ai/platform/compare/v1.19.6...v1.19.7) (2026-03-14)


### Bug Fixes

* enhance migrations with safe helpers, remove deprecated patterns, and guard against missing resources ([f66ec23](https://github.com/project-david-ai/platform/commit/f66ec239930c9d5c71439a629330646e31afde79))

## [1.19.6](https://github.com/project-david-ai/platform/compare/v1.19.5...v1.19.6) (2026-03-14)


### Bug Fixes

* add safe index creation and deletion helpers to safe_ddl utilities ([8c5fc91](https://github.com/project-david-ai/platform/commit/8c5fc91c0f2b6efa90dcb945880a3ca2e75e15b4))
* expand CI workflow to include testing, coverage, and Docker image build/publish steps ([289c984](https://github.com/project-david-ai/platform/commit/289c984425425a6a54ad62b12a3f624d31bd7427))
* simplify CI workflow, consolidate dependency installation, and clean up comments ([cc76525](https://github.com/project-david-ai/platform/commit/cc7652543eae733d879f73f6cd2b25b0d666f8ea))
* streamline workflows for linting, testing, releasing, and publishing to PyPI ([0d8be0b](https://github.com/project-david-ai/platform/commit/0d8be0b8e25d8ccf3ef9ce90df70ddc9bc19ad22))

## [1.19.5](https://github.com/project-david-ai/platform/compare/v1.19.4...v1.19.5) (2026-03-14)


### Bug Fixes

* add guards for missing tables/columns in migrations and enhance safe-DDL usage for indices, constraints, and FKs ([ff59ebd](https://github.com/project-david-ai/platform/commit/ff59ebd2a58b0af96b590359a910e946dd9a0d22))

## [1.19.4](https://github.com/project-david-ai/platform/compare/v1.19.3...v1.19.4) (2026-03-14)


### Bug Fixes

* add automatic multimodal message normalization for OpenAI compatibility in DefaultBase and GptOssBase workers ([b3f5647](https://github.com/project-david-ai/platform/commit/b3f5647b2df61b9e9e633a88a3ff21c9c39ed4c3))
* add interactive `configure` command, improve HF_TOKEN handling, and eliminate static placeholder secrets ([a89bea2](https://github.com/project-david-ai/platform/commit/a89bea2d73f4225f1fe0293b59d0a46aa32c639a))
* enforce Hyperbolic's single-image constraint, update integration tests, refactor multimodal normalization, and bump projectdavid to 1.77.6 ([568c03a](https://github.com/project-david-ai/platform/commit/568c03a6f0e6655ac4de85dc36ec5da972adf62f))
* enforce secure secret generation, add HF_TOKEN validation, and improve HuggingFace cache path resolution ([de579c3](https://github.com/project-david-ai/platform/commit/de579c32a939d42a88251e48f0a339d3114bdc1d))
* migrate `generate_docker_compose.py` to `docker_manager.py`, add interactive .env management, validate secrets, improve Docker configuration and HuggingFace support ([7adad40](https://github.com/project-david-ai/platform/commit/7adad40de814c04d9eec9d00ab56145660b853f1))
* restructure integration tests, add Ollama and Hyperbolic multimodal SDK tests, and enhance Qwen worker NLP support ([0c8f71e](https://github.com/project-david-ai/platform/commit/0c8f71edaff870cc03ea8b3944d5c5b4baea6b91))

## [1.19.3](https://github.com/project-david-ai/platform/compare/v1.19.2...v1.19.3) (2026-03-14)


### Bug Fixes

* add automatic multimodal message normalization for OpenAI compatibility in DefaultBase and GptOssBase workers ([680c059](https://github.com/project-david-ai/platform/commit/680c05999ab63dc2ee3cc5ddab850d53cafcd178))
* enforce Hyperbolic's single-image constraint, update integration tests, refactor multimodal normalization, and bump projectdavid to 1.77.6 ([4af8512](https://github.com/project-david-ai/platform/commit/4af85127558af8b213075d0c0743ff0f2d3b364a))
* restructure integration tests, add Ollama and Hyperbolic multimodal SDK tests, and enhance Qwen worker NLP support ([82cd1ff](https://github.com/project-david-ai/platform/commit/82cd1ff51a17e0670b00a1ab6b171ab108831637))

## [1.19.2](https://github.com/project-david-ai/platform/compare/v1.19.1...v1.19.2) (2026-03-13)


### Bug Fixes

* bump projectdavid to version 1.76.2 in requirements files ([6fa5489](https://github.com/project-david-ai/platform/commit/6fa5489f44d6d0edc7d576b5a2098b5b876966bc))
* bump projectdavid to version 1.76.2 in requirements files ([92b5e64](https://github.com/project-david-ai/platform/commit/92b5e64020e7460734dd39c5ccd2460838ce4550))
* cut back multi-modal message change ([534bc11](https://github.com/project-david-ai/platform/commit/534bc11388bc95ecbc2d86d85f61551f8ca1d69e))
* cut back multi-modal message change ([8ae8326](https://github.com/project-david-ai/platform/commit/8ae83267b08b135a366054ff1fe064658e0365a5))
* refactor multimodal handling in vLLMRawStream, improve content merging logic, and enhance conversation truncation ([910ec19](https://github.com/project-david-ai/platform/commit/910ec1920d1586210a9ffdb9572e00f0e882e6e6))
* refactor multimodal handling in vLLMRawStream, improve content merging logic, and enhance conversation truncation ([3ebc5da](https://github.com/project-david-ai/platform/commit/3ebc5daa44d4df7b567b1890bb8d38708750bd07))
* remove obsolete SSEManager and cleanup import formatting in multiple files ([86718a2](https://github.com/project-david-ai/platform/commit/86718a2854a48d8d3933432509810382af13ff5f))
* remove obsolete SSEManager and cleanup import formatting in multiple files ([2728e30](https://github.com/project-david-ai/platform/commit/2728e30c8d20db694c7d93757c52f3ed3b3c9a32))

## [1.19.1](https://github.com/project-david-ai/platform/compare/v1.19.0...v1.19.1) (2026-03-12)


### Bug Fixes

* bump projectdavid to version 1.76.2 in requirements files ([6be28da](https://github.com/project-david-ai/platform/commit/6be28dab4f5c4a007819bac0213f4f732ddeb3a3))
* cut back multi-modal message change ([6309353](https://github.com/project-david-ai/platform/commit/630935379a2a0355c5688968520058cf92ebb194))

# [1.19.0](https://github.com/project-david-ai/platform/compare/v1.18.1...v1.19.0) (2026-03-12)


### Bug Fixes

*  Optimize the FastAPI Endpoint Bridge ([4ea9408](https://github.com/project-david-ai/platform/commit/4ea9408fdbbbbe39c3ff82bc45fb8b14ea2bcd3a))
*  Update client to projectdavid==1.39.8 ([1a47dd0](https://github.com/project-david-ai/platform/commit/1a47dd05b192f793f84bbb9b4e7f340b44df916e))
* _set_up_context_window          ← async, called from stream() ([1629b54](https://github.com/project-david-ai/platform/commit/1629b54b91e76f383907b644ed9643f75ce774ab))
* Add block that peforms ([110ef78](https://github.com/project-david-ai/platform/commit/110ef780ea61ab9f43eae7df398a91c7d2d8a24d))
* Add decision telemetry to the main DB ([42b0dac](https://github.com/project-david-ai/platform/commit/42b0dac38b871e27870b327809bb734b04944fb6))
* Add decision_telemetry param to _set_up_context_window ([fc32423](https://github.com/project-david-ai/platform/commit/fc3242309d1fbc3440a474c06e037143f4b9e934))
* Add deep_research toggle to the assistant_cache.py ([2d4686a](https://github.com/project-david-ai/platform/commit/2d4686ae6ce715ea25e96b757e4722c50f9ac31e))
* Add delegate_engineer_task to delegate_engineer_task ([250ec12](https://github.com/project-david-ai/platform/commit/250ec12c39a417e29a99777193da5b853b830163))
* Add delete_ephemeral_thread class attribute. ([302ea82](https://github.com/project-david-ai/platform/commit/302ea828bc10164c2c22e01852c5cbdec7f2567a))
* Add ephemeral clean up. ([ed8788c](https://github.com/project-david-ai/platform/commit/ed8788ca13bd43f20be42d89aca99f3fd2bda543))
* add GITHUB_TOKEN parameter to checkout steps in CI workflow ([3333241](https://github.com/project-david-ai/platform/commit/33332410ced019e8befc30faab6f9c064632e263))
* add missing function call formatting instructions to L4_SENIOR_ENGINEER_INSTRUCTIONS ([b40eed5](https://github.com/project-david-ai/platform/commit/b40eed5f1fec8d495f324b627cd2db30bf3db7b1))
* Add missing tools to the PLATFORM_TOOLS list. ([ee48c7e](https://github.com/project-david-ai/platform/commit/ee48c7e9301cd25e7434ee2a7f644c263e3f7e5e))
* Add model compatibility table ([a688993](https://github.com/project-david-ai/platform/commit/a688993b064cef11ef7ffa8735229ccdbe8a6234))
* add NetworkInventoryMixin to mixins ([f8a3bbc](https://github.com/project-david-ai/platform/commit/f8a3bbc06ca50461f84f1a72623dd603f41229f8))
* Add new Quen models to SUBMODEL_CLASS_MAP in TogetherAI handler. ([3a57300](https://github.com/project-david-ai/platform/commit/3a57300d03b1aa1a97435f5100ac487fb3a4daeb))
* Add new Quen models to SUBMODEL_CLASS_MAP in TogetherAI handler. ([3f9a3dd](https://github.com/project-david-ai/platform/commit/3f9a3dd66aaed99e38198249a6c3b9731b25d36e))
* add status messages to web_search_mixin.py ([9c4c168](https://github.com/project-david-ai/platform/commit/9c4c1687232b5b334f759a8aa5bd333e877bede6))
* add status messages to web_search_mixin.py ([0799a90](https://github.com/project-david-ai/platform/commit/0799a908839abbff4a57ab0a2d7997a58edff8c9))
* Add structured function call, with tool_call ID to DeepSeek dialogue ([14e79c9](https://github.com/project-david-ai/platform/commit/14e79c91a51081c5ceaa43a33d0a89f8d6c28d20))
* Add support for unicode function calls in delta_normalizer.py ([fd3576f](https://github.com/project-david-ai/platform/commit/fd3576f688a5fc785f37f8badc1eac09f34755b0))
* Add tool inventory mixin to tool routing. ([de5004d](https://github.com/project-david-ai/platform/commit/de5004ded89bff3d48dbab7323bbc8aee856b229))
* Add tool name to tool output metadata ([981e5ac](https://github.com/project-david-ai/platform/commit/981e5ace76001044ee591f4f6852000049ab6334))
* Add TOOL_DECISION_PROTOCOL to CORE_INSTRUCTIONS ([70d97f3](https://github.com/project-david-ai/platform/commit/70d97f3118d4b61368b0161f858d238245c01e1e))
* Add URL support for code execution generated files ([efb3b2e](https://github.com/project-david-ai/platform/commit/efb3b2ee209865d0d7b42901b8712f9112c2b360))
* Alembic Revision ID: ce0a8a7e9d41 ([3751662](https://github.com/project-david-ai/platform/commit/3751662adb67d03fecb7b470e50fdfe7a123b95f))
* align worker ScratchpadEvent intercept payload with _scratchpad_status() contract ([347cc23](https://github.com/project-david-ai/platform/commit/347cc232af2038acb73d7d60655bf1349876a4d5))
* align worker ScratchpadEvent intercept payload with _scratchpad_status() contract ([3cbb759](https://github.com/project-david-ai/platform/commit/3cbb7596caa84d7df7d9a6eff1f9e5d885205e35))
* await handle_file_search instead of async iterating ([dc97f02](https://github.com/project-david-ai/platform/commit/dc97f0202a838a7ad6c9cd655383d8b2ca7cfe65))
* back out from clean up block change in delegation_mixin.py ([743a746](https://github.com/project-david-ai/platform/commit/743a74680de69aded58d2ca350ca42ba790fb971))
* back out of project The Engineer changes. ([51efb5b](https://github.com/project-david-ai/platform/commit/51efb5b8896304c360cfba08ea62f306c3b1196a))
* broken google worker import ([7d83f90](https://github.com/project-david-ai/platform/commit/7d83f90ae02f5a9e2a1897d5e392937396d75fbb))
* bubble worker ScratchpadEvents through senior stream ([71f1609](https://github.com/project-david-ai/platform/commit/71f160939d24b9d969f354cc69ea7336f2414d96))
* catch cold-load timeout on new ephemeral threads ([c0c2a66](https://github.com/project-david-ai/platform/commit/c0c2a667bb534ab73163481186485bc7415a0d60))
* catch cold-load timeout on new ephemeral threads ([70f8911](https://github.com/project-david-ai/platform/commit/70f8911e362aae5276b10bbe6a164701c0289380))
* change method list_thread_messages to get_formatted_messages ([67d142f](https://github.com/project-david-ai/platform/commit/67d142f8deb0efdd3d05d4da0654cad2406f7666))
* clean up obsolete code and replace bare `except` usage for consistency ([a92430f](https://github.com/project-david-ai/platform/commit/a92430f16f6ed42b72a9bfed4f9e0ce8b883b6ef))
* **code-interpreter:** resolve syntax normalization errors and prevent system prompt leakage ([5d493ed](https://github.com/project-david-ai/platform/commit/5d493ed102464acccb1943c6a1758c0561276595))
* **code-interpreter:** suppress raw errors from consumer stream; route via activity messages ([8159a13](https://github.com/project-david-ai/platform/commit/8159a134fb7c243bd5bc3b5bc1dd1ad899ac6f95))
* Consolidate SUBMODEL_CLASS_MAP in HyperbolicHandler ([40118cf](https://github.com/project-david-ai/platform/commit/40118cf5536519903f19bad2528bdaaae0c48c6e))
* Consolidate SUBMODEL_CLASS_MAP in TogetherAIHandler ([ef55252](https://github.com/project-david-ai/platform/commit/ef55252d0e9197c9a24534bccf46b6596e691cf1))
* correct decision_payload typo in Actions.create call in code_execution_mixin.py ([8d88eec](https://github.com/project-david-ai/platform/commit/8d88eec407a86062439b3c94cc1e66095b1fd896))
* correct decision_payload typo in Actions.create call in shell_execution_mixin.py ([f8d3e2f](https://github.com/project-david-ai/platform/commit/f8d3e2fdec8034f4211e7d8a2c7e79d533fd6e96))
* correct enable_decision_telemetry typo ([cb9af30](https://github.com/project-david-ai/platform/commit/cb9af304e3c50eaa62e8b6056935ff1b7eff0d0a))
* correct enable_decision_telemetry typo ([b29c466](https://github.com/project-david-ai/platform/commit/b29c4668230d551b83cfbd23344f7419ec79cfcd))
* correct named argument issue in scratchpad_mixin.py. The * means all arguments after self must be passed as keyword arguments, not positional arguments ([433d7c6](https://github.com/project-david-ai/platform/commit/433d7c6ad06634ccb89b3529577f3c8ab186455f))
* cut back to a non validated return  model in the return from get_pending_actions ([4dbc78e](https://github.com/project-david-ai/platform/commit/4dbc78e7634152e150e53aa536b5fcfd0aeb6887))
* Cut back to specific DeepSeek Stream worke ([58376dd](https://github.com/project-david-ai/platform/commit/58376ddb294c63b60f2242c1ab59babcf4f4e52f))
* Cut over /together-ai models to TogetherHermesDefaultWorker ([c64fd77](https://github.com/project-david-ai/platform/commit/c64fd776338482f05c24f60c71e860667526131a))
* Cut over /together-ai models to TogetherHermesDefaultWorker ([28aab2e](https://github.com/project-david-ai/platform/commit/28aab2e44dd80028609efa70e5d1a6bcb60cabbc))
* Cut over deepseek_base.py to self._execute_stream_request client. ([cebac81](https://github.com/project-david-ai/platform/commit/cebac81620741fe13130d4b6f0f9c357f9f0ebda))
* Cut over deepseek_base.py to self._execute_stream_request client. ([df6cff8](https://github.com/project-david-ai/platform/commit/df6cff8df421174a27f8455c1856065ed69e71c3))
* Cut over Quen Worker to level 3  assistant! ([e9681bb](https://github.com/project-david-ai/platform/commit/e9681bb23704e087cad53c6c74bff1308d8ad97d))
* Decouple and align JSON key mapping for DelegationMixin status events ([ddd5d80](https://github.com/project-david-ai/platform/commit/ddd5d80a626049023459ec407e2c700c9dcbddf9))
* Default False = Use Redis Cache (Efficient). ([c5631c0](https://github.com/project-david-ai/platform/commit/c5631c0b0537ba0c5f1ff0dc402b49714378fec6))
* delete api/code_execution_service.py ([3366dfc](https://github.com/project-david-ai/platform/commit/3366dfcb1fa6aa2f2f1233a1829c1f0d77ecb27f))
* delete async_client.py. ([9c1ca9f](https://github.com/project-david-ai/platform/commit/9c1ca9f0c1ab6cfa5d31ba296f569aa7e014b946))
* delete concrete steam method from llama worker class ([44e154d](https://github.com/project-david-ai/platform/commit/44e154d313f98c41a251d7bfb6eddb47f202d3e9))
* delete defunct event handling service. ([30c0db5](https://github.com/project-david-ai/platform/commit/30c0db53293914182960afc6c73ff4a07e1055c7))
* delete old /inference dir. ([baa5c2e](https://github.com/project-david-ai/platform/commit/baa5c2e755fab988e075b8edb6551601fb3db100))
* delete old_hb_gpt_oss.py ([6c803b0](https://github.com/project-david-ai/platform/commit/6c803b0e9c77cf7ce99dd5dbd77fa56382c58f2c))
* delete redundant code execution file handling instructions. ([5a37451](https://github.com/project-david-ai/platform/commit/5a3745148fc465e6174fdf1e6a1766f6e761ae6d))
* delete tools_router.py ([fe7a408](https://github.com/project-david-ai/platform/commit/fe7a408f07faa54e797ff8e6c86660bb6f305773))
* delete tools.py ([0a739f9](https://github.com/project-david-ai/platform/commit/0a739f9efc99bb61c858902934849d2d3f64005c))
* eliminate internal SDK HTTP round-trips across assistant layer ([7599bcf](https://github.com/project-david-ai/platform/commit/7599bcf1149c7c851cfb812358e22a919e9bc2cb))
* eliminate internal SDK HTTP round-trips across assistant layer ([b31a660](https://github.com/project-david-ai/platform/commit/b31a660ffc0107cee69c7b8625c6a38bea246360))
* enforce recursive troubleshooting and prevent speculative diagnosis in SE_ANTI_STALL ([2944385](https://github.com/project-david-ai/platform/commit/294438511325054491b9ef59c1f1b4adbc2e9248))
* enforce scroll limits and search-first gate to prevent doom-scrolling ([015eca9](https://github.com/project-david-ai/platform/commit/015eca963288ec8a43650e57649fce9aaf5fa622))
* Engineering instructions locked down! ([44acb05](https://github.com/project-david-ai/platform/commit/44acb05da24fb510570a7ba3c2ebf3d1ff1d45f7))
* Engineering instructions locked down! ([d59e6c9](https://github.com/project-david-ai/platform/commit/d59e6c95f3e407d53f07f195d84b7164c1cae573))
* enhance Level 3 recursion and fix web state MRO issues ([ff25f66](https://github.com/project-david-ai/platform/commit/ff25f668414fb7e9e3302a716d81495c4faf6c9c))
* ensure `user_id` is passed in thread operations for ownership validation ([46be1b5](https://github.com/project-david-ai/platform/commit/46be1b542a89727ed49b19c258d638cd22fca568))
* filter  "<|channel|>analysis" ([a5dd32a](https://github.com/project-david-ai/platform/commit/a5dd32a57fd3a879b1774ce62ea170ac54f1d0c7))
* Fix computer tool, crashing the stream issue ([8c60c81](https://github.com/project-david-ai/platform/commit/8c60c811f357d4b5b0866c1688d2332691246c8f))
* fix(files): resolve signed url download 401 error and path duplication ([1d3ae6d](https://github.com/project-david-ai/platform/commit/1d3ae6d1b1f4a5e6fe56b450b8e342aaf4885f66))
* function call response issues ([2b6086f](https://github.com/project-david-ai/platform/commit/2b6086f88479a43a00d8c451b838e7c7687cb30b))
* function calls correctly set and parsed in gpt oss ([8b276d7](https://github.com/project-david-ai/platform/commit/8b276d702350e3c525e3a554250ff711f4353b82))
* gpt-oss function call 2nd turn issue ([524b693](https://github.com/project-david-ai/platform/commit/524b6938d594c297ce87d8f5901d04b56b621b83))
* GPT-oss level 3 compliant. ([d9e4b24](https://github.com/project-david-ai/platform/commit/d9e4b2475acdc6be5c873b206b6696eca7d85d4c))
* GPT-oss level 3 compliant. ([2872954](https://github.com/project-david-ai/platform/commit/28729544fda3b2f09377e6926a75f1e577742c72))
* GPT-oss level 3 compliant. ([c01a740](https://github.com/project-david-ai/platform/commit/c01a740b80dade3131900ba46f6cf3a622297e84))
* **gpt-oss-worker:** preserve ephemeral supervisor identity during DB persistence ([56d54e9](https://github.com/project-david-ai/platform/commit/56d54e92fdc4fa43a540df105fe5af55fb88b89e))
* **hermes-worker:** preserve ephemeral supervisor identity during DB persistence ([dc0ef86](https://github.com/project-david-ai/platform/commit/dc0ef86282715c71ead4b50ab8d13d840a781355))
* housekeeping daemons for expired runs and thread cleanup ([9ea441f](https://github.com/project-david-ai/platform/commit/9ea441f85af9d299b570d97c2fb2ddda30704f72))
* Implement cache invalidation for assistants cached_assistant ([7f4d3c5](https://github.com/project-david-ai/platform/commit/7f4d3c52d80b00844a8f07a12c9d7901b190ff2e))
* Implement DecisionEvent type ([1992583](https://github.com/project-david-ai/platform/commit/1992583e424f22a2170e9bd9075d0345d5786a7d))
* Implement dotenv for secrets in alembic .env.py ([a437017](https://github.com/project-david-ai/platform/commit/a4370176c279f29140d6f742c523b4045c018332))
* Implement Hermes style function call dialogue structuring for DeepCognito models ([2a28d79](https://github.com/project-david-ai/platform/commit/2a28d794acff457e40500d3cd7dbdb5b9ce9bb06))
* Implement Hermes style function call dialogue structuring for DeepCognito models ([3ea46d8](https://github.com/project-david-ai/platform/commit/3ea46d885749e43c1b9a06c8e6875f06979e96e5))
* Implement level 2 recovery for platform tools. ([ca48507](https://github.com/project-david-ai/platform/commit/ca4850760d1b42312d7bd02fcb04f58a2691115d))
* implement level 3 instructions_assembly.py ([0f24f94](https://github.com/project-david-ai/platform/commit/0f24f947da6bc523f605ca025b1dd81a306e5a0a))
* Implement native tool call response for gpt-oss Hermes class models. ([6846044](https://github.com/project-david-ai/platform/commit/684604440b5f6692b47f610cf411d88adf41f324))
* Implement native tool call response for llama. ([610b4ed](https://github.com/project-david-ai/platform/commit/610b4edaf336c8cd99198a8e393e6cbfca7f3fc0))
* implement native tool calling on Quen Model calls. ([6347d12](https://github.com/project-david-ai/platform/commit/6347d12ebcf912c9348ee142f76f4af43dc2d823))
* Implement new action state signalling. Previously the client end had to use ActionService.get_pending_actions in a constant poll from the client side before the client new that a run had an action to service. This was cumbersome , slow, and caused needless churn on the DB. The new method propagates the action.id direct to the client side via a manifest payload as soon as the action is created. ([14c1066](https://github.com/project-david-ai/platform/commit/14c1066abd684461ebb0ac00a5ca918560ef57e6))
* implement new architecture for hot_code. Complex parsing and replay from each worker is needless when hot_code can be replayed direct from the handler. ([e49ea75](https://github.com/project-david-ai/platform/commit/e49ea75e0987b05052a2b286b0339a28726708ea))
* implement new instruction hirarchy for  _build_system_message ([fafd1c5](https://github.com/project-david-ai/platform/commit/fafd1c527de70e66512ef50e8dbe6b53e984ad8c))
* implement new instruction hirarchy for  _build_system_message ([5c01a9d](https://github.com/project-david-ai/platform/commit/5c01a9db1834be053af0d7807dd7920d2b509eb2))
* Implement new scripts ([2ba916a](https://github.com/project-david-ai/platform/commit/2ba916a961ce592ab053ff3cf526852d005a1d4b))
* implement process_hot_code_buffer in CodeExecutionMixin. ([e7d5db7](https://github.com/project-david-ai/platform/commit/e7d5db769e0bd5531efadb02c7123c654b438004))
* implement Senior/Junior agent duo with full inventory resolution ([d84cf6b](https://github.com/project-david-ai/platform/commit/d84cf6bd287ea716fb7d75afe14bee17cb921d60))
* Implement serp search tool. ([4c3d5aa](https://github.com/project-david-ai/platform/commit/4c3d5aafc78e9b9b62c6db58ce1fc8dfd5e1853f))
* Implement structured tool call detection for gpt-oss ([4c6934d](https://github.com/project-david-ai/platform/commit/4c6934d78e88f95846ed811d6821f198dba29d3c))
* Implemented the Stream method as a concreate class in the base orchestrator_core.py. Should greatly increase the speed of model onboarding! ([f775bc0](https://github.com/project-david-ai/platform/commit/f775bc0634adbd05b5b21b747f1789455cdf4995))
* import PLATFORM_TOOLS from common library ([89dd4c0](https://github.com/project-david-ai/platform/commit/89dd4c0d03094373b511ec4e303c8750705d451b))
* Improve code intepreter file generation instructions. ([7b5f18a](https://github.com/project-david-ai/platform/commit/7b5f18a3387cf9583e7a35ecfbf6a1b8f13a9f53))
* improve exception handling and cleanup redundant code ([9987132](https://github.com/project-david-ai/platform/commit/998713240785e4ef85762165b730b0cb786cead4))
* Improve speed and smoothness of streaming. ([0e16437](https://github.com/project-david-ai/platform/commit/0e164378fdc5722ee19f75319a3b32d47c27ee30))
* Improve speed and smoothness of streaming. ([b3e9191](https://github.com/project-david-ai/platform/commit/b3e919132480bea23337161e93dca7484c94ee77))
* improve variable naming in exception handling logic in `fix_ruff.py` ([6f89aa5](https://github.com/project-david-ai/platform/commit/6f89aa5891afebb3a1ed0af98e423e3f531eac47))
* Integrate save dll rendering functions ino  alembic . We no longer have to manually refactor auto generated alembic scripts. ([0516264](https://github.com/project-david-ai/platform/commit/0516264243d5467a1ed61184827a565bdf058f1b))
* issue with hyperbolic/DeepSeek data flow ([4b7a0e5](https://github.com/project-david-ai/platform/commit/4b7a0e54859e15cf95fa1544952780ac7a189839))
* issue with oss reasoning chunks ([9c4e88f](https://github.com/project-david-ai/platform/commit/9c4e88f02946f9ca65477dcbad1a358f2d211ea3))
* issue with oss reasoning chunks ([d228581](https://github.com/project-david-ai/platform/commit/d228581cc001e152bd81ba71949a404cac5ccd47))
* JsonUtilsMixin update. ([c73cd3f](https://github.com/project-david-ai/platform/commit/c73cd3f1bd8eae46f7c8bd3c43f3d227a599066d))
* Major speed improvements in inference. ([dc06f29](https://github.com/project-david-ai/platform/commit/dc06f294b0ae9d68bc5733053db26eec660152c1))
* Make assistant.tools as the source of truth for the assistants tools array ([fd05c4e](https://github.com/project-david-ai/platform/commit/fd05c4e00f0a8cba161ccdce979c85cec8e7c8c7))
* Migrate all TogetherAI workers to unifed asynch client ([9f088e4](https://github.com/project-david-ai/platform/commit/9f088e4cca93ddac3c180773f2c20d6df552fe9e))
* Migrate all workers to decision telemetry algorithm. ([bb29acb](https://github.com/project-david-ai/platform/commit/bb29acbac15a77d4b97f3034b12ee56b0ac24a07))
* Migrate b697008df93a ([4c12971](https://github.com/project-david-ai/platform/commit/4c12971fe3aa882ebd8fef19439f749130baba77))
* Migrate DB dda6fd28f45c ([4e65be6](https://github.com/project-david-ai/platform/commit/4e65be6dc816b87cbd973a0e6bf2c07ce91bd005))
* Migrate deepseek_base.py to asynchronous mode. ([5205f0a](https://github.com/project-david-ai/platform/commit/5205f0a6f6a9e621d0d7ad9ed1bb227a4b43a5fd))
* Migrate deepseek_base.py to asynchronous mode. ([5fdc74b](https://github.com/project-david-ai/platform/commit/5fdc74bb8401127ea83c25ab725883477a7f06e1))
* Migrate default_base.py.py to asynchronous mode. ([e878fb6](https://github.com/project-david-ai/platform/commit/e878fb6505eb89594e7112c7153880e46ca188d3))
* Migrate GPT-oss worker to home brew client, works! ([ae862c8](https://github.com/project-david-ai/platform/commit/ae862c8aa4149a77e6931ef50988d32b91eb47c7))
* Migrate hermes_type_default_base.py to asynchronous mode. ([82ed344](https://github.com/project-david-ai/platform/commit/82ed3442c3f34d8c2ff23f495f18afafa64ac086))
* Migrate nvidia_base.py.py to asynchronous mode. ([abf1171](https://github.com/project-david-ai/platform/commit/abf117115e5074b6b4aa2124d54a750bd882b1f6))
* migrate quen_base.py to the new function call architecture ([f1c9bae](https://github.com/project-david-ai/platform/commit/f1c9bae342af9b4fdbb7138ced4087a5cb0ccee4))
* Migrate qwen_base.py to asynchronous mode. ([0bc71e9](https://github.com/project-david-ai/platform/commit/0bc71e95cf4246c655834a1c90b4c153d0ec53d9))
* Migrate service_now_base.py to asynchronous mode. ([baeb2ea](https://github.com/project-david-ai/platform/commit/baeb2ead8c214ede60944189f443b179dbfd10fa))
* Migrate to json status events ([21cb9cb](https://github.com/project-david-ai/platform/commit/21cb9cb9f7b123b232a76805be7d0476ab06a1ee))
* migrate to native execution and fix cache initialization ([2ef44f8](https://github.com/project-david-ai/platform/commit/2ef44f8f38b91b49e40f5f0e7afb4971f1183808))
* Migrate to new base worker arhitecture ([69f0b02](https://github.com/project-david-ai/platform/commit/69f0b024a41fe48855e65ad7a4cdbed9e5853494))
* Migrate to new message cache architecture ([435bdd9](https://github.com/project-david-ai/platform/commit/435bdd961e15e7746d1e7fe7eece70d38f22ff10))
* Migrate to new singleton client ([150ecee](https://github.com/project-david-ai/platform/commit/150ecee94882aa5fff29ad87d7f30fc6f0022d64))
* Migrate to new singleton client ([13713d4](https://github.com/project-david-ai/platform/commit/13713d4d23822a52aa2cf7e356961b6db465cff1))
* migrate to projectdavid client 1.67.1 ([f803de5](https://github.com/project-david-ai/platform/commit/f803de508dbd91cdea2b6b3a98479337e564f0ea))
* Migrate to projectdavid v1.42.0 ([eaa467c](https://github.com/project-david-ai/platform/commit/eaa467cba691e084262d45d7bb4604ac1744269f))
* Migrate to projectdavid v1.42.0 ([eabb4f3](https://github.com/project-david-ai/platform/commit/eabb4f3b9465eeaf41e5f5829c078b358ba078f3))
* Migrate to projectdavid==1.49.1 ([e603321](https://github.com/project-david-ai/platform/commit/e603321ec25244192b42d4ec28b09dcc87857da5))
* migrate tool mixins to native execution service ([02d75f2](https://github.com/project-david-ai/platform/commit/02d75f28bf250501525d4be886731b7bcfb81769))
* migrate web tool execution to NativeExecutionService ([e3fde1b](https://github.com/project-david-ai/platform/commit/e3fde1bf19692a3e6d1523f5585a30aba727caf7))
* Move assistant cache setup to core Class ([52d4be8](https://github.com/project-david-ai/platform/commit/52d4be806c506a22a6cb377b9c99b70057319f9c))
* Move assistant cache setup to core Class ([de54203](https://github.com/project-david-ai/platform/commit/de542035b2f66f7a3f949fc8164293aef37e1eec))
* Move AsyncHyperbolicClient to client factory. ([9ba074c](https://github.com/project-david-ai/platform/commit/9ba074c33925aeed4a5b713c11166f599bdfe16a))
* move cache_utils.py and cached_assistant.py --> /cache ([11e435a](https://github.com/project-david-ai/platform/commit/11e435a6d3ef94a0ad9e1cb4e98a58ede219a58e))
* Move unified_async_client.py to /clients package ([726d73d](https://github.com/project-david-ai/platform/commit/726d73dc08a765dce2a928e05c05c03bc25bee41))
* Once and ephemeral supervisor has run, the assistants cache must be cleared and reinitiated so  that the latest persona is the cache to avoid context and tool contamination. ([b85cce2](https://github.com/project-david-ai/platform/commit/b85cce288877e5a0fd106cd5ac397384ab46a194))
* order senior Engineers tools. ([8d51d4a](https://github.com/project-david-ai/platform/commit/8d51d4a0200da53f44c8687161e2ffb11cded40d))
* orphaned-thread purge daemon for GDPR compliance ([1bb0157](https://github.com/project-david-ai/platform/commit/1bb0157c4b8e5fd5a4a5936b04ce588cf4a0795c))
* pass `user_id` for ownership validation in run operations ([49a090f](https://github.com/project-david-ai/platform/commit/49a090f90760a857e97375178af206af0328f33d))
* pass snapshot_id to refresh_snapshot, not snapshot_name ([8d7c2e3](https://github.com/project-david-ai/platform/commit/8d7c2e3ac5d9b8bdfda3e1d4ed25cca0e4c0fce0))
* phantom file generation issue in CodeExecutionMixin ([f3d7f44](https://github.com/project-david-ai/platform/commit/f3d7f44cb526e7e78537d3639170ef289f7ea45b))
* pin tempfile output dir in CODE_FILE_HANDLING system instruction ([f49f5cf](https://github.com/project-david-ai/platform/commit/f49f5cf4672263f013e28f30a975fdff9d548e6d))
* preserve ephemeral supervisor identity during DB persistence ([6debbf9](https://github.com/project-david-ai/platform/commit/6debbf90e7e9b8f0a072ccffe344921b2fcb6bf6))
* Push ac1498a9642c_remove_tools_table_and_associations.py ([495fa01](https://github.com/project-david-ai/platform/commit/495fa012217203e6e8e626a647ea8e14f629da97))
* record_tool_decision.py ([e15cf1e](https://github.com/project-david-ai/platform/commit/e15cf1e76bf6742c173d278562a7715218f22120))
* refactor _resolve_and_prioritize_platform_tools with new silent tool mandatory_platform_tools ([fdeb54c](https://github.com/project-david-ai/platform/commit/fdeb54cbbdb17129fe2c4e97b3e7868da669fb39))
* Reinstated partial code_interpreter instructions to _build_native_tools_system_message ([694a3fa](https://github.com/project-david-ai/platform/commit/694a3fa6343901f80b94afeb433deafc5bd1d722))
* remove `model_compatibility_report.md` and bump `projectdavid` to v1.74.7 ([a8423bf](https://github.com/project-david-ai/platform/commit/a8423bf47e7ace75b7bacab6e426b86ac200ce1f))
* remove `vector_store_assistants` relationship and update `messages` constraints ([afb9c54](https://github.com/project-david-ai/platform/commit/afb9c54dccbb9da995ab12fe57e0acc931d9e9d1))
* remove assistant_id from engineer router endpoints ([7e66173](https://github.com/project-david-ai/platform/commit/7e6617397905f8c68d5ff7a4910621eeb18a72dc))
* remove assistant_id from engineer router endpoints ([9d7d708](https://github.com/project-david-ai/platform/commit/9d7d70802d117936970a657653fb31b3b8537108))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([4358043](https://github.com/project-david-ai/platform/commit/4358043993d81973f5d63f78ad689d2fb0a29b2a))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([937b704](https://github.com/project-david-ai/platform/commit/937b7042dee1feeae2073e094461407fb1f4e55d))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([2f58759](https://github.com/project-david-ai/platform/commit/2f58759c497249b40d62e6ebbe8e64bd52340810))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([3ac11e3](https://github.com/project-david-ai/platform/commit/3ac11e3afa4ff54ca2058328702ac821ece800e6))
* Remove device inventory instructions from SE_TRIAGE_PROTOCOL ([45bca41](https://github.com/project-david-ai/platform/commit/45bca418199e326ae4528b56753659700df4cceb))
* Remove device inventory tools from the senior engineers tool registry. ([b1b7c00](https://github.com/project-david-ai/platform/commit/b1b7c00a3398b2b2d9968d29968ec4a396526a2a))
* Remove duplicate api arg from worker signature. ([e5f1356](https://github.com/project-david-ai/platform/commit/e5f1356662e78002edadcb314a6c71ab224da458))
* Remove hot_code handling from Quen worker! ([447b8a6](https://github.com/project-david-ai/platform/commit/447b8a648f259a8d0445b292d208473905e7a895))
* Remove local curl searches from main api ([336c4f6](https://github.com/project-david-ai/platform/commit/336c4f67d12acaf00652675a034542baf6651cd7))
* remove obsolete integration test file ([4060a98](https://github.com/project-david-ai/platform/commit/4060a98cfb81604182185f55a97db27723e8a3d5))
* Remove Tools table from models.py ([f073286](https://github.com/project-david-ai/platform/commit/f073286592196efc8cd33d7f73a9c06c13845a5d))
* replace bare `except` with `except Exception` and clean up redundant imports ([279fe56](https://github.com/project-david-ai/platform/commit/279fe56b2a51c6c543ce7d71efc24da7b09dba07))
* replace unattended_file_search with _search_vs_async ([1b816d5](https://github.com/project-david-ai/platform/commit/1b816d50746101abb49ca7ce6018b953ec5a654f))
* resolve 500 error on engineer inventory ingest ([d87c744](https://github.com/project-david-ai/platform/commit/d87c7444b6854ecdb5d97edd46eb4a30c486c820))
* resolve 500 error on engineer inventory ingest ([fdd33ae](https://github.com/project-david-ai/platform/commit/fdd33ae645951cf3ea8cd58b8a2dcb3741f71dfa))
* resolve all deep search issues ([e354cfc](https://github.com/project-david-ai/platform/commit/e354cfc828c538a1468ba9817a43c08a7668c99d))
* resolve batfish snapshot ownership across Senior/Junior worker boundary ([c4b252d](https://github.com/project-david-ai/platform/commit/c4b252d4bce8577e13793babfb77eb05125717ce))
* resolve batfish snapshot ownership across Senior/Junior worker boundary ([3613180](https://github.com/project-david-ai/platform/commit/3613180a82bb262a67e4a53d4deebce281f9f09d))
* resolve batfish snapshot ownership across Senior/Junior worker boundary ([eecbab0](https://github.com/project-david-ai/platform/commit/eecbab0590aee7acf600f602d9498fb813b2024e))
* Resolve computer tool auth token issues. ([a9c2074](https://github.com/project-david-ai/platform/commit/a9c20744a4344c3c8fdd9d9daabd9bce65cb78c3))
* Resolve computer tool auth token issues. ([8426ac7](https://github.com/project-david-ai/platform/commit/8426ac7bb1a8227217b9a86583a431af09c7cefd))
* resolve computer tool dispatch and SDK action call regressions ([777c391](https://github.com/project-david-ai/platform/commit/777c391deed0032a2a18b639d97da679d3b3386f))
* resolve computer tool dispatch and SDK action call regressions ([85604c6](https://github.com/project-david-ai/platform/commit/85604c66bc480d4ecad8a70b1fea8387140e96d1))
* Resolve deepseek_base.py  and child issues. ([153d073](https://github.com/project-david-ai/platform/commit/153d073096210342ae9c2e07118949a89a845f9e))
* Resolve forward stream delay issue ([09deac8](https://github.com/project-david-ai/platform/commit/09deac88058cb4d7e0b4e819d5a9029f29604195))
* Resolve GPT oss streaming issue by creating an instance of  _get_client_instance, not OpenAI ([ab14433](https://github.com/project-david-ai/platform/commit/ab14433528d2a56d15ae700f6da7faf857c6a9ca))
* Resolve gpt-oss-120b reasoning tokens not generating issue. ([a78e11c](https://github.com/project-david-ai/platform/commit/a78e11c325394f5fb3ff7ae1e67c4377d691aa4e))
* Resolve gpt-oss-120b reasoning tokens not generating issue. ([90bc1c4](https://github.com/project-david-ai/platform/commit/90bc1c4f6463cf8bb37ceb6b466bd26506b86277))
* Resolve handover issues ([297dcee](https://github.com/project-david-ai/platform/commit/297dcee3e2cd24394bfde5a60c1050220b1dfe11))
* Resolve hyperbolic/meta-llama- models function calling issues. Previously, the model did not respond to the first turn. This was due to passing in the tool call array in the request payload. ([3e19625](https://github.com/project-david-ai/platform/commit/3e196257f5aa230682cb9ef915b45f13a5adf591))
* resolve identity swap failures, worker flag extraction, and SDK reliance ([322cceb](https://github.com/project-david-ai/platform/commit/322cceba85c66159ca97866a4732686df9ecb94c))
* Resolve issues with fc call polling. ([a85af1d](https://github.com/project-david-ai/platform/commit/a85af1dbe5ad6b4f38a42e8d23877fda3d337992))
* Resolve ongoing issues with consumer side function call handling race condition ([32826ea](https://github.com/project-david-ai/platform/commit/32826ead80416914b311c73e42e6d7fa76f6eac7))
* Resolve ongoing issues with consumer side function call handling race condition ([f2e6e88](https://github.com/project-david-ai/platform/commit/f2e6e886cefe04f471c52c0ed697bbe2784908c6))
* Resolve the issue contaminating the research supervisors context with unwanted tools ([07df062](https://github.com/project-david-ai/platform/commit/07df062dd29f28a72f4944fe50409fe45946984e))
* Resolve Threads relationship issue in models.py ([2943b62](https://github.com/project-david-ai/platform/commit/2943b625105895e214b1e0bf34acb81c4e399091))
* restore assistant_id ownership to process_conversation ([bc91e55](https://github.com/project-david-ai/platform/commit/bc91e55d43d4754ec80c366c90847dfde42a9ab4))
* Restore gpt-oss hot code interleave. ([9d00699](https://github.com/project-david-ai/platform/commit/9d00699a21e140c47ee0c88a746b4ff512fdf206))
* Restore real time streaming of research workers output ([8c73d6b](https://github.com/project-david-ai/platform/commit/8c73d6b501a979eb3faccbd149c696e5a62347b5))
* resurrect soft-deleted snapshots on create instead of raising 409 ([8c2abaf](https://github.com/project-david-ai/platform/commit/8c2abafa3189a4bd745b70ead03caa9a359db99e))
* Run model intake reports ([cc149e7](https://github.com/project-david-ai/platform/commit/cc149e7cc60be49dfe9894101a4f4abd6a6c94be))
* Run model intake reports ([dfeb5ab](https://github.com/project-david-ai/platform/commit/dfeb5ab9e96af1d61a812a57c9d2071c7d45a467))
* smarter Turn 2 prompt + anti-loop failure handling ([6e9dfc3](https://github.com/project-david-ai/platform/commit/6e9dfc378c4a1c683d5b460a560a370273a80f56))
* stamp scratchpad thread id in the junior research assistants run object. ([a5de48b](https://github.com/project-david-ai/platform/commit/a5de48bc8d59424fc853e1273d201d7f4332e653))
* standardise WebEvent emission across backend, SDK, and frontend ([f92cbb4](https://github.com/project-david-ai/platform/commit/f92cbb4e35dd36000ceb7a1d9e7c69f8818ef261))
* streamline vector store endpoints and remove deprecated assistant relations ([7673c76](https://github.com/project-david-ai/platform/commit/7673c76972518d98232e38a9f91e12f0acf2968d))
* streamline worker delegation logic and improve stream handling ([8e4413f](https://github.com/project-david-ai/platform/commit/8e4413f9b31e538d386c9b1c3539d5520aef12c8))
* strengthen ownership guards in inference router ([63c3c8d](https://github.com/project-david-ai/platform/commit/63c3c8dde2fafe62818040b661d53c053fd0cbe6))
* strictly enforce Change Request formatting and ban conversational preambles ([65adabb](https://github.com/project-david-ai/platform/commit/65adabb308d980b68c2aa36c49b1447af2ac48fa))
* successfully ported deepseek_base.py ([c8d444d](https://github.com/project-david-ai/platform/commit/c8d444dbe576e8da829d392636f38139ce7c1541))
* support dynamic `ollama_base_url` via metadata in orchestration worker ([7d3bc4c](https://github.com/project-david-ai/platform/commit/7d3bc4c7101b0d84df3e90f20c2e935808706e55))
* synchronize and alphabetize mixin imports and inheritance ([e9eba55](https://github.com/project-david-ai/platform/commit/e9eba5556b9dedbd54a948f472e3a4017615af28))
* synchronize and alphabetize mixin imports and inheritance ([70d8aaf](https://github.com/project-david-ai/platform/commit/70d8aafc23d0cc214e4498b87c5fa1277616bba8))
* synchronize worker classes with Qwen gold standard logic ([678b13c](https://github.com/project-david-ai/platform/commit/678b13c335f0e6f7853a2e1353ce101afc253b8f))
* Temporarily remove status event messaging from shell_execution_mixin.py ([0a2c406](https://github.com/project-david-ai/platform/commit/0a2c4068b02171069004d28f68aa9c8146612e3b))
* Temporarily remove status event messaging from shell_execution_mixin.py ([48f2702](https://github.com/project-david-ai/platform/commit/48f2702709f3d3b185c33556532f4fdb9a49bcea))
* Temporarily remove status event messaging from shell_execution_mixin.py ([81fda55](https://github.com/project-david-ai/platform/commit/81fda5534a188dbe3be66fbe273750a966c090fa))
* The JE and SE are hallucinating solutions, update instruction set ([486e5cd](https://github.com/project-david-ai/platform/commit/486e5cdba57d18c895398dfe0ed25ec1e1175b99))
* TogetherAI level 3 tests. ([48fdff4](https://github.com/project-david-ai/platform/commit/48fdff43d2aa4064634a20247e1771023ebc0780))
* TogetherAI level 3 tests. ([371569a](https://github.com/project-david-ai/platform/commit/371569a3cb0613144d4de95663a510140ff8e1c6))
* tool role tool_call_id bug ([502788e](https://github.com/project-david-ai/platform/commit/502788e7d17b4e74ac3f2aa35d654ec9a4f412fa))
* Tweak unified_inference_test.py ([cdb38a5](https://github.com/project-david-ai/platform/commit/cdb38a500b246b55afe42fd3ecf16e89fb06ce9e))
* Update client to projectdavid==1.39.4 ([af886ff](https://github.com/project-david-ai/platform/commit/af886ffbbdb7e84c9d9b988f5657c5d06291fda6))
* update client to projectdavid==1.51.2 ([4c21d26](https://github.com/project-david-ai/platform/commit/4c21d261619ecd547c7fce049c734286ce4924ce))
* update client to projectdavid==1.51.2 ([ee0554a](https://github.com/project-david-ai/platform/commit/ee0554aa8487ed200a5bdc216f11c9b9cb333ed1))
* update client to projectdavid==1.60.0 ([ea9efc9](https://github.com/project-david-ai/platform/commit/ea9efc912f19f7b2faa80259c9ce7ca3806fb5b1))
* update client to projectdavid==1.73.0 ([69803ee](https://github.com/project-david-ai/platform/commit/69803ee3a225503e4b6bfe1f92de65ea2957149e))
* Update delta_normalizer.py with level 3 planning protocol tags ([2e3edd2](https://github.com/project-david-ai/platform/commit/2e3edd2184f91e0d758524211a6ac9e077b55194))
* update GitHub Actions workflows to use latest action versions and add placeholder test file ([53846af](https://github.com/project-david-ai/platform/commit/53846afc90709e4e51d911f0b6ab613a66e0c20d))
* update integration test scripts. ([cd566c9](https://github.com/project-david-ai/platform/commit/cd566c98cf236a75316e3756a946f6a8064fb281))
* Update process_conversation with decision state handling. ([f38482f](https://github.com/project-david-ai/platform/commit/f38482f7a9e9de101d878d560c756851ff31b99d))
* update projectdavid client to : 1.59.0 ([8c53820](https://github.com/project-david-ai/platform/commit/8c538200fee0c85a293713341772860019a8163b))
* Update projectdavid client to projectdavid==1.66.0 ([c1387c8](https://github.com/project-david-ai/platform/commit/c1387c85a835f675481bec4763fb94f9ae8ce409))
* Update projectdavid client to projectdavid==1.73.1 ([5f4100b](https://github.com/project-david-ai/platform/commit/5f4100b40a169d062704f33c296f199a301807dc))
* Update projectdavid sdk to projectdavid==1.47.5 ([6149b5f](https://github.com/project-david-ai/platform/commit/6149b5f1891570228d95f8653931d8f4de00b4a5))
* update to projectdavid==1.54.4 ([a5b0619](https://github.com/project-david-ai/platform/commit/a5b0619a4cbb0182bc4619ea0fa0050eb8baf32e))
* update to projectdavid==1.60.3 ([aef1a1e](https://github.com/project-david-ai/platform/commit/aef1a1e256a00967d20e018325e80268885186d0))
* Upgrade client to projectdavid==1.39.6 ([c4b1f9f](https://github.com/project-david-ai/platform/commit/c4b1f9febecb69f17f77f10298918f19fbdab97e))
* Upgrade client to projectdavid==1.39.6 ([bb48ceb](https://github.com/project-david-ai/platform/commit/bb48ceb463ad0c6f692681b3cb381f3dd08ed895))
* Upgrade projectdavid 1.53.0 ([091a83b](https://github.com/project-david-ai/platform/commit/091a83b21a06eb58af9fab1d4cda945c163adeb1))
* use self.assistant_id in process_tool_calls dispatch to reflect post-swap identity ([255098f](https://github.com/project-david-ai/platform/commit/255098f5563b0c9def2f4d92f0b18ae1fd2db833))
* use self.assistant_id in process_tool_calls dispatch to reflect post-swap identity ([8c41a8a](https://github.com/project-david-ai/platform/commit/8c41a8a7b1d5b58a663ae8b26fa1e2148e8ffbe0))
* We must explicitly set the cwd (Current Working Directory) of the Python subprocess to self.generated_files_dir. ([9fb5991](https://github.com/project-david-ai/platform/commit/9fb59917970d319324e934cfd215573c983e7073))
* web_search working. ([04728be](https://github.com/project-david-ai/platform/commit/04728bed82a711b9671e18f6177842ed9a5b80e3))
* web_search working. ([3dfd3e3](https://github.com/project-david-ai/platform/commit/3dfd3e3af9b691a5c6c5ce7db219d53d1de35574))
* web_search working. ([b0b9afd](https://github.com/project-david-ai/platform/commit/b0b9afd73509214e8c3fbec4e3250de372445db6))


### Features

* _get_enriched_topology ([0382732](https://github.com/project-david-ai/platform/commit/038273272c8e000bb106e5ae23a0f6fae64143cb))
* 9314d4058f78_add_agentic_state_columns_to_assistant_.py adds some level 3 agentic components to the master DB ([1c8d684](https://github.com/project-david-ai/platform/commit/1c8d684f4f930b272418a76047e70145c04a970f))
* Add  get_enriched_topology endpoint ([433d300](https://github.com/project-david-ai/platform/commit/433d3006078e8042eb158b0bd996f1ccc37e0b26))
* Add  get_enriched_topology endpoint ([b3796da](https://github.com/project-david-ai/platform/commit/b3796da3d001516ecf5f4c922ebe83133accc686))
* add `owner_id` to `Thread` table and enhance thread management ([242bb36](https://github.com/project-david-ai/platform/commit/242bb36aad05b2fa0d2fb32736c267da8d48d2cb))
* add assistant ownership validation to `completions` endpoint ([aa6f2be](https://github.com/project-david-ai/platform/commit/aa6f2bec8ae7a56fe8018213c11956f6ea60bf37))
* add BatfishSnapshot to models ([8f22a13](https://github.com/project-david-ai/platform/commit/8f22a13b215f42d7e58031ae18b0ecbcc897930a))
* Add engineer flag ([18f2a0b](https://github.com/project-david-ai/platform/commit/18f2a0b79dad4d44e23d10ef495f633a1b91e118))
* Add Engineers to context build logic ([09863ea](https://github.com/project-david-ai/platform/commit/09863eafdeeb2439755692a0a4ae22eb82413a4c))
* add hard-purge daemons for soft-deleted files and vector stores ([7c12eea](https://github.com/project-david-ai/platform/commit/7c12eea89e1cb939d1353340a6d440f07134179f))
* add owner_id to assistants with SafeDDL guards ([7591325](https://github.com/project-david-ai/platform/commit/7591325c942c254296bb3ff1a177aa88e639e5db))
* add pytest options and ignore integration tests ([040ba83](https://github.com/project-david-ai/platform/commit/040ba839676e56a2ac7429c63f8e58930bfca0b2))
* add run_batfish_tool and run_all_batfish_tools to tool router ([576c2a3](https://github.com/project-david-ai/platform/commit/576c2a3ab474b2b69063822d3b0c5f4cb6fcfa98))
* add tool_resources to cached assistant payload ([54f0dd4](https://github.com/project-david-ai/platform/commit/54f0dd44d830df54439a56c8a680f2ecaeeb046f))
* add viewer support for shell sessions and enhance file harvest logic ([6e53b60](https://github.com/project-david-ai/platform/commit/6e53b6076a7ad9b55162266d2a6c31e429143dc3))
* add vLLM orchestration support and documentation ([c133522](https://github.com/project-david-ai/platform/commit/c1335229d9fffd215ead8c8bfdf6edb35e850110))
* batfish_router.py ([a231f10](https://github.com/project-david-ai/platform/commit/a231f10e655c015ff4daa62dc837fdb246cd99b7))
* batfish_router.py ([0f96559](https://github.com/project-david-ai/platform/commit/0f96559475f879a02a353ba6bcdecdb9f0987055))
* complete orchestration alignment and stream_sync implementation ([8cb5728](https://github.com/project-david-ai/platform/commit/8cb5728457a0fba29188ec28dde314507b6dd963))
* cut over to projectdavid==1.39.0: bringing new models online: ([5a9ed31](https://github.com/project-david-ai/platform/commit/5a9ed3183a5244e40826a422ba67b117a9e53c6e))
* deep research ([6a56cc5](https://github.com/project-david-ai/platform/commit/6a56cc5dfe4196eea0f926206301fa20cffa26dc))
* DeepSearch paradigm 2 ([28403bf](https://github.com/project-david-ai/platform/commit/28403bf7d9f8e681f8b8ba12eb403359035f9059))
* **default-worker:** implement async stream_sync wrapper and queue isolation ([65fd483](https://github.com/project-david-ai/platform/commit/65fd483e3e44b1aaf5b4cb0f3a66ca1aa458150e))
* docker compose exec api alembic revision --autogenerate -m "add ID to BatfishSnapshot to models" ([6573524](https://github.com/project-david-ai/platform/commit/6573524a328114ba2633addceaa6e6bfb5f08b09))
* enforce ownership validation for thread operations ([1c81ca9](https://github.com/project-david-ai/platform/commit/1c81ca9bafc4841f45b4838e6ddf8511d564819d))
* enforce secure file generation with mandatory tempfile.tempdir and path confinement ([4d522d6](https://github.com/project-david-ai/platform/commit/4d522d62bcf09b689741dcc5e67e78a56fb61403))
* **engineer:** bridge client and service layers for inventory tools ([fc20e8a](https://github.com/project-david-ai/platform/commit/fc20e8a198b1884ad8940e273cefd500d737476f))
* enhance Docker service management with --exclude flag and improve exception handling ([5897dde](https://github.com/project-david-ai/platform/commit/5897dde4fd8d28495cfa655bb971a848e4d181f4))
* extend `docker-compose.dev.example.yml` with Redis, browser, SearxNG, Jaeger, Otel Collector, and Ollama services ([afa7dc3](https://github.com/project-david-ai/platform/commit/afa7dc34514482648f7fea3a97fb9b4bd6f4ca3c))
* Extend DelegationMixin for Engineering delegation. ([afc3363](https://github.com/project-david-ai/platform/commit/afc3363f60f19cc41dae41e8674eb3cf9a966868))
* fan out across tool_resources vector stores ([7f82a17](https://github.com/project-david-ai/platform/commit/7f82a175307346a8da71b36764274f821850ed20))
* implement auto-cleanup of ephemeral files after streaming ([ba48f7c](https://github.com/project-david-ai/platform/commit/ba48f7c8bc984f4bd1e978317455ba5ea6b662bb))
* implement auto-cleanup of ephemeral files after streaming ([d17465e](https://github.com/project-david-ai/platform/commit/d17465e6da9432ef1a2074ec06c2bd4321d29746))
* implement delegated deep search model mapping ([10e27a0](https://github.com/project-david-ai/platform/commit/10e27a0b7c99a98dec39a388bb4a847c55f786ef))
* Implement Engineering tool defs ([54d1a0a](https://github.com/project-david-ai/platform/commit/54d1a0a1fd11139dcefed0ff563c5c5254849f48))
* implement expired file cleanup utility and daemon ([5584c42](https://github.com/project-david-ai/platform/commit/5584c4202e3af581d653eaaf9159f4b785c72f56))
* Implement full-stack real-time Scratchpad visualization ([decb876](https://github.com/project-david-ai/platform/commit/decb87678e439f2e599f64763bd807e5b755ba76))
* implement junior engineer second turn with intercepted tool execution ([89d8bbb](https://github.com/project-david-ai/platform/commit/89d8bbb06695c51790ce90677d22559eceb7c3af))
* implement NetworkEngineerMixin ([ba7b681](https://github.com/project-david-ai/platform/commit/ba7b68117b295b091110205118075ec0459a4295))
* Implement persistent research worker thread per session. ([733dcca](https://github.com/project-david-ai/platform/commit/733dcca316c67dea0786140b0339ab3fde2fa7f4))
* Integrate Ollama local inference into the stack. ([4d21102](https://github.com/project-david-ai/platform/commit/4d211026987a2de3a50d5318ff70ba96fa5fa35a))
* integrate Ollama local inference support into DeltaNormalizer ([8e0e8c1](https://github.com/project-david-ai/platform/commit/8e0e8c1a1e107a838942520e04fc0b82050c97c3))
* intercept Junior Engineer tool calls and emit as ToolInterceptEvent ([5f9d38f](https://github.com/project-david-ai/platform/commit/5f9d38f722ea1a8bc83c555e5316893c7f0cd6e1))
* introduce computer_shell_netfilter and enhance shell execution lifecycle ([77aeff2](https://github.com/project-david-ai/platform/commit/77aeff251b937339b8c945c2e10ff47624c08964))
* map engineer flag during Assistant creation in AssistantService ([afec101](https://github.com/project-david-ai/platform/commit/afec101a3d9f8ab482f6169aec4948c9d779ca5d))
* migrate DelegationMixin off HTTP SDK to NativeExecutionService ([a964414](https://github.com/project-david-ai/platform/commit/a964414d5b52de9ba1adaa5e412e05abd1786739))
* Migrate FileSearchMixin to direct VectoreStoreManager inference. ([308015b](https://github.com/project-david-ai/platform/commit/308015b8b28c47a317a69dc7bded085fad45cf1b))
* move tool-call suppression logic to inference handler ([8e56c5b](https://github.com/project-david-ai/platform/commit/8e56c5b47870803a5d1ca7fb073b6aacdcbf6898))
* move tool-call suppression logic to inference handler ([4152548](https://github.com/project-david-ai/platform/commit/415254882786c6de79280805378a86093789c418))
* pass delegated model through run metadata and update workers ([a76a203](https://github.com/project-david-ai/platform/commit/a76a203e271d20c57af28d03de0b66373df798c4))
* pass inference API key through run metadata and enhance worker stream handling ([ee1d410](https://github.com/project-david-ai/platform/commit/ee1d4103f261d370100b79cbd07f8620c70cbf1d))
* port latest orchestration features and stream_sync wrapper ([fb4faf5](https://github.com/project-david-ai/platform/commit/fb4faf566e51c75ef9965c31001f973d580589f9))
* prevent `docker_manager.py` from running inside Docker containers ([86e4ee7](https://github.com/project-david-ai/platform/commit/86e4ee79c8ba55a501f57679f2915e5e74a07727))
* publish with deep search ([08f464c](https://github.com/project-david-ai/platform/commit/08f464c8136afe1b8d4d7760a40fa88bf1f3d477))
* Refactor batfish_router ([a4e35a7](https://github.com/project-david-ai/platform/commit/a4e35a75166b894a54fc435df41d29393934ebfa))
* remove deprecated assistant bootstrap scripts and add DockerManager CLI scaffolding ([6c8bbce](https://github.com/project-david-ai/platform/commit/6c8bbcedf5813d444a6fb1c298552f7d908380d5))
* replace DDG SERP scrape with SearxNG meta-search engine ([0d72b81](https://github.com/project-david-ai/platform/commit/0d72b81ddbf62aa4bdba6ed633e94a77c1c0e887))
* replace shell command execution with sequential protocol; enhance session lifecycle with sandboxing and file harvest ([a89a226](https://github.com/project-david-ai/platform/commit/a89a2267355443a0390ba9660e4d8d419a82a6a9))
* **services:** map engineer flag during Assistant creation in AssistantService ([e6a25de](https://github.com/project-david-ai/platform/commit/e6a25de2d82933ef79154f71ebd8e46b672b9b31))
* soft-delete functionality for files with Recycle Bin support ([a6bf1c2](https://github.com/project-david-ai/platform/commit/a6bf1c2b06e7179313fd35f380fed0448d1effa7))
* soft-delete functionality for files with Recycle Bin support ([4c33903](https://github.com/project-david-ai/platform/commit/4c33903f52186e19460d9b076c6c3f17b7a45e11))
* tenant-isolated snapshot pipeline with ID-first design ([7c1e2bf](https://github.com/project-david-ai/platform/commit/7c1e2bfe7c2c70632eb4049ebf5c4de63631ef85))
* track full ordered tool dispatch log per run ([4a43c00](https://github.com/project-david-ai/platform/commit/4a43c00b8530129e2bcbe7515cf9918dab5edb95))
* unify orchestration logic, add stream_sync, and upgrade run retrieval ([bda1975](https://github.com/project-david-ai/platform/commit/bda1975c06949085667d14d637e737a3f2d011d4))
* unify orchestration logic, add stream_sync, and upgrade run retrieval ([1e2bf1c](https://github.com/project-david-ai/platform/commit/1e2bf1c8204b5982321001da0b5e78a991cbe627))
* upgrade projectdavid SDK to projectdavid==1.37.0 ([a3305b1](https://github.com/project-david-ai/platform/commit/a3305b103186385678bb1db0cdc5ad36784c331b))
* wire worker scratchpad to shared supervisor thread ([5cdc43a](https://github.com/project-david-ai/platform/commit/5cdc43a16de6b7310e6a17c362f5c3fc0f8c82ca))
* wrap inventory_cache in a service layer ([49ba796](https://github.com/project-david-ai/platform/commit/49ba796db3e62a3ac2369db69c42e4f30c33c435))

# [1.19.0](https://github.com/project-david-ai/platform/compare/v1.18.1...v1.19.0) (2026-03-12)


### Bug Fixes

*  Optimize the FastAPI Endpoint Bridge ([4ea9408](https://github.com/project-david-ai/platform/commit/4ea9408fdbbbbe39c3ff82bc45fb8b14ea2bcd3a))
*  Update client to projectdavid==1.39.8 ([1a47dd0](https://github.com/project-david-ai/platform/commit/1a47dd05b192f793f84bbb9b4e7f340b44df916e))
* _set_up_context_window          ← async, called from stream() ([1629b54](https://github.com/project-david-ai/platform/commit/1629b54b91e76f383907b644ed9643f75ce774ab))
* Add block that peforms ([110ef78](https://github.com/project-david-ai/platform/commit/110ef780ea61ab9f43eae7df398a91c7d2d8a24d))
* Add decision telemetry to the main DB ([42b0dac](https://github.com/project-david-ai/platform/commit/42b0dac38b871e27870b327809bb734b04944fb6))
* Add decision_telemetry param to _set_up_context_window ([fc32423](https://github.com/project-david-ai/platform/commit/fc3242309d1fbc3440a474c06e037143f4b9e934))
* Add deep_research toggle to the assistant_cache.py ([2d4686a](https://github.com/project-david-ai/platform/commit/2d4686ae6ce715ea25e96b757e4722c50f9ac31e))
* Add delegate_engineer_task to delegate_engineer_task ([250ec12](https://github.com/project-david-ai/platform/commit/250ec12c39a417e29a99777193da5b853b830163))
* Add delete_ephemeral_thread class attribute. ([302ea82](https://github.com/project-david-ai/platform/commit/302ea828bc10164c2c22e01852c5cbdec7f2567a))
* Add ephemeral clean up. ([ed8788c](https://github.com/project-david-ai/platform/commit/ed8788ca13bd43f20be42d89aca99f3fd2bda543))
* add missing function call formatting instructions to L4_SENIOR_ENGINEER_INSTRUCTIONS ([b40eed5](https://github.com/project-david-ai/platform/commit/b40eed5f1fec8d495f324b627cd2db30bf3db7b1))
* Add missing tools to the PLATFORM_TOOLS list. ([ee48c7e](https://github.com/project-david-ai/platform/commit/ee48c7e9301cd25e7434ee2a7f644c263e3f7e5e))
* Add model compatibility table ([a688993](https://github.com/project-david-ai/platform/commit/a688993b064cef11ef7ffa8735229ccdbe8a6234))
* add NetworkInventoryMixin to mixins ([f8a3bbc](https://github.com/project-david-ai/platform/commit/f8a3bbc06ca50461f84f1a72623dd603f41229f8))
* Add new Quen models to SUBMODEL_CLASS_MAP in TogetherAI handler. ([3a57300](https://github.com/project-david-ai/platform/commit/3a57300d03b1aa1a97435f5100ac487fb3a4daeb))
* Add new Quen models to SUBMODEL_CLASS_MAP in TogetherAI handler. ([3f9a3dd](https://github.com/project-david-ai/platform/commit/3f9a3dd66aaed99e38198249a6c3b9731b25d36e))
* add status messages to web_search_mixin.py ([9c4c168](https://github.com/project-david-ai/platform/commit/9c4c1687232b5b334f759a8aa5bd333e877bede6))
* add status messages to web_search_mixin.py ([0799a90](https://github.com/project-david-ai/platform/commit/0799a908839abbff4a57ab0a2d7997a58edff8c9))
* Add structured function call, with tool_call ID to DeepSeek dialogue ([14e79c9](https://github.com/project-david-ai/platform/commit/14e79c91a51081c5ceaa43a33d0a89f8d6c28d20))
* Add support for unicode function calls in delta_normalizer.py ([fd3576f](https://github.com/project-david-ai/platform/commit/fd3576f688a5fc785f37f8badc1eac09f34755b0))
* Add tool inventory mixin to tool routing. ([de5004d](https://github.com/project-david-ai/platform/commit/de5004ded89bff3d48dbab7323bbc8aee856b229))
* Add tool name to tool output metadata ([981e5ac](https://github.com/project-david-ai/platform/commit/981e5ace76001044ee591f4f6852000049ab6334))
* Add TOOL_DECISION_PROTOCOL to CORE_INSTRUCTIONS ([70d97f3](https://github.com/project-david-ai/platform/commit/70d97f3118d4b61368b0161f858d238245c01e1e))
* Add URL support for code execution generated files ([efb3b2e](https://github.com/project-david-ai/platform/commit/efb3b2ee209865d0d7b42901b8712f9112c2b360))
* Alembic Revision ID: ce0a8a7e9d41 ([3751662](https://github.com/project-david-ai/platform/commit/3751662adb67d03fecb7b470e50fdfe7a123b95f))
* align worker ScratchpadEvent intercept payload with _scratchpad_status() contract ([347cc23](https://github.com/project-david-ai/platform/commit/347cc232af2038acb73d7d60655bf1349876a4d5))
* align worker ScratchpadEvent intercept payload with _scratchpad_status() contract ([3cbb759](https://github.com/project-david-ai/platform/commit/3cbb7596caa84d7df7d9a6eff1f9e5d885205e35))
* await handle_file_search instead of async iterating ([dc97f02](https://github.com/project-david-ai/platform/commit/dc97f0202a838a7ad6c9cd655383d8b2ca7cfe65))
* back out from clean up block change in delegation_mixin.py ([743a746](https://github.com/project-david-ai/platform/commit/743a74680de69aded58d2ca350ca42ba790fb971))
* back out of project The Engineer changes. ([51efb5b](https://github.com/project-david-ai/platform/commit/51efb5b8896304c360cfba08ea62f306c3b1196a))
* broken google worker import ([7d83f90](https://github.com/project-david-ai/platform/commit/7d83f90ae02f5a9e2a1897d5e392937396d75fbb))
* bubble worker ScratchpadEvents through senior stream ([71f1609](https://github.com/project-david-ai/platform/commit/71f160939d24b9d969f354cc69ea7336f2414d96))
* catch cold-load timeout on new ephemeral threads ([c0c2a66](https://github.com/project-david-ai/platform/commit/c0c2a667bb534ab73163481186485bc7415a0d60))
* catch cold-load timeout on new ephemeral threads ([70f8911](https://github.com/project-david-ai/platform/commit/70f8911e362aae5276b10bbe6a164701c0289380))
* change method list_thread_messages to get_formatted_messages ([67d142f](https://github.com/project-david-ai/platform/commit/67d142f8deb0efdd3d05d4da0654cad2406f7666))
* clean up obsolete code and replace bare `except` usage for consistency ([a92430f](https://github.com/project-david-ai/platform/commit/a92430f16f6ed42b72a9bfed4f9e0ce8b883b6ef))
* **code-interpreter:** resolve syntax normalization errors and prevent system prompt leakage ([5d493ed](https://github.com/project-david-ai/platform/commit/5d493ed102464acccb1943c6a1758c0561276595))
* **code-interpreter:** suppress raw errors from consumer stream; route via activity messages ([8159a13](https://github.com/project-david-ai/platform/commit/8159a134fb7c243bd5bc3b5bc1dd1ad899ac6f95))
* Consolidate SUBMODEL_CLASS_MAP in HyperbolicHandler ([40118cf](https://github.com/project-david-ai/platform/commit/40118cf5536519903f19bad2528bdaaae0c48c6e))
* Consolidate SUBMODEL_CLASS_MAP in TogetherAIHandler ([ef55252](https://github.com/project-david-ai/platform/commit/ef55252d0e9197c9a24534bccf46b6596e691cf1))
* correct decision_payload typo in Actions.create call in code_execution_mixin.py ([8d88eec](https://github.com/project-david-ai/platform/commit/8d88eec407a86062439b3c94cc1e66095b1fd896))
* correct decision_payload typo in Actions.create call in shell_execution_mixin.py ([f8d3e2f](https://github.com/project-david-ai/platform/commit/f8d3e2fdec8034f4211e7d8a2c7e79d533fd6e96))
* correct enable_decision_telemetry typo ([cb9af30](https://github.com/project-david-ai/platform/commit/cb9af304e3c50eaa62e8b6056935ff1b7eff0d0a))
* correct enable_decision_telemetry typo ([b29c466](https://github.com/project-david-ai/platform/commit/b29c4668230d551b83cfbd23344f7419ec79cfcd))
* correct named argument issue in scratchpad_mixin.py. The * means all arguments after self must be passed as keyword arguments, not positional arguments ([433d7c6](https://github.com/project-david-ai/platform/commit/433d7c6ad06634ccb89b3529577f3c8ab186455f))
* cut back to a non validated return  model in the return from get_pending_actions ([4dbc78e](https://github.com/project-david-ai/platform/commit/4dbc78e7634152e150e53aa536b5fcfd0aeb6887))
* Cut back to specific DeepSeek Stream worke ([58376dd](https://github.com/project-david-ai/platform/commit/58376ddb294c63b60f2242c1ab59babcf4f4e52f))
* Cut over /together-ai models to TogetherHermesDefaultWorker ([c64fd77](https://github.com/project-david-ai/platform/commit/c64fd776338482f05c24f60c71e860667526131a))
* Cut over /together-ai models to TogetherHermesDefaultWorker ([28aab2e](https://github.com/project-david-ai/platform/commit/28aab2e44dd80028609efa70e5d1a6bcb60cabbc))
* Cut over deepseek_base.py to self._execute_stream_request client. ([cebac81](https://github.com/project-david-ai/platform/commit/cebac81620741fe13130d4b6f0f9c357f9f0ebda))
* Cut over deepseek_base.py to self._execute_stream_request client. ([df6cff8](https://github.com/project-david-ai/platform/commit/df6cff8df421174a27f8455c1856065ed69e71c3))
* Cut over Quen Worker to level 3  assistant! ([e9681bb](https://github.com/project-david-ai/platform/commit/e9681bb23704e087cad53c6c74bff1308d8ad97d))
* Decouple and align JSON key mapping for DelegationMixin status events ([ddd5d80](https://github.com/project-david-ai/platform/commit/ddd5d80a626049023459ec407e2c700c9dcbddf9))
* Default False = Use Redis Cache (Efficient). ([c5631c0](https://github.com/project-david-ai/platform/commit/c5631c0b0537ba0c5f1ff0dc402b49714378fec6))
* delete api/code_execution_service.py ([3366dfc](https://github.com/project-david-ai/platform/commit/3366dfcb1fa6aa2f2f1233a1829c1f0d77ecb27f))
* delete async_client.py. ([9c1ca9f](https://github.com/project-david-ai/platform/commit/9c1ca9f0c1ab6cfa5d31ba296f569aa7e014b946))
* delete concrete steam method from llama worker class ([44e154d](https://github.com/project-david-ai/platform/commit/44e154d313f98c41a251d7bfb6eddb47f202d3e9))
* delete defunct event handling service. ([30c0db5](https://github.com/project-david-ai/platform/commit/30c0db53293914182960afc6c73ff4a07e1055c7))
* delete old /inference dir. ([baa5c2e](https://github.com/project-david-ai/platform/commit/baa5c2e755fab988e075b8edb6551601fb3db100))
* delete old_hb_gpt_oss.py ([6c803b0](https://github.com/project-david-ai/platform/commit/6c803b0e9c77cf7ce99dd5dbd77fa56382c58f2c))
* delete redundant code execution file handling instructions. ([5a37451](https://github.com/project-david-ai/platform/commit/5a3745148fc465e6174fdf1e6a1766f6e761ae6d))
* delete tools_router.py ([fe7a408](https://github.com/project-david-ai/platform/commit/fe7a408f07faa54e797ff8e6c86660bb6f305773))
* delete tools.py ([0a739f9](https://github.com/project-david-ai/platform/commit/0a739f9efc99bb61c858902934849d2d3f64005c))
* eliminate internal SDK HTTP round-trips across assistant layer ([7599bcf](https://github.com/project-david-ai/platform/commit/7599bcf1149c7c851cfb812358e22a919e9bc2cb))
* eliminate internal SDK HTTP round-trips across assistant layer ([b31a660](https://github.com/project-david-ai/platform/commit/b31a660ffc0107cee69c7b8625c6a38bea246360))
* enforce recursive troubleshooting and prevent speculative diagnosis in SE_ANTI_STALL ([2944385](https://github.com/project-david-ai/platform/commit/294438511325054491b9ef59c1f1b4adbc2e9248))
* enforce scroll limits and search-first gate to prevent doom-scrolling ([015eca9](https://github.com/project-david-ai/platform/commit/015eca963288ec8a43650e57649fce9aaf5fa622))
* Engineering instructions locked down! ([44acb05](https://github.com/project-david-ai/platform/commit/44acb05da24fb510570a7ba3c2ebf3d1ff1d45f7))
* Engineering instructions locked down! ([d59e6c9](https://github.com/project-david-ai/platform/commit/d59e6c95f3e407d53f07f195d84b7164c1cae573))
* enhance Level 3 recursion and fix web state MRO issues ([ff25f66](https://github.com/project-david-ai/platform/commit/ff25f668414fb7e9e3302a716d81495c4faf6c9c))
* ensure `user_id` is passed in thread operations for ownership validation ([46be1b5](https://github.com/project-david-ai/platform/commit/46be1b542a89727ed49b19c258d638cd22fca568))
* filter  "<|channel|>analysis" ([a5dd32a](https://github.com/project-david-ai/platform/commit/a5dd32a57fd3a879b1774ce62ea170ac54f1d0c7))
* Fix computer tool, crashing the stream issue ([8c60c81](https://github.com/project-david-ai/platform/commit/8c60c811f357d4b5b0866c1688d2332691246c8f))
* fix(files): resolve signed url download 401 error and path duplication ([1d3ae6d](https://github.com/project-david-ai/platform/commit/1d3ae6d1b1f4a5e6fe56b450b8e342aaf4885f66))
* function call response issues ([2b6086f](https://github.com/project-david-ai/platform/commit/2b6086f88479a43a00d8c451b838e7c7687cb30b))
* function calls correctly set and parsed in gpt oss ([8b276d7](https://github.com/project-david-ai/platform/commit/8b276d702350e3c525e3a554250ff711f4353b82))
* gpt-oss function call 2nd turn issue ([524b693](https://github.com/project-david-ai/platform/commit/524b6938d594c297ce87d8f5901d04b56b621b83))
* GPT-oss level 3 compliant. ([d9e4b24](https://github.com/project-david-ai/platform/commit/d9e4b2475acdc6be5c873b206b6696eca7d85d4c))
* GPT-oss level 3 compliant. ([2872954](https://github.com/project-david-ai/platform/commit/28729544fda3b2f09377e6926a75f1e577742c72))
* GPT-oss level 3 compliant. ([c01a740](https://github.com/project-david-ai/platform/commit/c01a740b80dade3131900ba46f6cf3a622297e84))
* **gpt-oss-worker:** preserve ephemeral supervisor identity during DB persistence ([56d54e9](https://github.com/project-david-ai/platform/commit/56d54e92fdc4fa43a540df105fe5af55fb88b89e))
* **hermes-worker:** preserve ephemeral supervisor identity during DB persistence ([dc0ef86](https://github.com/project-david-ai/platform/commit/dc0ef86282715c71ead4b50ab8d13d840a781355))
* housekeeping daemons for expired runs and thread cleanup ([9ea441f](https://github.com/project-david-ai/platform/commit/9ea441f85af9d299b570d97c2fb2ddda30704f72))
* Implement cache invalidation for assistants cached_assistant ([7f4d3c5](https://github.com/project-david-ai/platform/commit/7f4d3c52d80b00844a8f07a12c9d7901b190ff2e))
* Implement DecisionEvent type ([1992583](https://github.com/project-david-ai/platform/commit/1992583e424f22a2170e9bd9075d0345d5786a7d))
* Implement dotenv for secrets in alembic .env.py ([a437017](https://github.com/project-david-ai/platform/commit/a4370176c279f29140d6f742c523b4045c018332))
* Implement Hermes style function call dialogue structuring for DeepCognito models ([2a28d79](https://github.com/project-david-ai/platform/commit/2a28d794acff457e40500d3cd7dbdb5b9ce9bb06))
* Implement Hermes style function call dialogue structuring for DeepCognito models ([3ea46d8](https://github.com/project-david-ai/platform/commit/3ea46d885749e43c1b9a06c8e6875f06979e96e5))
* Implement level 2 recovery for platform tools. ([ca48507](https://github.com/project-david-ai/platform/commit/ca4850760d1b42312d7bd02fcb04f58a2691115d))
* implement level 3 instructions_assembly.py ([0f24f94](https://github.com/project-david-ai/platform/commit/0f24f947da6bc523f605ca025b1dd81a306e5a0a))
* Implement native tool call response for gpt-oss Hermes class models. ([6846044](https://github.com/project-david-ai/platform/commit/684604440b5f6692b47f610cf411d88adf41f324))
* Implement native tool call response for llama. ([610b4ed](https://github.com/project-david-ai/platform/commit/610b4edaf336c8cd99198a8e393e6cbfca7f3fc0))
* implement native tool calling on Quen Model calls. ([6347d12](https://github.com/project-david-ai/platform/commit/6347d12ebcf912c9348ee142f76f4af43dc2d823))
* Implement new action state signalling. Previously the client end had to use ActionService.get_pending_actions in a constant poll from the client side before the client new that a run had an action to service. This was cumbersome , slow, and caused needless churn on the DB. The new method propagates the action.id direct to the client side via a manifest payload as soon as the action is created. ([14c1066](https://github.com/project-david-ai/platform/commit/14c1066abd684461ebb0ac00a5ca918560ef57e6))
* implement new architecture for hot_code. Complex parsing and replay from each worker is needless when hot_code can be replayed direct from the handler. ([e49ea75](https://github.com/project-david-ai/platform/commit/e49ea75e0987b05052a2b286b0339a28726708ea))
* implement new instruction hirarchy for  _build_system_message ([fafd1c5](https://github.com/project-david-ai/platform/commit/fafd1c527de70e66512ef50e8dbe6b53e984ad8c))
* implement new instruction hirarchy for  _build_system_message ([5c01a9d](https://github.com/project-david-ai/platform/commit/5c01a9db1834be053af0d7807dd7920d2b509eb2))
* Implement new scripts ([2ba916a](https://github.com/project-david-ai/platform/commit/2ba916a961ce592ab053ff3cf526852d005a1d4b))
* implement process_hot_code_buffer in CodeExecutionMixin. ([e7d5db7](https://github.com/project-david-ai/platform/commit/e7d5db769e0bd5531efadb02c7123c654b438004))
* implement Senior/Junior agent duo with full inventory resolution ([d84cf6b](https://github.com/project-david-ai/platform/commit/d84cf6bd287ea716fb7d75afe14bee17cb921d60))
* Implement serp search tool. ([4c3d5aa](https://github.com/project-david-ai/platform/commit/4c3d5aafc78e9b9b62c6db58ce1fc8dfd5e1853f))
* Implement structured tool call detection for gpt-oss ([4c6934d](https://github.com/project-david-ai/platform/commit/4c6934d78e88f95846ed811d6821f198dba29d3c))
* Implemented the Stream method as a concreate class in the base orchestrator_core.py. Should greatly increase the speed of model onboarding! ([f775bc0](https://github.com/project-david-ai/platform/commit/f775bc0634adbd05b5b21b747f1789455cdf4995))
* import PLATFORM_TOOLS from common library ([89dd4c0](https://github.com/project-david-ai/platform/commit/89dd4c0d03094373b511ec4e303c8750705d451b))
* Improve code intepreter file generation instructions. ([7b5f18a](https://github.com/project-david-ai/platform/commit/7b5f18a3387cf9583e7a35ecfbf6a1b8f13a9f53))
* improve exception handling and cleanup redundant code ([9987132](https://github.com/project-david-ai/platform/commit/998713240785e4ef85762165b730b0cb786cead4))
* Improve speed and smoothness of streaming. ([0e16437](https://github.com/project-david-ai/platform/commit/0e164378fdc5722ee19f75319a3b32d47c27ee30))
* Improve speed and smoothness of streaming. ([b3e9191](https://github.com/project-david-ai/platform/commit/b3e919132480bea23337161e93dca7484c94ee77))
* improve variable naming in exception handling logic in `fix_ruff.py` ([6f89aa5](https://github.com/project-david-ai/platform/commit/6f89aa5891afebb3a1ed0af98e423e3f531eac47))
* Integrate save dll rendering functions ino  alembic . We no longer have to manually refactor auto generated alembic scripts. ([0516264](https://github.com/project-david-ai/platform/commit/0516264243d5467a1ed61184827a565bdf058f1b))
* issue with hyperbolic/DeepSeek data flow ([4b7a0e5](https://github.com/project-david-ai/platform/commit/4b7a0e54859e15cf95fa1544952780ac7a189839))
* issue with oss reasoning chunks ([9c4e88f](https://github.com/project-david-ai/platform/commit/9c4e88f02946f9ca65477dcbad1a358f2d211ea3))
* issue with oss reasoning chunks ([d228581](https://github.com/project-david-ai/platform/commit/d228581cc001e152bd81ba71949a404cac5ccd47))
* JsonUtilsMixin update. ([c73cd3f](https://github.com/project-david-ai/platform/commit/c73cd3f1bd8eae46f7c8bd3c43f3d227a599066d))
* Major speed improvements in inference. ([dc06f29](https://github.com/project-david-ai/platform/commit/dc06f294b0ae9d68bc5733053db26eec660152c1))
* Make assistant.tools as the source of truth for the assistants tools array ([fd05c4e](https://github.com/project-david-ai/platform/commit/fd05c4e00f0a8cba161ccdce979c85cec8e7c8c7))
* Migrate all TogetherAI workers to unifed asynch client ([9f088e4](https://github.com/project-david-ai/platform/commit/9f088e4cca93ddac3c180773f2c20d6df552fe9e))
* Migrate all workers to decision telemetry algorithm. ([bb29acb](https://github.com/project-david-ai/platform/commit/bb29acbac15a77d4b97f3034b12ee56b0ac24a07))
* Migrate b697008df93a ([4c12971](https://github.com/project-david-ai/platform/commit/4c12971fe3aa882ebd8fef19439f749130baba77))
* Migrate DB dda6fd28f45c ([4e65be6](https://github.com/project-david-ai/platform/commit/4e65be6dc816b87cbd973a0e6bf2c07ce91bd005))
* Migrate deepseek_base.py to asynchronous mode. ([5205f0a](https://github.com/project-david-ai/platform/commit/5205f0a6f6a9e621d0d7ad9ed1bb227a4b43a5fd))
* Migrate deepseek_base.py to asynchronous mode. ([5fdc74b](https://github.com/project-david-ai/platform/commit/5fdc74bb8401127ea83c25ab725883477a7f06e1))
* Migrate default_base.py.py to asynchronous mode. ([e878fb6](https://github.com/project-david-ai/platform/commit/e878fb6505eb89594e7112c7153880e46ca188d3))
* Migrate GPT-oss worker to home brew client, works! ([ae862c8](https://github.com/project-david-ai/platform/commit/ae862c8aa4149a77e6931ef50988d32b91eb47c7))
* Migrate hermes_type_default_base.py to asynchronous mode. ([82ed344](https://github.com/project-david-ai/platform/commit/82ed3442c3f34d8c2ff23f495f18afafa64ac086))
* Migrate nvidia_base.py.py to asynchronous mode. ([abf1171](https://github.com/project-david-ai/platform/commit/abf117115e5074b6b4aa2124d54a750bd882b1f6))
* migrate quen_base.py to the new function call architecture ([f1c9bae](https://github.com/project-david-ai/platform/commit/f1c9bae342af9b4fdbb7138ced4087a5cb0ccee4))
* Migrate qwen_base.py to asynchronous mode. ([0bc71e9](https://github.com/project-david-ai/platform/commit/0bc71e95cf4246c655834a1c90b4c153d0ec53d9))
* Migrate service_now_base.py to asynchronous mode. ([baeb2ea](https://github.com/project-david-ai/platform/commit/baeb2ead8c214ede60944189f443b179dbfd10fa))
* Migrate to json status events ([21cb9cb](https://github.com/project-david-ai/platform/commit/21cb9cb9f7b123b232a76805be7d0476ab06a1ee))
* migrate to native execution and fix cache initialization ([2ef44f8](https://github.com/project-david-ai/platform/commit/2ef44f8f38b91b49e40f5f0e7afb4971f1183808))
* Migrate to new base worker arhitecture ([69f0b02](https://github.com/project-david-ai/platform/commit/69f0b024a41fe48855e65ad7a4cdbed9e5853494))
* Migrate to new message cache architecture ([435bdd9](https://github.com/project-david-ai/platform/commit/435bdd961e15e7746d1e7fe7eece70d38f22ff10))
* Migrate to new singleton client ([150ecee](https://github.com/project-david-ai/platform/commit/150ecee94882aa5fff29ad87d7f30fc6f0022d64))
* Migrate to new singleton client ([13713d4](https://github.com/project-david-ai/platform/commit/13713d4d23822a52aa2cf7e356961b6db465cff1))
* migrate to projectdavid client 1.67.1 ([f803de5](https://github.com/project-david-ai/platform/commit/f803de508dbd91cdea2b6b3a98479337e564f0ea))
* Migrate to projectdavid v1.42.0 ([eaa467c](https://github.com/project-david-ai/platform/commit/eaa467cba691e084262d45d7bb4604ac1744269f))
* Migrate to projectdavid v1.42.0 ([eabb4f3](https://github.com/project-david-ai/platform/commit/eabb4f3b9465eeaf41e5f5829c078b358ba078f3))
* Migrate to projectdavid==1.49.1 ([e603321](https://github.com/project-david-ai/platform/commit/e603321ec25244192b42d4ec28b09dcc87857da5))
* migrate tool mixins to native execution service ([02d75f2](https://github.com/project-david-ai/platform/commit/02d75f28bf250501525d4be886731b7bcfb81769))
* migrate web tool execution to NativeExecutionService ([e3fde1b](https://github.com/project-david-ai/platform/commit/e3fde1bf19692a3e6d1523f5585a30aba727caf7))
* Move assistant cache setup to core Class ([52d4be8](https://github.com/project-david-ai/platform/commit/52d4be806c506a22a6cb377b9c99b70057319f9c))
* Move assistant cache setup to core Class ([de54203](https://github.com/project-david-ai/platform/commit/de542035b2f66f7a3f949fc8164293aef37e1eec))
* Move AsyncHyperbolicClient to client factory. ([9ba074c](https://github.com/project-david-ai/platform/commit/9ba074c33925aeed4a5b713c11166f599bdfe16a))
* move cache_utils.py and cached_assistant.py --> /cache ([11e435a](https://github.com/project-david-ai/platform/commit/11e435a6d3ef94a0ad9e1cb4e98a58ede219a58e))
* Move unified_async_client.py to /clients package ([726d73d](https://github.com/project-david-ai/platform/commit/726d73dc08a765dce2a928e05c05c03bc25bee41))
* Once and ephemeral supervisor has run, the assistants cache must be cleared and reinitiated so  that the latest persona is the cache to avoid context and tool contamination. ([b85cce2](https://github.com/project-david-ai/platform/commit/b85cce288877e5a0fd106cd5ac397384ab46a194))
* order senior Engineers tools. ([8d51d4a](https://github.com/project-david-ai/platform/commit/8d51d4a0200da53f44c8687161e2ffb11cded40d))
* orphaned-thread purge daemon for GDPR compliance ([1bb0157](https://github.com/project-david-ai/platform/commit/1bb0157c4b8e5fd5a4a5936b04ce588cf4a0795c))
* pass `user_id` for ownership validation in run operations ([49a090f](https://github.com/project-david-ai/platform/commit/49a090f90760a857e97375178af206af0328f33d))
* pass snapshot_id to refresh_snapshot, not snapshot_name ([8d7c2e3](https://github.com/project-david-ai/platform/commit/8d7c2e3ac5d9b8bdfda3e1d4ed25cca0e4c0fce0))
* phantom file generation issue in CodeExecutionMixin ([f3d7f44](https://github.com/project-david-ai/platform/commit/f3d7f44cb526e7e78537d3639170ef289f7ea45b))
* pin tempfile output dir in CODE_FILE_HANDLING system instruction ([f49f5cf](https://github.com/project-david-ai/platform/commit/f49f5cf4672263f013e28f30a975fdff9d548e6d))
* preserve ephemeral supervisor identity during DB persistence ([6debbf9](https://github.com/project-david-ai/platform/commit/6debbf90e7e9b8f0a072ccffe344921b2fcb6bf6))
* Push ac1498a9642c_remove_tools_table_and_associations.py ([495fa01](https://github.com/project-david-ai/platform/commit/495fa012217203e6e8e626a647ea8e14f629da97))
* record_tool_decision.py ([e15cf1e](https://github.com/project-david-ai/platform/commit/e15cf1e76bf6742c173d278562a7715218f22120))
* refactor _resolve_and_prioritize_platform_tools with new silent tool mandatory_platform_tools ([fdeb54c](https://github.com/project-david-ai/platform/commit/fdeb54cbbdb17129fe2c4e97b3e7868da669fb39))
* Reinstated partial code_interpreter instructions to _build_native_tools_system_message ([694a3fa](https://github.com/project-david-ai/platform/commit/694a3fa6343901f80b94afeb433deafc5bd1d722))
* remove `model_compatibility_report.md` and bump `projectdavid` to v1.74.7 ([a8423bf](https://github.com/project-david-ai/platform/commit/a8423bf47e7ace75b7bacab6e426b86ac200ce1f))
* remove `vector_store_assistants` relationship and update `messages` constraints ([afb9c54](https://github.com/project-david-ai/platform/commit/afb9c54dccbb9da995ab12fe57e0acc931d9e9d1))
* remove assistant_id from engineer router endpoints ([7e66173](https://github.com/project-david-ai/platform/commit/7e6617397905f8c68d5ff7a4910621eeb18a72dc))
* remove assistant_id from engineer router endpoints ([9d7d708](https://github.com/project-david-ai/platform/commit/9d7d70802d117936970a657653fb31b3b8537108))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([4358043](https://github.com/project-david-ai/platform/commit/4358043993d81973f5d63f78ad689d2fb0a29b2a))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([937b704](https://github.com/project-david-ai/platform/commit/937b7042dee1feeae2073e094461407fb1f4e55d))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([2f58759](https://github.com/project-david-ai/platform/commit/2f58759c497249b40d62e6ebbe8e64bd52340810))
* remove Codecov step from CI workflow and simplify Pytest coverage configuration ([3ac11e3](https://github.com/project-david-ai/platform/commit/3ac11e3afa4ff54ca2058328702ac821ece800e6))
* Remove device inventory instructions from SE_TRIAGE_PROTOCOL ([45bca41](https://github.com/project-david-ai/platform/commit/45bca418199e326ae4528b56753659700df4cceb))
* Remove device inventory tools from the senior engineers tool registry. ([b1b7c00](https://github.com/project-david-ai/platform/commit/b1b7c00a3398b2b2d9968d29968ec4a396526a2a))
* Remove duplicate api arg from worker signature. ([e5f1356](https://github.com/project-david-ai/platform/commit/e5f1356662e78002edadcb314a6c71ab224da458))
* Remove hot_code handling from Quen worker! ([447b8a6](https://github.com/project-david-ai/platform/commit/447b8a648f259a8d0445b292d208473905e7a895))
* Remove local curl searches from main api ([336c4f6](https://github.com/project-david-ai/platform/commit/336c4f67d12acaf00652675a034542baf6651cd7))
* Remove Tools table from models.py ([f073286](https://github.com/project-david-ai/platform/commit/f073286592196efc8cd33d7f73a9c06c13845a5d))
* replace bare `except` with `except Exception` and clean up redundant imports ([279fe56](https://github.com/project-david-ai/platform/commit/279fe56b2a51c6c543ce7d71efc24da7b09dba07))
* replace unattended_file_search with _search_vs_async ([1b816d5](https://github.com/project-david-ai/platform/commit/1b816d50746101abb49ca7ce6018b953ec5a654f))
* resolve 500 error on engineer inventory ingest ([d87c744](https://github.com/project-david-ai/platform/commit/d87c7444b6854ecdb5d97edd46eb4a30c486c820))
* resolve 500 error on engineer inventory ingest ([fdd33ae](https://github.com/project-david-ai/platform/commit/fdd33ae645951cf3ea8cd58b8a2dcb3741f71dfa))
* resolve all deep search issues ([e354cfc](https://github.com/project-david-ai/platform/commit/e354cfc828c538a1468ba9817a43c08a7668c99d))
* resolve batfish snapshot ownership across Senior/Junior worker boundary ([c4b252d](https://github.com/project-david-ai/platform/commit/c4b252d4bce8577e13793babfb77eb05125717ce))
* resolve batfish snapshot ownership across Senior/Junior worker boundary ([3613180](https://github.com/project-david-ai/platform/commit/3613180a82bb262a67e4a53d4deebce281f9f09d))
* resolve batfish snapshot ownership across Senior/Junior worker boundary ([eecbab0](https://github.com/project-david-ai/platform/commit/eecbab0590aee7acf600f602d9498fb813b2024e))
* Resolve computer tool auth token issues. ([a9c2074](https://github.com/project-david-ai/platform/commit/a9c20744a4344c3c8fdd9d9daabd9bce65cb78c3))
* Resolve computer tool auth token issues. ([8426ac7](https://github.com/project-david-ai/platform/commit/8426ac7bb1a8227217b9a86583a431af09c7cefd))
* resolve computer tool dispatch and SDK action call regressions ([777c391](https://github.com/project-david-ai/platform/commit/777c391deed0032a2a18b639d97da679d3b3386f))
* resolve computer tool dispatch and SDK action call regressions ([85604c6](https://github.com/project-david-ai/platform/commit/85604c66bc480d4ecad8a70b1fea8387140e96d1))
* Resolve deepseek_base.py  and child issues. ([153d073](https://github.com/project-david-ai/platform/commit/153d073096210342ae9c2e07118949a89a845f9e))
* Resolve forward stream delay issue ([09deac8](https://github.com/project-david-ai/platform/commit/09deac88058cb4d7e0b4e819d5a9029f29604195))
* Resolve GPT oss streaming issue by creating an instance of  _get_client_instance, not OpenAI ([ab14433](https://github.com/project-david-ai/platform/commit/ab14433528d2a56d15ae700f6da7faf857c6a9ca))
* Resolve gpt-oss-120b reasoning tokens not generating issue. ([a78e11c](https://github.com/project-david-ai/platform/commit/a78e11c325394f5fb3ff7ae1e67c4377d691aa4e))
* Resolve gpt-oss-120b reasoning tokens not generating issue. ([90bc1c4](https://github.com/project-david-ai/platform/commit/90bc1c4f6463cf8bb37ceb6b466bd26506b86277))
* Resolve handover issues ([297dcee](https://github.com/project-david-ai/platform/commit/297dcee3e2cd24394bfde5a60c1050220b1dfe11))
* Resolve hyperbolic/meta-llama- models function calling issues. Previously, the model did not respond to the first turn. This was due to passing in the tool call array in the request payload. ([3e19625](https://github.com/project-david-ai/platform/commit/3e196257f5aa230682cb9ef915b45f13a5adf591))
* resolve identity swap failures, worker flag extraction, and SDK reliance ([322cceb](https://github.com/project-david-ai/platform/commit/322cceba85c66159ca97866a4732686df9ecb94c))
* Resolve issues with fc call polling. ([a85af1d](https://github.com/project-david-ai/platform/commit/a85af1dbe5ad6b4f38a42e8d23877fda3d337992))
* Resolve ongoing issues with consumer side function call handling race condition ([32826ea](https://github.com/project-david-ai/platform/commit/32826ead80416914b311c73e42e6d7fa76f6eac7))
* Resolve ongoing issues with consumer side function call handling race condition ([f2e6e88](https://github.com/project-david-ai/platform/commit/f2e6e886cefe04f471c52c0ed697bbe2784908c6))
* Resolve the issue contaminating the research supervisors context with unwanted tools ([07df062](https://github.com/project-david-ai/platform/commit/07df062dd29f28a72f4944fe50409fe45946984e))
* Resolve Threads relationship issue in models.py ([2943b62](https://github.com/project-david-ai/platform/commit/2943b625105895e214b1e0bf34acb81c4e399091))
* restore assistant_id ownership to process_conversation ([bc91e55](https://github.com/project-david-ai/platform/commit/bc91e55d43d4754ec80c366c90847dfde42a9ab4))
* Restore gpt-oss hot code interleave. ([9d00699](https://github.com/project-david-ai/platform/commit/9d00699a21e140c47ee0c88a746b4ff512fdf206))
* Restore real time streaming of research workers output ([8c73d6b](https://github.com/project-david-ai/platform/commit/8c73d6b501a979eb3faccbd149c696e5a62347b5))
* resurrect soft-deleted snapshots on create instead of raising 409 ([8c2abaf](https://github.com/project-david-ai/platform/commit/8c2abafa3189a4bd745b70ead03caa9a359db99e))
* Run model intake reports ([cc149e7](https://github.com/project-david-ai/platform/commit/cc149e7cc60be49dfe9894101a4f4abd6a6c94be))
* Run model intake reports ([dfeb5ab](https://github.com/project-david-ai/platform/commit/dfeb5ab9e96af1d61a812a57c9d2071c7d45a467))
* smarter Turn 2 prompt + anti-loop failure handling ([6e9dfc3](https://github.com/project-david-ai/platform/commit/6e9dfc378c4a1c683d5b460a560a370273a80f56))
* stamp scratchpad thread id in the junior research assistants run object. ([a5de48b](https://github.com/project-david-ai/platform/commit/a5de48bc8d59424fc853e1273d201d7f4332e653))
* standardise WebEvent emission across backend, SDK, and frontend ([f92cbb4](https://github.com/project-david-ai/platform/commit/f92cbb4e35dd36000ceb7a1d9e7c69f8818ef261))
* streamline vector store endpoints and remove deprecated assistant relations ([7673c76](https://github.com/project-david-ai/platform/commit/7673c76972518d98232e38a9f91e12f0acf2968d))
* streamline worker delegation logic and improve stream handling ([8e4413f](https://github.com/project-david-ai/platform/commit/8e4413f9b31e538d386c9b1c3539d5520aef12c8))
* strengthen ownership guards in inference router ([63c3c8d](https://github.com/project-david-ai/platform/commit/63c3c8dde2fafe62818040b661d53c053fd0cbe6))
* strictly enforce Change Request formatting and ban conversational preambles ([65adabb](https://github.com/project-david-ai/platform/commit/65adabb308d980b68c2aa36c49b1447af2ac48fa))
* successfully ported deepseek_base.py ([c8d444d](https://github.com/project-david-ai/platform/commit/c8d444dbe576e8da829d392636f38139ce7c1541))
* support dynamic `ollama_base_url` via metadata in orchestration worker ([7d3bc4c](https://github.com/project-david-ai/platform/commit/7d3bc4c7101b0d84df3e90f20c2e935808706e55))
* synchronize and alphabetize mixin imports and inheritance ([e9eba55](https://github.com/project-david-ai/platform/commit/e9eba5556b9dedbd54a948f472e3a4017615af28))
* synchronize and alphabetize mixin imports and inheritance ([70d8aaf](https://github.com/project-david-ai/platform/commit/70d8aafc23d0cc214e4498b87c5fa1277616bba8))
* synchronize worker classes with Qwen gold standard logic ([678b13c](https://github.com/project-david-ai/platform/commit/678b13c335f0e6f7853a2e1353ce101afc253b8f))
* Temporarily remove status event messaging from shell_execution_mixin.py ([0a2c406](https://github.com/project-david-ai/platform/commit/0a2c4068b02171069004d28f68aa9c8146612e3b))
* Temporarily remove status event messaging from shell_execution_mixin.py ([48f2702](https://github.com/project-david-ai/platform/commit/48f2702709f3d3b185c33556532f4fdb9a49bcea))
* Temporarily remove status event messaging from shell_execution_mixin.py ([81fda55](https://github.com/project-david-ai/platform/commit/81fda5534a188dbe3be66fbe273750a966c090fa))
* The JE and SE are hallucinating solutions, update instruction set ([486e5cd](https://github.com/project-david-ai/platform/commit/486e5cdba57d18c895398dfe0ed25ec1e1175b99))
* TogetherAI level 3 tests. ([48fdff4](https://github.com/project-david-ai/platform/commit/48fdff43d2aa4064634a20247e1771023ebc0780))
* TogetherAI level 3 tests. ([371569a](https://github.com/project-david-ai/platform/commit/371569a3cb0613144d4de95663a510140ff8e1c6))
* tool role tool_call_id bug ([502788e](https://github.com/project-david-ai/platform/commit/502788e7d17b4e74ac3f2aa35d654ec9a4f412fa))
* Tweak unified_inference_test.py ([cdb38a5](https://github.com/project-david-ai/platform/commit/cdb38a500b246b55afe42fd3ecf16e89fb06ce9e))
* Update client to projectdavid==1.39.4 ([af886ff](https://github.com/project-david-ai/platform/commit/af886ffbbdb7e84c9d9b988f5657c5d06291fda6))
* update client to projectdavid==1.51.2 ([4c21d26](https://github.com/project-david-ai/platform/commit/4c21d261619ecd547c7fce049c734286ce4924ce))
* update client to projectdavid==1.51.2 ([ee0554a](https://github.com/project-david-ai/platform/commit/ee0554aa8487ed200a5bdc216f11c9b9cb333ed1))
* update client to projectdavid==1.60.0 ([ea9efc9](https://github.com/project-david-ai/platform/commit/ea9efc912f19f7b2faa80259c9ce7ca3806fb5b1))
* update client to projectdavid==1.73.0 ([69803ee](https://github.com/project-david-ai/platform/commit/69803ee3a225503e4b6bfe1f92de65ea2957149e))
* Update delta_normalizer.py with level 3 planning protocol tags ([2e3edd2](https://github.com/project-david-ai/platform/commit/2e3edd2184f91e0d758524211a6ac9e077b55194))
* update GitHub Actions workflows to use latest action versions and add placeholder test file ([53846af](https://github.com/project-david-ai/platform/commit/53846afc90709e4e51d911f0b6ab613a66e0c20d))
* update integration test scripts. ([cd566c9](https://github.com/project-david-ai/platform/commit/cd566c98cf236a75316e3756a946f6a8064fb281))
* Update process_conversation with decision state handling. ([f38482f](https://github.com/project-david-ai/platform/commit/f38482f7a9e9de101d878d560c756851ff31b99d))
* update projectdavid client to : 1.59.0 ([8c53820](https://github.com/project-david-ai/platform/commit/8c538200fee0c85a293713341772860019a8163b))
* Update projectdavid client to projectdavid==1.66.0 ([c1387c8](https://github.com/project-david-ai/platform/commit/c1387c85a835f675481bec4763fb94f9ae8ce409))
* Update projectdavid client to projectdavid==1.73.1 ([5f4100b](https://github.com/project-david-ai/platform/commit/5f4100b40a169d062704f33c296f199a301807dc))
* Update projectdavid sdk to projectdavid==1.47.5 ([6149b5f](https://github.com/project-david-ai/platform/commit/6149b5f1891570228d95f8653931d8f4de00b4a5))
* update to projectdavid==1.54.4 ([a5b0619](https://github.com/project-david-ai/platform/commit/a5b0619a4cbb0182bc4619ea0fa0050eb8baf32e))
* update to projectdavid==1.60.3 ([aef1a1e](https://github.com/project-david-ai/platform/commit/aef1a1e256a00967d20e018325e80268885186d0))
* Upgrade client to projectdavid==1.39.6 ([c4b1f9f](https://github.com/project-david-ai/platform/commit/c4b1f9febecb69f17f77f10298918f19fbdab97e))
* Upgrade client to projectdavid==1.39.6 ([bb48ceb](https://github.com/project-david-ai/platform/commit/bb48ceb463ad0c6f692681b3cb381f3dd08ed895))
* Upgrade projectdavid 1.53.0 ([091a83b](https://github.com/project-david-ai/platform/commit/091a83b21a06eb58af9fab1d4cda945c163adeb1))
* use self.assistant_id in process_tool_calls dispatch to reflect post-swap identity ([255098f](https://github.com/project-david-ai/platform/commit/255098f5563b0c9def2f4d92f0b18ae1fd2db833))
* use self.assistant_id in process_tool_calls dispatch to reflect post-swap identity ([8c41a8a](https://github.com/project-david-ai/platform/commit/8c41a8a7b1d5b58a663ae8b26fa1e2148e8ffbe0))
* We must explicitly set the cwd (Current Working Directory) of the Python subprocess to self.generated_files_dir. ([9fb5991](https://github.com/project-david-ai/platform/commit/9fb59917970d319324e934cfd215573c983e7073))
* web_search working. ([04728be](https://github.com/project-david-ai/platform/commit/04728bed82a711b9671e18f6177842ed9a5b80e3))
* web_search working. ([3dfd3e3](https://github.com/project-david-ai/platform/commit/3dfd3e3af9b691a5c6c5ce7db219d53d1de35574))
* web_search working. ([b0b9afd](https://github.com/project-david-ai/platform/commit/b0b9afd73509214e8c3fbec4e3250de372445db6))


### Features

* _get_enriched_topology ([0382732](https://github.com/project-david-ai/platform/commit/038273272c8e000bb106e5ae23a0f6fae64143cb))
* 9314d4058f78_add_agentic_state_columns_to_assistant_.py adds some level 3 agentic components to the master DB ([1c8d684](https://github.com/project-david-ai/platform/commit/1c8d684f4f930b272418a76047e70145c04a970f))
* Add  get_enriched_topology endpoint ([433d300](https://github.com/project-david-ai/platform/commit/433d3006078e8042eb158b0bd996f1ccc37e0b26))
* Add  get_enriched_topology endpoint ([b3796da](https://github.com/project-david-ai/platform/commit/b3796da3d001516ecf5f4c922ebe83133accc686))
* add `owner_id` to `Thread` table and enhance thread management ([242bb36](https://github.com/project-david-ai/platform/commit/242bb36aad05b2fa0d2fb32736c267da8d48d2cb))
* add assistant ownership validation to `completions` endpoint ([aa6f2be](https://github.com/project-david-ai/platform/commit/aa6f2bec8ae7a56fe8018213c11956f6ea60bf37))
* add BatfishSnapshot to models ([8f22a13](https://github.com/project-david-ai/platform/commit/8f22a13b215f42d7e58031ae18b0ecbcc897930a))
* Add engineer flag ([18f2a0b](https://github.com/project-david-ai/platform/commit/18f2a0b79dad4d44e23d10ef495f633a1b91e118))
* Add Engineers to context build logic ([09863ea](https://github.com/project-david-ai/platform/commit/09863eafdeeb2439755692a0a4ae22eb82413a4c))
* add hard-purge daemons for soft-deleted files and vector stores ([7c12eea](https://github.com/project-david-ai/platform/commit/7c12eea89e1cb939d1353340a6d440f07134179f))
* add owner_id to assistants with SafeDDL guards ([7591325](https://github.com/project-david-ai/platform/commit/7591325c942c254296bb3ff1a177aa88e639e5db))
* add pytest options and ignore integration tests ([040ba83](https://github.com/project-david-ai/platform/commit/040ba839676e56a2ac7429c63f8e58930bfca0b2))
* add run_batfish_tool and run_all_batfish_tools to tool router ([576c2a3](https://github.com/project-david-ai/platform/commit/576c2a3ab474b2b69063822d3b0c5f4cb6fcfa98))
* add tool_resources to cached assistant payload ([54f0dd4](https://github.com/project-david-ai/platform/commit/54f0dd44d830df54439a56c8a680f2ecaeeb046f))
* add viewer support for shell sessions and enhance file harvest logic ([6e53b60](https://github.com/project-david-ai/platform/commit/6e53b6076a7ad9b55162266d2a6c31e429143dc3))
* add vLLM orchestration support and documentation ([c133522](https://github.com/project-david-ai/platform/commit/c1335229d9fffd215ead8c8bfdf6edb35e850110))
* batfish_router.py ([a231f10](https://github.com/project-david-ai/platform/commit/a231f10e655c015ff4daa62dc837fdb246cd99b7))
* batfish_router.py ([0f96559](https://github.com/project-david-ai/platform/commit/0f96559475f879a02a353ba6bcdecdb9f0987055))
* complete orchestration alignment and stream_sync implementation ([8cb5728](https://github.com/project-david-ai/platform/commit/8cb5728457a0fba29188ec28dde314507b6dd963))
* cut over to projectdavid==1.39.0: bringing new models online: ([5a9ed31](https://github.com/project-david-ai/platform/commit/5a9ed3183a5244e40826a422ba67b117a9e53c6e))
* deep research ([6a56cc5](https://github.com/project-david-ai/platform/commit/6a56cc5dfe4196eea0f926206301fa20cffa26dc))
* DeepSearch paradigm 2 ([28403bf](https://github.com/project-david-ai/platform/commit/28403bf7d9f8e681f8b8ba12eb403359035f9059))
* **default-worker:** implement async stream_sync wrapper and queue isolation ([65fd483](https://github.com/project-david-ai/platform/commit/65fd483e3e44b1aaf5b4cb0f3a66ca1aa458150e))
* docker compose exec api alembic revision --autogenerate -m "add ID to BatfishSnapshot to models" ([6573524](https://github.com/project-david-ai/platform/commit/6573524a328114ba2633addceaa6e6bfb5f08b09))
* enforce ownership validation for thread operations ([1c81ca9](https://github.com/project-david-ai/platform/commit/1c81ca9bafc4841f45b4838e6ddf8511d564819d))
* enforce secure file generation with mandatory tempfile.tempdir and path confinement ([4d522d6](https://github.com/project-david-ai/platform/commit/4d522d62bcf09b689741dcc5e67e78a56fb61403))
* **engineer:** bridge client and service layers for inventory tools ([fc20e8a](https://github.com/project-david-ai/platform/commit/fc20e8a198b1884ad8940e273cefd500d737476f))
* enhance Docker service management with --exclude flag and improve exception handling ([5897dde](https://github.com/project-david-ai/platform/commit/5897dde4fd8d28495cfa655bb971a848e4d181f4))
* extend `docker-compose.dev.example.yml` with Redis, browser, SearxNG, Jaeger, Otel Collector, and Ollama services ([afa7dc3](https://github.com/project-david-ai/platform/commit/afa7dc34514482648f7fea3a97fb9b4bd6f4ca3c))
* Extend DelegationMixin for Engineering delegation. ([afc3363](https://github.com/project-david-ai/platform/commit/afc3363f60f19cc41dae41e8674eb3cf9a966868))
* fan out across tool_resources vector stores ([7f82a17](https://github.com/project-david-ai/platform/commit/7f82a175307346a8da71b36764274f821850ed20))
* implement auto-cleanup of ephemeral files after streaming ([ba48f7c](https://github.com/project-david-ai/platform/commit/ba48f7c8bc984f4bd1e978317455ba5ea6b662bb))
* implement auto-cleanup of ephemeral files after streaming ([d17465e](https://github.com/project-david-ai/platform/commit/d17465e6da9432ef1a2074ec06c2bd4321d29746))
* implement delegated deep search model mapping ([10e27a0](https://github.com/project-david-ai/platform/commit/10e27a0b7c99a98dec39a388bb4a847c55f786ef))
* Implement Engineering tool defs ([54d1a0a](https://github.com/project-david-ai/platform/commit/54d1a0a1fd11139dcefed0ff563c5c5254849f48))
* implement expired file cleanup utility and daemon ([5584c42](https://github.com/project-david-ai/platform/commit/5584c4202e3af581d653eaaf9159f4b785c72f56))
* Implement full-stack real-time Scratchpad visualization ([decb876](https://github.com/project-david-ai/platform/commit/decb87678e439f2e599f64763bd807e5b755ba76))
* implement junior engineer second turn with intercepted tool execution ([89d8bbb](https://github.com/project-david-ai/platform/commit/89d8bbb06695c51790ce90677d22559eceb7c3af))
* implement NetworkEngineerMixin ([ba7b681](https://github.com/project-david-ai/platform/commit/ba7b68117b295b091110205118075ec0459a4295))
* Implement persistent research worker thread per session. ([733dcca](https://github.com/project-david-ai/platform/commit/733dcca316c67dea0786140b0339ab3fde2fa7f4))
* Integrate Ollama local inference into the stack. ([4d21102](https://github.com/project-david-ai/platform/commit/4d211026987a2de3a50d5318ff70ba96fa5fa35a))
* integrate Ollama local inference support into DeltaNormalizer ([8e0e8c1](https://github.com/project-david-ai/platform/commit/8e0e8c1a1e107a838942520e04fc0b82050c97c3))
* intercept Junior Engineer tool calls and emit as ToolInterceptEvent ([5f9d38f](https://github.com/project-david-ai/platform/commit/5f9d38f722ea1a8bc83c555e5316893c7f0cd6e1))
* introduce computer_shell_netfilter and enhance shell execution lifecycle ([77aeff2](https://github.com/project-david-ai/platform/commit/77aeff251b937339b8c945c2e10ff47624c08964))
* map engineer flag during Assistant creation in AssistantService ([afec101](https://github.com/project-david-ai/platform/commit/afec101a3d9f8ab482f6169aec4948c9d779ca5d))
* migrate DelegationMixin off HTTP SDK to NativeExecutionService ([a964414](https://github.com/project-david-ai/platform/commit/a964414d5b52de9ba1adaa5e412e05abd1786739))
* Migrate FileSearchMixin to direct VectoreStoreManager inference. ([308015b](https://github.com/project-david-ai/platform/commit/308015b8b28c47a317a69dc7bded085fad45cf1b))
* move tool-call suppression logic to inference handler ([8e56c5b](https://github.com/project-david-ai/platform/commit/8e56c5b47870803a5d1ca7fb073b6aacdcbf6898))
* move tool-call suppression logic to inference handler ([4152548](https://github.com/project-david-ai/platform/commit/415254882786c6de79280805378a86093789c418))
* pass delegated model through run metadata and update workers ([a76a203](https://github.com/project-david-ai/platform/commit/a76a203e271d20c57af28d03de0b66373df798c4))
* pass inference API key through run metadata and enhance worker stream handling ([ee1d410](https://github.com/project-david-ai/platform/commit/ee1d4103f261d370100b79cbd07f8620c70cbf1d))
* port latest orchestration features and stream_sync wrapper ([fb4faf5](https://github.com/project-david-ai/platform/commit/fb4faf566e51c75ef9965c31001f973d580589f9))
* prevent `docker_manager.py` from running inside Docker containers ([86e4ee7](https://github.com/project-david-ai/platform/commit/86e4ee79c8ba55a501f57679f2915e5e74a07727))
* publish with deep search ([08f464c](https://github.com/project-david-ai/platform/commit/08f464c8136afe1b8d4d7760a40fa88bf1f3d477))
* Refactor batfish_router ([a4e35a7](https://github.com/project-david-ai/platform/commit/a4e35a75166b894a54fc435df41d29393934ebfa))
* remove deprecated assistant bootstrap scripts and add DockerManager CLI scaffolding ([6c8bbce](https://github.com/project-david-ai/platform/commit/6c8bbcedf5813d444a6fb1c298552f7d908380d5))
* replace DDG SERP scrape with SearxNG meta-search engine ([0d72b81](https://github.com/project-david-ai/platform/commit/0d72b81ddbf62aa4bdba6ed633e94a77c1c0e887))
* replace shell command execution with sequential protocol; enhance session lifecycle with sandboxing and file harvest ([a89a226](https://github.com/project-david-ai/platform/commit/a89a2267355443a0390ba9660e4d8d419a82a6a9))
* **services:** map engineer flag during Assistant creation in AssistantService ([e6a25de](https://github.com/project-david-ai/platform/commit/e6a25de2d82933ef79154f71ebd8e46b672b9b31))
* soft-delete functionality for files with Recycle Bin support ([a6bf1c2](https://github.com/project-david-ai/platform/commit/a6bf1c2b06e7179313fd35f380fed0448d1effa7))
* soft-delete functionality for files with Recycle Bin support ([4c33903](https://github.com/project-david-ai/platform/commit/4c33903f52186e19460d9b076c6c3f17b7a45e11))
* tenant-isolated snapshot pipeline with ID-first design ([7c1e2bf](https://github.com/project-david-ai/platform/commit/7c1e2bfe7c2c70632eb4049ebf5c4de63631ef85))
* track full ordered tool dispatch log per run ([4a43c00](https://github.com/project-david-ai/platform/commit/4a43c00b8530129e2bcbe7515cf9918dab5edb95))
* unify orchestration logic, add stream_sync, and upgrade run retrieval ([bda1975](https://github.com/project-david-ai/platform/commit/bda1975c06949085667d14d637e737a3f2d011d4))
* unify orchestration logic, add stream_sync, and upgrade run retrieval ([1e2bf1c](https://github.com/project-david-ai/platform/commit/1e2bf1c8204b5982321001da0b5e78a991cbe627))
* upgrade projectdavid SDK to projectdavid==1.37.0 ([a3305b1](https://github.com/project-david-ai/platform/commit/a3305b103186385678bb1db0cdc5ad36784c331b))
* wire worker scratchpad to shared supervisor thread ([5cdc43a](https://github.com/project-david-ai/platform/commit/5cdc43a16de6b7310e6a17c362f5c3fc0f8c82ca))
* wrap inventory_cache in a service layer ([49ba796](https://github.com/project-david-ai/platform/commit/49ba796db3e62a3ac2369db69c42e4f30c33c435))

## [1.18.1](https://github.com/frankie336/entities_api/compare/v1.18.0...v1.18.1) (2025-07-06)


### Bug Fixes

* update_assistant association issues. ([d57bf31](https://github.com/frankie336/entities_api/commit/d57bf3142245c47fb3969d7b92c42f37c43077d6))

# [1.18.0](https://github.com/frankie336/entities_api/compare/v1.17.0...v1.18.0) (2025-06-17)


### Features

* Release. ([cf082d0](https://github.com/frankie336/entities_api/commit/cf082d09aaee97ef7bd56384e06142ebf1af96f3))

# [1.17.0](https://github.com/frankie336/entities_api/compare/v1.16.0...v1.17.0) (2025-06-16)


### Features

* Structured Error Payloads. ([383c2b0](https://github.com/frankie336/entities_api/commit/383c2b0715a0a47c42fcfdd090329555277203c7))

# [1.16.0](https://github.com/frankie336/entities_api/compare/v1.15.1...v1.16.0) (2025-06-16)


### Bug Fixes

* --Integrate mixins architecture. This major upgrade splits the original base class into a series of smaller mixin classes for better maintainability and less cognitive load. ([80e54ca](https://github.com/frankie336/entities_api/commit/80e54caacecbb7276009639f40fa05deca2fb38f))


### Features

* --Integrate mixins architecture. This major upgrade splits the original base class into a series of smaller mixin classes for better maintainability and less cognitive load. ([d02411b](https://github.com/frankie336/entities_api/commit/d02411b0d2aa0c322bdd4a655c713b146bb0b1ac))
* Restore function call processing in the hyperbolicDs1 class. Function calls json objects can now be suppressed. Improve Mixin Architecture. ([c9a5b1b](https://github.com/frankie336/entities_api/commit/c9a5b1b40301db6e051608788b0e0fec88c85f35))

## [1.15.1](https://github.com/frankie336/entities_api/compare/v1.15.0...v1.15.1) (2025-05-25)


### Bug Fixes

* cache export completely disabled ([dab9b81](https://github.com/frankie336/entities_api/commit/dab9b81212ca4fcc18cd27e87acd1e5c99fbe268))

# [1.15.0](https://github.com/frankie336/entities_api/compare/v1.14.0...v1.15.0) (2025-05-25)


### Bug Fixes

* --Integrate mixins architecture. This major upgrade splits the original base class into a series of smaller mixin classes for better maintainability and less cognitive load. ([ac5d439](https://github.com/frankie336/entities_api/commit/ac5d439ce5c8595232dc32f5e1ae8b081d6ab7fa))


### Features

* --Integrate mixins architecture. This major upgrade splits the original base class into a series of smaller mixin classes for better maintainability and less cognitive load. ([cdab8fc](https://github.com/frankie336/entities_api/commit/cdab8fcead838fe61e357c0dc549d819be10799c))

# [1.14.0](https://github.com/frankie336/entities_api/compare/v1.13.0...v1.14.0) (2025-05-25)


### Features

There  are some major changes and enhancements to vector store creation and life cycle management (RAG).
 Creating a vector store
No longer requires you manually pass the user id into the creaction method 

```python
vs = client.vectors.create_vector_store(
    name="movielens-complete-demo",
    user_id=USER_ID,
)
```

Becomes:

```python
vs = client.vectors.create_vector_store(
    name="movielens-complete-demo",
    
)
```

**Search Methods**

Several new search method have been added:
vector_file_search_raw
Search hits are returned in a raw format with similarity scoring. There is no further post processing, formatting or ranking. This is most appropriate where you need to apply custom or third party ranking and or post processing.  

**Example:**

````python
hits = client.vectors.vector_file_search_raw(
    vector_store_id="vect_GsSezuKiXy11rFssDcRFAg",
    query_text=query,
    top_k=top_k,
    vector_store_host=host_override,
)
````

**Simple_vector_file_search**

Search hits are returned wrapped in an envelope that provides anotation and citations per hit. This is most appropriate for bodies of text where you might need the assistant to provide authorities and citations; a legal document for example. 

**Example**

```python
hits = client.vectors.simple_vector_file_search(
    vector_store_id=STORE_ID,
    query_text=query,
    top_k=top_k,
)
```

**attended_file_search**

Search results are synthesized by an integrated agent; results are passed to the Large Language model. The output comes with AI insights and organization. Additionally, result rankings are enhanced by a second pass through a ranking model. Suited for cumilitative research (deep research) and multi agent   tasks.   

**Example:**

```
hits = client.vectors.attended_file_search(
    vector_store_id=STORE_ID,
    query_text=query,
    top_k=top_k,
)
```

**unattended_file_search**

Search hits are returned wrapped in an envelope that provides anotation and citations per hit. Additionally, result rankings are enhanced by a second pass through a ranking model

**Example:**

```python
 hits = client.vectors.unattended_file_search(
    vector_store_id=STORE_ID,
    query_text=query,
    top_k=top_k,
)
```


* --Integrate mixins architecture. This major upgrade splits the original base class into a series of smaller mixin classes for better maintainability and less cognitive load. ([88216e0](https://github.com/frankie336/entities_api/commit/88216e04f5de357cc35910c32bf9420813f8affa))
* --Integrate mixins architecture. This major upgrade splits the original base class into a series of smaller mixin classes for better maintainability and less cognitive load. ([bf29576](https://github.com/frankie336/entities_api/commit/bf29576c936d85fe88d98f8b01cd2c649f344140))

# [1.13.0](https://github.com/frankie336/entities_api/compare/v1.12.0...v1.13.0) (2025-05-11)


### Features

* allow an admin to choose the owner ([c3e57dd](https://github.com/frankie336/entities_api/commit/c3e57ddef780da23c2169b1ca0d16fc1bbe11db2))

# [1.12.0](https://github.com/frankie336/entities_api/compare/v1.11.1...v1.12.0) (2025-04-30)


### Features

* add tools_resources ([3b399fd](https://github.com/frankie336/entities_api/commit/3b399fd528e8296470e2acd12dc6e00b47171fe6))

## [1.11.1](https://github.com/frankie336/entities_api/compare/v1.11.0...v1.11.1) (2025-04-29)


### Bug Fixes

* restore attach_tool_to_assistant method ([ba87d17](https://github.com/frankie336/entities_api/commit/ba87d172dd3dda2d5ba9e38e09f2a57ca9585eaa))

# [1.11.0](https://github.com/frankie336/entities_api/compare/v1.10.0...v1.11.0) (2025-04-29)


### Bug Fixes

* restore attach_tool_to_assistant method ([4d6eb2e](https://github.com/frankie336/entities_api/commit/4d6eb2e7e94570425f9c20a3ef742f504f252d55))


### Features

* Add canonical_id reserved ID's to base_tools ([17b7214](https://github.com/frankie336/entities_api/commit/17b7214f7ff7a7f992dfac0a51a70be883b094df))

# [1.10.0](https://github.com/frankie336/entities_api/compare/v1.9.0...v1.10.0) (2025-04-28)


### Features

* multi-stage requirements  hashing wiht improved function call handling. Introducing the platform_tools param in the assistants data structure. Introducing hybrid hashed and non hashed requirements file for problematic packages. ([8ae4b19](https://github.com/frankie336/entities_api/commit/8ae4b192ba2b56e8899850fa0ac71647c170e808))
* multi-stage requirements  hashing wiht improved function call handling. Introducing the platform_tools param in the assistants data structure. Introducing hybrid hashed and non hashed requirements file for problematic packages. ([a2f517a](https://github.com/frankie336/entities_api/commit/a2f517ab3e1df81ea7058e0b0c86be327479ddbe))
* multi-stage requirements  hashing wiht improved function call handling. Introducing the platform_tools param in the assistants data structure. Introducing hybrid hashed and non hashed requirements file for problematic packages. ([79e4c65](https://github.com/frankie336/entities_api/commit/79e4c65a9a6c6088631b262765d973776164f398))
* multi-stage requirements  hashing wiht improved function call handling. Introducing the platform_tools param in the assistants data structure. Introducing hybrid hashed and non hashed requirements file for problematic packages. ([267dbb9](https://github.com/frankie336/entities_api/commit/267dbb9aa382b6607ac44e5ba585ca0dce98d6b2))
* multi-stage requirements  hashing wiht improved function call handling. Introducing the platform_tools param in the assistants data structure. Introducing hybrid hashed and non hashed requirements file for problematic packages. ([76efb7d](https://github.com/frankie336/entities_api/commit/76efb7db6a16c1e73fca8c296c85da17bc83a4d9))

# [1.9.0](https://github.com/frankie336/entities_api/compare/v1.8.0...v1.9.0) (2025-04-23)


### Features

* function-call-tags ([6e67d57](https://github.com/frankie336/entities_api/commit/6e67d57808d16b8960ec1cd29499549215d26044))
* Improved function call handling. Working towards full call structure filtering. ([4c4c67f](https://github.com/frankie336/entities_api/commit/4c4c67f4fefd18776ad49892a30d7e27f36ca255))

# [1.8.0](https://github.com/frankie336/entities_api/compare/v1.7.1...v1.8.0) (2025-04-23)


### Features

* Asynchronous-REDIS ([8fc3de9](https://github.com/frankie336/entities_api/commit/8fc3de941163e36ad682448f9d7c4d6aae492788))

## [1.7.1](https://github.com/frankie336/entities_api/compare/v1.7.0...v1.7.1) (2025-04-19)


### Bug Fixes

* generate_dev_docker_compose ([570bda1](https://github.com/frankie336/entities_api/commit/570bda1ce25b66bb08b025899a23da953f4b7870))

# [1.7.0](https://github.com/frankie336/entities_api/compare/v1.6.0...v1.7.0) (2025-04-19)


### Features

* shunt-streams-to-redis-phase-1 ([1cab6f5](https://github.com/frankie336/entities_api/commit/1cab6f5596e7643b31f867729ae4a450c317f5f7))
Shunting streams to Redis 
We are in the process of shunting streaming content to the Redis server. This lays the ground  for state of the art utility performance enhancement, error recovery, edge device integration, and agentic features 


# [1.6.0](https://github.com/frankie336/entities_api/compare/v1.5.0...v1.6.0) (2025-04-19)


### Features

* basic_vector_embeddings_search ([7ce4fab](https://github.com/frankie336/entities_api/commit/7ce4fab60f7c0d5870f5c8f0cbb18fe8020299ea))
* move context-window-to-redis-primary ([475f19b](https://github.com/frankie336/entities_api/commit/475f19b13f63e3ce3340e26aee52469f88be1ad1))

We have moved the primary message history and system message to a redis server. 
We use a data driven method of building the context window for each assistant. On each turn the latest messages from the assistant and user are appended onto the dialogue. This includes tool responses. The context window can quickly grows in a multi turn conversation. Fetching this from the database adds more and more latency as the size of the thread  grows. 
Implementing Redis
 We moved the primary message history,  tool definitions, and system message to a Redis server. At the same time, a redis server is now introduced into the docker container estate; the first in a series of changes that will leverage powerful features of Redis for a true scalable, enterprise AI inference platform. 
No Database change 
There is no database change, so you can retain existing data in Dev and not have to make any Alembic update s
Whilst the solution has been tested, please let us know if you experience any abnormalities; these will be rapidly fixed.


# [1.5.0](https://github.com/frankie336/entities_api/compare/v1.4.0...v1.5.0) (2025-04-18)


### Features

* basic_vector_embeddings_search ([e5def53](https://github.com/frankie336/entities_api/commit/e5def53adbe148f6d0feb843f3458659a8542c91))

# [1.4.0](https://github.com/frankie336/entities_api/compare/v1.3.2...v1.4.0) (2025-04-17)


### Features

* Integrate DeepSeek API. ([e6c1dae](https://github.com/frankie336/entities_api/commit/e6c1daeb2201ae63d4e07daf67dc923b00e69a2a))

## [1.3.2](https://github.com/frankie336/entities_api/compare/v1.3.1...v1.3.2) (2025-04-17)


### Bug Fixes

* upgrade pd client. ([ae0569b](https://github.com/frankie336/entities_api/commit/ae0569bece4174112225fa5ed2181160f2f5961f))

## [1.3.1](https://github.com/frankie336/entities_api/compare/v1.3.0...v1.3.1) (2025-04-17)


### Bug Fixes

* TogetherAIHandler routing issues. ([be6c386](https://github.com/frankie336/entities_api/commit/be6c38684f81ddd5759c1adad7e636f873d08dbc))

# [1.3.0](https://github.com/frankie336/entities_api/compare/v1.2.1...v1.3.0) (2025-04-17)


### Bug Fixes

* Normalize your route keys to lowercase in SUBMODEL_CLASS_MAP ([a8d1f51](https://github.com/frankie336/entities_api/commit/a8d1f51719e3c9f79527145ead9329562db6e9ce))


### Features

* Add support for new models ([4523191](https://github.com/frankie336/entities_api/commit/45231911a3f24f0af7989197c6e4fb0bd017551b))

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [1.2.0](https://github.com/frankie336/entities_api/compare/v1.1.0...v1.2.0) (2025-04-15)

## [1.2.1](https://github.com/frankie336/entities_api/compare/v1.2.0...v1.2.1) (2025-04-16)


### Bug Fixes

* Moved admin .env ([3c2834e](https://github.com/frankie336/entities_api/commit/3c2834eb90256349c22c24ea9146c750013f1c36))


### ✨ Added
- Introduced `scripts/generate_docker_compose.py` and `scripts/generate_docker_compose_dev.py`:
  - Automatically generate `docker-compose.yml` and `.env` files if they do not exist.
  - Inject unique, secure values for `MYSQL_ROOT_PASSWORD`, `MYSQL_PASSWORD`, and `DEFAULT_SECRET_KEY`.
  - Generate and map a `unique_secret` for custom Docker network binding.

- Added fallback `.example` templates:
  - `docker-compose.dev.example.yml` – now tracked in source control.
  - Redacts all secrets and replaces them with `REPLACE_ME` tokens for dev visibility and safety.

### 🔧 Changed
- `start.py` (DockerManager):
  - Aligned `.env` generation to source values directly from `docker-compose.yml` (or fallback defaults).
  - Dynamically constructs `DATABASE_URL` and `SPECIAL_DB_URL` using parsed credentials.
  - Added logic to detect `docker-compose.yml` presence and skip regeneration if already defined.
  - Ensured platform-aware path handling for mounted volumes (`SHARED_PATH` detection).
  - Added validation for `docker-compose.dev.yml` parsing via PyYAML.

### 🧪 Improved
- Hardened Docker secret management:
  - Secrets are never committed to source control.
  - All auto-generated credentials use `uuid.uuid4().hex` or `secrets.token_urlsafe(...)` for high entropy.
  - `.dockerignore` and `.gitignore` now explicitly exclude sensitive runtime files.



### Bug Fixes

* env generate_docker_compose.py ([9ddde9f](https://github.com/frankie336/entities_api/commit/9ddde9f170f1f3624585baf3fdab29e61dd1023d))


### Features

* dynamic-unique-env-secrets ([13b556f](https://github.com/frankie336/entities_api/commit/13b556fdbca04bfdf65de44263345f45dafd0bea))

# [1.1.0](https://github.com/frankie336/entities_api/compare/v1.0.0...v1.1.0) (2025-04-15)


### Features

* New CI flow18! ([ff06107](https://github.com/frankie336/entities_api/commit/ff061078e69d5fb75f3abf739ed1ba8eaccb3888))
* New CI flow19! ([05e2bd3](https://github.com/frankie336/entities_api/commit/05e2bd36a3c125276e15d0de35e2402ba0f18e39))

# 1.0.0 (2025-04-14)


### Bug Fixes

* ASSISTANTS_BASE_URL eroneus back slash / ([6e70fd9](https://github.com/frankie336/entities_api/commit/6e70fd9d841de89c5da2055b35ac6405321493d5))
* **download:** allow inline rendering of image MIME types via conditional Content-Disposition ([d833909](https://github.com/frankie336/entities_api/commit/d833909e7bcb66508a9a8884148fe476c9a8549c))
* **download:** allow inline rendering of image MIME types via conditional Content-Disposition ([7b45b3b](https://github.com/frankie336/entities_api/commit/7b45b3ba08d035b9b9da58b31432354a0598d02f))
* remove comments from .env generation ([7894e1e](https://github.com/frankie336/entities_api/commit/7894e1e361f6c931cca178cb3bbd5e5b79ae37b3))
* users route. ([de6683b](https://github.com/frankie336/entities_api/commit/de6683bacb7cf3a4a26d20a447e8fffd4a99859d))


### Features

* add routes.md ([a2a52fa](https://github.com/frankie336/entities_api/commit/a2a52faa8fc9eb553e0b95ca64a902fb85d471a0))
* add routes.md ([f5ae598](https://github.com/frankie336/entities_api/commit/f5ae598417739109e5d9b57e7ee4668265dc0110))
* add routes.md ([eb03440](https://github.com/frankie336/entities_api/commit/eb03440140137163058cc2b1f79e2a15da0336a4))
* Add support for deepseek-ai/DeepSeek-V3-0324 ([87ac173](https://github.com/frankie336/entities_api/commit/87ac173ac2b332ddef9bf8c1db6633e380a15fd6))
* Add support for deepseek-ai/DeepSeek-V3-0324 ([350503f](https://github.com/frankie336/entities_api/commit/350503f4c6692fb9b0dab5c7305dde3ada6883c5))
* bootstrap default assistant_set_up_service.py. ([6cea7bb](https://github.com/frankie336/entities_api/commit/6cea7bbe82f2c1e3f3c6c92c6fbce455573b766b))
* bootstrap default assistant_set_up_service.py. ([10518ff](https://github.com/frankie336/entities_api/commit/10518ff6bd3b7c9a08546353ac1b22869b8edbcb))
* enhance code interpreter output with smart file rendering ([6f50b50](https://github.com/frankie336/entities_api/commit/6f50b50a384298ddcc57c8957d9f26f4af16c455))
* Implement API key protected routes ([197aec4](https://github.com/frankie336/entities_api/commit/197aec4771e3097817c10200b8a32bbc98ffd0c1))
* Implement new provider routing. ([45c16fa](https://github.com/frankie336/entities_api/commit/45c16faed545d86d5cb9ca68e87127f201102c80))
* Implement new provider routing. ([3624b52](https://github.com/frankie336/entities_api/commit/3624b52d537cd322d9081ce58bcd77b30a78e100))
* Integrate new models: ([32b954f](https://github.com/frankie336/entities_api/commit/32b954fbc8f456b6c78db2630df7ccab9c0d315f))
* multi-stage-containers. ([092f1f7](https://github.com/frankie336/entities_api/commit/092f1f7b3f23767a5733f507d7713161b60c895e))
* multi-stage-containers. ([a4dc666](https://github.com/frankie336/entities_api/commit/a4dc6663244dcc99f4bd893d62ee25958431b34c))
* New CI flow! ([486d875](https://github.com/frankie336/entities_api/commit/486d87574180c50a9c42ce4e4158a2e23ab8d6a2))
* New CI flow! ([1428ce7](https://github.com/frankie336/entities_api/commit/1428ce743efdc73bc35eca75994da766341cebae))
* New CI flow10! ([57d2f3e](https://github.com/frankie336/entities_api/commit/57d2f3e29d86a01210b625a3e2f5f5f7aa12acca))
* New CI flow11! ([49e408a](https://github.com/frankie336/entities_api/commit/49e408ab9074150520250176b9411d8015bcf537))
* New CI flow13! ([59ea8df](https://github.com/frankie336/entities_api/commit/59ea8df6ffc1e0bcfdd52e4b1481a722b81955cb))
* New CI flow14! ([1e7a303](https://github.com/frankie336/entities_api/commit/1e7a30345c651e34e4cd178d153d43c7cbec71f6))
* New CI flow15! ([293e133](https://github.com/frankie336/entities_api/commit/293e1332da53d7ec745f090d849b7b182ffe29ef))
* New CI flow16! ([f2b49fd](https://github.com/frankie336/entities_api/commit/f2b49fd267ecab89bc239d04cd948269d692123b))
* New CI flow17! ([f31b927](https://github.com/frankie336/entities_api/commit/f31b927e80590b491932a74acd6506a3b42a7286))
* New CI flow2! ([a6db2c1](https://github.com/frankie336/entities_api/commit/a6db2c117c6d41a4808701d42d444a80e41f17c7))
* New CI flow3! ([6bb1357](https://github.com/frankie336/entities_api/commit/6bb13570a1f24f1a15e4905dcfbfdf80706423a6))
* New CI flow4! ([5d229be](https://github.com/frankie336/entities_api/commit/5d229bedbe92f534cd703ded228f00c60b059c45))
* New CI flow5! ([fb95024](https://github.com/frankie336/entities_api/commit/fb950244852f33e40c5f01b165b4645d89e3a0f0))
* New CI flow6! ([05f8d0d](https://github.com/frankie336/entities_api/commit/05f8d0d0360eb28ab339a60b8c7a164ed7ce580a))
* New CI flow7! ([4e704b4](https://github.com/frankie336/entities_api/commit/4e704b42cf321ff1c402a668bded7e86e69c1a7a))
* New CI flow8! ([1633e76](https://github.com/frankie336/entities_api/commit/1633e76d71e49412b00e505f134e31ac6295a806))
* New CI flow9! ([8cdc5f9](https://github.com/frankie336/entities_api/commit/8cdc5f9044ce9d94d11b28ae53043c928cd6e0d1))

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
