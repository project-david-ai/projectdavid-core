# ProjectDavid Core — Migration Chain

## Chain Topology

The migration history has one branch point and one merge point.
All migrations after the merge run on a single linear chain.

```
ROOT
 │
 ├─ eed80604f05c  baseline_schema (root — no parent)
 │
 ├─ e844e0ceaba2  baseline_schema (creates users, assistants, threads, core tables)
 │
 ├─ 0f2bbee38b41  add_platform_tools_tool_resources
 ├─ c8adf7bf8fa8  drop_platform_tools_from_assistants
 ├─ ffbf9622eece  convert_timestamps_to_epoch_time
 ├─ dc84d53c3c46  restore_missing_columns
 ├─ c34a9a215b5d  convert_truncation_strategy_to_varchar
 ├─ 9314d4058f78  add_agentic_state_columns_to_assistants
 ├─ ac1498a9642c  remove_tools_table_and_associations
 ├─ 3b439a3fa9f6  fix_run_instructions_length
 ├─ 4c257388838e  add_tool_name_to_actions
 ├─ 741d86dd5ac8  add_decision_telemetry_to_actions
 ├─ 6413efed1b18  add_decision_telemetry_to_assistants
 ├─ 04607cbac68b  add_web_access_to_assistants
 ├─ b697008df93a  add_deep_research_to_assistants
 ├─ dda6fd28f45c  add_audit_logs_table_and_assistant_soft_delete
 ├─ ce0a8a7e9d41  add_engineer_column_to_assistants
 ├─ b50938f6bd99  add_batfishsnapshot_to_models
 ├─ 1e55188b6b26  add_owner_id_to_assistants
 │
 ├─ 1c9784351972  add_owner_id_to_thread_table
 │         │
 │         ├─── BRANCH A ──────────────────────────────────────────┐
 │         │    222cafa3baac  fix_fk_delete_rules_for_gdpr          │
 │         │                                                         │
 │         └─── BRANCH B ──────────────────────────────────────────┤
 │              3a42e4f129e4  remove_assistant_vector_store         │
 │              26a927cbe516  remove_thread_vector_store_relationship│
 │                                                                   │
 │         ┌─── MERGE ◄────────────────────────────────────────────┘
 │         │
 ├─ 9351530d20ab  merge_gdpr_fk_and_main  ◄── MERGE POINT
 │
 ├─ 52be510eb9c8  fix_join_table_fk_delete_rules
 ├─ 3e16915ae60f  add_soft_delete_to_files
 ├─ ba35b4620058  add_soft_delete_to_vectorstore
 │
 ├─ 005820173bc4  add_fine_tuning_tables          ◄── FIXED (depends_on: 9351530d20ab)
 │                FK: datasets → users
 │                FK: training_jobs → users
 │                FK: fine_tuned_models → users
 │
 ├─ 53ed443a77c1  move_fine_tuning_tables_to_training
 ├─ c6d0aaad984f  implement_fine_tuning_tables_in_training_models
 ├─ 33111f6ac0b8  add_file_id_to_datasets_make_storage
 ├─ 66b1d150d350  remove_training_data_from_models
 ├─ d98d34517e5f  add_training_data_to_training_models2
 ├─ 02ce09f95a67  add_fine_tuning_tables (v2)
 ├─ 05cf57b50101  updated_at_column_to_match_other_models
 ├─ a4d6ae115898  adding_updated_at_and_deleted_at
 │
 └─ 81522f7beef3  add_cluster_management_tables_and_fields  ◄── HEAD
```

---

## Full Migration Table

| Order | Revision | Description | Parent | depends_on | Notes |
|---|---|---|---|---|---|
| 1 | `eed80604f05c` | baseline_schema | — | — | **Root. No parent.** |
| 2 | `e844e0ceaba2` | baseline_schema | `eed80604f05c` | — | **Creates users, assistants, threads, all core tables** |
| 3 | `0f2bbee38b41` | add_platform_tools_tool_resources | `e844e0ceaba2` | — | |
| 4 | `c8adf7bf8fa8` | drop_platform_tools_from_assistants | `0f2bbee38b41` | — | |
| 5 | `ffbf9622eece` | convert_timestamps_to_epoch_time | `c8adf7bf8fa8` | — | |
| 6 | `dc84d53c3c46` | restore_missing_columns | `ffbf9622eece` | — | |
| 7 | `c34a9a215b5d` | convert_truncation_strategy_to_varchar | `dc84d53c3c46` | — | |
| 8 | `9314d4058f78` | add_agentic_state_columns_to_assistants | `c34a9a215b5d` | — | |
| 9 | `ac1498a9642c` | remove_tools_table_and_associations | `9314d4058f78` | — | |
| 10 | `3b439a3fa9f6` | fix_run_instructions_length | `ac1498a9642c` | — | |
| 11 | `4c257388838e` | add_tool_name_to_actions | `3b439a3fa9f6` | — | |
| 12 | `741d86dd5ac8` | add_decision_telemetry_to_actions | `4c257388838e` | — | |
| 13 | `6413efed1b18` | add_decision_telemetry_to_assistants | `741d86dd5ac8` | — | |
| 14 | `04607cbac68b` | add_web_access_to_assistants | `6413efed1b18` | — | |
| 15 | `b697008df93a` | add_deep_research_to_assistants | `04607cbac68b` | — | |
| 16 | `dda6fd28f45c` | add_audit_logs_table_and_assistant_soft_delete | `b697008df93a` | — | FK: audit_logs → users |
| 17 | `ce0a8a7e9d41` | add_engineer_column_to_assistants | `dda6fd28f45c` | — | |
| 18 | `b50938f6bd99` | add_batfishsnapshot_to_models | `ce0a8a7e9d41` | — | FK: batfish_snapshots → users |
| 19 | `1e55188b6b26` | add_owner_id_to_assistants | `b50938f6bd99` | — | |
| 20 | `1c9784351972` | add_owner_id_to_thread_table | `1e55188b6b26` | — | **Branch point** |
| 21a | `222cafa3baac` | fix_fk_delete_rules_for_gdpr | `1c9784351972` | — | **Branch A** |
| 21b | `3a42e4f129e4` | remove_assistant_vector_store | `1c9784351972` | — | **Branch B** |
| 22b | `26a927cbe516` | remove_thread_vector_store_relationship | `3a42e4f129e4` | — | **Branch B** |
| 23 | `9351530d20ab` | merge_gdpr_fk_and_main | `222cafa3baac` + `26a927cbe516` | — | **MERGE POINT** |
| 24 | `52be510eb9c8` | fix_join_table_fk_delete_rules | `9351530d20ab` | — | |
| 25 | `3e16915ae60f` | add_soft_delete_to_files | `52be510eb9c8` | — | |
| 26 | `ba35b4620058` | add_soft_delete_to_vectorstore | `3e16915ae60f` | — | |
| 27 | `005820173bc4` | add_fine_tuning_tables | `ba35b4620058` | **`9351530d20ab`** | ⚠️ **FIXED** — FK: datasets/training_jobs/fine_tuned_models → users |
| 28 | `53ed443a77c1` | move_fine_tuning_tables_to_training | `005820173bc4` | — | |
| 29 | `c6d0aaad984f` | implement_fine_tuning_tables_in_training_models | `53ed443a77c1` | — | |
| 30 | `33111f6ac0b8` | add_file_id_to_datasets_make_storage | `c6d0aaad984f` | — | |
| 31 | `66b1d150d350` | remove_training_data_from_models | `33111f6ac0b8` | — | FK: datasets/training_jobs → users |
| 32 | `d98d34517e5f` | add_training_data_to_training_models2 | `66b1d150d350` | — | FK: datasets → users |
| 33 | `02ce09f95a67` | add_fine_tuning_tables (v2) | `d98d34517e5f` | — | |
| 34 | `05cf57b50101` | updated_at_column_to_match_other_models | `02ce09f95a67` | — | |
| 35 | `a4d6ae115898` | adding_updated_at_and_deleted_at | `05cf57b50101` | — | |
| 36 | `81522f7beef3` | add_cluster_management_tables_and_fields | `a4d6ae115898` | — | **HEAD** |

---

## Key Facts

- **Root:** `eed80604f05c` — no parent, chain starts here
- **Baseline (creates core tables):** `e844e0ceaba2` — users, assistants, threads etc all created here
- **Branch point:** `1c9784351972` — splits into GDPR branch and vector store branch
- **Merge point:** `9351530d20ab` — rejoins both branches
- **Fixed migration:** `005820173bc4` — was on a side branch that did not pass through the merge point; added `depends_on: 9351530d20ab` to guarantee users table exists before FK creation
- **Head:** `81522f7beef3` — latest migration

## Rule for future migrations

Any migration that creates a table with a FK to `users`, `assistants`, `threads`,
or any other table created in `e844e0ceaba2` must either:

1. Have `down_revision` pointing to a migration that is **after** `9351530d20ab` in the chain, OR
2. Explicitly declare `depends_on = "9351530d20ab"`