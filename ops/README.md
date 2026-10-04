# ops/

Operational and maintenance scripts that aren't part of the running app. Each script's own
docstring or header has the details and options. Run Python scripts from the repo root, e.g.
`uv run python ops/training/backup_db.py`. Model-related scripts need `--extra train`.

## Top level

| Script | Purpose |
|--------|---------|
| `bootstrap_local.sh` | One-shot local setup: database, model artifacts and search indexes (README steps 3-5). Requires `uv sync --extra train`. |
| `test_training_locally.sh` | Older quick run of the training steps. Mostly superseded by `bootstrap_local.sh` and `training/automated_training.py`. |
| `purge_large_blobs_from_history.sh` | One-off: rewrites git history to remove large data blobs. Needs a force-push; read its header first. |

## training/

The production retraining pipeline. `automated_training.py` is the entry point and calls the others.

| Script | Purpose |
|--------|---------|
| `automated_training.py` | Orchestrates export, training, artifact staging, quality gate, versioned promotion and model server reload. |
| `backup_db.py` | Gzipped `mysqldump` to `data/backups/db/`, keeps the 7 newest. Runs before training. |
| `evaluate_gate.py` | Quality gate: decides whether a staged training run can be promoted. |
| `reload_signal.py` | Sends SIGHUP to the model server containers for a graceful Gunicorn reload. |
| `flush_cache.py` | Flushes stale Redis cache entries after promotion. |
| `notify.py` | Email, Slack and Healthchecks notifications for pipeline steps (each enabled by env vars). |
| `smoke_test_notify.py` | Manual check that notifications arrive. Not part of pytest. |

## meilisearch/

| Script | Purpose |
|--------|---------|
| `index_books_meili.py` | Full index of all books with enrichments and Bayesian scores (README step 5). |
| `update_bayes_pop.py` | Refreshes Bayesian popularity scores in the index. Called by `automated_training.py`. |

## enrichment/

Tools for the LLM enrichment pipeline (producer, Kafka, Spark, SQL). Most need Kafka and the
Spark loader from `docker/spark-loader/` running.

| Script | Purpose |
|--------|---------|
| `preflight_validation.py` | Checks the environment before an enrichment run (`--quick` skips slow checks). |
| `setup_kafka_topics.py` | Creates the enrichment Kafka topics. Run after starting Kafka. |
| `test_enrichment.py` | Runs the real enrichment pipeline with a `--limit` and reports basic stats. |
| `test_enrichment_integration.py` | Checks that producer output reaches SQL through Kafka and Spark. |
| `monitor_enrichment_pipeline.py` | Dashboard: consumer lag, error rates, commit times, per tags version. |
| `check_kafka_balance.py` | Kafka partition balance and consumer health. |
| `analyse_test_results.py` | Analyzes enrichment results from SQL and writes an HTML report. |
| `analyse_test_errors.py` | Analyzes enrichment errors from SQL. |
| `analyze_metadata_coverage.py` | Coverage of descriptions and OL subjects, used to design the quality tiers. |
| `cleanup_enrichment_version.py` | Removes all data for one tags version: SQL, Kafka, bronze, checkpoints, embeddings. Destructive. |
| `clear_kafka_test_data.sh` | Resets test data: recreates topics and clears Spark checkpoints. Destructive. |
| `run_embedding_worker.sh` | Runs the incremental semantic embedding worker. Production host only (hardcoded paths and conda env). |

## ol_subjects/

The one-time Open Library subject cleaning pipeline. Each step reads the previous step's output
in `data/ol_subjects/`. The starting input, `books_with_genres.pkl`, isn't in the repo.

1. `convert_subjects_to_jsonl.py`: `books_with_genres.pkl` to `ol_subjects.jsonl`
2. `clean_subjects_v1.py`: split compound subjects, normalize text, dedupe
3. `clean_subjects_v2.py`: drop translations, split "X in art" / "X in literature" style subjects
4. `clean_subjects_v3.py`: drop fictitious-character, review and award subjects
5. `clean_subjects_v4.py`: map variants to canonical subjects, drop meaningless ones
6. `clean_subjects_v5.py`: keep only works present in the `books` table (needs the database)
7. `combine_v5_duplicates.py`: merge duplicate work IDs into one record
8. `ingest_ol_subjects_to_db.py`: load into the `ol_subjects` and `book_ol_subjects` tables

Analysis helpers: `analyze_subjects.py` (importable module for checking a cleaning step) and
`analyze_subject_lengths.py` (subject length distribution of the final output).

## models/ and migrations/

One-time artifact layout migrations, kept for reference. All have `--dry-run`.

| Script | Purpose |
|--------|---------|
| `models/migrate_artifacts.py` | Renames and reorganizes the original flat `models/data/` artifacts. See `models/migrate_artifacts_readme.md`. |
| `migrations/migrate_flat_artifacts.py` | Moves the flat artifact layout into the versioned store. |
| `migrations/migrate_training_data_to_artifacts.py` | Copies the training data snapshot into each artifact version. |
