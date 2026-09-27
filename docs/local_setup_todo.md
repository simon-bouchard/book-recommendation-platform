# Local setup / reproducibility — status and final test

Tracking doc for the "make a fresh clone fully runnable" effort.

## Resolved

Everything below has been individually tested and fixed, either already on
`master` or on the `chore/local-setup-reproducibility` branch (not yet
merged/pushed):

- `data/import_csvs.py`, `data/import_enrichment_csvs.py` — tested, verified
  against live DB, perf-fixed (`INSERT IGNORE` + per-chunk commit)
- Full training pipeline (`export_training_data` → subject-embedding
  training → `precompute_embs` → `precompute_bayesian` →
  `build_metadata_lookup` → `train_als` → `build_similarity_indices`) — all
  individually tested; fixed: checkpoint filename mismatch, wrong step
  order, a misleading log path, a pandas `SettingWithCopyWarning`
- `app/semantic_index/builders/build_enriched_index.py` — fixed a real
  tone-resolution bug (v1/v2 ontology ID offset mismatch)
- `ops/meilisearch/index_books_meili.py` — fixed a real bug (primary key
  auto-detected as `cover_id` instead of `item_idx`, silently dropping ~16%
  of the catalog)
- `ops/bootstrap_local.sh` — tested as a full single-script run (on a
  separate desktop machine); fixed: bare `python` → `uv run python`,
  composite-PK DDL not emitted (`app/table_models.py`), added
  `cryptography` dependency for MySQL's `caching_sha2_password`
- `.env.example` — tested via a controlled swap (both the
  model-servers/Meilisearch env file and the backend's `.env`, one at a
  time, restored after); confirmed the documented 7-key set is genuinely
  sufficient for the app + model servers to work correctly. Added the
  missing `ARTIFACTS_DIR`, `OTEL_SDK_DISABLED`, and a Chatbot section
  listing LLM provider vars with empty placeholders
- `requirements/embedder.txt`, `requirements/semantic.txt` — fixed: both
  were pulling full CUDA-bundled torch builds with no GPU to use them,
  now pull CPU-only wheels
- `app/search/engine.py` — fixed a real bug: Meilisearch being unavailable
  crashed with `AttributeError` (`_fallback_search` was never implemented)
  instead of a clean 503
- README — documented GPU incompatibility (`train-gpu` needs sm_70+,
  Pascal cards like GTX 1060 crash), clarified Jaeger needs a separate
  `docker compose up` (not part of the main stack, by design), consolidated
  a duplicated "Book Enrichment" section

## Deferred by decision, not readiness

- [ ] MySQL + Redis containerization — blocked on the prod port-collision
  risk (native MySQL/Redis likely already bound to standard ports on prod)
- [ ] `/etc/bookrec.env` (prod) vs `.env` (app + scripts) — cannot change
  what the model-server/Meilisearch compose files point to; prod depends
  on that exact path
- [ ] Git history purge (`ops/purge_large_blobs_from_history.sh`) — script
  ready, one-time hygiene, not blocking

## Closed

- [x] GitHub issue — confirmed posted and closed
- [x] Kaggle column descriptions not showing on the dataset page —
  "no matter," not revisiting

---

## Final test: full README, genuinely fresh environment (desktop)

**Goal:** validate the entire README Local Setup section, start to finish,
on an environment with none of this machine's accumulated state — no
pre-existing `.env`, no `/etc/bookrec.env`, no already-imported data, no
already-built artifacts. Follow the README literally, as a new contributor
would, without shortcuts — including the full (uncapped `SUBJ_EPOCHS`)
subject-embedding training run, accepting that it will take a long time on
CPU (a single epoch took 20+ minutes on an 8-core CPU box earlier in this
effort).

**Before starting:** push `chore/local-setup-reproducibility` to origin —
it only exists locally right now. The desktop must check out this branch,
not `master`, or none of the fixes above are present.

### Setup

1. Fresh `git clone`, `git checkout chore/local-setup-reproducibility`, into
   a new directory (not reusing any existing checkout).
2. `cp .env.example .env`, fill in real values.
3. Create a fresh, empty MySQL database and Redis instance (reuse
   already-installed MySQL/Redis server software if present — the README
   doesn't claim to install those for you, only to use them).
4. Download the 7 CSVs from the Kaggle dataset into `data/`.
5. `uv sync --extra train` (or `--extra train-gpu` if the desktop's GPU
   supports it — the GTX 1060 does not, see the GPU note in the README).

### Run every README step in order, no skipping

Steps 1-5 (data → training pipeline → search indexes), then step 6 (start
services): `docker compose -f docker/compose/docker-compose.yml up -d` +
`uv run uvicorn main:app`. For step 6, the known `/etc/bookrec.env` gap
applies — create that file with the same values as `.env` (matching what
the README's "Known gap" note says) rather than treating it as a blocker.

`ops/bootstrap_local.sh` can be used for steps 3-5 in one shot, or run
individually for easier debugging — either is fine.

### Report back

Same format as the earlier bootstrap smoke test report: what was run, what
happened at each step (including timings for anything slow), the exact
text of any error or warning encountered, and confirmation that real
functionality works at the end (search, similarity, signup/login — not
just health checks). Fix nothing silently; report first.

If new bugs are found: push fix commits to `chore/local-setup-reproducibility`
(or a new branch off it) rather than fixing locally and not sharing.

**Once this test passes (or after fixing whatever it finds):** merge
`chore/local-setup-reproducibility` into `master` and delete this file.
