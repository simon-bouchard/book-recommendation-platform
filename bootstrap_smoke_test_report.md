# Bootstrap Smoke Test Report
**Date:** 2026-09-27  
**Machine:** GTX 1060 3GB (sm_61/Pascal), WSL Ubuntu, 8-core, SSH+tmux  
**Command:** `CUDA_VISIBLE_DEVICES="" SUBJ_EPOCHS=1 ops/bootstrap_local.sh`

---

## Result: PASS (CPU fallback)

All three pipeline sections completed without error. All four required output files
exist and are non-empty:

| File | Size |
|------|------|
| `models/artifacts/staging/attention/subject_attention_perdim.pth` | 884K |
| `models/artifacts/staging/embeddings/book_subject_embeddings.npy` | 61M |
| `models/artifacts/staging/scoring/bayesian_scores.npy` | 976K |
| `models/artifacts/staging/similarity/subject/index.faiss` | 126M |

---

## GPU outcome

The GTX 1060 (CUDA capability sm_61/Pascal) is **not supported** by PyTorch 2.8.0+cu128,
which requires sm_70 (Volta) or newer. PyTorch detected the card, printed a warning, then
crashed on the first CUDA kernel call:

```
NVIDIA GeForce GTX 1060 3GB with CUDA capability sm_61 is not compatible with the current
PyTorch installation. The current PyTorch install supports CUDA capabilities
sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
...
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Retry with `CUDA_VISIBLE_DEVICES=""` forced CPU. Training completed on CPU with
`SUBJ_EPOCHS=1`; wall-clock time for the subject embedding step was not separately
measured (see future work below).

**To use this GPU:** install a PyTorch build with sm_61 support (CUDA 11.x era,
e.g. `torch==1.x` or `torch==2.0` with `cu117`). The `train-gpu` extra in
`pyproject.toml` currently pins a cu128 build.

---

## Bugs found and fixed

### 1. `ops/bootstrap_local.sh` — bare `python` instead of `uv run python`

All script invocations after the precondition check used bare `python`, which resolves
to the system interpreter rather than the project venv. Fixed by replacing every
`python` call with `uv run python` throughout.

### 2. `app/table_models.py` — composite PKs defined via `__mapper_args__` emit no DDL

Four models used `__mapper_args__ = {"primary_key": [...]}` to declare composite primary
keys. This sets ORM-level identity only and does **not** emit `PRIMARY KEY` in the
`CREATE TABLE` DDL. MySQL therefore created those tables with no primary key, which
caused `create_tables.py` to fail immediately:

```
(1822, "Failed to add the foreign key constraint. Missing index for constraint
'book_genres_ibfk_1' in the referenced table 'genres'")
```

Affected models: `Genre`, `BookGenre`, `BookVibe`, `EnrichmentError`.  
Fix: moved `primary_key=True` onto the column definitions directly.

### 3. `data/import_enrichment_csvs.py` — single deferred commit causes O(n²) slowdown

The script used a single `db.commit()` at the very end of a 48-chunk import loop.
Each `db.add_all()` + `db.flush()` accumulated objects in SQLAlchemy's identity map
without ever clearing it. By batch 30 the per-batch time had grown from ~13s to ~65s.

Fix: commit per chunk (clears identity map automatically). Also switched link-table
inserts to `mysql_insert(...).prefix_with("IGNORE")` (SQLAlchemy core, bypasses identity
map entirely, and makes reruns safe — a cancelled import can be resumed without hitting
duplicate key errors on already-committed chunks).

---

## Future work / known gaps

- **GPU support:** `train-gpu` extra needs a cu117/cu118 PyTorch build to support sm_61.
  Alternatively, document that Pascal and older are unsupported.
- **`train_subject_embs_contrastive.py` CPU timing:** not measured cleanly this run due
  to multiple restarts. Worth a dedicated timed run with `time` or internal logging.
- **`import_csvs.py` accumulation:** the same O(n²) identity map pattern exists in the
  core CSV import. Attempts to fix it with `expunge_all()` per chunk were observed to be
  slower in practice (likely because `expunge_all()` itself has overhead, and the
  accumulation doesn't hurt badly enough on this dataset size to justify it). Left as
  original. If dataset grows significantly, revisit with `bulk_insert_mappings` instead.
- **MySQL native password:** Docker MySQL 8.0 defaults to `caching_sha2_password`,
  which requires the `cryptography` package (not in `pyproject.toml`). Worked around
  by starting the container with `--default-authentication-plugin=mysql_native_password`.
  Either add `cryptography` as a dependency or document this container flag.
