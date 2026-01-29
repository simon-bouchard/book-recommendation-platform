# ops/models/

Operational scripts for model artifact management and maintenance.

## Overview

This directory contains one-time operational tasks and administrative scripts for managing model artifacts. These are separate from production code to maintain clear separation between runtime code and maintenance operations.

## Scripts

### migrate_artifacts.py

Migration script to reorganize model artifacts from the old flat structure to the new organized hierarchy.

**What it does:**
1. Renames artifacts to descriptive names (e.g., `book_embs.npy` → `book_subject_embeddings.npy`)
2. Organizes into subdirectories (`embeddings/`, `attention/`, `scoring/`)
3. Verifies file integrity with SHA256 checksums
4. Creates detailed JSON report of all operations

**Usage:**

```bash
# Preview changes (safe, no modifications)
python ops/models/migrate_artifacts.py --dry-run

# Execute migration with confirmation
python ops/models/migrate_artifacts.py --execute

# Execute migration without confirmation (use with caution)
python ops/models/migrate_artifacts.py --execute --force
```

**Migration Mapping:**

```
OLD STRUCTURE (models/data/)              NEW STRUCTURE (models/artifacts/)
├── book_embs.npy                    →    ├── embeddings/
├── book_ids.json                          │   ├── book_subject_embeddings.npy
├── book_als_emb.npy                       │   ├── book_subject_ids.json
├── book_als_ids.json                      │   ├── book_als_factors.npy
├── user_als_emb.npy                       │   ├── book_als_ids.json
├── user_als_ids.json                      │   ├── user_als_factors.npy
├── bayesian_tensor.npy                    │   └── user_als_ids.json
├── gbt_cold.pickle                        ├── attention/
├── gbt_warm.pickle                        │   ├── subject_attention_scalar.pth
├── subject_attention_components.pth       │   ├── subject_attention_perdim.pth
├── subject_attention_components_*.pth     │   ├── subject_attention_selfattn.pth
└── ...                                    │   └── subject_attention_selfattn_perdim.pth
                                           └── scoring/
                                               ├── bayesian_scores.npy
                                               ├── gbt_cold.pickle
                                               └── gbt_warm.pickle
```

**Safety Features:**

- Dry-run mode to preview all changes
- SHA256 checksum verification before and after
- Detailed JSON report of all operations
- Automatic rollback on checksum mismatch
- Confirmation prompt before executing

**Reports:**

Migration reports are saved as:
```
ops/models/migration_report_[mode]_[timestamp].json
```

Each report contains:
- Timestamp of migration
- List of all file operations
- Checksums before and after
- Verification status
- Any errors encountered

**Example Output:**

```
================================================================================
MODEL ARTIFACTS MIGRATION - DRY RUN
================================================================================

Checking source files:
--------------------------------------------------------------------------------
  [FOUND] book_embs.npy                                  (   12.34 MB)
  [FOUND] book_ids.json                                  (    0.15 MB)
  [FOUND] book_als_emb.npy                               (    4.56 MB)
  [SKIP]  subject_attention_components_scalar.pth        (not found)
--------------------------------------------------------------------------------
Found: 3 files | Missing: 10 files

Computing checksums for source files:
--------------------------------------------------------------------------------
  Hashing book_embs.npy... a3f5c8d9e2b1...
  Hashing book_ids.json... 7b2e9f4a6c8d...
  Hashing book_als_emb.npy... e1d8c4b7a5f2...

Computed 3 checksums

Creating directory structure:
--------------------------------------------------------------------------------
  [DRY RUN] Would create: models/artifacts/embeddings
  [DRY RUN] Would create: models/artifacts/attention
  [DRY RUN] Would create: models/artifacts/scoring

Migrating files:
--------------------------------------------------------------------------------
  [DRY RUN] book_embs.npy                                  -> book_subject_embeddings.npy
  [DRY RUN] book_ids.json                                  -> book_subject_ids.json
  [DRY RUN] book_als_emb.npy                               -> book_als_factors.npy
--------------------------------------------------------------------------------
Migrated: 3 files | Skipped: 10 files

[DRY RUN] Skipping checksum verification

================================================================================
MIGRATION SUMMARY
================================================================================
Mode: DRY RUN
Total files: 13
Files to migrate: 3

Report saved to: ops/models/migration_report_dry_run_20251217_123456.json
================================================================================

Dry run complete. Run with --execute to perform actual migration.
```

## When to Use This

**Before:**
- Running automated training pipeline
- Deploying to production
- After pulling trained models from Azure VM

**Recommended workflow:**
1. Run `--dry-run` to preview
2. Review the report JSON
3. Run `--execute` to migrate
4. Verify application still works
5. Commit changes

## Related Documentation

- See `models/core/README.md` for path definitions
- See `models/data/README.md` for loading functions
- See `naming_references.md` for complete artifact naming guide
