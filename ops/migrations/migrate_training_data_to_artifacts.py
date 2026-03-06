# ops/migrations/migrate_training_data_to_artifacts.py
"""
One-time migration that copies the training data snapshot from
models/training/data/ into the data/ subdirectory of every existing versioned
artifact directory and into staging/data/.

After this migration, data/ is treated as a fully versioned artifact subdir.
Each version carries its own data snapshot, ensuring model weights and the
data they were trained on are always in sync — including across rollbacks.

Run from the project root:
    python ops/migrations/migrate_training_data_to_artifacts.py [--dry-run] [--cleanup]
"""

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.core.artifact_registry import (
    VersionManifest,
    _compute_checksums,
    _read_manifest,
    _write_manifest,
)
from models.core.paths import PATHS

_EXPECTED_PKL_NAMES = frozenset(
    {
        "interactions.pkl",
        "users.pkl",
        "books.pkl",
        "book_subjects.pkl",
        "user_fav_subjects.pkl",
        "subjects.pkl",
    }
)

_EXPORT_SENTINEL = ".export_complete"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MigrationPlan:
    """
    Describes what the migration will do before any changes are made.

    Built during the inspection phase and either printed (dry-run) or
    executed (live run).
    """

    source_dir: Path
    source_pkl_names: List[str]
    versions_to_migrate: List[str]
    versions_already_done: List[str]
    migrate_staging: bool
    no_source_data: bool = False
    no_versions: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def nothing_to_do(self) -> bool:
        """True if there is no work to perform."""
        return self.no_source_data or (not self.versions_to_migrate and not self.migrate_staging)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _collect_source_pkls(source_dir: Path) -> List[str]:
    """
    Return the sorted names of pkl files present in the source directory.

    Only files whose names are in _EXPECTED_PKL_NAMES are included. Files not
    in that set are ignored with a note — they may be leftovers from old runs.
    """
    return sorted(
        f.name for f in source_dir.iterdir() if f.is_file() and f.name in _EXPECTED_PKL_NAMES
    )


def _version_data_dir(version_id: str) -> Path:
    """Return the data/ subdir path for a given version."""
    return PATHS.versions_dir / version_id / "data"


def _is_version_already_done(version_id: str) -> bool:
    """
    Return True if this version already has a populated data/ subdir.

    A version is considered done if its data/ directory exists and contains
    at least one pkl file. An empty directory is treated as incomplete.
    """
    data_dir = _version_data_dir(version_id)
    if not data_dir.exists():
        return False
    return any(data_dir.glob("*.pkl"))


def _staging_already_done() -> bool:
    """Return True if staging/data/ already contains pkl files."""
    if not PATHS.staging_data_dir.exists():
        return False
    return any(PATHS.staging_data_dir.glob("*.pkl"))


def build_plan() -> MigrationPlan:
    """
    Inspect the current state and build a MigrationPlan without touching anything.

    Returns:
        A MigrationPlan describing what the migration will do.
    """
    source_dir = PATHS.training_root / "data"
    warnings: List[str] = []

    if not source_dir.exists() or not any(source_dir.glob("*.pkl")):
        return MigrationPlan(
            source_dir=source_dir,
            source_pkl_names=[],
            versions_to_migrate=[],
            versions_already_done=[],
            migrate_staging=False,
            no_source_data=True,
        )

    source_pkl_names = _collect_source_pkls(source_dir)

    missing_pkls = _EXPECTED_PKL_NAMES - set(source_pkl_names)
    if missing_pkls:
        warnings.append(
            f"The following expected pkl files are absent from the source: "
            f"{sorted(missing_pkls)}. They will not be present in migrated versions."
        )

    if not PATHS.versions_dir.exists() or not any(PATHS.versions_dir.iterdir()):
        return MigrationPlan(
            source_dir=source_dir,
            source_pkl_names=source_pkl_names,
            versions_to_migrate=[],
            versions_already_done=[],
            migrate_staging=not _staging_already_done(),
            no_versions=True,
            warnings=warnings,
        )

    all_version_ids = sorted(
        d.name
        for d in PATHS.versions_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )

    versions_to_migrate = [v for v in all_version_ids if not _is_version_already_done(v)]
    versions_already_done = [v for v in all_version_ids if _is_version_already_done(v)]

    if versions_already_done:
        warnings.append(
            f"The following versions already have a data/ subdir and will be skipped: "
            f"{versions_already_done}."
        )

    return MigrationPlan(
        source_dir=source_dir,
        source_pkl_names=source_pkl_names,
        versions_to_migrate=versions_to_migrate,
        versions_already_done=versions_already_done,
        migrate_staging=not _staging_already_done(),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _copy_data_into_dir(source_dir: Path, dest_dir: Path) -> None:
    """
    Copy all pkl files from source_dir into dest_dir.

    dest_dir is created if it does not exist. Existing files are overwritten
    so this is safe to call on a partially populated directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for pkl in source_dir.iterdir():
        if pkl.is_file() and pkl.name in _EXPECTED_PKL_NAMES:
            shutil.copy2(pkl, dest_dir / pkl.name)


def _rewrite_manifest(version_id: str) -> None:
    """
    Recompute checksums for a version and rewrite its manifest.json.

    Called after injecting data/ into a version directory so that the manifest
    reflects the complete artifact set. All existing manifest fields are
    preserved; only the checksums dict is updated.

    Args:
        version_id: The version whose manifest should be updated.
    """
    version_dir = PATHS.versions_dir / version_id
    existing = _read_manifest(version_dir)
    updated_checksums = _compute_checksums(version_dir)

    updated = VersionManifest(
        version_id=existing.version_id,
        created_at=existing.created_at,
        git_commit=existing.git_commit,
        pad_idx=existing.pad_idx,
        attn_strategy=existing.attn_strategy,
        metrics=existing.metrics,
        checksums=updated_checksums,
    )
    _write_manifest(version_dir, updated)


def _migrate_version(version_id: str, source_dir: Path) -> None:
    """
    Copy pkl files into a version's data/ subdir and rewrite its manifest.

    Args:
        version_id: The version to migrate.
        source_dir: Directory containing the source pkl files.
    """
    dest_dir = _version_data_dir(version_id)
    print(f"  Copying data into version '{version_id}'...")
    _copy_data_into_dir(source_dir, dest_dir)
    print(f"  Rewriting manifest for version '{version_id}'...")
    _rewrite_manifest(version_id)


def _migrate_staging(source_dir: Path) -> None:
    """
    Copy pkl files into staging/data/ and write the export sentinel.

    Writing the sentinel marks staging/data/ as complete, which means
    staging is immediately promotable after migration if all other staging
    subdirs are also populated.

    Args:
        source_dir: Directory containing the source pkl files.
    """
    dest_dir = PATHS.staging_data_dir
    print("  Copying data into staging/data/...")
    _copy_data_into_dir(source_dir, dest_dir)
    (dest_dir / _EXPORT_SENTINEL).touch()
    print(f"  Wrote sentinel: {dest_dir / _EXPORT_SENTINEL}")


# ---------------------------------------------------------------------------
# Plan display
# ---------------------------------------------------------------------------


def _print_plan(plan: MigrationPlan) -> None:
    """Print a human-readable description of the migration plan."""
    print("Migration plan (dry-run — no changes will be made):")
    print()

    if plan.no_source_data:
        print(f"  No pkl files found in source directory: '{plan.source_dir}'")
        print("  Nothing to do.")
        return

    print(f"  Source directory  : {plan.source_dir}")
    print(f"  Source pkl files  : {plan.source_pkl_names}")
    print()

    if plan.no_versions:
        print("  No registered versions found.")
    else:
        if plan.versions_to_migrate:
            print(f"  Versions to migrate ({len(plan.versions_to_migrate)}):")
            for v in plan.versions_to_migrate:
                print(f"    {v}  ->  {_version_data_dir(v)}/")
        if plan.versions_already_done:
            print(f"  Versions already done (will be skipped):")
            for v in plan.versions_already_done:
                print(f"    {v}")

    print()
    if plan.migrate_staging:
        print(f"  Staging           : {PATHS.staging_data_dir}/  (will be populated)")
        print(
            f"  Sentinel          : {PATHS.staging_data_dir / _EXPORT_SENTINEL}  (will be written)"
        )
    else:
        print(f"  Staging           : already populated, will be skipped")

    if plan.warnings:
        print()
        print("  Warnings:")
        for w in plan.warnings:
            print(f"    - {w}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _verify_cleanup_safe(source_dir: Path, versions_to_verify: List[str]) -> bool:
    """
    Check whether it is safe to remove the source training data directory.

    Verifies that every registered version has the same set of pkl filenames as
    the source. All versions must pass for cleanup to proceed — a single
    mismatch aborts the whole cleanup to avoid partial data loss.

    Args:
        source_dir: The original training data directory.
        versions_to_verify: All version IDs to check, regardless of whether they
                            were migrated in this session or a previous one.

    Returns:
        True if safe to remove source_dir, False otherwise.
    """
    source_pkls = {f.name for f in source_dir.iterdir() if f.name in _EXPECTED_PKL_NAMES}
    safe = True

    for version_id in versions_to_verify:
        dest_dir = _version_data_dir(version_id)
        if not dest_dir.exists():
            print(f"  WARNING: data/ dir missing for version '{version_id}'. Aborting cleanup.")
            safe = False
            continue

        dest_pkls = {f.name for f in dest_dir.iterdir() if f.is_file() and f.name.endswith(".pkl")}
        missing = source_pkls - dest_pkls
        if missing:
            print(
                f"  WARNING: Version '{version_id}' is missing pkl files: {sorted(missing)}. "
                "Aborting cleanup."
            )
            safe = False

    return safe


def _run_cleanup(source_dir: Path) -> None:
    """
    Remove the source training data directory after verifying all registered versions.

    Collects every version currently on disk and verifies each has a complete
    data/ subdir before deleting the source. This means cleanup is safe to run
    independently of the migration itself — you can migrate in one session and
    clean up in another once you have verified the results.

    All versions must pass the safety check before anything is deleted.

    Args:
        source_dir: The original training data directory to remove.
    """
    print()
    print("Running cleanup of source training data directory...")

    if not source_dir.exists():
        print(f"  '{source_dir}' does not exist. Nothing to remove.")
        return

    all_version_ids = (
        sorted(
            d.name
            for d in PATHS.versions_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        )
        if PATHS.versions_dir.exists()
        else []
    )

    if not all_version_ids:
        print(
            "  No registered versions found. "
            "Skipping cleanup to avoid removing data before it has been versioned."
        )
        return

    if _verify_cleanup_safe(source_dir, all_version_ids):
        shutil.rmtree(source_dir)
        print(f"  Removed '{source_dir}'.")
    else:
        print()
        print(
            "Source directory was not removed. Inspect the warnings above and "
            "remove it manually once you have verified the versioned artifacts."
        )


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute_migration(plan: MigrationPlan) -> List[str]:
    """
    Execute the migration described by the plan.

    Copies pkl files into each version's data/ subdir, rewrites their
    manifests, and populates staging/data/. Cleanup of the source directory
    is handled separately by _run_cleanup() so it can be run independently
    in a subsequent invocation.

    Args:
        plan: A MigrationPlan produced by build_plan().

    Returns:
        List of version IDs that were migrated in this run.
    """
    if plan.no_source_data:
        print(f"No pkl files found in '{plan.source_dir}'. Nothing to do.")
        return []

    if plan.warnings:
        for warning in plan.warnings:
            print(f"Warning: {warning}")
        print()

    versions_migrated: List[str] = []

    if plan.no_versions:
        print("No registered versions found. Skipping version migration.")
    elif not plan.versions_to_migrate:
        print("All versions already have a data/ subdir. Skipping version migration.")
    else:
        print(f"Migrating {len(plan.versions_to_migrate)} version(s)...")
        for version_id in plan.versions_to_migrate:
            _migrate_version(version_id, plan.source_dir)
            versions_migrated.append(version_id)
        print()

    if plan.migrate_staging:
        print("Migrating staging/data/...")
        _migrate_staging(plan.source_dir)
        print()
    else:
        print("Staging/data/ already populated. Skipping.")

    print("Migration complete.")
    if versions_migrated:
        print(f"  Versions migrated : {versions_migrated}")
    print(f"  Staging data dir  : {PATHS.staging_data_dir}")

    return versions_migrated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_training_data_to_artifacts",
        description=(
            "Copy the training data snapshot from models/training/data/ into "
            "the data/ subdir of every registered version and into staging/data/. "
            "Safe to run multiple times — already-migrated versions are skipped."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what the migration would do without making any changes.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "After a successful migration, remove models/training/data/. "
            "Only proceeds if every migrated version passes a file-set verification. "
            "Has no effect in dry-run mode."
        ),
    )
    return parser


def main() -> None:
    """Entry point for the migration CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    plan = build_plan()

    if args.dry_run:
        _print_plan(plan)
        return

    execute_migration(plan)

    if args.cleanup:
        _run_cleanup(plan.source_dir)


if __name__ == "__main__":
    main()
