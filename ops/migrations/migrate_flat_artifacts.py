# ops/migrations/migrate_flat_artifacts.py
"""
One-time migration from the legacy flat artifact layout to the versioned layout.

Copies existing flat artifacts (models/artifacts/{embeddings,attention,scoring}/)
into the versioned store under a generated version ID derived from file mtimes,
writes the active_version pointer, and scaffolds the staging/ directory.

Run from the project root:
    python ops/migrations/migrate_flat_artifacts.py [--dry-run] [--cleanup]
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.core.artifact_registry import VersionManifest, register_existing_version
from models.core.paths import _VERSIONED_SUBDIRS, PATHS

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

    version_id: str
    subdirs_to_migrate: List[str]
    flat_file_counts: Dict[str, int]
    already_migrated: bool = False
    no_artifacts: bool = False
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _collect_flat_files(artifacts_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan each versioned subdir in the legacy flat layout.

    Returns a dict mapping subdir name to a sorted list of files found inside
    it. Subdirs that do not exist or are empty are omitted from the result.
    """
    result: Dict[str, List[Path]] = {}
    for subdir_name in _VERSIONED_SUBDIRS:
        subdir = artifacts_dir / subdir_name
        if not subdir.exists():
            continue
        files = sorted(f for f in subdir.rglob("*") if f.is_file())
        if files:
            result[subdir_name] = files
    return result


def _derive_version_id(flat_files: Dict[str, List[Path]]) -> str:
    """
    Derive a version ID from the flat artifact files.

    Timestamp comes from the most recent mtime across all files, formatted
    as YYYYMMDD-HHMM (UTC). The hash suffix is the first 6 characters of
    the MD5 of the sorted relative filenames, giving a stable fingerprint
    of which artifacts were present.

    Args:
        flat_files: Dict of subdir name to list of Path objects, as returned
                    by _collect_flat_files().

    Returns:
        Version ID string, e.g. '20260110-0900-c8d1e4'.
    """
    all_files: List[Path] = [f for files in flat_files.values() for f in files]

    latest_mtime = max(f.stat().st_mtime for f in all_files)
    timestamp = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).strftime("%Y%m%d-%H%M")

    artifacts_dir = PATHS.artifacts_dir
    sorted_names = sorted(str(f.relative_to(artifacts_dir)) for f in all_files)
    name_hash = hashlib.md5("\n".join(sorted_names).encode()).hexdigest()[:6]

    return f"{timestamp}-{name_hash}"


def build_plan(artifacts_dir: Path) -> MigrationPlan:
    """
    Inspect the current state and build a MigrationPlan without touching anything.

    Args:
        artifacts_dir: The top-level artifacts directory (PATHS.artifacts_dir).

    Returns:
        A MigrationPlan describing what the migration will do.
    """
    if PATHS.active_version_file.exists():
        active_id = PATHS.active_version_file.read_text().strip()
        return MigrationPlan(
            version_id=active_id,
            subdirs_to_migrate=[],
            flat_file_counts={},
            already_migrated=True,
        )

    flat_files = _collect_flat_files(artifacts_dir)

    if not flat_files:
        return MigrationPlan(
            version_id="",
            subdirs_to_migrate=[],
            flat_file_counts={},
            no_artifacts=True,
        )

    version_id = _derive_version_id(flat_files)
    warnings: List[str] = []

    missing_subdirs = [s for s in _VERSIONED_SUBDIRS if s not in flat_files]
    if missing_subdirs:
        warnings.append(
            f"The following subdirs have no files and will not be included: "
            f"{missing_subdirs}. This is expected if those components have "
            f"never been trained."
        )

    return MigrationPlan(
        version_id=version_id,
        subdirs_to_migrate=list(flat_files.keys()),
        flat_file_counts={k: len(v) for k, v in flat_files.items()},
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _print_plan(plan: MigrationPlan) -> None:
    """Print a human-readable description of the migration plan."""
    print("Migration plan (dry-run — no changes will be made):")
    print()

    if plan.already_migrated:
        print(f"  Already migrated. Active version: '{plan.version_id}'")
        print("  Nothing to do.")
        return

    if plan.no_artifacts:
        print("  No flat artifact files found.")
        print("  The directory structure will be scaffolded (staging/, versions/).")
        print("  No active_version pointer will be written.")
        return

    print(f"  Version ID      : {plan.version_id}")
    print("  Subdirs to copy :")
    for subdir, count in plan.flat_file_counts.items():
        print(f"    {subdir}/  ({count} file(s))")
    print(f"  Destination     : {PATHS.versions_dir / plan.version_id}/")
    print(f"  Pointer file    : {PATHS.active_version_file}")
    print(f"  Staging dir     : {PATHS.staging_dir}/")

    if plan.warnings:
        print()
        print("  Warnings:")
        for w in plan.warnings:
            print(f"    - {w}")


def _assemble_source_temp_dir(
    flat_files: Dict[str, List[Path]],
    artifacts_dir: Path,
) -> Path:
    """
    Copy the flat artifact subdirs into a temporary directory.

    register_existing_version expects a source directory whose contents
    map directly to the versioned subdirs. Building a temp dir avoids
    passing artifacts_dir directly, which may contain staging/, versions/,
    and other non-artifact content by the time the migration runs.

    The caller is responsible for deleting the temp dir.

    Args:
        flat_files: Dict of subdir name to list of Path objects.
        artifacts_dir: The legacy flat artifacts directory.

    Returns:
        Path to the temporary directory.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="bookrec_migration_"))

    for subdir_name in flat_files:
        src_subdir = artifacts_dir / subdir_name
        dst_subdir = tmp_dir / subdir_name
        shutil.copytree(src_subdir, dst_subdir)

    return tmp_dir


def _verify_cleanup_safe(subdir_name: str, version_id: str, expected_count: int) -> bool:
    """
    Check whether it is safe to delete a flat subdir.

    Compares the file count in the versioned copy against the expected count
    from the flat source. Returns False and prints a warning if they differ.

    Args:
        subdir_name: The artifact subdir name (e.g. 'embeddings').
        version_id: The version that was just registered.
        expected_count: Number of files that were in the flat source subdir.

    Returns:
        True if safe to delete, False otherwise.
    """
    versioned_subdir = PATHS.versions_dir / version_id / subdir_name
    if not versioned_subdir.exists():
        print(
            f"  WARNING: Versioned subdir '{versioned_subdir}' does not exist. "
            f"Skipping cleanup of '{subdir_name}/'."
        )
        return False

    versioned_count = sum(1 for f in versioned_subdir.rglob("*") if f.is_file())
    if versioned_count != expected_count:
        print(
            f"  WARNING: File count mismatch in '{subdir_name}/'. "
            f"Expected {expected_count}, found {versioned_count} in versioned copy. "
            f"Skipping cleanup to avoid data loss."
        )
        return False

    return True


def _run_cleanup(plan: MigrationPlan, artifacts_dir: Path) -> None:
    """
    Delete the original flat subdirs after verifying the versioned copies.

    Each subdir is verified independently — a failed check on one does not
    prevent cleanup of the others.

    Args:
        plan: The executed MigrationPlan (must not be dry-run or no-op).
        artifacts_dir: The top-level artifacts directory.
    """
    print()
    print("Running cleanup of legacy flat directories...")
    any_skipped = False

    for subdir_name, expected_count in plan.flat_file_counts.items():
        if _verify_cleanup_safe(subdir_name, plan.version_id, expected_count):
            flat_subdir = artifacts_dir / subdir_name
            shutil.rmtree(flat_subdir)
            print(f"  Removed '{flat_subdir}'.")
        else:
            any_skipped = True

    if any_skipped:
        print()
        print(
            "Some flat directories were not removed. Inspect the warnings above "
            "and remove them manually once you have verified the versioned artifacts."
        )


def execute_migration(plan: MigrationPlan, cleanup: bool) -> Optional[VersionManifest]:
    """
    Execute the migration described by the plan.

    Scaffolds staging/ and versions/, assembles a temp source directory,
    delegates to register_existing_version, then optionally cleans up
    the legacy flat dirs.

    Args:
        plan: A MigrationPlan produced by build_plan().
        cleanup: If True, attempt to remove legacy flat dirs after migration.

    Returns:
        The registered VersionManifest, or None if no migration was needed.
    """
    if plan.already_migrated:
        print(f"Already migrated. Active version: '{plan.version_id}'. Nothing to do.")
        return None

    PATHS.ensure_artifact_dirs()

    if plan.no_artifacts:
        print("No flat artifact files found. Directory structure has been scaffolded.")
        print("Run a full training cycle to populate artifacts and promote staging.")
        return None

    if plan.warnings:
        for warning in plan.warnings:
            print(f"Warning: {warning}")

    flat_files = _collect_flat_files(PATHS.artifacts_dir)
    tmp_dir: Optional[Path] = None

    try:
        print(f"Assembling temporary source directory for version '{plan.version_id}'...")
        tmp_dir = _assemble_source_temp_dir(flat_files, PATHS.artifacts_dir)

        manifest = register_existing_version(
            source_dir=tmp_dir,
            version_id=plan.version_id,
        )
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temporary directory '{tmp_dir}'.")

    print()
    print("Migration complete.")
    print(f"  Version ID   : {manifest.version_id}")
    print(f"  Created at   : {manifest.created_at}")
    print(f"  Artifacts    : {PATHS.versions_dir / manifest.version_id}")
    print(f"  Pointer      : {PATHS.active_version_file}")

    if cleanup:
        _run_cleanup(plan, PATHS.artifacts_dir)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_flat_artifacts",
        description=(
            "Migrate legacy flat model artifacts into the versioned artifact store. "
            "Safe to run multiple times — exits cleanly if already migrated."
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
            "After a successful migration, delete the original flat artifact "
            "directories. Each directory is verified against its versioned copy "
            "before deletion. Has no effect in dry-run mode."
        ),
    )
    return parser


def main() -> None:
    """Entry point for the migration CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    plan = build_plan(PATHS.artifacts_dir)

    if args.dry_run:
        _print_plan(plan)
        return

    if plan.already_migrated:
        print(f"Already migrated. Active version: '{plan.version_id}'.")
        if args.cleanup:
            flat_files = _collect_flat_files(PATHS.artifacts_dir)
            if flat_files:
                plan.subdirs_to_migrate = list(flat_files.keys())
                plan.flat_file_counts = {k: len(v) for k, v in flat_files.items()}
                _run_cleanup(plan, PATHS.artifacts_dir)
            else:
                print("No legacy flat directories found. Nothing to clean up.")
        return

    execute_migration(plan, cleanup=args.cleanup)


if __name__ == "__main__":
    main()
