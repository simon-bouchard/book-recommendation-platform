# ops/models/migrate_artifacts.py
"""
Migration script to reorganize model artifacts with new naming conventions.

This script:
1. Renames artifacts to descriptive names (book_embs.npy -> book_subject_embeddings.npy)
2. Reorganizes into subdirectories (embeddings/, attention/, scoring/)
3. Verifies integrity with SHA256 checksums
4. Creates detailed migration report
"""

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.core import PATHS


def compute_checksum(filepath: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        filepath: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


class ArtifactMigration:
    """
    Manages migration of model artifacts to new structure.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize migration.

        Args:
            dry_run: If True, only simulate changes without modifying files
        """
        self.dry_run = dry_run
        self.old_data_dir = PROJECT_ROOT / "models" / "data"

        # Migration mapping: old_path -> new_path
        self.file_migrations: List[Tuple[Path, Path]] = []
        self.dir_migrations: List[Tuple[Path, Path]] = []
        self.checksums_before: Dict[str, str] = {}
        self.checksums_after: Dict[str, str] = {}
        self.errors: List[str] = []

        self._build_migration_map()

    def _build_migration_map(self) -> None:
        """Build mapping of old paths to new paths."""

        # File migrations: Embeddings
        self.file_migrations.extend(
            [
                (self.old_data_dir / "book_embs.npy", PATHS.book_subject_embeddings),
                (self.old_data_dir / "book_ids.json", PATHS.book_subject_ids),
                (self.old_data_dir / "book_als_emb.npy", PATHS.book_als_factors),
                (self.old_data_dir / "book_als_ids.json", PATHS.book_als_ids),
                (self.old_data_dir / "user_als_emb.npy", PATHS.user_als_factors),
                (self.old_data_dir / "user_als_ids.json", PATHS.user_als_ids),
            ]
        )

        # File migrations: Attention models
        attention_variants = ["scalar", "perdim", "selfattn", "selfattn_perdim"]
        for variant in attention_variants:
            if variant == "scalar":
                old_name = "subject_attention_components.pth"
            else:
                old_name = f"subject_attention_components_{variant}.pth"

            old_path = self.old_data_dir / old_name
            new_path = PATHS.get_attention_path(variant)
            self.file_migrations.append((old_path, new_path))

        # File migrations: Scoring
        self.file_migrations.extend(
            [
                (self.old_data_dir / "bayesian_tensor.npy", PATHS.bayesian_scores),
                (self.old_data_dir / "gbt_cold.pickle", PATHS.gbt_cold),
                (self.old_data_dir / "gbt_warm.pickle", PATHS.gbt_warm),
            ]
        )

        # Directory migrations: Semantic indexes
        self.dir_migrations.extend(
            [
                (self.old_data_dir / "baseline", PATHS.semantic_index_baseline),
                (self.old_data_dir / "baseline_clean", PATHS.semantic_index_baseline_clean),
                (self.old_data_dir / "enriched_v1", PATHS.semantic_index_enriched_v1),
                (
                    self.old_data_dir / "enriched_v1_subjects",
                    PATHS.semantic_index_enriched_v1_subjects,
                ),
                (self.old_data_dir / "enriched_v2", PATHS.semantic_index_enriched_v2),
                (
                    self.old_data_dir / "enriched_v2_subjects",
                    PATHS.semantic_index_enriched_v2_subjects,
                ),
            ]
        )

    def check_files_exist(self) -> bool:
        """
        Check which source files and directories exist.

        Returns:
            True if at least one file/directory exists, False otherwise
        """
        found_count = 0
        missing_count = 0

        print("\nChecking source files:")
        print("-" * 80)

        for old_path, new_path in self.file_migrations:
            if old_path.exists():
                size_mb = old_path.stat().st_size / (1024 * 1024)
                print(f"  [FOUND] {old_path.name:45s} ({size_mb:7.2f} MB)")
                found_count += 1
            else:
                print(f"  [SKIP]  {old_path.name:45s} (not found)")
                missing_count += 1

        print("\nChecking source directories:")
        print("-" * 80)

        for old_path, new_path in self.dir_migrations:
            if old_path.exists() and old_path.is_dir():
                # Calculate total size of directory
                total_size = sum(f.stat().st_size for f in old_path.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                file_count = sum(1 for f in old_path.rglob("*") if f.is_file())
                print(f"  [FOUND] {old_path.name:35s} ({size_mb:7.2f} MB, {file_count} files)")
                found_count += 1
            else:
                print(f"  [SKIP]  {old_path.name:35s} (not found)")
                missing_count += 1

        print("-" * 80)
        print(f"Found: {found_count} items | Missing: {missing_count} items")

        return found_count > 0

    def compute_checksums_before(self) -> None:
        """Compute checksums of all existing source files (not directories)."""
        print("\nComputing checksums for source files:")
        print("-" * 80)

        for old_path, _ in self.file_migrations:
            if old_path.exists():
                print(f"  Hashing {old_path.name}...", end=" ", flush=True)
                checksum = compute_checksum(old_path)
                self.checksums_before[str(old_path)] = checksum
                print(f"{checksum[:16]}...")

        print(f"\nComputed {len(self.checksums_before)} checksums")
        print("Note: Directory contents are moved as-is without checksum verification")

    def create_directories(self) -> None:
        """Create target directory structure."""
        print("\nCreating directory structure:")
        print("-" * 80)

        directories = [
            PATHS.embeddings_dir,
            PATHS.attention_dir,
            PATHS.scoring_dir,
            PATHS.semantic_indexes_dir,
        ]

        for directory in directories:
            if self.dry_run:
                print(f"  [DRY RUN] Would create: {directory}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"  [CREATED] {directory}")

    def migrate_files(self) -> None:
        """Move and rename files and directories to new locations."""
        print("\nMigrating files:")
        print("-" * 80)

        migrated_count = 0
        skipped_count = 0

        for old_path, new_path in self.file_migrations:
            if not old_path.exists():
                skipped_count += 1
                continue

            if self.dry_run:
                print(f"  [DRY RUN] {old_path.name:45s} -> {new_path.name}")
                migrated_count += 1
            else:
                try:
                    # Ensure parent directory exists
                    new_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move file
                    shutil.move(str(old_path), str(new_path))
                    print(f"  [MOVED]   {old_path.name:45s} -> {new_path.name}")
                    migrated_count += 1

                except Exception as e:
                    error_msg = f"Failed to move {old_path.name}: {e}"
                    self.errors.append(error_msg)
                    print(f"  [ERROR]   {error_msg}")

        print("-" * 80)
        print(f"Migrated: {migrated_count} files | Skipped: {skipped_count} files")

        print("\nMigrating directories:")
        print("-" * 80)

        dir_migrated = 0
        dir_skipped = 0

        for old_path, new_path in self.dir_migrations:
            if not old_path.exists() or not old_path.is_dir():
                dir_skipped += 1
                continue

            if self.dry_run:
                file_count = sum(1 for f in old_path.rglob("*") if f.is_file())
                print(f"  [DRY RUN] {old_path.name:35s} -> {new_path.name} ({file_count} files)")
                dir_migrated += 1
            else:
                try:
                    # Ensure parent directory exists
                    new_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move entire directory
                    shutil.move(str(old_path), str(new_path))
                    file_count = sum(1 for f in new_path.rglob("*") if f.is_file())
                    print(
                        f"  [MOVED]   {old_path.name:35s} -> {new_path.name} ({file_count} files)"
                    )
                    dir_migrated += 1

                except Exception as e:
                    error_msg = f"Failed to move directory {old_path.name}: {e}"
                    self.errors.append(error_msg)
                    print(f"  [ERROR]   {error_msg}")

        print("-" * 80)
        print(f"Migrated: {dir_migrated} directories | Skipped: {dir_skipped} directories")

    def verify_checksums(self) -> bool:
        """
        Verify checksums of migrated files match originals.
        Directories are not verified (moved as-is).

        Returns:
            True if all checksums match, False otherwise
        """
        if self.dry_run:
            print("\n[DRY RUN] Skipping checksum verification")
            return True

        print("\nVerifying file integrity:")
        print("-" * 80)

        all_match = True
        verified_count = 0

        for old_path, new_path in self.file_migrations:
            old_checksum = self.checksums_before.get(str(old_path))

            if old_checksum is None:
                continue

            if not new_path.exists():
                print(f"  [ERROR]   {new_path.name:45s} not found after migration!")
                all_match = False
                continue

            print(f"  Verifying {new_path.name}...", end=" ", flush=True)
            new_checksum = compute_checksum(new_path)
            self.checksums_after[str(new_path)] = new_checksum

            if old_checksum == new_checksum:
                print("OK")
                verified_count += 1
            else:
                print("MISMATCH!")
                all_match = False
                self.errors.append(
                    f"Checksum mismatch for {new_path.name}: "
                    f"{old_checksum[:16]}... != {new_checksum[:16]}..."
                )

        print("-" * 80)
        if all_match:
            print(f"All {verified_count} files verified successfully")
            print("Note: Directory contents moved as-is (not verified)")
        else:
            print("VERIFICATION FAILED - Some checksums do not match!")

        return all_match

    def create_report(self) -> Dict:
        """
        Create detailed migration report.

        Returns:
            Dictionary with migration details
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "total_files": len(self.file_migrations),
            "total_directories": len(self.dir_migrations),
            "migrated_files": len(self.checksums_after) if not self.dry_run else "N/A",
            "errors": self.errors,
            "file_migrations": [],
            "directory_migrations": [],
        }

        # File migrations
        for old_path, new_path in self.file_migrations:
            migration_info = {
                "old_path": str(old_path.relative_to(PROJECT_ROOT)),
                "new_path": str(new_path.relative_to(PROJECT_ROOT)),
                "exists_before": old_path.exists() if self.dry_run else None,
                "exists_after": new_path.exists() if not self.dry_run else None,
            }

            if not self.dry_run:
                old_checksum = self.checksums_before.get(str(old_path))
                new_checksum = self.checksums_after.get(str(new_path))

                if old_checksum:
                    migration_info["checksum_before"] = old_checksum
                if new_checksum:
                    migration_info["checksum_after"] = new_checksum
                    migration_info["verified"] = old_checksum == new_checksum

            report["file_migrations"].append(migration_info)

        # Directory migrations
        for old_path, new_path in self.dir_migrations:
            migration_info = {
                "old_path": str(old_path.relative_to(PROJECT_ROOT)),
                "new_path": str(new_path.relative_to(PROJECT_ROOT)),
                "exists_before": old_path.exists() if self.dry_run else None,
                "exists_after": new_path.exists() if not self.dry_run else None,
            }

            if old_path.exists() and old_path.is_dir():
                file_count = sum(1 for f in old_path.rglob("*") if f.is_file())
                migration_info["file_count"] = file_count

            report["directory_migrations"].append(migration_info)

        return report

    def save_report(self, report: Dict) -> Path:
        """
        Save migration report to file.

        Args:
            report: Migration report dictionary

        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "dry_run" if self.dry_run else "actual"
        report_filename = f"migration_report_{mode}_{timestamp}.json"
        report_path = PROJECT_ROOT / "ops" / "models" / report_filename

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path

    def run(self) -> bool:
        """
        Execute the migration process.

        Returns:
            True if migration successful, False otherwise
        """
        mode = "DRY RUN" if self.dry_run else "ACTUAL MIGRATION"
        print("=" * 80)
        print(f"MODEL ARTIFACTS MIGRATION - {mode}")
        print("=" * 80)

        # Check source files
        if not self.check_files_exist():
            print("\nNo source files found. Nothing to migrate.")
            return False

        # Compute checksums before migration
        self.compute_checksums_before()

        # Create directory structure
        self.create_directories()

        # Migrate files
        self.migrate_files()

        # Verify checksums (if not dry run)
        if not self.dry_run:
            verification_ok = self.verify_checksums()

            if not verification_ok:
                print("\nERROR: Checksum verification failed!")
                print("Some files may be corrupted. Check the report for details.")

                # Create error report
                report = self.create_report()
                report_path = self.save_report(report)
                print(f"\nError report saved to: {report_path}")

                return False

        # Create and save report
        report = self.create_report()
        report_path = self.save_report(report)

        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"Mode: {mode}")
        print(f"Total files: {len(self.file_migrations)}")
        print(f"Total directories: {len(self.dir_migrations)}")

        if self.dry_run:
            print(f"Files to migrate: {len(self.checksums_before)}")
            dirs_to_migrate = sum(1 for old, _ in self.dir_migrations if old.exists())
            print(f"Directories to migrate: {dirs_to_migrate}")
        else:
            print(f"Files migrated: {len(self.checksums_after)}")
            dirs_migrated = sum(1 for _, new in self.dir_migrations if new.exists())
            print(f"Directories migrated: {dirs_migrated}")
            print(f"Errors: {len(self.errors)}")

        print(f"\nReport saved to: {report_path}")
        print("=" * 80)

        if self.errors:
            print("\nERRORS ENCOUNTERED:")
            for error in self.errors:
                print(f"  - {error}")
            return False

        return True


def main():
    """Main entry point for migration script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate model artifacts to new structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without modifying files
  python ops/models/migrate_artifacts.py --dry-run

  # Execute migration
  python ops/models/migrate_artifacts.py --execute

  # Force execution without confirmation
  python ops/models/migrate_artifacts.py --execute --force
        """,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files (default)"
    )

    parser.add_argument(
        "--execute", action="store_true", help="Execute the migration (modifies files)"
    )

    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    # Default to dry run if neither specified
    dry_run = not args.execute

    if not dry_run and not args.force:
        print("\n" + "!" * 80)
        print("WARNING: This will modify files in your repository!")
        print("!" * 80)
        print("\nThis migration will:")
        print("  1. Rename model artifacts to new naming convention")
        print("  2. Move files to new directory structure")
        print("  3. Verify integrity with checksums")
        print("\nRecommendation: Run with --dry-run first to preview changes.")

        response = input("\nProceed with migration? [yes/NO]: ").strip().lower()

        if response != "yes":
            print("\nMigration cancelled.")
            return

    # Run migration
    migration = ArtifactMigration(dry_run=dry_run)
    success = migration.run()

    if not success:
        sys.exit(1)

    if dry_run:
        print("\nDry run complete. Run with --execute to perform actual migration.")
    else:
        print("\nMigration complete!")


if __name__ == "__main__":
    main()
