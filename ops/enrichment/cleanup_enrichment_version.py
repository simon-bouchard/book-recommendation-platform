#!/usr/bin/env python3
# ops/cleanup_enrichment_version.py
"""
Comprehensive version cleanup: SQL + Kafka + Bronze + Checkpoints + Embeddings
"""

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pymysql


def load_env():
    from dotenv import load_dotenv

    env_files = [Path("/etc/bookrec.env"), Path(".env")]
    for env_file in env_files:
        if env_file.exists():
            print(f"Loading: {env_file}")
            load_dotenv(env_file, override=True)
            return
    print("⚠️  No .env found")


def get_db_connection():
    jdbc_url = os.getenv("JDBC_URL", "jdbc:mysql://127.0.0.1:3306/bookrec_db")
    user = os.getenv("JDBC_USER", "bookrec")
    password = os.getenv("JDBC_PASS")

    if not password:
        password = input("Enter MySQL password: ")

    parsed = urlparse(jdbc_url.replace("jdbc:", "", 1))
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=user,
        password=password,
        database=parsed.path.lstrip("/") or "bookrec_db",
        charset="utf8mb4",
    )
    return conn


def cleanup_sql(version: str, dry_run: bool = False):
    print("\n" + "=" * 70)
    print("SQL CLEANUP")
    print("=" * 70)

    conn = get_db_connection()
    cur = conn.cursor()

    # More comprehensive table list
    tables = ["book_tones", "book_genres", "book_vibes", "book_llm_subjects", "enrichment_errors"]

    print(f"\nChecking {version} data in SQL...")
    counts = {}
    total = 0

    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE tags_version = %s", (version,))
        count = cur.fetchone()[0]
        counts[table] = count
        total += count
        print(f"  {table}: {count:,}")

    # Check orphaned subjects (only linked to this version)
    cur.execute(
        """
        SELECT COUNT(DISTINCT ls.llm_subject_idx)
        FROM llm_subjects ls
        WHERE EXISTS (
            SELECT 1 FROM book_llm_subjects bls
            WHERE bls.llm_subject_idx = ls.llm_subject_idx
            AND bls.tags_version = %s
        )
        AND NOT EXISTS (
            SELECT 1 FROM book_llm_subjects bls
            WHERE bls.llm_subject_idx = ls.llm_subject_idx
            AND bls.tags_version != %s
        )
    """,
        (version, version),
    )
    orphaned_subjects = cur.fetchone()[0]

    # Check orphaned vibes (only linked to this version)
    cur.execute(
        """
        SELECT COUNT(DISTINCT v.vibe_id)
        FROM vibes v
        WHERE EXISTS (
            SELECT 1 FROM book_vibes bv
            WHERE bv.vibe_id = v.vibe_id
            AND bv.tags_version = %s
        )
        AND NOT EXISTS (
            SELECT 1 FROM book_vibes bv
            WHERE bv.vibe_id = v.vibe_id
            AND bv.tags_version != %s
        )
    """,
        (version, version),
    )
    orphaned_vibes = cur.fetchone()[0]

    if orphaned_subjects > 0:
        print(f"  ⚠️  {orphaned_subjects:,} subjects only linked to {version}")
        counts["llm_subjects"] = orphaned_subjects
        total += orphaned_subjects

    if orphaned_vibes > 0:
        print(f"  ⚠️  {orphaned_vibes:,} vibes only linked to {version}")
        counts["vibes"] = orphaned_vibes
        total += orphaned_vibes

    if total == 0:
        print(f"\n✅ No {version} data found in SQL")
        cur.close()
        conn.close()
        return

    if dry_run:
        print(f"\n[DRY RUN] Would delete {total:,} rows")
        cur.close()
        conn.close()
        return

    print(f"\n⚠️  About to delete {total:,} rows with tags_version='{version}'")
    confirm = input("Delete? (yes/no): ")

    if confirm != "yes":
        print("Skipped SQL cleanup")
        cur.close()
        conn.close()
        return

    # Delete in correct order (respecting foreign keys)
    deleted_total = 0

    # 1. Delete link tables first
    for table in tables:
        if counts.get(table, 0) > 0:
            cur.execute(f"DELETE FROM {table} WHERE tags_version = %s", (version,))
            deleted = cur.rowcount
            deleted_total += deleted
            print(f"  ✅ Deleted from {table}: {deleted:,}")

    # 2. Delete orphaned subjects (only after links are gone)
    if orphaned_subjects > 0:
        cur.execute("""
            DELETE ls FROM llm_subjects ls
            WHERE NOT EXISTS (
                SELECT 1 FROM book_llm_subjects bls
                WHERE bls.llm_subject_idx = ls.llm_subject_idx
            )
        """)
        deleted = cur.rowcount
        deleted_total += deleted
        print(f"  ✅ Deleted orphaned llm_subjects: {deleted:,}")

    # 3. Delete orphaned vibes (only after links are gone)
    if orphaned_vibes > 0:
        cur.execute("""
            DELETE v FROM vibes v
            WHERE NOT EXISTS (
                SELECT 1 FROM book_vibes bv
                WHERE bv.vibe_id = v.vibe_id
            )
        """)
        deleted = cur.rowcount
        deleted_total += deleted
        print(f"  ✅ Deleted orphaned vibes: {deleted:,}")

    conn.commit()
    print(f"\n✅ SQL cleanup complete: {deleted_total:,} rows deleted")

    cur.close()
    conn.close()


def cleanup_bronze(version: str, dry_run: bool = False):
    print("\n" + "=" * 70)
    print("BRONZE ARCHIVE CLEANUP (MinIO)")
    print("=" * 70)

    try:
        import subprocess

        # Check if mc is installed
        result = subprocess.run(["which", "mc"], capture_output=True)
        if result.returncode != 0:
            print("⚠️  MinIO client (mc) not installed")
            print("   Install: wget https://dl.min.io/client/mc/release/linux-amd64/mc")
            print("   Then: chmod +x mc && sudo mv mc /usr/local/bin/")
            return

        # Configure MinIO alias if not exists
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        minio_access = os.getenv("MINIO_ACCESS_KEY", "admin")
        minio_secret = os.getenv("MINIO_SECRET_KEY", "minio123456")

        subprocess.run(
            ["mc", "alias", "set", "myminio", minio_endpoint, minio_access, minio_secret],
            capture_output=True,
        )

        # Check version partitions
        paths_to_delete = [
            f"myminio/enrichment-bronze/enrich.results.v1/tags_version={version}/",
            f"myminio/enrichment-bronze/enrich.errors.v1/tags_version={version}/",
        ]

        found_data = False
        for path in paths_to_delete:
            result = subprocess.run(
                ["mc", "ls", path, "--recursive"], capture_output=True, text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                file_count = len(result.stdout.strip().split("\n"))
                print(f"  {path}: {file_count} files")
                found_data = True
            else:
                print(f"  {path}: no data")

        if not found_data:
            print(f"\n✅ No {version} data in bronze archive")
            return

        if dry_run:
            print(f"\n[DRY RUN] Would delete {version} partitions from bronze")
            return

        print(f"\n⚠️  About to delete {version} partitions from bronze archive")
        confirm = input("Delete? (yes/no): ")

        if confirm != "yes":
            print("Skipped bronze cleanup")
            return

        for path in paths_to_delete:
            result = subprocess.run(
                ["mc", "rm", "--recursive", "--force", path], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✅ Deleted: {path}")
            else:
                print(f"  ⚠️  Error deleting {path}: {result.stderr}")

        print("\n✅ Bronze cleanup complete")

    except Exception as e:
        print(f"⚠️  Bronze cleanup error: {e}")


def cleanup_kafka_tombstones(version: str, dry_run: bool = False):
    print("\n" + "=" * 70)
    print("KAFKA TOMBSTONES")
    print("=" * 70)

    try:
        from kafka import KafkaProducer

        # Get version item_idx from SQL
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT item_idx FROM book_genres WHERE tags_version=%s", (version,))
        version_items = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        if not version_items:
            print(f"✅ No {version} items to tombstone")
            return

        print(f"Found {len(version_items)} {version} items in SQL")

        if dry_run:
            print(f"[DRY RUN] Would send tombstones for {len(version_items)} keys")
            return

        print(f"\n⚠️  About to send {len(version_items)} tombstones to Kafka")
        print("   (Compaction will remove records over 1-2 hours)")
        confirm = input("Send tombstones? (yes/no): ")

        if confirm != "yes":
            print("Skipped Kafka tombstones")
            return

        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        producer = KafkaProducer(
            bootstrap_servers=bootstrap, key_serializer=lambda k: str(k).encode("utf-8")
        )

        for item_idx in version_items:
            key = f"{item_idx}:{version}"
            producer.send("enrich.results.v1", key=key, value=None)
            producer.send("enrich.errors.v1", key=key, value=None)

        producer.flush()
        print(f"✅ Sent {len(version_items)} tombstones")
        print("  Wait 1-2 hours for compaction to complete")

    except ImportError:
        print("⚠️  kafka-python not installed, skipping tombstones")
    except Exception as e:
        print(f"⚠️  Kafka error: {e}")


def cleanup_checkpoints(version: str, dry_run: bool = False):
    print("\n" + "=" * 70)
    print("SPARK CHECKPOINTS")
    print("=" * 70)

    checkpoint_base = Path(os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/enrichment"))
    checkpoint_dir = checkpoint_base / version

    if not checkpoint_dir.exists():
        print(f"✅ No checkpoint for {version}")
        return

    size_bytes = sum(f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file())
    size_mb = size_bytes / (1024 * 1024)
    print(f"Found: {checkpoint_dir} ({size_mb:.2f} MB)")

    if dry_run:
        print("[DRY RUN] Would delete checkpoint")
        return

    confirm = input("Delete checkpoint? (yes/no): ")
    if confirm == "yes":
        shutil.rmtree(checkpoint_dir)
        print(f"✅ Deleted checkpoint ({size_mb:.2f} MB freed)")
    else:
        print("Skipped checkpoint cleanup")


def cleanup_embeddings(version: str, dry_run: bool = False):
    """
    Clean up embedding artifacts for a specific tags_version.

    Removes:
    - NPZ batch files in accumulator/
    - Fingerprint database entries
    - Metadata for the version
    - Built FAISS indices (if they exist)
    """
    print("\n" + "=" * 70)
    print("EMBEDDING CLEANUP")
    print("=" * 70)

    # Determine output directory based on version
    # v1 -> models/data/enriched_v1, v2 -> models/data/enriched_v2
    output_dir = Path(f"models/data/enriched_{version}")

    if not output_dir.exists():
        print(f"✅ No embedding data for {version}")
        return

    # 1. Check accumulator directory
    accumulator_dir = output_dir / "accumulator"
    batch_files = []
    if accumulator_dir.exists():
        batch_files = list(accumulator_dir.glob("batch_*.npz"))

    # 2. Check fingerprints database
    fingerprints_db = output_dir / "fingerprints.db"
    fingerprint_count = 0
    if fingerprints_db.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(fingerprints_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fingerprints WHERE tags_version = ?", (version,))
            fingerprint_count = cursor.fetchone()[0]
            conn.close()
        except Exception as e:
            print(f"  ⚠️  Error reading fingerprints: {e}")

    # 3. Check metadata
    metadata_file = output_dir / "metadata.json"
    metadata_exists = metadata_file.exists()

    # 4. Check for built indices
    builds_dir = output_dir / "builds"
    build_dirs = []
    if builds_dir.exists():
        build_dirs = [d for d in builds_dir.iterdir() if d.is_dir()]

    # Calculate total size
    total_size = 0
    if output_dir.exists():
        for f in output_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)

    print(f"\nFound embedding data for {version}:")
    print(f"  Location: {output_dir}")
    print(f"  Batch files: {len(batch_files):,}")
    print(f"  Fingerprints: {fingerprint_count:,}")
    print(f"  Metadata: {'Yes' if metadata_exists else 'No'}")
    print(f"  Built indices: {len(build_dirs)}")
    if size_gb >= 1:
        print(f"  Total size: {size_gb:.2f} GB")
    else:
        print(f"  Total size: {size_mb:.2f} MB")

    if len(batch_files) == 0 and fingerprint_count == 0:
        print(f"\n✅ No significant embedding data to clean for {version}")
        return

    if dry_run:
        print("\n[DRY RUN] Would delete:")
        print(f"  - {len(batch_files)} batch files")
        print(f"  - {fingerprint_count} fingerprint entries")
        if metadata_exists:
            print("  - metadata.json")
        if build_dirs:
            print(f"  - {len(build_dirs)} built indices")
        if size_gb >= 1:
            print(f"  Total: {size_gb:.2f} GB")
        else:
            print(f"  Total: {size_mb:.2f} MB")
        return

    print(f"\n⚠️  About to delete embedding data for {version}")
    if size_gb >= 1:
        print(f"   Total size: {size_gb:.2f} GB")
    else:
        print(f"   Total size: {size_mb:.2f} MB")
    confirm = input("Delete? (yes/no): ")

    if confirm != "yes":
        print("Skipped embedding cleanup")
        return

    deleted_items = 0

    # Delete accumulator batch files
    if batch_files:
        for batch_file in batch_files:
            batch_file.unlink()
            deleted_items += 1
        print(f"  ✅ Deleted {len(batch_files)} batch files")

    # Delete fingerprints for this version
    if fingerprint_count > 0 and fingerprints_db.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(fingerprints_db)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fingerprints WHERE tags_version = ?", (version,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"  ✅ Deleted {deleted} fingerprint entries")
            deleted_items += deleted
        except Exception as e:
            print(f"  ⚠️  Error deleting fingerprints: {e}")

    # Delete metadata if it exists
    if metadata_exists:
        metadata_file.unlink()
        print("  ✅ Deleted metadata.json")
        deleted_items += 1

    # Delete built indices
    if build_dirs:
        for build_dir in build_dirs:
            shutil.rmtree(build_dir)
            deleted_items += 1
        print(f"  ✅ Deleted {len(build_dirs)} built indices")

    # If the entire output directory is now empty, remove it
    if output_dir.exists() and not list(output_dir.iterdir()):
        output_dir.rmdir()
        print(f"  ✅ Removed empty directory: {output_dir}")

    if size_gb >= 1:
        print(f"\n✅ Embedding cleanup complete: {size_gb:.2f} GB freed")
    else:
        print(f"\n✅ Embedding cleanup complete: {size_mb:.2f} MB freed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive version cleanup (enrichment + embeddings)"
    )
    parser.add_argument("version", help="Version to clean (e.g., v2)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--skip-sql", action="store_true", help="Skip SQL cleanup")
    parser.add_argument("--skip-bronze", action="store_true", help="Skip bronze archive cleanup")
    parser.add_argument("--skip-kafka", action="store_true", help="Skip Kafka tombstones")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Skip Spark checkpoints")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding cleanup")

    args = parser.parse_args()

    print("=" * 70)
    print("COMPREHENSIVE VERSION CLEANUP")
    print("=" * 70)
    print(f"Version: {args.version}")
    print(f"Dry run: {args.dry_run}")

    load_env()

    if not args.skip_sql:
        cleanup_sql(args.version, args.dry_run)

    if not args.skip_bronze:
        cleanup_bronze(args.version, args.dry_run)

    if not args.skip_kafka:
        cleanup_kafka_tombstones(args.version, args.dry_run)

    if not args.skip_checkpoints:
        cleanup_checkpoints(args.version, args.dry_run)

    if not args.skip_embeddings:
        cleanup_embeddings(args.version, args.dry_run)

    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
