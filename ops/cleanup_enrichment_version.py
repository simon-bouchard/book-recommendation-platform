#!/usr/bin/env python3
# ops/cleanup_enrichment_version.py
"""Clean up enrichment data for a specific version"""

import os
import sys
import shutil
import time
from pathlib import Path
import pymysql
from urllib.parse import urlparse

# Load environment variables
def load_env():
    """Load environment variables from .env files"""
    try:
        from dotenv import load_dotenv
        
        env_files = [
            Path('/etc/bookrec.env'),
            Path('.env'),
            Path(__file__).parent.parent / '.env',
        ]
        
        loaded = False
        for env_file in env_files:
            if env_file.exists():
                print(f"Loading environment from: {env_file}")
                load_dotenv(env_file, override=True)
                loaded = True
                break
        
        if not loaded:
            print("⚠️  No .env file found, using system environment variables")
        
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables")


def get_db_connection():
    """Get database connection with environment variables"""
    jdbc_url = os.getenv("JDBC_URL")
    user = os.getenv("JDBC_USER")
    password = os.getenv("JDBC_PASS")
    
    if not jdbc_url:
        print("⚠️  JDBC_URL not set, using default: localhost:3306/bookrec_db")
        jdbc_url = "jdbc:mysql://127.0.0.1:3306/bookrec_db"
    
    if not user:
        print("⚠️  JDBC_USER not set, using default: bookrec")
        user = "bookrec"
    
    if not password:
        print("⚠️  JDBC_PASS not set")
        password = input("Enter MySQL password: ")
    
    print(f"\nConnecting to database:")
    print(f"  URL: {jdbc_url}")
    print(f"  User: {user}")
    
    parsed = urlparse(jdbc_url.replace("jdbc:", "", 1))
    
    try:
        conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=user,
            password=password,
            database=parsed.path.lstrip("/") or "bookrec_db",
            charset="utf8mb4"
        )
        print("✓ Connected successfully\n")
        return conn
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)


def cleanup_sql(version: str, dry_run: bool = False):
    """Clean up SQL tables for a version"""
    print("\n" + "="*70)
    print("SQL TABLES CLEANUP")
    print("="*70)
    
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        
        print(f"Current counts for tags_version='{version}':")
        tables = ['book_tones', 'book_genres', 'book_vibes', 'book_llm_subjects', 'enrichment_errors']
        
        total = 0
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE tags_version = %s", (version,))
            count = cur.fetchone()[0]
            total += count
            print(f"  {table}: {count:,}")
        
        if total == 0:
            print(f"\n✓ No data found for version '{version}'")
            return
        
        if dry_run:
            print(f"\nDRY RUN: Would delete {total:,} total rows")
            return
        
        print(f"\nTotal rows to delete: {total:,}")
        confirm = input(f"Delete all {version} data from SQL? (yes/no): ")
        if confirm != "yes":
            print("Skipped SQL cleanup")
            return
        
        deleted_total = 0
        for table in tables:
            cur.execute(f"DELETE FROM {table} WHERE tags_version = %s", (version,))
            deleted = cur.rowcount
            deleted_total += deleted
            if deleted > 0:
                print(f"  Deleted from {table}: {deleted:,} rows")
        
        conn.commit()
        print(f"\n✓ SQL cleanup complete - deleted {deleted_total:,} rows")
        
    finally:
        cur.close()
        conn.close()


def cleanup_kafka_topics(dry_run: bool = False):
    """Delete and recreate Kafka topics"""
    print("\n" + "="*70)
    print("KAFKA TOPICS CLEANUP")
    print("="*70)
    
    try:
        from kafka.admin import KafkaAdminClient
        from kafka.errors import UnknownTopicOrPartitionError
    except ImportError:
        print("❌ kafka-python not installed")
        print("   Install with: pip install kafka-python")
        return
    
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topics = ["enrich.results.v1", "enrich.errors.v1"]
    
    print(f"Bootstrap servers: {bootstrap_servers}")
    print(f"Topics to recreate: {', '.join(topics)}")
    
    if dry_run:
        print("\nDRY RUN: Would delete and recreate topics")
        return
    
    print("\n⚠️  WARNING: This will delete ALL data in these topics (all versions)")
    print("   Alternative: Let Kafka compaction remove old data naturally")
    confirm = input("Delete and recreate topics? (yes/no): ")
    
    if confirm != "yes":
        print("Skipped Kafka cleanup")
        return
    
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='cleanup-script'
        )
        
        # Check which topics exist
        existing_topics = admin.list_topics()
        topics_to_delete = [t for t in topics if t in existing_topics]
        
        if not topics_to_delete:
            print("✓ Topics don't exist yet")
            admin.close()
            return
        
        # Delete topics
        print(f"\nDeleting topics: {', '.join(topics_to_delete)}")
        try:
            admin.delete_topics(topics_to_delete, timeout_ms=30000)
            print("  Deletion initiated...")
            
            # Wait for deletion to complete
            print("  Waiting for deletion to complete (max 30s)...")
            for i in range(30):
                time.sleep(1)
                remaining_topics = admin.list_topics()
                still_exist = [t for t in topics_to_delete if t in remaining_topics]
                
                if not still_exist:
                    print(f"  ✓ Topics deleted after {i+1}s")
                    break
                
                if i % 5 == 0 and i > 0:
                    print(f"  ... still waiting ({i}s)")
            else:
                print("  ⚠️  Timeout waiting for deletion, proceeding anyway")
        
        except UnknownTopicOrPartitionError:
            print("  ✓ Topics already deleted")
        
        admin.close()
        
        # Recreate topics
        print("\nRecreating topics...")
        import subprocess
        
        # Try to run the setup script
        setup_script = Path(__file__).parent / "setup_kafka_topics.py"
        if setup_script.exists():
            result = subprocess.run(
                [sys.executable, str(setup_script)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print("✓ Topics recreated successfully")
            else:
                print(f"❌ Error recreating topics:")
                print(result.stderr)
        else:
            print(f"⚠️  Setup script not found: {setup_script}")
            print("   Manually run: python ops/setup_kafka_topics.py")
        
    except Exception as e:
        print(f"❌ Error managing Kafka topics: {e}")
        import traceback
        traceback.print_exc()


def cleanup_checkpoints(version: str, dry_run: bool = False):
    """Clean up Spark checkpoints"""
    print("\n" + "="*70)
    print("SPARK CHECKPOINTS CLEANUP")
    print("="*70)
    
    checkpoint_base = Path(os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/enrichment"))
    checkpoint_dir = checkpoint_base / version
    
    print(f"Checking: {checkpoint_dir}")
    
    if not checkpoint_dir.exists():
        print(f"✓ No checkpoint found for {version}")
        return
    
    size_bytes = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"\nFound checkpoint: {checkpoint_dir}")
    print(f"Size: {size_mb:.2f} MB")
    
    if dry_run:
        print("DRY RUN: Would delete checkpoint directory")
        return
    
    confirm = input(f"Delete checkpoint for {version}? (yes/no): ")
    if confirm == "yes":
        shutil.rmtree(checkpoint_dir)
        print(f"✓ Checkpoint deleted ({size_mb:.2f} MB freed)")
    else:
        print("Skipped checkpoint cleanup")


def cleanup_jsonl(version: str, dry_run: bool = False):
    """Clean up JSONL files"""
    print("\n" + "="*70)
    print("JSONL FILES CLEANUP")
    print("="*70)
    
    jsonl_path = Path(os.getenv("ENRICH_JSONL_PATH", f"data/enrichment_{version}.jsonl"))
    
    print(f"Checking: {jsonl_path}")
    
    if not jsonl_path.exists():
        print(f"✓ No JSONL file found for {version}")
        return
    
    size_bytes = jsonl_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    with open(jsonl_path, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"\nFound JSONL file: {jsonl_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Lines: {line_count:,}")
    
    if dry_run:
        print("DRY RUN: Would delete JSONL file")
        return
    
    confirm = input(f"Delete JSONL file? (yes/no): ")
    if confirm == "yes":
        jsonl_path.unlink()
        print(f"✓ JSONL file deleted ({size_mb:.2f} MB freed)")
    else:
        print("Skipped JSONL cleanup")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up enrichment version data")
    parser.add_argument("version", help="Version to clean (e.g., v2)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--skip-sql", action="store_true", help="Skip SQL cleanup")
    parser.add_argument("--skip-kafka", action="store_true", help="Skip Kafka cleanup")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Skip checkpoint cleanup")
    parser.add_argument("--skip-jsonl", action="store_true", help="Skip JSONL cleanup")
    parser.add_argument("--kafka-only", action="store_true", help="Only delete/recreate Kafka topics")
    parser.add_argument("--env-file", help="Path to .env file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ENRICHMENT VERSION CLEANUP")
    print("="*70)
    print(f"Version: {args.version}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load environment variables
    if args.env_file:
        from dotenv import load_dotenv
        print(f"Loading environment from: {args.env_file}")
        load_dotenv(args.env_file, override=True)
    else:
        load_env()
    
    print()
    
    # Kafka-only mode
    if args.kafka_only:
        cleanup_kafka_topics(args.dry_run)
        return
    
    # Full cleanup
    if not args.skip_sql:
        cleanup_sql(args.version, args.dry_run)
    
    if not args.skip_kafka:
        cleanup_kafka_topics(args.dry_run)
    
    if not args.skip_checkpoints:
        cleanup_checkpoints(args.version, args.dry_run)
    
    if not args.skip_jsonl:
        cleanup_jsonl(args.version, args.dry_run)
    
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print(f"  1. export ENRICHMENT_JOB_TAG_VERSION={args.version}")
    print("  2. Start Spark consumer: docker-compose -f docker/spark-loader/docker-compose.yml up")
    print("  3. Run enrichment: python -m app.enrichment.runner_kafka --limit 100")
    print("  4. Monitor: python ops/monitor_enrichment_pipeline.py")
    print("="*70)


if __name__ == "__main__":
    main()
