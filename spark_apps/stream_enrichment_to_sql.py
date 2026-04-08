# spark_apps/stream_enrichment_to_sql.py
"""
Spark Structured Streaming consumer for enrichment pipeline.
Direct JDBC upserts with chunking, parameterization, and safety.

FIXES:
- Orphaned subjects bug (wrong variable + race condition)
- Parameter ordering for vibes and subjects
- Proper commit ordering for FK dependencies
"""

import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

# Configuration
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RESULTS_TOPIC = "enrich.results.v1"
ERRORS_TOPIC = "enrich.errors.v1"
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/enrichment")
VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v1")
ONTOLOGY_VERSION = os.getenv("ENRICHMENT_ONTOLOGY_VERSION", "v2")
REPLAY_MODE = os.getenv("REPLAY_FROM_EARLIEST", "0") == "1"

JDBC_URL = os.getenv("JDBC_URL", "jdbc:mysql://127.0.0.1:3306/bookrec_db")
JDBC_USER = os.getenv("JDBC_USER", "bookrec")
JDBC_PASS = os.getenv("JDBC_PASS", "secret")

# Batch size for chunked inserts (stay under max_allowed_packet)
CHUNK_SIZE = 1000

# Schemas
results_schema = StructType(
    [
        StructField("item_idx", IntegerType(), False),
        StructField("subjects", ArrayType(StringType()), False),
        StructField("tone_ids", ArrayType(IntegerType()), False),
        StructField("genre", StringType(), False),
        StructField("vibe", StringType(), False),
        StructField("tags_version", StringType(), False),
        StructField("scores", MapType(StringType(), StringType()), True),
        StructField("timestamp", LongType(), True),
        StructField("metadata", MapType(StringType(), StringType()), True),
    ]
)

errors_schema = StructType(
    [
        StructField("item_idx", IntegerType(), False),
        StructField("tags_version", StringType(), False),
        StructField("timestamp", LongType(), True),
        StructField("stage", StringType(), False),
        StructField("error_code", StringType(), False),
        StructField("error_field", StringType(), True),
        StructField("error_msg", StringType(), False),
        StructField("title", StringType(), True),
        StructField("author", StringType(), True),
        StructField("attempted", MapType(StringType(), StringType()), True),
        StructField("run_id", StringType(), True),
    ]
)


def batch_parameterized_insert(cur, sql: str, params_list: list, chunk_size: int = CHUNK_SIZE):
    """
    Execute parameterized INSERT in chunks to avoid max_allowed_packet.

    Args:
        cur: Database cursor
        sql: SQL with %s placeholders
        params_list: List of parameter tuples
        chunk_size: Max rows per executemany
    """
    if not params_list:
        return

    total = len(params_list)
    for i in range(0, total, chunk_size):
        chunk = params_list[i : i + chunk_size]
        cur.executemany(sql, chunk)

        if total > chunk_size and i + chunk_size < total:
            print(f"      ... {i + len(chunk)}/{total}")


def process_results_batch(batch_df, batch_id):
    """
    Process enrichment results with direct JDBC upserts.
    Uses chunking, parameterized queries, and proper FK handling.

    CRITICAL FIX: Commits dictionary tables BEFORE link tables to avoid orphans.
    """
    if batch_df.isEmpty():
        return

    count = batch_df.count()
    print(f"\n{'=' * 80}")
    print(f"[Batch {batch_id}] Processing {count} enrichment results")
    print(f"{'=' * 80}")

    # Collect batch data
    batch_data = batch_df.collect()

    # Connect to database
    from urllib.parse import urlparse

    import pymysql

    parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=JDBC_USER,
        password=JDBC_PASS,
        database=parsed.path.lstrip("/") or "bookrec_db",
        charset="utf8mb4",
        autocommit=False,
    )

    start_time = time.time()

    try:
        cur = conn.cursor()

        # ====================================================================
        # PHASE 1: UPSERT DICTIONARY TABLES (must commit before links!)
        # ====================================================================

        # 1a. Upsert LLM subjects dictionary
        print("  [1/6] Upserting LLM subjects dictionary...")
        subjects_set = set()
        for row in batch_data:
            subjects_set.update(row.subjects or [])

        if subjects_set:
            subject_dict_params = [(s,) for s in subjects_set]
            batch_parameterized_insert(
                cur, "INSERT IGNORE INTO llm_subjects(subject) VALUES (%s)", subject_dict_params
            )
            print(f"    → {len(subjects_set)} unique subjects")

        # 1b. Upsert vibes dictionary
        print("  [2/6] Upserting vibes dictionary...")
        vibes_set = {row.vibe for row in batch_data if row.vibe}
        if vibes_set:
            vibe_dict_params = [(v,) for v in vibes_set]
            batch_parameterized_insert(
                cur, "INSERT IGNORE INTO vibes(text) VALUES (%s)", vibe_dict_params
            )
            print(f"    → {len(vibes_set)} unique vibes")

        # 🔥 CRITICAL: Commit dictionaries so FKs resolve correctly
        print("  [CHECKPOINT] Committing dictionaries before link tables...")
        conn.commit()

        # ====================================================================
        # PHASE 2: UPSERT LINK TABLES (now dictionaries are committed)
        # ====================================================================

        # 2a. Upsert book_tones (many-to-many with tags_version)
        print("  [3/6] Upserting book tones...")
        tone_params = []
        for row in batch_data:
            for tone_id in row.tone_ids or []:
                tone_params.append((row.item_idx, tone_id, row.tags_version))

        if tone_params:
            batch_parameterized_insert(
                cur,
                """INSERT INTO book_tones(item_idx, tone_id, tags_version)
                   VALUES (%s, %s, %s)
                   ON DUPLICATE KEY UPDATE tone_id=VALUES(tone_id)""",
                tone_params,
            )
            print(f"    → {len(tone_params)} tone links")

        # 2b. Upsert book_genres (one per book per version)
        print("  [4/6] Upserting book genres...")
        genre_params = [
            (row.item_idx, row.genre, row.tags_version, ONTOLOGY_VERSION)
            for row in batch_data
            if row.genre
        ]
        if genre_params:
            batch_parameterized_insert(
                cur,
                """INSERT INTO book_genres(item_idx, genre_slug, tags_version, genre_ontology_version)
                   VALUES (%s, %s, %s, %s)  # ← Now 4 parameters instead of 3
                   ON DUPLICATE KEY UPDATE genre_slug=VALUES(genre_slug)""",
                genre_params,
            )
            print(f"    → {len(genre_params)} genre assignments")

        # 2c. Upsert book_vibes (batch resolution of vibe_id)
        print("  [5/6] Upserting book vibes...")
        vibe_link_params = []
        for row in batch_data:
            if row.vibe:
                vibe_link_params.append((row.item_idx, row.tags_version, row.vibe))

        if vibe_link_params:
            batch_parameterized_insert(
                cur,
                """INSERT INTO book_vibes(item_idx, vibe_id, tags_version)
                   SELECT %s, vibe_id, %s FROM vibes WHERE text = %s
                   ON DUPLICATE KEY UPDATE vibe_id=VALUES(vibe_id)""",
                vibe_link_params,
            )
            print(f"    → {len(vibe_link_params)} vibe assignments")

        # 2d. Upsert book_llm_subjects (batch resolution of llm_subject_idx)
        # 🔥 FIX: Build CORRECT parameter list for book-subject links
        print("  [6/6] Upserting book LLM subjects...")
        subject_link_params = []
        for row in batch_data:
            for subject in row.subjects or []:
                subject_link_params.append((row.item_idx, row.tags_version, subject))

        if subject_link_params:
            batch_parameterized_insert(
                cur,
                """INSERT IGNORE INTO book_llm_subjects(item_idx, llm_subject_idx, tags_version)
                   SELECT %s, llm_subject_idx, %s FROM llm_subjects WHERE subject = %s""",
                subject_link_params,  # ✅ CORRECT: per-book subject links
            )
            print(f"    → {len(subject_link_params)} subject links")

            # 🔍 Debugging: Check for orphaned subjects
            cur.execute("""
                SELECT COUNT(*) FROM llm_subjects s
                LEFT JOIN book_llm_subjects bs ON s.llm_subject_idx = bs.llm_subject_idx
                WHERE bs.llm_subject_idx IS NULL
            """)
            orphan_count = cur.fetchone()[0]
            if orphan_count > 0:
                print(f"    ⚠️  WARNING: {orphan_count} orphaned subjects in dictionary")

        # Commit link tables
        conn.commit()

        elapsed = time.time() - start_time
        print(f"\n✓ Batch {batch_id} committed successfully")
        print(f"  Books: {count} | Time: {elapsed:.2f}s | Rate: {count / elapsed:.1f} books/s")
        print(f"{'=' * 80}\n")

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Batch {batch_id} failed: {e}")
        print("  Rolling back transaction...")
        import traceback

        traceback.print_exc()
        print(f"{'=' * 80}\n")
        raise
    finally:
        cur.close()
        conn.close()


def process_errors_batch(batch_df, batch_id):
    """Process error events with run_history tracking."""
    if batch_df.isEmpty():
        return

    count = batch_df.count()
    print(f"\n[Error Batch {batch_id}] Processing {count} error records...")

    errors_data = batch_df.collect()

    from datetime import datetime
    from urllib.parse import urlparse

    import pymysql

    parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=JDBC_USER,
        password=JDBC_PASS,
        database=parsed.path.lstrip("/") or "bookrec_db",
        charset="utf8mb4",
        autocommit=False,
    )

    try:
        cur = conn.cursor()

        error_params = []
        for row in errors_data:
            timestamp_dt = (
                datetime.fromtimestamp(row.timestamp / 1000) if row.timestamp else datetime.utcnow()
            )

            run_id = row.run_id if hasattr(row, "run_id") else None

            # Serialize attempted to JSON
            attempted_json = None
            if row.attempted:
                try:
                    if isinstance(row.attempted, dict):
                        attempted_json = json.dumps(row.attempted)
                    elif isinstance(row.attempted, str):
                        json.loads(row.attempted)  # Validate
                        attempted_json = row.attempted
                except (json.JSONDecodeError, TypeError, AttributeError):
                    attempted_json = None

            # Build run_history JSON
            run_history_json = (
                json.dumps(
                    [
                        {
                            "run_id": run_id,
                            "timestamp": timestamp_dt.isoformat(),
                            "error_code": row.error_code,
                        }
                    ]
                )
                if run_id
                else None
            )

            error_params.append(
                (
                    row.item_idx,
                    timestamp_dt,
                    timestamp_dt,
                    1,
                    row.stage,
                    row.error_code,
                    row.error_field,
                    row.error_msg[:512] if row.error_msg else None,
                    row.tags_version,
                    row.title[:256] if row.title else None,
                    row.author[:256] if row.author else None,
                    attempted_json,
                    run_id,
                    run_history_json,
                )
            )

        # Execute INSERT
        batch_parameterized_insert(
            cur,
            """INSERT INTO enrichment_errors(
                   item_idx, first_seen_at, last_seen_at, occurrence_count,
                   stage, error_code, error_field, error_msg, tags_version,
                   title, author, attempted, last_run_id, run_history
               )
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
                   last_seen_at = VALUES(last_seen_at),
                   occurrence_count = occurrence_count + 1,
                   stage = VALUES(stage),
                   error_code = VALUES(error_code),
                   error_msg = VALUES(error_msg),
                   attempted = VALUES(attempted),
                   last_run_id = VALUES(last_run_id),
                   run_history = CASE
                       WHEN run_history IS NULL THEN VALUES(run_history)
                       ELSE JSON_ARRAY_APPEND(
                           run_history,
                           '$',
                           JSON_OBJECT(
                               'run_id', VALUES(last_run_id),
                               'timestamp', VALUES(last_seen_at),
                               'error_code', VALUES(error_code)
                           )
                       )
                   END""",
            error_params,
        )

        conn.commit()
        print(f"✅ Error batch {batch_id} committed ({count} errors)\n")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error batch {batch_id} failed: {e}\n")
        import traceback

        traceback.print_exc()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    """
    Main streaming job: consume from Kafka and write to SQL.
    """
    spark = (
        SparkSession.builder.appName("enrichment-streaming")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # Checkpoint configuration
    if REPLAY_MODE:
        checkpoint_dir = f"{CHECKPOINT_DIR}-replay-{int(time.time())}"
        starting_offsets = "earliest"
        print("\n⚠️  REPLAY MODE: Starting from earliest offsets")
    else:
        checkpoint_dir = f"{CHECKPOINT_DIR}/{VERSION_TAG}"
        starting_offsets = "latest"

    print("\n" + "=" * 80)
    print("ENRICHMENT STREAMING CONSUMER")
    print("=" * 80)
    print(f"Kafka Bootstrap: {KAFKA_BOOTSTRAP}")
    print(f"JDBC URL: {JDBC_URL}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Tags Version: {VERSION_TAG}")
    print(f"Starting Offsets: {starting_offsets}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print("=" * 80 + "\n")

    # Read results stream
    results_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", RESULTS_TOPIC)
        .option("startingOffsets", starting_offsets)
        .option("maxOffsetsPerTrigger", 1000)
        .option("kafka.group.id", "cg.enrichment.sql.v1")
        .load()
        .select(F.from_json(F.col("value").cast("string"), results_schema).alias("data"))
        .select("data.*")
    )

    # Read errors stream
    errors_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", ERRORS_TOPIC)
        .option("startingOffsets", starting_offsets)
        .option("maxOffsetsPerTrigger", 500)
        .option("kafka.group.id", "cg.enrichment.errors.v1")
        .load()
        .select(F.from_json(F.col("value").cast("string"), errors_schema).alias("data"))
        .select("data.*")
    )

    # Start results query
    (
        results_stream.writeStream.foreachBatch(process_results_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{checkpoint_dir}/results")
        .start()
    )

    # Start errors query
    (
        errors_stream.writeStream.foreachBatch(process_errors_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{checkpoint_dir}/errors")
        .start()
    )

    print("✓ Streaming queries started")
    print(f"  - Results: {RESULTS_TOPIC} → SQL")
    print(f"  - Errors: {ERRORS_TOPIC} → SQL")
    print("\nPress Ctrl+C to stop...\n")

    # Wait for termination
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
