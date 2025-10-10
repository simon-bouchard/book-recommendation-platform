# spark_apps/stream_enrichment_to_sql.py
"""
Spark Structured Streaming consumer for enrichment pipeline.
Direct JDBC upserts with chunking, parameterization, and safety.
"""
import os, json
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    ArrayType, MapType
)

# Configuration
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RESULTS_TOPIC = "enrich.results.v1"
ERRORS_TOPIC = "enrich.errors.v1"
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/enrichment")
VERSION_TAG = os.getenv("ENRICHMENT_JOB_TAG_VERSION", "v1")
REPLAY_MODE = os.getenv("REPLAY_FROM_EARLIEST", "0") == "1"

JDBC_URL = os.getenv("JDBC_URL", "jdbc:mysql://127.0.0.1:3306/bookrec_db")
JDBC_USER = os.getenv("JDBC_USER", "bookrec")
JDBC_PASS = os.getenv("JDBC_PASS", "secret")

# Batch size for chunked inserts (stay under max_allowed_packet)
CHUNK_SIZE = 1000

# Schemas
results_schema = StructType([
    StructField("item_idx", IntegerType(), False),
    StructField("subjects", ArrayType(StringType()), False),
    StructField("tone_ids", ArrayType(IntegerType()), False),
    StructField("genre", StringType(), False),
    StructField("vibe", StringType(), False),
    StructField("tags_version", StringType(), False),
    StructField("scores", MapType(StringType(), StringType()), True),
    StructField("timestamp", LongType(), True),
    StructField("metadata", MapType(StringType(), StringType()), True),
])

errors_schema = StructType([
    StructField("item_idx", IntegerType(), False),
    StructField("tags_version", StringType(), False),
    StructField("timestamp", LongType(), True),
    StructField("stage", StringType(), False),
    StructField("error_code", StringType(), False),
    StructField("error_field", StringType(), True),
    StructField("error_msg", StringType(), False),
    StructField("title", StringType(), True),
    StructField("author", StringType(), True),
    StructField("attempted", StringType(), True),  # ✅ Changed to StringType - treat as JSON string
])


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
        chunk = params_list[i:i + chunk_size]
        cur.executemany(sql, chunk)
        
        if total > chunk_size and i + chunk_size < total:
            print(f"      ... {i + len(chunk)}/{total}")


def process_results_batch(batch_df, batch_id):
    """
    Process enrichment results with direct JDBC upserts.
    Uses chunking, parameterized queries, and batch resolution.
    """
    if batch_df.isEmpty():
        return
    
    count = batch_df.count()
    print(f"\n{'='*80}")
    print(f"[Batch {batch_id}] Processing {count} enrichment results")
    print(f"{'='*80}")
    
    # Collect batch data
    batch_data = batch_df.collect()
    
    # Connect to database
    import pymysql
    from urllib.parse import urlparse
    
    parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=JDBC_USER,
        password=JDBC_PASS,
        database=parsed.path.lstrip("/") or "bookrec_db",
        charset="utf8mb4",
        autocommit=False
    )
    
    start_time = time.time()
    
    try:
        cur = conn.cursor()
        
        # 1. Upsert LLM subjects dictionary
        print(f"  [1/6] Upserting LLM subjects...")
        subjects_set = set()
        for row in batch_data:
            subjects_set.update(row.subjects or [])
        
        if subjects_set:
            subject_params = [(s,) for s in subjects_set]
            batch_parameterized_insert(
                cur,
                "INSERT IGNORE INTO llm_subjects(subject) VALUES (%s)",
                subject_params
            )
            print(f"    → {len(subjects_set)} unique subjects")
        
        # 2. Upsert vibes dictionary
        print(f"  [2/6] Upserting vibes...")
        vibes_set = {row.vibe for row in batch_data if row.vibe}
        if vibes_set:
            vibe_params = [(v,) for v in vibes_set]
            batch_parameterized_insert(
                cur,
                "INSERT IGNORE INTO vibes(text) VALUES (%s)",
                vibe_params
            )
            print(f"    → {len(vibes_set)} unique vibes")
        
        # 3. Upsert book_tones (many-to-many with tags_version)
        print(f"  [3/6] Upserting book tones...")
        tone_params = []
        for row in batch_data:
            for tone_id in (row.tone_ids or []):
                tone_params.append((row.item_idx, tone_id, row.tags_version))
        
        if tone_params:
            batch_parameterized_insert(
                cur,
                """INSERT INTO book_tones(item_idx, tone_id, tags_version)
                   VALUES (%s, %s, %s)
                   ON DUPLICATE KEY UPDATE tone_id=VALUES(tone_id)""",
                tone_params
            )
            print(f"    → {len(tone_params)} tone links")
        
        # 4. Upsert book_genres (one per book per version)
        print(f"  [4/6] Upserting book genres...")
        genre_params = [
            (row.item_idx, row.genre, row.tags_version)
            for row in batch_data if row.genre
        ]
        if genre_params:
            batch_parameterized_insert(
                cur,
                """INSERT INTO book_genres(item_idx, genre_slug, tags_version)
                   VALUES (%s, %s, %s)
                   ON DUPLICATE KEY UPDATE genre_slug=VALUES(genre_slug)""",
                genre_params
            )
            print(f"    → {len(genre_params)} genre assignments")
        
        # 5. Upsert book_vibes (batch resolution of vibe_id)
        print(f"  [5/6] Upserting book vibes...")
        vibe_params = [
            (row.item_idx, row.vibe, row.tags_version)
            for row in batch_data if row.vibe
        ]
        if vibe_params:
            batch_parameterized_insert(
                cur,
                """INSERT INTO book_vibes(item_idx, vibe_id, tags_version)
                   SELECT %s, vibe_id, %s FROM vibes WHERE text = %s
                   ON DUPLICATE KEY UPDATE vibe_id=VALUES(vibe_id)""",
                vibe_params
            )
            print(f"    → {len(vibe_params)} vibe assignments")
        
        # 6. Upsert book_llm_subjects (batch resolution of llm_subject_idx)
        print(f"  [6/6] Upserting book LLM subjects...")
        subject_params = []
        for row in batch_data:
            for subject in (row.subjects or []):
                subject_params.append((row.item_idx, subject, row.tags_version))
        
        if subject_params:
            batch_parameterized_insert(
                cur,
                """INSERT IGNORE INTO book_llm_subjects(item_idx, llm_subject_idx, tags_version)
                   SELECT %s, llm_subject_idx, %s FROM llm_subjects WHERE subject = %s""",
                subject_params
            )
            print(f"    → {len(subject_params)} subject links")
        
        # Commit transaction
        conn.commit()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Batch {batch_id} committed successfully")
        print(f"  Books: {count} | Time: {elapsed:.2f}s | Rate: {count/elapsed:.1f} books/s")
        print(f"{'='*80}\n")
        
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Batch {batch_id} failed: {e}")
        print(f"  Rolling back transaction...")
        print(f"{'='*80}\n")
        raise
    finally:
        cur.close()
        conn.close()

def process_errors_batch(batch_df, batch_id):
    """
    Process error events from DLQ with direct upsert.
    """
    if batch_df.isEmpty():
        return
    
    count = batch_df.count()
    print(f"\n[Error Batch {batch_id}] Processing {count} error records...")
    
    # Collect error data
    errors_data = batch_df.collect()
    
    # Connect to database
    import pymysql
    from urllib.parse import urlparse
    from datetime import datetime
    
    parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=JDBC_USER,
        password=JDBC_PASS,
        database=parsed.path.lstrip("/") or "bookrec_db",
        charset="utf8mb4",
        autocommit=False
    )
    
    try:
        cur = conn.cursor()
        
        # Prepare error parameters
        error_params = []
        for row in errors_data:
            timestamp_dt = datetime.fromtimestamp(row.timestamp / 1000) if row.timestamp else datetime.utcnow()
            run_id = row.get("run_id")
            
            # ✅ FIX: Properly serialize attempted to JSON
            attempted_json = None
            if row.attempted:
                try:
                    if isinstance(row.attempted, str):
                        # Already a string - validate it's JSON
                        json.loads(row.attempted)
                        attempted_json = row.attempted
                    else:
                        # It's a dict/map - serialize to JSON
                        attempted_json = json.dumps(row.attempted)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"  ⚠️  Invalid attempted for item_idx={row.item_idx}: {e}")
                    attempted_json = None
            
            error_params.append((
                row.item_idx,
                timestamp_dt,  # first_seen_at
                timestamp_dt,  # last_seen_at
                1,  # occurrence_count
                row.stage,
                row.error_code,
                row.error_field,
                row.error_msg[:512] if row.error_msg else None,
                row.tags_version,
                row.title[:256] if row.title else None,
                row.author[:256] if row.author else None,
                attempted_json,  
                run_id,
            ))
        
        # Batch upsert errors
        batch_parameterized_insert(
            cur,
            """INSERT INTO enrichment_errors(
                   item_idx, first_seen_at, last_seen_at, occurrence_count,
                   stage, error_code, error_field, error_msg, tags_version,
                   title, author, attempted, last_run_id
               )
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
                   last_seen_at = VALUES(last_seen_at),
                   occurrence_count = occurrence_count + 1,
                   stage = VALUES(stage),
                   error_code = VALUES(error_code),
                   error_msg = VALUES(error_msg),
                   last_run_id = VALUES(last_run_id)""",
            error_params
        )
        
        conn.commit()
        print(f"✓ Error batch {batch_id} committed successfully ({count} errors)\n")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error batch {batch_id} failed: {e}\n")
        raise
    finally:
        cur.close()
        conn.close()

def main():
    """
    Main streaming job: consume from Kafka and write to SQL.
    """
    spark = (SparkSession.builder
        .appName("enrichment-streaming")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Checkpoint configuration
    if REPLAY_MODE:
        checkpoint_dir = f"{CHECKPOINT_DIR}-replay-{int(time.time())}"
        starting_offsets = "earliest"
        print(f"\n⚠️  REPLAY MODE: Starting from earliest offsets")
    else:
        checkpoint_dir = f"{CHECKPOINT_DIR}/{VERSION_TAG}"
        starting_offsets = "latest"
    
    print("\n" + "="*80)
    print("ENRICHMENT STREAMING CONSUMER")
    print("="*80)
    print(f"Kafka Bootstrap: {KAFKA_BOOTSTRAP}")
    print(f"JDBC URL: {JDBC_URL}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Tags Version: {VERSION_TAG}")
    print(f"Starting Offsets: {starting_offsets}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print("="*80 + "\n")
    
    # Read results stream
    results_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", RESULTS_TOPIC)
        .option("startingOffsets", starting_offsets)
        .option("maxOffsetsPerTrigger", 1000)
        .option("kafka.group.id", "cg.enrichment.sql.v1")
        .load()
        .select(F.from_json(F.col("value").cast("string"), results_schema).alias("data"))
        .select("data.*"))
    
    # Read errors stream
    errors_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", ERRORS_TOPIC)
        .option("startingOffsets", starting_offsets)
        .option("maxOffsetsPerTrigger", 500)
        .option("kafka.group.id", "cg.enrichment.errors.v1")
        .load()
        .select(F.from_json(F.col("value").cast("string"), errors_schema).alias("data"))
        .select("data.*"))
    
    # Start results query
    results_query = (results_stream
        .writeStream
        .foreachBatch(process_results_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{checkpoint_dir}/results")
        .start())
    
    # Start errors query
    errors_query = (errors_stream
        .writeStream
        .foreachBatch(process_errors_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{checkpoint_dir}/errors")
        .start())
    
    print("✓ Streaming queries started")
    print(f"  - Results: {RESULTS_TOPIC} → SQL")
    print(f"  - Errors: {ERRORS_TOPIC} → SQL")
    print("\nPress Ctrl+C to stop...\n")
    
    # Wait for termination
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
