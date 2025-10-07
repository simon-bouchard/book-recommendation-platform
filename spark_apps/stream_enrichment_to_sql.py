# spark_apps/stream_enrichment_to_sql.py
"""
Spark Structured Streaming consumer for enrichment pipeline.
Reads from Kafka topics and writes to SQL with idempotent upserts.
FIXED: Now properly handles tags_version for versioned enrichment.
"""
import os
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType

# Configuration
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RESULTS_TOPIC = "enrich.results.v1"
ERRORS_TOPIC = "enrich.errors.v1"
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/enrichment")

JDBC_URL = os.getenv("JDBC_URL", "jdbc:mysql://127.0.0.1:3306/bookrec_db")
JDBC_USER = os.getenv("JDBC_USER", "bookrec")
JDBC_PASS = os.getenv("JDBC_PASS", "secret")
JDBC_PROPS = {"user": JDBC_USER, "password": JDBC_PASS, "driver": "com.mysql.cj.jdbc.Driver"}

# Schemas for parsing JSON from Kafka
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
    StructField("attempted", MapType(StringType(), StringType()), True),
])


def process_results_batch(batch_df, batch_id):
    """
    Process batch with direct JDBC upserts - no staging tables needed.
    Each batch is self-contained and idempotent.
    """
    if batch_df.isEmpty():
        return
    
    print(f"Processing batch {batch_id} with {batch_df.count()} records")
    
    # Collect data structures (small batches are OK in memory)
    batch_data = batch_df.collect()
    
    # Build connection
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
    
    try:
        cur = conn.cursor()
        
        # 1. Upsert LLM subjects (dictionary)
        subjects_set = set()
        for row in batch_data:
            subjects_set.update(row.subjects or [])
        
        if subjects_set:
            subject_values = ",".join([f"({pymysql.escape_string(s)})" for s in subjects_set])
            cur.execute(f"INSERT IGNORE INTO llm_subjects(subject) VALUES {subject_values}")
        
        # 2. Upsert vibes (dictionary)
        vibes_set = {row.vibe for row in batch_data if row.vibe}
        if vibes_set:
            vibe_values = ",".join([f"({pymysql.escape_string(v)})" for v in vibes_set])
            cur.execute(f"INSERT IGNORE INTO vibes(text) VALUES {vibe_values}")
        
        # 3. Upsert book links (with tags_version)
        # Book tones
        tone_values = []
        for row in batch_data:
            for tone_id in (row.tone_ids or []):
                tone_values.append(f"({row.item_idx}, {tone_id}, '{row.tags_version}')")
        
        if tone_values:
            cur.execute(f"""
                INSERT INTO book_tones(item_idx, tone_id, tags_version)
                VALUES {','.join(tone_values)}
                ON DUPLICATE KEY UPDATE tone_id=VALUES(tone_id)
            """)
        
        # Book genres
        genre_values = [
            f"({row.item_idx}, '{row.genre}', '{row.tags_version}')"
            for row in batch_data if row.genre
        ]
        if genre_values:
            cur.execute(f"""
                INSERT INTO book_genres(item_idx, genre_slug, tags_version)
                VALUES {','.join(genre_values)}
                ON DUPLICATE KEY UPDATE genre_slug=VALUES(genre_slug)
            """)
        
        # Book vibes (need to resolve vibe_id from text)
        for row in batch_data:
            if row.vibe:
                cur.execute("""
                    INSERT INTO book_vibes(item_idx, vibe_id, tags_version)
                    SELECT %s, vibe_id, %s FROM vibes WHERE text = %s
                    ON DUPLICATE KEY UPDATE vibe_id=VALUES(vibe_id)
                """, (row.item_idx, row.tags_version, row.vibe))
        
        # Book subjects (need to resolve llm_subject_idx)
        for row in batch_data:
            for subject in (row.subjects or []):
                cur.execute("""
                    INSERT IGNORE INTO book_llm_subjects(item_idx, llm_subject_idx, tags_version)
                    SELECT %s, llm_subject_idx, %s FROM llm_subjects WHERE subject = %s
                """, (row.item_idx, row.tags_version, subject))
        
        conn.commit()
        print(f"✓ Batch {batch_id} committed successfully")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error processing batch {batch_id}: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def process_errors_batch(batch_df, batch_id):
    """
    Process a micro-batch of error events from DLQ.
    """
    if batch_df.isEmpty():
        return
    
    print(f"Processing errors batch {batch_id} with {batch_df.count()} records")
    
    # Aggregate errors by item_idx (keep most recent)
    errors_agg = (batch_df
        .withColumn("first_seen_at", F.from_unixtime(F.col("timestamp") / 1000))
        .withColumn("last_seen_at", F.from_unixtime(F.col("timestamp") / 1000))
        .withColumn("occurrence_count", F.lit(1))
        .select(
            "item_idx", "first_seen_at", "last_seen_at", "occurrence_count",
            "stage", "error_code", "error_field", "error_msg", "tags_version",
            "title", "author", "attempted"
        )
        .dropDuplicates(["item_idx"]))
    
    # Write to staging
    try:
        errors_agg.write.mode("overwrite").option("truncate", "true") \
            .jdbc(JDBC_URL, "tmp_enrichment_errors_load", properties=JDBC_PROPS)
        
        # Merge into errors table
        _merge_errors()
        
        print(f"✓ Errors batch {batch_id} committed successfully")
        
    except Exception as e:
        print(f"✗ Error processing errors batch {batch_id}: {e}")
        raise

def _merge_errors():
    """
    Merge errors from staging into enrichment_errors table.
    """
    import pymysql
    from urllib.parse import urlparse
    
    parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3306
    db_name = parsed.path.lstrip("/") or "bookrec_db"
    
    conn = pymysql.connect(
        host=host, port=port, user=JDBC_USER, password=JDBC_PASS,
        database=db_name, charset="utf8mb4", autocommit=False
    )
    
    try:
        cur = conn.cursor()
        
        # Create errors table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS enrichment_errors (
                item_idx INT PRIMARY KEY,
                first_seen_at DATETIME,
                last_seen_at DATETIME,
                occurrence_count INT DEFAULT 1,
                stage VARCHAR(64),
                error_code VARCHAR(64),
                error_field VARCHAR(128),
                error_msg TEXT,
                tags_version VARCHAR(32),
                title VARCHAR(256),
                author VARCHAR(256),
                attempted JSON
            ) ENGINE=InnoDB
        """)
        
        # Upsert: update occurrence_count and last_seen_at if exists
        cur.execute("""
            INSERT INTO enrichment_errors
            SELECT * FROM tmp_enrichment_errors_load
            ON DUPLICATE KEY UPDATE
                last_seen_at = VALUES(last_seen_at),
                occurrence_count = occurrence_count + 1,
                stage = VALUES(stage),
                error_code = VALUES(error_code),
                error_msg = VALUES(error_msg)
        """)
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
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
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        .getOrCreate())
    
    # Read results stream
    results_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", RESULTS_TOPIC)
        .option("startingOffsets", "latest")  # or "earliest" for backfill
        .option("maxOffsetsPerTrigger", 1000)  # Backpressure control
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
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 500)
        .option("kafka.group.id", "cg.enrichment.errors.v1")
        .load()
        .select(F.from_json(F.col("value").cast("string"), errors_schema).alias("data"))
        .select("data.*"))
    
    print("✓ Streaming queries started")
    print(f"  - Results: {RESULTS_TOPIC} → SQL")
    print(f"  - Errors: {ERRORS_TOPIC} → SQL")
    print(f"  - Checkpoint: {CHECKPOINT_DIR}")
    
    # Wait for termination
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
