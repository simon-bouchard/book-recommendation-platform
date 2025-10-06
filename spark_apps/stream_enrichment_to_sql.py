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
    Process a micro-batch of enrichment results.
    Writes to staging tables, then executes idempotent SQL upserts.
    """
    if batch_df.isEmpty():
        return
    
    print(f"Processing results batch {batch_id} with {batch_df.count()} records")
    
    # Transform to staging DataFrames - NOW INCLUDING tags_version
    
    # 1. Tones (many-to-many)
    tones = (batch_df
        .withColumn("tone_id", F.explode_outer("tone_ids"))
        .select("item_idx", "tone_id", "tags_version")
        .filter(F.col("tone_id").isNotNull())
        .dropDuplicates())
    
    # 2. Genre (one per book per version)
    genres = (batch_df
        .select("item_idx", F.col("genre").alias("genre_slug"), "tags_version")
        .filter(F.col("genre_slug").isNotNull())
        .dropDuplicates(["item_idx", "tags_version"]))
    
    # 3. Vibe (one per book per version)
    vibes_raw = (batch_df
        .select("item_idx", F.col("vibe").alias("vibe_text"), "tags_version")
        .filter(F.col("vibe_text").isNotNull())
        .dropDuplicates(["item_idx", "tags_version"]))
    
    # 4. LLM subjects (many-to-many)
    subjects = (batch_df
        .withColumn("llm_subject", F.explode_outer("subjects"))
        .select("item_idx", "llm_subject", "tags_version")
        .filter(F.col("llm_subject").isNotNull())
        .withColumn("llm_subject", F.lower(F.trim(F.col("llm_subject"))))
        .filter(F.length("llm_subject") > 0)
        .dropDuplicates())
    
    # Write to staging tables
    try:
        # Subject dictionary (no version - subjects are global)
        llm_subjects_distinct = subjects.select("llm_subject").dropDuplicates()
        if not llm_subjects_distinct.isEmpty():
            llm_subjects_distinct.write.mode("overwrite").option("truncate", "true") \
                .jdbc(JDBC_URL, "tmp_llm_subjects_load", properties=JDBC_PROPS)
        
        # Vibe texts (no version - vibes are global)
        vibe_texts = vibes_raw.select(F.col("vibe_text").alias("text")).dropDuplicates()
        if not vibe_texts.isEmpty():
            vibe_texts.write.mode("overwrite").option("truncate", "true") \
                .jdbc(JDBC_URL, "tmp_vibes_load", properties=JDBC_PROPS)
        
        # Book vibes staging (WITH tags_version)
        book_vibes_staging = vibes_raw.select("item_idx", "vibe_text", "tags_version") \
            .dropDuplicates(["item_idx", "tags_version"])
        if not book_vibes_staging.isEmpty():
            book_vibes_staging.write.mode("overwrite").option("truncate", "true") \
                .jdbc(JDBC_URL, "tmp_book_vibes_load", properties=JDBC_PROPS)
        
        # Link tables (WITH tags_version)
        if not tones.isEmpty():
            tones.write.mode("overwrite").option("truncate", "true") \
                .jdbc(JDBC_URL, "tmp_book_tones_load", properties=JDBC_PROPS)
        
        if not genres.isEmpty():
            genres.write.mode("overwrite").option("truncate", "true") \
                .jdbc(JDBC_URL, "tmp_book_genres_load", properties=JDBC_PROPS)
        
        if not subjects.isEmpty():
            subjects.write.mode("overwrite").option("truncate", "true") \
                .jdbc(JDBC_URL, "tmp_book_llm_subjects_load", properties=JDBC_PROPS)
        
        # Execute SQL merges
        _execute_sql_merges()
        
        print(f"✓ Batch {batch_id} committed successfully")
        
    except Exception as e:
        print(f"✗ Error processing batch {batch_id}: {e}")
        raise


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


def _execute_sql_merges():
    """
    Execute idempotent SQL merges from staging to final tables.
    NOW VERSION-AWARE: Includes tags_version in all upserts.
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
        
        # Subject dictionary (global, no version)
        cur.execute("""
            INSERT IGNORE INTO llm_subjects(subject)
            SELECT DISTINCT llm_subject FROM tmp_llm_subjects_load
        """)
        
        # Vibe dictionary (global, no version)
        cur.execute("""
            INSERT IGNORE INTO vibes(text)
            SELECT DISTINCT text FROM tmp_vibes_load
        """)
        
        # Book -> Vibe (WITH tags_version)
        cur.execute("""
            INSERT INTO book_vibes(item_idx, vibe_id, tags_version)
            SELECT b.item_idx, v.vibe_id, b.tags_version
            FROM tmp_book_vibes_load b
            JOIN vibes v ON v.text = b.vibe_text
            ON DUPLICATE KEY UPDATE vibe_id = VALUES(vibe_id)
        """)
        
        # Book -> Genre (WITH tags_version)
        cur.execute("""
            INSERT INTO book_genres(item_idx, genre_slug, tags_version)
            SELECT item_idx, genre_slug, tags_version FROM tmp_book_genres_load
            ON DUPLICATE KEY UPDATE genre_slug = VALUES(genre_slug)
        """)
        
        # Book -> Tone (WITH tags_version)
        cur.execute("""
            INSERT IGNORE INTO book_tones(item_idx, tone_id, tags_version)
            SELECT item_idx, tone_id, tags_version FROM tmp_book_tones_load
        """)
        
        # Book -> LLM Subject (WITH tags_version)
        cur.execute("""
            INSERT IGNORE INTO book_llm_subjects(item_idx, llm_subject_idx, tags_version)
            SELECT t.item_idx, s.llm_subject_idx, t.tags_version
            FROM tmp_book_llm_subjects_load t
            JOIN llm_subjects s ON s.subject = t.llm_subject
        """)
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


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
    
    # Start results query
    results_query = (results_stream
        .writeStream
        .foreachBatch(process_results_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/results")
        .start())
    
    # Start errors query
    errors_query = (errors_stream
        .writeStream
        .foreachBatch(process_errors_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/errors")
        .start())
    
    print("✓ Streaming queries started")
    print(f"  - Results: {RESULTS_TOPIC} → SQL")
    print(f"  - Errors: {ERRORS_TOPIC} → SQL")
    print(f"  - Checkpoint: {CHECKPOINT_DIR}")
    
    # Wait for termination
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
