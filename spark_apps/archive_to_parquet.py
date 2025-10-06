# spark_apps/archive_to_parquet.py
"""
Archive Kafka events to MinIO using Kafka's built-in timestamp for partitioning.
"""
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, ArrayType, MapType

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/archival")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123456")

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
    if batch_df.isEmpty():
        return
    
    count = batch_df.count()
    print(f"\nProcessing results batch {batch_id} with {count} records")
    batch_df.select("item_idx", "kafka_timestamp", "year", "month", "day").show(5)
    
    try:
        (batch_df
            .write
            .mode("append")
            .partitionBy("tag_version", "year", "month", "day")
            .parquet("s3a://enrichment-bronze/enrich.results.v1/"))
        
        print(f"✓ Batch {batch_id} written")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def process_errors_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    
    count = batch_df.count()
    print(f"\nProcessing errors batch {batch_id} with {count} records")
    
    try:
        (batch_df
            .write
            .mode("append")
            .partitionBy("year", "month", "day")
            .parquet("s3a://enrichment-bronze/enrich.errors.v1/"))
        
        print(f"✓ Batch {batch_id} written")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def main():
    spark = (SparkSession.builder
        .appName("enrichment-bronze-archival")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*80)
    print("BRONZE ARCHIVAL - Using Kafka Timestamp")
    print("="*80)
    
    # Read results - NO DIVISION, timestamp is already a TIMESTAMP type
    results_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "enrich.results.v1")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 10000)
        .load()
        .select(
            F.from_json(F.col("value").cast("string"), results_schema).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        .withColumn("year", F.year(F.col("kafka_timestamp")))
        .withColumn("month", F.month(F.col("kafka_timestamp")))
        .withColumn("day", F.dayofmonth(F.col("kafka_timestamp"))))
    
    # Read errors
    errors_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "enrich.errors.v1")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 5000)
        .load()
        .select(
            F.from_json(F.col("value").cast("string"), errors_schema).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        .withColumn("year", F.year(F.col("kafka_timestamp")))
        .withColumn("month", F.month(F.col("kafka_timestamp")))
        .withColumn("day", F.dayofmonth(F.col("kafka_timestamp"))))
    
    # Write
    results_query = (results_stream
        .writeStream
        .foreachBatch(process_results_batch)
        .outputMode("append")
        .trigger(processingTime="5 minutes")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/results-parquet")
        .start())
    
    errors_query = (errors_stream
        .writeStream
        .foreachBatch(process_errors_batch)
        .outputMode("append")
        .trigger(processingTime="5 minutes")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/errors-parquet")
        .start())
    
    print("✓ Archival started")
    print(f"  Results → s3a://enrichment-bronze/enrich.results.v1/")
    print(f"  Errors  → s3a://enrichment-bronze/enrich.errors.v1/")
    print("\n")
    
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
