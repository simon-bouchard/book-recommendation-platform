# spark_apps/archive_to_parquet.py
"""
Spark Structured Streaming job to archive Kafka events to MinIO as Parquet.
Runs alongside the main SQL consumer, purely for bronze archival.
"""
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/archival")

# MinIO configuration (S3-compatible)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123456")

# Schemas (same as SQL consumer)
results_schema = StructType([
    StructField("item_idx", IntegerType(), False),
    StructField("subjects", ArrayType(StringType()), False),
    StructField("tone_ids", ArrayType(IntegerType()), False),
    StructField("genre", StringType(), False),
    StructField("vibe", StringType(), False),
    StructField("tags_version", StringType(), False),
    StructField("scores", MapType(StringType(), StringType()), True),
    StructField("timestamp", IntegerType(), True),
    StructField("metadata", MapType(StringType(), StringType()), True),
])

errors_schema = StructType([
    StructField("item_idx", IntegerType(), False),
    StructField("tags_version", StringType(), False),
    StructField("timestamp", IntegerType(), True),
    StructField("stage", StringType(), False),
    StructField("error_code", StringType(), False),
    StructField("error_field", StringType(), True),
    StructField("error_msg", StringType(), False),
    StructField("title", StringType(), True),
    StructField("author", StringType(), True),
    StructField("attempted", MapType(StringType(), StringType()), True),
])


def main():
    """Archive Kafka topics to Parquet in MinIO"""
    
    spark = (SparkSession.builder
        .appName("enrichment-bronze-archival")
        .config("spark.sql.session.timeZone", "UTC")
        # MinIO/S3 configuration
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate())
    
    # Read results stream
    results_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "enrich.results.v1")
        .option("startingOffsets", "earliest")  # Archive everything
        .option("maxOffsetsPerTrigger", 10000)  # Higher throughput for archival
        .load()
        .select(
            F.from_json(F.col("value").cast("string"), results_schema).alias("data")
        )
        .select("data.*")
        # Add partitioning columns from the timestamp field in the data
        .withColumn("year", F.year(F.from_unixtime(F.col("timestamp") / 1000)))
        .withColumn("month", F.month(F.from_unixtime(F.col("timestamp") / 1000)))
        .withColumn("day", F.dayofmonth(F.from_unixtime(F.col("timestamp") / 1000)))
    )
    
    # Read errors stream
    errors_stream = (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "enrich.errors.v1")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 5000)
        .load()
        .select(
            F.from_json(F.col("value").cast("string"), errors_schema).alias("data")
        )
        .select("data.*")
        # Add partitioning columns
        .withColumn("year", F.year(F.from_unixtime(F.col("timestamp") / 1000)))
        .withColumn("month", F.month(F.from_unixtime(F.col("timestamp") / 1000)))
        .withColumn("day", F.dayofmonth(F.from_unixtime(F.col("timestamp") / 1000)))
    )
    
    # Write results to Parquet
    results_query = (results_stream
        .writeStream
        .format("parquet")
        .option("path", "s3a://enrichment-bronze/enrich.results.v1/")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/results-parquet")
        .partitionBy("year", "month", "day")
        .outputMode("append")
        .trigger(processingTime="5 minutes")  # Batch every 5 minutes
        .start())
    
    # Write errors to Parquet
    errors_query = (errors_stream
        .writeStream
        .format("parquet")
        .option("path", "s3a://enrichment-bronze/enrich.errors.v1/")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/errors-parquet")
        .partitionBy("year", "month", "day")
        .outputMode("append")
        .trigger(processingTime="5 minutes")
        .start())
    
    print("✓ Bronze archival started")
    print(f"  - Results → s3a://enrichment-bronze/enrich.results.v1/")
    print(f"  - Errors → s3a://enrichment-bronze/enrich.errors.v1/")
    print(f"  - Format: Parquet (Snappy compression)")
    print(f"  - Partitioned by: year/month/day")
    
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
