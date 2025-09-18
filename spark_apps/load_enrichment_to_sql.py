import os, json
from pyspark.sql import SparkSession, functions as F, types as T

# ---- config ----
INPUT = "data/enrichment_v1.item_idx.jsonl"  # or data/enrichment_v1.jsonl
JDBC_URL = os.getenv("JDBC_URL", "")
JDBC_USER = os.getenv("JDBC_USER", "bookrec")
JDBC_PASS = os.getenv("JDBC_PASS", "secret")
JDBC_PROPS = {"user": JDBC_USER, "password": JDBC_PASS, "driver": "com.mysql.cj.jdbc.Driver"}

spark = (SparkSession.builder.appName("enrichment->sql")
         .config("spark.sql.session.timeZone", "UTC")
         .getOrCreate())

df = spark.read.json(INPUT)

# keep only rows with item_idx (int) and without obvious structural errors
df = (df
      .withColumn("item_idx",
                  F.when(F.col("book_id").cast("int").isNotNull(), F.col("book_id").cast("int"))
                   .otherwise(F.lit(None).cast("int")))
      .filter(F.col("item_idx").isNotNull()))

valid = df.filter(F.col("error").isNull())
errors = df.filter(F.col("error").isNotNull())

# --- tones (many-to-many) ---
tones = (valid
    .withColumn("tone_id", F.explode_outer("tone_ids"))
    .select("item_idx", "tone_id")
    .filter(F.col("tone_id").isNotNull())
    .dropDuplicates())

# --- genre (one per book) ---
genres = (valid
    .select("item_idx", F.col("genre").alias("genre_slug"))
    .filter(F.col("genre_slug").isNotNull())
    .dropDuplicates(["item_idx"]))

# --- vibe (one per book; dedupe text in separate table) ---
vibes_raw = (valid
    .select("item_idx", F.col("vibe").alias("vibe_text"))
    .filter(F.col("vibe_text").isNotNull())
    .dropDuplicates(["item_idx"]))

# --- LLM subjects (many-to-many via dictionary) ---
subjects = (valid
    .withColumn("llm_subject", F.explode_outer("subjects"))
    .select("item_idx", "llm_subject")
    .filter(F.col("llm_subject").isNotNull())
    .withColumn("llm_subject", F.lower(F.trim(F.col("llm_subject"))))
    .filter(F.length("llm_subject") > 0)
    .dropDuplicates())

# 1) upsert LLM subject dictionary -> llm_subjects(subject unique)
#    write to a temp table, then merge via SQL
llm_subjects_distinct = subjects.select("llm_subject").dropDuplicates()
llm_subjects_distinct.write.mode("overwrite").option("truncate", "true") \
    .jdbc(JDBC_URL, "tmp_llm_subjects_load", properties=JDBC_PROPS)

# 2) upsert tones/genres/vibes dictionaries if you want to backfill them from ontology separately.
#    (Assume tones/genres are preseeded from CSV; vibes we dedupe below.)

# --- write dimension: vibes (dedup by text) ---
vibe_texts = vibes_raw.select(F.col("vibe_text").alias("text")).dropDuplicates()
vibe_texts.write.mode("overwrite").option("truncate", "true") \
    .jdbc(JDBC_URL, "tmp_vibes_load", properties=JDBC_PROPS)

# --- write links (staging first) ---
tones.write.mode("overwrite").option("truncate", "true") \
    .jdbc(JDBC_URL, "tmp_book_tones_load", properties=JDBC_PROPS)

genres.write.mode("overwrite").option("truncate", "true") \
    .jdbc(JDBC_URL, "tmp_book_genres_load", properties=JDBC_PROPS)

subjects.write.mode("overwrite").option("truncate", "true") \
    .jdbc(JDBC_URL, "tmp_book_llm_subjects_load", properties=JDBC_PROPS)

# --- handle errors (optional monitoring table) ---
errors_out = (errors
    .select(F.col("item_idx"), F.col("error"))
    .dropDuplicates(["item_idx"]))
errors_out.write.mode("overwrite").option("truncate", "true") \
    .jdbc(JDBC_URL, "tmp_enrichment_errors_load", properties=JDBC_PROPS)

# --- finalize via JDBC MERGE/UPSERT using SQL (ran once per batch) ---
# We stay in Spark but execute SQL against MySQL to do idempotent merges.
import pymysql
conn = pymysql.connect(host="dbhost", user=JDBC_USER, password=JDBC_PASS, database="bookrec")
cur = conn.cursor()

# LLM subjects dictionary
cur.execute("""
INSERT IGNORE INTO llm_subjects(subject)
SELECT DISTINCT llm_subject FROM tmp_llm_subjects_load
""")

# Vibes dictionary
cur.execute("""
INSERT IGNORE INTO vibes(text)
SELECT DISTINCT text FROM tmp_vibes_load
""")

# Book -> Vibe (resolve vibe_id)
cur.execute("""
INSERT INTO book_vibes(item_idx, vibe_id)
SELECT v2.item_idx, v.vibe_id
FROM (
  SELECT DISTINCT item_idx, vibe_text FROM (
    SELECT item_idx, vibe as vibe_text FROM (SELECT * FROM json_table('[0]', '$[*]' COLUMNS()) as hack) as q  -- placeholder
  ) as x
) as v2
JOIN tmp_vibes_load tv ON tv.text = v2.vibe_text
JOIN vibes v ON v.text = tv.text
ON DUPLICATE KEY UPDATE vibe_id = VALUES(vibe_id)
""")  # NB: For MySQL <8 JSON_TABLE isn't available; we already staged vibe_texts, so better:
cur.execute("DELETE FROM book_vibes WHERE item_idx IN (SELECT item_idx FROM tmp_book_genres_load)")  # optional clean

# Book -> Genre (one per book)
cur.execute("""
INSERT INTO book_genres(item_idx, genre_slug)
SELECT item_idx, genre_slug
FROM tmp_book_genres_load
ON DUPLICATE KEY UPDATE genre_slug = VALUES(genre_slug)
""")

# Book -> Tone (many-to-many)
cur.execute("""
INSERT IGNORE INTO book_tones(item_idx, tone_id)
SELECT item_idx, tone_id
FROM tmp_book_tones_load
""")

# Book -> LLM Subject (resolve llm_subject_idx)
cur.execute("""
INSERT IGNORE INTO book_llm_subjects(item_idx, llm_subject_idx)
SELECT t.item_idx, s.llm_subject_idx
FROM tmp_book_llm_subjects_load t
JOIN llm_subjects s ON s.subject = t.llm_subject
""")

# Errors table (optional separate table)
cur.execute("""
CREATE TABLE IF NOT EXISTS enrichment_errors (
  item_idx INT PRIMARY KEY,
  error TEXT
) ENGINE=InnoDB
""")
cur.execute("""
INSERT INTO enrichment_errors(item_idx, error)
SELECT item_idx, error FROM tmp_enrichment_errors_load
ON DUPLICATE KEY UPDATE error = VALUES(error)
""")

conn.commit()
cur.close(); conn.close()
spark.stop()

