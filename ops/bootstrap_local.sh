#!/bin/bash
# ops/bootstrap_local.sh
#
# Runs the "Database" through "Search indexes" steps of the README's Local
# Setup section, in order: create tables, import the Kaggle dataset CSVs,
# train the model artifacts from scratch, and build the search indexes.
#
# Preconditions (not automated here, see README.md#local-setup):
#   - .env exists and has a valid DATABASE_URL (see .env.example)
#   - the 7 CSVs from the Kaggle dataset are in data/
#   - a MySQL instance is running and reachable at DATABASE_URL
#
# Run from the repo root: ops/bootstrap_local.sh

set -e  # Exit on error

CSV_FILES=(
    "data/books.csv"
    "data/authors.csv"
    "data/users.csv"
    "data/interactions.csv"
    "data/books_to_subjects.csv"
    "data/users_to_subjects.csv"
    "data/book_enrichment_v2.csv"
)

echo "=== Checking preconditions ==="

if [ ! -f .env ]; then
    echo "Error: .env not found. Copy .env.example to .env and fill it in first." >&2
    exit 1
fi

missing=0
for f in "${CSV_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Missing: $f" >&2
        missing=1
    fi
done
if [ "$missing" -eq 1 ]; then
    echo "Download the dataset from https://www.kaggle.com/datasets/simonbouchardk/book-recommendation-platform-data and place the CSVs in data/." >&2
    exit 1
fi

echo "Creating database (if it doesn't already exist)..."
python -c "
import os
from urllib.parse import urlsplit
from dotenv import load_dotenv
import pymysql

load_dotenv()
url = urlsplit(os.environ['DATABASE_URL'])
db_name = url.path.lstrip('/')
conn = pymysql.connect(host=url.hostname, port=url.port or 3306, user=url.username, password=url.password or '')
conn.cursor().execute(f'CREATE DATABASE IF NOT EXISTS \`{db_name}\`')
conn.close()
print(f'Database \"{db_name}\" ready.')
"

echo "=== 1. Database ==="
python data/create_tables.py
python data/import_csvs.py
python data/import_enrichment_csvs.py

echo "=== 2. Model artifacts ==="
echo "-> train_subject_embs_contrastive.py (one-time subject embedding bootstrap)"
python models/training/train_subject_embs_contrastive.py --pad-idx "${PAD_IDX:-0}"
echo "-> export_training_data"
python -m models.training.export_training_data
echo "-> precompute_embs.py"
python models/training/precompute_embs.py --pad-idx "${PAD_IDX:-0}"
echo "-> precompute_bayesian.py"
python models/training/precompute_bayesian.py --pad-idx "${PAD_IDX:-0}"
echo "-> build_metadata_lookup.py"
python models/training/build_metadata_lookup.py
echo "-> train_als.py"
python models/training/train_als.py --pad-idx "${PAD_IDX:-0}"
echo "-> build_similarity_indices.py"
python models/training/build_similarity_indices.py

echo "=== 3. Search indexes ==="
echo "-> semantic index"
python app/semantic_index/builders/build_enriched_index.py --tags-version v2 --full \
    --output models/artifacts/semantic_indexes/enriched_v2

echo
echo "=== Done ==="
echo "Data imported, model artifacts and semantic index built."
echo "Meilisearch indexing (ops/meilisearch/index_books_meili.py) and starting"
echo "services (README.md#local-setup, step 6) are not run by this script."
