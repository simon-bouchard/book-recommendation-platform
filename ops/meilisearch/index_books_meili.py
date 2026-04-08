# ops/meilisearch/index_books_meili.py
"""
Index all books in Meilisearch with enrichments and Bayesian scores.

Uses new artifact loading infrastructure. Preserves original subject_ids
while adding v2 enrichments (genres, tones, vibes, LLM subjects).
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from meilisearch import Client
from sqlalchemy import text
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from models.data.loaders import (
    get_item_idx_to_row,
    load_bayesian_scores,
    load_book_meta,
    load_book_subject_embeddings,
)

load_dotenv()
client = Client("http://localhost:7700", os.getenv("MEILI_MASTER_KEY"))
index = client.index("books")

TAGS_VERSION = "v2"
BATCH_SIZE = 5000


def configure_index():
    """Configure Meilisearch index settings."""
    index.update_settings(
        {
            "searchableAttributes": [
                "title",
                "author",
                "description",
                "genre_slugs",
                "tone_slugs",
                "vibe_texts",
                "llm_subject_slugs",
            ],
            "filterableAttributes": [
                "subject_ids",
                "genre_slugs",
                "tone_slugs",
                "vibe_texts",
                "llm_subject_slugs",
                "year",
                "tags_version",
            ],
            "sortableAttributes": ["bayes_pop", "num_ratings", "avg_rating", "year"],
            "rankingRules": ["words", "typo", "proximity", "attribute", "sort", "exactness"],
            "faceting": {
                "maxValuesPerFacet": 200000,
                "sortFacetValuesBy": {"subject_ids": "count"},
            },
        }
    )
    print("Meilisearch settings applied (subject_ids preserved)")


def fetch_all_data(db):
    """Fetch all book data and enrichments from database."""
    print("Bulk-fetching all data...")

    books_df = pd.read_sql(
        text("""
        SELECT
            b.item_idx, b.title, b.description, b.year, b.cover_id, b.isbn,
            a.name AS author
        FROM books b
        LEFT JOIN authors a ON b.author_idx = a.author_idx
    """),
        db.bind,
    )

    # Original subjects - never touch
    subject_df = pd.read_sql(
        text("""
        SELECT item_idx, subject_idx
        FROM book_subjects
    """),
        db.bind,
    )

    # v2 enrichments only
    genre_df = pd.read_sql(
        text("""
        SELECT bg.item_idx, g.slug AS genre_slug
        FROM book_genres bg
        JOIN genres g ON bg.genre_slug = g.slug AND bg.genre_ontology_version = g.ontology_version
        WHERE bg.tags_version = :version
    """),
        db.bind,
        params={"version": TAGS_VERSION},
    )

    tone_df = pd.read_sql(
        text("""
        SELECT bt.item_idx, t.slug AS tone_slug
        FROM book_tones bt
        JOIN tones t ON bt.tone_id = t.tone_id
        WHERE bt.tags_version = :version
    """),
        db.bind,
        params={"version": TAGS_VERSION},
    )

    vibe_df = pd.read_sql(
        text("""
        SELECT bv.item_idx, v.text AS vibe_text
        FROM book_vibes bv
        JOIN vibes v ON bv.vibe_id = v.vibe_id
        WHERE bv.tags_version = :version
    """),
        db.bind,
        params={"version": TAGS_VERSION},
    )

    llm_subject_df = pd.read_sql(
        text("""
        SELECT bl.item_idx, ls.subject AS llm_subject_slug
        FROM book_llm_subjects bl
        JOIN llm_subjects ls ON bl.llm_subject_idx = ls.llm_subject_idx
        WHERE bl.tags_version = :version
    """),
        db.bind,
        params={"version": TAGS_VERSION},
    )

    return books_df, subject_df, genre_df, tone_df, vibe_df, llm_subject_df


def build_documents(books_df, subject_df, genre_df, tone_df, vibe_df, llm_subject_df):
    """Build Meilisearch documents with all enrichments and scores."""
    print("Building documents (vectorized, safe)...")

    # Load model artifacts
    book_meta = load_book_meta(use_cache=True)
    bayesian_scores = load_bayesian_scores(use_cache=True)
    _, book_ids = load_book_subject_embeddings(use_cache=True)
    item_to_row = get_item_idx_to_row(book_ids)

    # Group everything by item_idx
    subject_groups = subject_df.groupby("item_idx")["subject_idx"].apply(list)
    genre_groups = genre_df.groupby("item_idx")["genre_slug"].apply(list)
    tone_groups = tone_df.groupby("item_idx")["tone_slug"].apply(list)
    vibe_groups = vibe_df.groupby("item_idx")["vibe_text"].apply(list)
    llm_groups = llm_subject_df.groupby("item_idx")["llm_subject_slug"].apply(list)

    df = books_df.set_index("item_idx")

    # Helper: map grouped series -> empty list if missing
    def safe_map(series):
        return df.index.to_series().map(series).apply(lambda x: x if isinstance(x, list) else [])

    # Original subject_ids - untouched, exactly as before
    df["subject_ids"] = safe_map(subject_groups)

    # New v2 enrichments - additive only
    df["genre_slugs"] = safe_map(genre_groups)
    df["tone_slugs"] = safe_map(tone_groups)
    df["vibe_texts"] = safe_map(vibe_groups)
    df["llm_subject_slugs"] = safe_map(llm_groups)

    # Ratings & bayes from daily training pipeline
    df = df.join(book_meta[["book_num_ratings", "book_avg_rating"]], how="left")
    df["num_ratings"] = df["book_num_ratings"].fillna(0).astype(int)
    df["avg_rating"] = df["book_avg_rating"].round(3)

    # Map Bayesian scores
    df["bayes_pop"] = df.index.map(
        lambda idx: round(float(bayesian_scores[item_to_row.get(idx, -1)]), 6)
        if item_to_row.get(idx) is not None and 0 <= item_to_row.get(idx, -1) < len(bayesian_scores)
        else 0.0
    )

    df["tags_version"] = TAGS_VERSION

    # Clean text fields
    df = df.reset_index()
    df["title"] = df["title"].fillna("").str.strip()
    df["author"] = df["author"].fillna("")
    df["description"] = df["description"].fillna("")

    return df.to_dict("records")


def main():
    """Main indexing workflow."""
    configure_index()

    db = SessionLocal()
    try:
        books_df, subject_df, genre_df, tone_df, vibe_df, llm_subject_df = fetch_all_data(db)
        docs = build_documents(books_df, subject_df, genre_df, tone_df, vibe_df, llm_subject_df)

        print(f"Pushing {len(docs):,} documents to Meilisearch...")
        for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Indexing"):
            index.add_documents(docs[i : i + BATCH_SIZE])

        print("Done! v2 enrichments added - original subject_ids fully preserved")
    finally:
        db.close()


if __name__ == "__main__":
    main()
