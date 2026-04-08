# ops/meilisearch/update_bayes_pop.py
"""
Update Bayesian popularity scores in Meilisearch after training.

Run this after automated training to refresh bayes_pop for all books.
Uses new artifact loading infrastructure.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from meilisearch import Client
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.data.loaders import (
    get_item_idx_to_row,
    load_bayesian_scores,
    load_book_subject_embeddings,
)

load_dotenv()
client = Client("http://localhost:7700", os.getenv("MEILI_MASTER_KEY"))
index = client.index("books")

BATCH_SIZE = 5000


def refresh_bayes_pop_after_training():
    """
    Update bayes_pop for all books in Meilisearch.

    Run this after training pipeline to reflect new Bayesian popularity scores.
    """
    print("Loading artifacts...")

    # Load Bayesian scores and book IDs
    bayesian_scores = load_bayesian_scores(use_cache=False)
    _, book_ids = load_book_subject_embeddings(use_cache=False)
    item_to_row = get_item_idx_to_row(book_ids)

    print(f"Loaded {len(book_ids)} books with Bayesian scores")

    print("Building bayes_pop updates...")
    updates = []

    for item_idx, row_idx in tqdm(item_to_row.items(), desc="Preparing updates"):
        if 0 <= row_idx < len(bayesian_scores):
            updates.append(
                {"item_idx": item_idx, "bayes_pop": round(float(bayesian_scores[row_idx]), 6)}
            )

    print(f"Updating bayes_pop for {len(updates)} books in Meilisearch...")

    for i in tqdm(range(0, len(updates), BATCH_SIZE), desc="Indexing"):
        batch = updates[i : i + BATCH_SIZE]
        index.update_documents(batch)

    print(f"✅ Updated bayes_pop for {len(updates)} books")


if __name__ == "__main__":
    refresh_bayes_pop_after_training()
