# ops/meilisearch/index_books_meili.py
import sys, os
from pathlib import Path
from meilisearch import Client
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import Book
from models.shared_utils import ModelStore
from tqdm import tqdm

BATCH_SIZE = 2000

load_dotenv()
secret_key = os.getenv("MEILI_MASTER_KEY")

client = Client("http://localhost:7700", secret_key)
index = client.index("books")

model_store = ModelStore()

def configure_index():
    index.update_settings({
        "searchableAttributes": [
            "title",
            "author",
            "subjects",
            "description"
        ],
        "sortableAttributes": [
            "bayes_pop",
            "year"
        ],
        "filterableAttributes": [
            "subjects",
            "year"
        ],
         "rankingRules": [
             "words",
             "typo",
             "proximity",
             "attribute",
             "sort",
             "exactness",
         ]
    })
    print("✓ Index settings applied.")


def book_to_doc(b, item_to_row, bayes_tensor):
    row = item_to_row.get(b.item_idx)
    bayes_pop = float(bayes_tensor[row]) if row is not None else 0.0

    subjects = [s.subject.subject for s in b.subjects]
    author = b.author.name if getattr(b, "author", None) else None

    return {
        "item_idx": b.item_idx,
        "title": b.title,
        "author": author,
        "description": b.description,
        "subjects": subjects,
        "year": b.year,
        "bayes_pop": bayes_pop,
        "cover_id": b.cover_id,
        "isbn": b.isbn,
    }

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def main():
    db = SessionLocal()

    print("→ Configuring index settings...")
    configure_index()

    print("→ Loading model-store metadata...")
    item_to_row = model_store.get_item_idx_to_row()
    bayes_tensor = model_store.get_bayesian_tensor()

    print("→ Querying books from SQL...")
    books = db.query(Book).all()

    docs = []
    for b in books:
        docs.append(book_to_doc(b, item_to_row, bayes_tensor))

    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"→ Sending {len(docs)} documents to Meilisearch in {total_batches} batches...")

    for batch_idx, batch in enumerate(
        tqdm(chunked(docs, BATCH_SIZE), total=total_batches, desc="Indexing batches"),
        start=1
    ):
        index.add_documents(batch)

    print("✓ Indexing complete.")


if __name__ == "__main__":
    main()

