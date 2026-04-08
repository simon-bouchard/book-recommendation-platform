# app/semantic_index/builders/build_baseline_clean_index.py
"""
Build baseline-clean semantic search index (no description, subjects only).
Tests whether descriptions add noise to baseline embeddings.
Includes ALL books (same coverage as baseline-old) for fair comparison.
"""

import sys
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # Go up to project root
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.semantic_index.index_store import IndexStore

# Import text templates from templates folder
from app.semantic_index.templates.text_templates import (
    build_baseline_clean_text,
    validate_text_not_empty,
)
from app.table_models import Author, Book, BookOLSubject, OLSubject


def fetch_baseline_clean_books(limit: int = None) -> Generator[Tuple[int, str, Dict], None, None]:
    """
    Fetch books with raw OL subjects (no description).
    Includes ALL books for fair comparison with baseline-old.

    Yields:
        (item_idx, embedding_text, metadata_dict) tuples

    Notes:
        - Books without subjects get text: "{title} — {author}" (no subjects section)
        - Books with subjects get text: "{title} — {author} | subjects: {subjects}"
        - This ensures same book coverage as baseline-old for fair comparison
    """
    print(f"\n{'=' * 80}")
    print("Fetching baseline books (no description, includes all books)")
    print(f"{'=' * 80}\n")

    with SessionLocal() as db:
        # Query books with titles
        query = (
            db.query(Book.item_idx, Book.title, Author.name.label("author"))
            .outerjoin(Author, Book.author_idx == Author.author_idx)
            .filter(Book.title.isnot(None))
            .order_by(Book.item_idx)
        )

        if limit:
            query = query.limit(limit)

        books = query.all()
        total = len(books)

        print(f"Processing {total:,} books...")

        processed = 0
        skipped = 0

        for book in books:
            item_idx = book.item_idx
            title = (book.title or "").strip()
            author = (book.author or "").strip()

            if not title:
                skipped += 1
                continue

            # Fetch OL subjects
            subjects_query = (
                db.query(OLSubject.subject)
                .join(BookOLSubject, OLSubject.ol_subject_idx == BookOLSubject.ol_subject_idx)
                .filter(BookOLSubject.item_idx == item_idx)
                .all()
            )
            ol_subjects = [s for (s,) in subjects_query if s]

            # Build embedding text (no description)
            # Include all books, even without subjects - fair comparison with baseline-old
            text = build_baseline_clean_text(title=title, author=author, ol_subjects=ol_subjects)

            # Validate
            try:
                validate_text_not_empty(text, f"baseline-clean item_idx={item_idx}")
            except ValueError as e:
                print(f"⚠️  Skipping item_idx={item_idx}: {e}")
                skipped += 1
                continue

            meta = {
                "title": title,
                "author": author,
                "num_subjects": len(ol_subjects),
                "tags_version": "baseline_clean",
            }

            processed += 1

            if processed % 1000 == 0:
                print(f"  Processed: {processed:,}/{total:,} ({skipped:,} skipped)")

            yield item_idx, text, meta

        print(f"\n✅ Final: {processed:,} books processed, {skipped:,} skipped")


def embed_texts_batch(
    texts: List[Tuple[int, str, Dict]],
    embedder=None,
    use_multiprocess=False,
    model=None,
    num_processes=4,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Embed texts in batch.

    Args:
        texts: List of (item_idx, text, meta) tuples
        embedder: Embedding function (for single-process mode)
        use_multiprocess: Use multi-process encoding (faster on multi-core CPU)
        model: SentenceTransformer model (required if use_multiprocess=True)
        num_processes: Number of processes for multiprocessing

    Returns:
        (embeddings, ids, metadata) tuple
    """
    if not texts:
        raise ValueError("No texts to embed")

    ids = []
    metas = []
    text_strings = []

    for item_idx, text, meta in texts:
        ids.append(item_idx)
        metas.append(meta)
        text_strings.append(text)

    print(f"\nEmbedding {len(text_strings):,} texts...")

    if use_multiprocess:
        print(f"🚀 Using multi-process encoding ({num_processes} processes)...")
        pool = model.start_multi_process_pool(target_devices=["cpu"] * num_processes)

        embeddings = model.encode_multi_process(
            text_strings,
            pool,
            batch_size=32,  # Per process
            chunk_size=1000,
            show_progress_bar=True,
        )

        model.stop_multi_process_pool(pool)
        embeddings = np.array(embeddings)
    else:
        print("Embedding texts (single process)...")
        embeddings = embedder(
            text_strings, batch_size=256, show_progress_bar=True, convert_to_numpy=True
        )

    ids_array = np.array(ids, dtype=np.int64)

    print(f"✅ Created {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})")

    return embeddings, ids_array, metas


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """Build FAISS HNSW index."""
    print("\nBuilding FAISS HNSW index...")

    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 80
    index.add(embeddings.astype("float32"))

    print(f"✅ FAISS index built ({index.ntotal:,} vectors)")

    return index


def main(
    output_dir: str,
    embedder=None,
    use_multiprocess=False,
    model=None,
    num_processes=4,
    limit: int = None,
):
    """
    Build baseline-clean semantic search index.

    Includes ALL books (same coverage as baseline-old) for fair comparison.
    Only difference from baseline-old: no description field in embedding text.

    Args:
        output_dir: Output directory
        embedder: Embedding function (for single-process mode)
        use_multiprocess: Use multi-process encoding (faster on multi-core CPU)
        model: SentenceTransformer model (for multi-process mode)
        num_processes: Number of processes for multiprocessing
        limit: Optional limit for testing
    """
    print(f"\n{'=' * 80}")
    print("BASELINE-CLEAN INDEX BUILDER")
    print(f"{'=' * 80}")
    print("Variant: No description, OL subjects only")
    print(f"Output: {output_dir}")
    if use_multiprocess:
        print(f"Mode: Multi-process ({num_processes} workers)")
    else:
        print("Mode: Single-process")
    if limit:
        print(f"Limit: {limit} books (testing mode)")
    print(f"{'=' * 80}\n")

    # Fetch books
    texts_generator = fetch_baseline_clean_books(limit=limit)
    texts = list(texts_generator)

    if not texts:
        raise RuntimeError("No valid baseline books found")

    # Embed
    embeddings, ids, metas = embed_texts_batch(
        texts,
        embedder=embedder,
        use_multiprocess=use_multiprocess,
        model=model,
        num_processes=num_processes,
    )

    # Build index
    index = build_faiss_index(embeddings)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    IndexStore(out_path).save(
        out_path / "semantic.faiss",
        out_path / "semantic_ids.npy",
        out_path / "semantic_meta.json",
        index,
        ids,
        metas,
    )

    print(f"\n{'=' * 80}")
    print(f"✅ Index saved to: {output_dir}")
    print("Files created:")
    print("  - semantic.faiss")
    print("  - semantic_ids.npy")
    print("  - semantic_meta.json")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build baseline-clean semantic search index (no description)"
    )

    parser.add_argument(
        "--output", default="models/data/baseline_clean", help="Output directory for index files"
    )

    parser.add_argument(
        "--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name"
    )

    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multi-process encoding (faster on multi-core CPU, 2-3x speedup)",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes for multi-process mode (default: 4)",
    )

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of books (for testing)"
    )

    args = parser.parse_args()

    # Load embedding model
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {args.embedder}...")
    model = SentenceTransformer(args.embedder)
    print("✅ Model loaded\n")

    # Create embedder function for single-process mode
    def embedder(texts, **kwargs):
        return model.encode(texts, **kwargs) if not args.multiprocess else None

    # Build index
    main(
        output_dir=args.output,
        embedder=embedder,
        use_multiprocess=args.multiprocess,
        model=model,
        num_processes=args.num_processes,
        limit=args.limit,
    )
