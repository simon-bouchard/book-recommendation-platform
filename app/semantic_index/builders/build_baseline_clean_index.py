# app/semantic_index/builders/build_baseline_clean_index.py
"""
Build baseline-clean semantic search index (no description, subjects only).
Tests whether descriptions add noise to baseline embeddings.
"""

from pathlib import Path
import sys
import numpy as np
import faiss
from sqlalchemy import func
from typing import Generator, Tuple, Dict, List

ROOT = Path(__file__).resolve().parents[3]  # Go up to project root
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import Book, Author, OLSubject, BookOLSubject
from app.semantic_index.index_store import IndexStore

# Import text templates from templates folder
from app.semantic_index.templates.text_templates import (
    build_baseline_clean_text,
    validate_text_not_empty
)


def fetch_baseline_clean_books(
    limit: int = None
) -> Generator[Tuple[int, str, Dict], None, None]:
    """
    Fetch books with raw OL subjects (no description).
    
    Yields:
        (item_idx, embedding_text, metadata_dict) tuples
    """
    print(f"\n{'='*80}")
    print("Fetching baseline books (no description)")
    print(f"{'='*80}\n")
    
    with SessionLocal() as db:
        # Query books with titles
        query = (
            db.query(
                Book.item_idx,
                Book.title,
                Author.name.label("author")
            )
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
            
            # Skip books without OL subjects
            if not ol_subjects:
                skipped += 1
                continue
            
            # Build embedding text (no description)
            text = build_baseline_clean_text(
                title=title,
                author=author,
                ol_subjects=ol_subjects
            )
            
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
                "tags_version": "baseline_clean"
            }
            
            processed += 1
            
            if processed % 1000 == 0:
                print(f"  Processed: {processed:,}/{total:,} ({skipped:,} skipped)")
            
            yield item_idx, text, meta
        
        print(f"\n✅ Final: {processed:,} books processed, {skipped:,} skipped")


def embed_texts_batch(
    texts: List[Tuple[int, str, Dict]],
    embedder
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Embed texts in batch."""
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
    
    embeddings = embedder(
        text_strings,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
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


def main(output_dir: str, embedder, limit: int = None):
    """
    Build baseline-clean semantic search index.
    
    Args:
        output_dir: Output directory
        embedder: Embedding function
        limit: Optional limit for testing
    """
    print(f"\n{'='*80}")
    print("BASELINE-CLEAN INDEX BUILDER")
    print(f"{'='*80}")
    print(f"Variant: No description, OL subjects only")
    print(f"Output: {output_dir}")
    if limit:
        print(f"Limit: {limit} books (testing mode)")
    print(f"{'='*80}\n")
    
    # Fetch books
    texts_generator = fetch_baseline_clean_books(limit=limit)
    texts = list(texts_generator)
    
    if not texts:
        raise RuntimeError("No valid baseline books found")
    
    # Embed
    embeddings, ids, metas = embed_texts_batch(texts, embedder)
    
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
    
    print(f"\n{'='*80}")
    print(f"✅ Index saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - semantic.faiss")
    print(f"  - semantic_ids.npy")
    print(f"  - semantic_meta.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build baseline-clean semantic search index (no description)"
    )
    
    parser.add_argument(
        "--output",
        default="models/data/baseline_clean",
        help="Output directory for index files"
    )
    
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of books (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load embedding model
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {args.embedder}...")
    model = SentenceTransformer(args.embedder)
    embedder = lambda texts, **kwargs: model.encode(texts, **kwargs)
    print("✅ Model loaded\n")
    
    # Build index
    main(
        output_dir=args.output,
        embedder=embedder,
        limit=args.limit
    )
