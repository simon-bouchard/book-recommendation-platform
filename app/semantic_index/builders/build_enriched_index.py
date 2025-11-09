# app/semantic_index/builders/build_enriched_index.py
"""
Unified builder for enriched semantic search indexes from SQL database.
Handles v1/v2, full/subjects-only variants. Reads directly from database tables.
"""

from pathlib import Path
import sys
import numpy as np
import faiss
from sqlalchemy import func
from typing import Generator, Tuple, Dict, List, Any

ROOT = Path(__file__).resolve().parents[3]  # Go up to project root
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import Book, Author, BookLLMSubject, BookTone, BookGenre, BookVibe
from app.semantic_index.index_store import IndexStore

# Import text templates from templates folder
from app.semantic_index.templates.text_templates import (
    build_v1_full_text,
    build_v1_subjects_text,
    build_v2_full_text,
    build_v2_subjects_text,
    validate_text_not_empty
)


def load_ontology_mappings(tags_version: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load ontology CSVs and create ID->name mappings.
    
    Returns:
        (tone_id_to_name, genre_id_to_name) tuple
    """
    import csv
    
    # Load tones
    tone_file = ROOT / "ontology" / f"tones_{tags_version}.csv"
    if not tone_file.exists():
        raise FileNotFoundError(f"Tone ontology not found: {tone_file}")
    
    tone_map = {}
    with open(tone_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tone_id = int(row["tone_id"])
            tone_slug = row["slug"]
            tone_map[tone_id] = tone_slug
    
    # Load genres (version-independent)
    genre_file = ROOT / "ontology" / "genres_v1.csv"
    if not genre_file.exists():
        raise FileNotFoundError(f"Genre ontology not found: {genre_file}")
    
    genre_map = {}
    with open(genre_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre_id = int(row["genre_id"])
            display_name = row["display_name"]
            genre_map[genre_id] = display_name
    
    print(f"✅ Loaded {len(tone_map)} tones and {len(genre_map)} genres")
    
    return tone_map, genre_map


def fetch_enriched_books(
    tags_version: str,
    include_metadata: bool,
    tone_map: Dict[int, str],
    genre_map: Dict[int, str],
    limit: int = None
) -> Generator[Tuple[int, str, Dict], None, None]:
    """
    Fetch enriched books from database and yield (item_idx, embedding_text, metadata).
    
    Args:
        tags_version: 'v1' or 'v2'
        include_metadata: True for full (genre/tones/vibe), False for subjects-only
        tone_map: tone_id -> tone_name mapping
        genre_map: genre_id -> genre_name mapping
        limit: Optional limit for testing
    
    Yields:
        (item_idx, embedding_text, metadata_dict) tuples
    """
    print(f"\n{'='*80}")
    print(f"Fetching books: tags_version={tags_version}, include_metadata={include_metadata}")
    print(f"{'='*80}\n")
    
    with SessionLocal() as db:
        # Base query: books with titles and authors
        query = (
            db.query(
                Book.item_idx,
                Book.title,
                Author.name.label("author")
            )
            .outerjoin(Author, Book.author_idx == Author.author_idx)
            .filter(Book.title.isnot(None))
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
            
            # Fetch LLM subjects
            subjects_query = (
                db.query(BookLLMSubject.subject)
                .filter(
                    BookLLMSubject.item_idx == item_idx,
                    BookLLMSubject.tags_version == tags_version
                )
                .all()
            )
            subjects = [s for (s,) in subjects_query if s]
            
            # Skip books without enrichment
            if not subjects:
                skipped += 1
                continue
            
            # Build embedding text based on variant
            if include_metadata:
                # Full variant: fetch genre, tones, vibe
                
                # Fetch genre
                genre_row = (
                    db.query(BookGenre.genre_id)
                    .filter(
                        BookGenre.item_idx == item_idx,
                        BookGenre.tags_version == tags_version
                    )
                    .first()
                )
                
                if not genre_row:
                    skipped += 1
                    continue  # Skip books without genre for full variant
                
                genre_id = genre_row[0]
                genre_name = genre_map.get(genre_id, f"Unknown-{genre_id}")
                
                # Fetch tones
                tone_rows = (
                    db.query(BookTone.tone_id)
                    .filter(
                        BookTone.item_idx == item_idx,
                        BookTone.tags_version == tags_version
                    )
                    .all()
                )
                tone_ids = [t for (t,) in tone_rows]
                tone_names = [tone_map.get(tid, f"unknown-{tid}") for tid in tone_ids]
                
                # Fetch vibe
                vibe_row = (
                    db.query(BookVibe.vibe)
                    .filter(
                        BookVibe.item_idx == item_idx,
                        BookVibe.tags_version == tags_version
                    )
                    .first()
                )
                vibe = vibe_row[0] if vibe_row else ""
                
                # Build full text
                if tags_version == "v1":
                    text = build_v1_full_text(
                        title=title,
                        author=author,
                        genre_name=genre_name,
                        subjects=subjects,
                        tone_names=tone_names,
                        vibe=vibe
                    )
                else:  # v2
                    text = build_v2_full_text(
                        title=title,
                        author=author,
                        genre_name=genre_name,
                        subjects=subjects,
                        tone_names=tone_names,
                        vibe=vibe
                    )
                
                meta = {
                    "title": title,
                    "author": author,
                    "genre": genre_name,
                    "tone_names": tone_names,
                    "subjects": subjects,
                    "vibe": vibe,
                    "tags_version": tags_version
                }
            
            else:
                # Subjects-only variant
                if tags_version == "v1":
                    text = build_v1_subjects_text(
                        title=title,
                        author=author,
                        subjects=subjects
                    )
                else:  # v2
                    text = build_v2_subjects_text(
                        title=title,
                        author=author,
                        subjects=subjects
                    )
                
                meta = {
                    "title": title,
                    "author": author,
                    "subjects": subjects,
                    "tags_version": tags_version
                }
            
            # Validate text not empty
            try:
                validate_text_not_empty(text, f"{tags_version} item_idx={item_idx}")
            except ValueError as e:
                print(f"⚠️  Skipping item_idx={item_idx}: {e}")
                skipped += 1
                continue
            
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
    num_processes=4
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Embed a list of (item_idx, text, meta) tuples.
    
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
        pool = model.start_multi_process_pool(target_devices=['cpu'] * num_processes)
        
        embeddings = model.encode_multi_process(
            text_strings,
            pool,
            batch_size=32,  # Per process
            chunk_size=1000,
            show_progress_bar=True
        )
        
        model.stop_multi_process_pool(pool)
        embeddings = np.array(embeddings)
    else:
        print("Embedding texts (single process)...")
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


def main(
    tags_version: str,
    include_metadata: bool,
    output_dir: str,
    embedder=None,
    use_multiprocess=False,
    model=None,
    num_processes=4,
    limit: int = None
):
    """
    Build enriched semantic search index from SQL.
    
    Args:
        tags_version: 'v1' or 'v2'
        include_metadata: True for full, False for subjects-only
        output_dir: Output directory
        embedder: Embedding function (for single-process mode)
        use_multiprocess: Use multi-process encoding (faster on multi-core CPU)
        model: SentenceTransformer model (for multi-process mode)
        num_processes: Number of processes for multiprocessing
        limit: Optional limit for testing
    """
    print(f"\n{'='*80}")
    print("ENRICHED INDEX BUILDER (SQL)")
    print(f"{'='*80}")
    print(f"Tags Version: {tags_version}")
    print(f"Variant: {'FULL (genre/tones/vibe)' if include_metadata else 'SUBJECTS-ONLY'}")
    print(f"Output: {output_dir}")
    if use_multiprocess:
        print(f"Mode: Multi-process ({num_processes} workers)")
    else:
        print(f"Mode: Single-process")
    if limit:
        print(f"Limit: {limit} books (testing mode)")
    print(f"{'='*80}\n")
    
    # Load ontologies
    tone_map, genre_map = load_ontology_mappings(tags_version)
    
    # Fetch and build texts
    texts_generator = fetch_enriched_books(
        tags_version=tags_version,
        include_metadata=include_metadata,
        tone_map=tone_map,
        genre_map=genre_map,
        limit=limit
    )
    
    # Collect all texts
    texts = list(texts_generator)
    
    if not texts:
        raise RuntimeError("No valid enriched books found")
    
    # Embed
    embeddings, ids, metas = embed_texts_batch(
        texts,
        embedder=embedder,
        use_multiprocess=use_multiprocess,
        model=model,
        num_processes=num_processes
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
        description="Build enriched semantic search index from SQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rebuild V1-Full (corrected with genre and tone names)
  python -m app.semantic_index.builders.build_enriched_index --tags-version v1 --full --output models/data/enriched_v1
  
  # Build V1-Full with multiprocessing (faster on multi-core CPU)
  python -m app.semantic_index.builders.build_enriched_index --tags-version v1 --full --multiprocess --num-processes 4 --output models/data/enriched_v1
  
  # Build V1-Subjects
  python -m app.semantic_index.builders.build_enriched_index --tags-version v1 --output models/data/enriched_v1_subjects
  
  # Build V2-Subjects (when v2 ready)
  python -m app.semantic_index.builders.build_enriched_index --tags-version v2 --output models/data/enriched_v2_subjects
  
  # Test with 100 books
  python -m app.semantic_index.builders.build_enriched_index --tags-version v1 --full --limit 100 --output test_output
        """
    )
    
    parser.add_argument(
        "--tags-version",
        required=True,
        choices=["v1", "v2"],
        help="Enrichment version (v1 or v2)"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include all metadata (genre/tones/vibe). Omit for subjects-only."
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for index files"
    )
    
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multi-process encoding (faster on multi-core CPU, 2-3x speedup)"
    )
    
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes for multi-process mode (default: 4)"
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
    print("✅ Model loaded\n")
    
    # Create embedder function for single-process mode
    embedder = lambda texts, **kwargs: model.encode(texts, **kwargs) if not args.multiprocess else None
    
    # Build index
    main(
        tags_version=args.tags_version,
        include_metadata=args.full,
        output_dir=args.output,
        embedder=embedder,
        use_multiprocess=args.multiprocess,
        model=model,
        num_processes=args.num_processes,
        limit=args.limit
    )
