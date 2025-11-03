"""
Build FAISS index from baseline enrichment (no LLM).
Simpler text construction than LLM-enriched version.
Optimized for CPU execution with multi-process support.
"""
from pathlib import Path
import json
import numpy as np
import faiss
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.semantic_index.index_store import IndexStore


def yield_baseline_texts(baseline_jsonl: Path):
    """
    Generator that yields (item_idx, text_for_embedding, metadata_dict).
    
    Baseline text format (simpler):
    - "{title} — {author} | {description} | subjects: {subjects}"
    
    No: tones, vibe, genre (LLM-only fields)
    """
    print(f"Reading baseline enrichment from: {baseline_jsonl}")
    
    count = 0
    skipped = 0
    
    with open(baseline_jsonl, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: JSON parse error: {e}")
                skipped += 1
                continue
            
            # Validate baseline format
            if obj.get("tags_version") != "baseline":
                print(f"⚠️  Line {line_num}: Not baseline (tags_version={obj.get('tags_version')})")
                skipped += 1
                continue
            
            item_idx = obj.get("book_id")
            if not isinstance(item_idx, int):
                print(f"⚠️  Line {line_num}: Invalid book_id type")
                skipped += 1
                continue
            
            # Extract fields
            title = (obj.get("title") or "").strip()
            author = (obj.get("author") or "").strip()
            description = (obj.get("description") or "").strip()
            subjects = obj.get("subjects", [])
            
            # Construct simple baseline text
            text_parts = []
            
            if title and author:
                text_parts.append(f"{title} — {author}")
            elif title:
                text_parts.append(title)
            
            if description:
                # Truncate very long descriptions
                desc = description[:500] if len(description) > 500 else description
                text_parts.append(desc)
            
            if subjects:
                subjects_str = ", ".join(subjects[:10])  # Limit to 10 subjects
                text_parts.append(f"subjects: {subjects_str}")
            
            if not text_parts:
                skipped += 1
                continue
            
            text = " | ".join(text_parts)
            
            meta = {
                "title": title,
                "author": author,
                "has_description": bool(description),
                "num_subjects": len(subjects),
                "tags_version": "baseline"
            }
            
            count += 1
            yield item_idx, text, meta
    
    print(f"✅ Processed {count:,} books ({skipped:,} skipped)")


def embed_texts(texts, embedder=None, use_multiprocess=False, model_name=None, num_processes=4):
    """
    Embed all book texts into dense vectors.
    Optimized for CPU execution.
    
    Args:
        texts: Generator of (item_idx, text, meta) tuples
        embedder: Embedding function (for single-process mode)
        use_multiprocess: Use multi-process encoding (faster on multi-core CPU)
        model_name: Required if use_multiprocess=True
        num_processes: Number of processes (default: 4 for 6 vCPUs)
    
    Returns: (embeddings array, ids array[int64], metadata list)
    """
    # Collect all texts
    all_texts = []
    ids = []
    metas = []
    
    print("Collecting texts...")
    for item_idx, txt, meta in texts:
        ids.append(item_idx)
        metas.append(meta)
        all_texts.append(txt)
    
    print(f"✅ Collected {len(all_texts):,} texts")
    
    if use_multiprocess:
        print(f"🚀 Using multi-process encoding ({num_processes} processes)...")
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name)
        pool = model.start_multi_process_pool(target_devices=['cpu'] * num_processes)
        
        embeddings = model.encode_multi_process(
            all_texts,
            pool,
            batch_size=32,  # Per process
            chunk_size=1000,
            show_progress_bar=True
        )
        
        model.stop_multi_process_pool(pool)
        embeddings = np.array(embeddings)
    else:
        print("Embedding texts (single process, CPU-optimized batching)...")
        embeddings = embedder(
            all_texts,
            batch_size=32,  # Smaller for CPU
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    ids_array = np.array(ids, dtype=np.int64)
    
    print(f"✅ Created {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})")
    
    return embeddings, ids_array, metas


def build_faiss(embeds: np.ndarray):
    """
    Build a FAISS HNSW index over the embedding vectors.
    """
    print("Building FAISS HNSW index...")
    
    d = embeds.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 80
    index.add(embeds.astype("float32"))
    
    print(f"✅ FAISS index built ({index.ntotal:,} vectors)")
    
    return index


def main(baseline_jsonl: str, out_dir: str, embedder=None, 
         use_multiprocess=False, model_name=None, num_processes=4):
    """
    Build baseline semantic search index.
    
    Args:
        baseline_jsonl: Path to baseline enrichment JSONL
        out_dir: Output directory for index files
        embedder: Embedding function (for single-process mode)
        use_multiprocess: Use multi-process encoding
        model_name: Model name (for multi-process mode)
        num_processes: Number of processes
    
    Steps:
      1. Read baseline enrichment JSONL
      2. Construct simple text inputs (no LLM fields)
      3. Embed texts (single or multi-process)
      4. Build FAISS index and save
    """
    print(f"\n{'='*80}")
    print("BASELINE INDEX BUILDER")
    print(f"{'='*80}")
    print(f"Input: {baseline_jsonl}")
    print(f"Output: {out_dir}")
    if use_multiprocess:
        print(f"Mode: Multi-process ({num_processes} workers)")
        print(f"Model: {model_name}")
    else:
        print(f"Mode: Single-process")
    print(f"{'='*80}\n")
    
    texts = yield_baseline_texts(Path(baseline_jsonl))
    
    # Convert generator to list for embedding
    text_list = list(texts)
    
    if not text_list:
        raise RuntimeError("No valid baseline records found.")
    
    embeds, ids, metas = embed_texts(
        text_list, 
        embedder=embedder,
        use_multiprocess=use_multiprocess,
        model_name=model_name,
        num_processes=num_processes
    )
    
    index = build_faiss(embeds)
    
    out_path = Path(out_dir)
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
    print(f"✅ Baseline index saved to: {out_dir}")
    print(f"Files created:")
    print(f"  - {out_path / 'semantic.faiss'}")
    print(f"  - {out_path / 'semantic_ids.npy'}")
    print(f"  - {out_path / 'semantic_meta.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build baseline FAISS index")
    parser.add_argument(
        "--input",
        default="data/baseline_enrichment.jsonl",
        help="Baseline enrichment JSONL"
    )
    parser.add_argument(
        "--output",
        default="models/data/baseline",
        help="Output directory for index"
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multi-process encoding (faster on multi-core CPU)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes for multi-process mode (default: 4)"
    )
    
    args = parser.parse_args()
    
    if args.multiprocess:
        # Multi-process mode - pass model name
        main(
            baseline_jsonl=args.input,
            out_dir=args.output,
            embedder=None,
            use_multiprocess=True,
            model_name=args.embedder,
            num_processes=args.num_processes
        )
    else:
        # Single-process mode - load model and create embedder function
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedder)
        embedder = lambda texts, **kwargs: model.encode(texts, **kwargs)
        
        main(
            baseline_jsonl=args.input,
            out_dir=args.output,
            embedder=embedder,
            use_multiprocess=False
        )
