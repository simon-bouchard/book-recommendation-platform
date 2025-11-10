#!/usr/bin/env python3
# app/semantic_index/embedding/consolidate_and_build_index.py
"""
Consolidate NPZ batch files and build FAISS index.

Reads all batch_*.npz files from accumulator directory, consolidates them,
and builds a FAISS HNSW index for semantic search.
"""
from pathlib import Path
import sys
import json
import numpy as np
import faiss
from typing import Tuple, List, Dict, Any
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_npz_batches(accumulator_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load and consolidate all NPZ batch files from accumulator directory.
    
    Returns:
        (embeddings, item_indices, metadata_list) tuple
    """
    print(f"\nLoading NPZ batches from: {accumulator_dir}")
    
    batch_files = sorted(accumulator_dir.glob("batch_*.npz"))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {accumulator_dir}")
    
    print(f"Found {len(batch_files)} batch files")
    
    all_embeddings = []
    all_item_indices = []
    all_metadata = []
    
    for i, batch_file in enumerate(batch_files, 1):
        print(f"  Loading {batch_file.name} ({i}/{len(batch_files)})...")
        
        data = np.load(batch_file, allow_pickle=True)
        
        embeddings = data['embeddings']
        item_indices = data['item_indices']
        metadata_json = data['metadata'].item() if 'metadata' in data else '{}'
        
        all_embeddings.append(embeddings)
        all_item_indices.append(item_indices)
        
        # Parse metadata JSON
        try:
            metadata_items = json.loads(metadata_json)
            all_metadata.extend(metadata_items)
        except:
            # If metadata is not a list, create placeholder
            for idx in item_indices:
                all_metadata.append({"item_idx": int(idx)})
    
    # Concatenate all batches
    consolidated_embeddings = np.vstack(all_embeddings)
    consolidated_indices = np.concatenate(all_item_indices)
    
    print(f"\n✅ Consolidated {len(batch_files)} batches")
    print(f"   Total embeddings: {len(consolidated_embeddings):,}")
    print(f"   Embedding dimension: {consolidated_embeddings.shape[1]}")
    
    return consolidated_embeddings, consolidated_indices, all_metadata


def remove_duplicates(
    embeddings: np.ndarray, 
    item_indices: np.ndarray, 
    metadata: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Remove duplicate item_idx entries, keeping the latest (last occurrence).
    
    Returns:
        (deduplicated_embeddings, deduplicated_indices, deduplicated_metadata)
    """
    print("\nRemoving duplicates...")
    
    original_count = len(item_indices)
    
    # Find unique indices (keep last occurrence)
    _, unique_idx = np.unique(item_indices[::-1], return_index=True)
    unique_idx = len(item_indices) - 1 - unique_idx  # Convert back to forward indices
    unique_idx = np.sort(unique_idx)
    
    dedup_embeddings = embeddings[unique_idx]
    dedup_indices = item_indices[unique_idx]
    dedup_metadata = [metadata[i] for i in unique_idx]
    
    duplicates_removed = original_count - len(dedup_indices)
    
    print(f"✅ Removed {duplicates_removed:,} duplicates")
    print(f"   Unique items: {len(dedup_indices):,}")
    
    return dedup_embeddings, dedup_indices, dedup_metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    """Build FAISS HNSW index from embeddings."""
    print("\nBuilding FAISS HNSW index...")
    
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)  # 32 = M parameter (neighbors per layer)
    index.hnsw.efConstruction = 80  # Construction-time search depth
    
    print(f"  Dimension: {d}")
    print(f"  Adding {len(embeddings):,} vectors...")
    
    index.add(embeddings.astype("float32"))
    
    print(f"✅ FAISS index built ({index.ntotal:,} vectors)")
    
    return index


def save_index(
    output_dir: Path,
    index: faiss.IndexHNSWFlat,
    item_indices: np.ndarray,
    metadata: List[Dict],
    tags_version: str,
    embed_version: str
):
    """Save FAISS index and metadata to disk."""
    print(f"\nSaving index to: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss_path = output_dir / "semantic.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"✅ Saved FAISS index: {faiss_path}")
    
    # Save item indices
    ids_path = output_dir / "semantic_ids.npy"
    np.save(ids_path, item_indices)
    print(f"✅ Saved item indices: {ids_path}")
    
    # Save metadata
    meta_path = output_dir / "semantic_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved metadata: {meta_path}")
    
    # Create info file
    info = {
        "tags_version": tags_version,
        "embed_version": embed_version,
        "embedding_dim": index.d,
        "total_items": int(index.ntotal),
        "index_type": "HNSW",
        "hnsw_m": 32,
        "hnsw_ef_construction": 80,
        "created_at": datetime.now().isoformat(),
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    info_path = output_dir / "index_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"✅ Saved index info: {info_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidate NPZ batches and build FAISS index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build v2 full index
  python -m app.semantic_index.embedding.consolidate_and_build_index \\
    --accumulator-dir models/data/enriched_v2/accumulator \\
    --output-dir models/data/enriched_v2 \\
    --tags-version v2
  
  # Build v2 subjects index
  python -m app.semantic_index.embedding.consolidate_and_build_index \\
    --accumulator-dir models/data/enriched_v2_subjects/accumulator \\
    --output-dir models/data/enriched_v2_subjects \\
    --tags-version v2 \\
    --embed-version subjects_v1
"""
    )
    
    parser.add_argument(
        "--accumulator-dir",
        required=True,
        help="Directory containing batch_*.npz files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for FAISS index"
    )
    parser.add_argument(
        "--tags-version",
        default="v2",
        help="Tags version (for metadata)"
    )
    parser.add_argument(
        "--embed-version",
        default="full_v1",
        help="Embedding version (for metadata)"
    )
    
    args = parser.parse_args()
    
    accumulator_dir = Path(args.accumulator_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("CONSOLIDATE NPZ BATCHES AND BUILD FAISS INDEX")
    print("=" * 80)
    print(f"Accumulator dir: {accumulator_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Tags version: {args.tags_version}")
    print(f"Embed version: {args.embed_version}")
    print("=" * 80)
    
    # Validate accumulator directory
    if not accumulator_dir.exists():
        print(f"❌ Accumulator directory not found: {accumulator_dir}")
        return 1
    
    # Load and consolidate batches
    try:
        embeddings, item_indices, metadata = load_npz_batches(accumulator_dir)
    except Exception as e:
        print(f"❌ Failed to load NPZ batches: {e}")
        return 1
    
    # Remove duplicates (keep latest)
    embeddings, item_indices, metadata = remove_duplicates(embeddings, item_indices, metadata)
    
    # Build FAISS index
    try:
        index = build_faiss_index(embeddings)
    except Exception as e:
        print(f"❌ Failed to build FAISS index: {e}")
        return 1
    
    # Save index
    try:
        save_index(
            output_dir, 
            index, 
            item_indices, 
            metadata,
            args.tags_version,
            args.embed_version
        )
    except Exception as e:
        print(f"❌ Failed to save index: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ INDEX BUILD COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  - {output_dir}/semantic.faiss")
    print(f"  - {output_dir}/semantic_ids.npy")
    print(f"  - {output_dir}/semantic_meta.json")
    print(f"  - {output_dir}/index_info.json")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
