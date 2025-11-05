# Location: app/semantic_index/embedding/worker.py
# Main orchestration loop for continuous embedding generation

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import sys
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal

from app.semantic_index.embedding.ontology_resolver import OntologyResolver
from app.semantic_index.embedding.fingerprint_tracker import FingerprintTracker
from app.semantic_index.embedding.enrichment_fetcher import EnrichmentFetcher
from app.semantic_index.embedding.embedding_client import EmbeddingClient
from app.semantic_index.embedding.accumulator_writer import AccumulatorWriter
from app.semantic_index.embedding.coverage_monitor import CoverageMonitor


class EmbeddingWorker:
    """
    Main worker for continuous embedding generation.
    
    Process:
    1. Fetch enriched items from SQL
    2. Check fingerprints (skip unchanged items)
    3. Build embedding text (resolve tones/genres)
    4. Generate embeddings in batches
    5. Write to accumulator NPZ files
    6. Update fingerprints and coverage stats
    
    Supports:
    - Incremental runs (only new/changed items)
    - Batch processing for memory efficiency
    - Progress tracking and error handling
    """
    
    def __init__(
        self,
        tags_version: str = "v2",
        batch_size: int = 128,
        output_dir: str = "models/data/enriched_v2",
        ontology_dir: str = "ontology",
        device: str = "cpu"
    ):
        self.tags_version = tags_version
        self.batch_size = batch_size
        
        print(f"Initializing EmbeddingWorker for tags_version={tags_version}")
        print(f"Output directory: {output_dir}")
        
        # Initialize components
        self.ontology = OntologyResolver(ontology_dir=ontology_dir)
        self.fingerprints = FingerprintTracker(db_path=f"{output_dir}/fingerprints.db")
        self.embedder = EmbeddingClient(batch_size=batch_size, device=device)
        self.writer = AccumulatorWriter(output_dir=output_dir)
        self.monitor = CoverageMonitor(output_dir=output_dir)
        
        # Database session (will be created per run)
        self.db: Optional[Session] = None
    
    def _identify_items_to_embed(self, items: List[dict]) -> List[dict]:
        """
        Filter items that need embedding (new or changed fingerprints).
        
        Returns: List of items that need embedding
        """
        if not items:
            return []
        
        # Get existing fingerprints
        item_indices = [item["item_idx"] for item in items]
        existing_fingerprints = self.fingerprints.get_existing_fingerprints(
            item_indices, 
            self.tags_version
        )
        
        # Identify items that need embedding
        items_to_embed = []
        
        for item in items:
            item_idx = item["item_idx"]
            current_fingerprint = self.fingerprints.compute_fingerprint(item)
            existing_fingerprint = existing_fingerprints.get(item_idx)
            
            if existing_fingerprint != current_fingerprint:
                items_to_embed.append(item)
        
        print(f"Fingerprint check: {len(items)} total, {len(items_to_embed)} need embedding")
        return items_to_embed
    
    def _build_embedding_texts(self, items: List[dict]) -> List[str]:
        """Build formatted text strings for embedding"""
        texts = []
        
        for item in items:
            # Resolve tone IDs and genre slug to names
            tone_names = self.ontology.resolve_tones(item.get("tone_ids", []))
            genre_name = self.ontology.resolve_genre(item.get("genre", ""))
            
            # Build embedding text
            text = self.embedder.build_embedding_text(item, tone_names, genre_name)
            texts.append(text)
        
        return texts
    
    def _process_batch(self, items: List[dict], run_id: str) -> int:
        """
        Process a batch of items: embed and write to disk.
        
        Returns: Number of items successfully embedded
        """
        if not items:
            return 0
        
        start_time = time.time()
        
        try:
            # Build embedding texts
            texts = self._build_embedding_texts(items)
            
            # Generate embeddings
            print(f"Encoding {len(texts)} texts...")
            embeddings = self.embedder.encode_batch(texts, show_progress=True)
            
            # Extract item indices
            item_indices = [item["item_idx"] for item in items]
            
            # Prepare metadata (keep essential fields only)
            metadata = []
            for item in items:
                meta = {
                    "item_idx": item["item_idx"],
                    "title": item["title"],
                    "author": item["author"],
                    "subjects": item["subjects"],
                    "tone_ids": item["tone_ids"],
                    "genre": item["genre"],
                    "vibe": item["vibe"]
                }
                metadata.append(meta)
            
            # Write batch to disk
            self.writer.write_batch(embeddings, item_indices, metadata)
            
            # Update fingerprints
            self.fingerprints.update_fingerprints(items, self.tags_version)
            
            # Update coverage
            elapsed = time.time() - start_time
            self.monitor.update_progress(len(items), run_id, elapsed)
            
            print(f"Batch complete: {len(items)} items in {elapsed:.2f}s")
            
            return len(items)
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            
            # Log errors for individual items
            for item in items:
                self.monitor.record_error(str(e), item["item_idx"])
            
            raise
    
    def run_full_load(self, limit: Optional[int] = None):
        """
        Run full initial load: embed all enriched items.
        
        Args:
            limit: Max items to process (None = all)
        """
        run_id = f"full_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n{'='*60}")
        print(f"FULL LOAD: {run_id}")
        print(f"{'='*60}\n")
        
        try:
            # Connect to database
            self.db = SessionLocal()
            fetcher = EnrichmentFetcher(self.db, self.tags_version)
            
            # Count total items
            total_items = fetcher.count_enriched_items()
            if limit:
                total_items = min(total_items, limit)
            
            print(f"Total enriched items: {total_items:,}")
            
            # Initialize monitoring
            self.monitor.initialize_run(total_items, self.tags_version)
            
            # Process in chunks
            offset = 0
            total_embedded = 0
            
            while offset < total_items:
                chunk_size = min(self.batch_size, total_items - offset)
                
                print(f"\nFetching items {offset:,} to {offset + chunk_size:,}...")
                
                # Fetch chunk
                items = fetcher.fetch_enriched_items(
                    limit=chunk_size,
                    offset=offset
                )
                
                if not items:
                    break
                
                # Filter by fingerprints
                items_to_embed = self._identify_items_to_embed(items)
                
                # Process batch
                if items_to_embed:
                    embedded = self._process_batch(items_to_embed, run_id)
                    total_embedded += embedded
                
                offset += len(items)
            
            print(f"\n{'='*60}")
            print(f"FULL LOAD COMPLETE")
            print(f"Total embedded: {total_embedded:,}")
            print(f"{'='*60}\n")
            
            self.monitor.print_stats()
        
        finally:
            if self.db:
                self.db.close()
    
    def run_incremental(self, limit: Optional[int] = None):
        """
        Run incremental update: only process new/changed items.
        
        Args:
            limit: Max items to check (None = all)
        """
        run_id = f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n{'='*60}")
        print(f"INCREMENTAL RUN: {run_id}")
        print(f"{'='*60}\n")
        
        try:
            # Connect to database
            self.db = SessionLocal()
            fetcher = EnrichmentFetcher(self.db, self.tags_version)
            
            # Fetch items (most recent first)
            items = fetcher.fetch_enriched_items(
                limit=limit,
                offset=0
            )
            
            print(f"Fetched {len(items):,} items to check")
            
            # Filter by fingerprints (only changed items)
            items_to_embed = self._identify_items_to_embed(items)
            
            if not items_to_embed:
                print("No items need embedding (all fingerprints match)")
                return
            
            print(f"Found {len(items_to_embed):,} items to embed")
            
            # Process in batches
            total_embedded = 0
            
            for i in range(0, len(items_to_embed), self.batch_size):
                batch = items_to_embed[i:i + self.batch_size]
                embedded = self._process_batch(batch, run_id)
                total_embedded += embedded
            
            print(f"\n{'='*60}")
            print(f"INCREMENTAL RUN COMPLETE")
            print(f"Total embedded: {total_embedded:,}")
            print(f"{'='*60}\n")
            
            self.monitor.print_stats()
        
        finally:
            if self.db:
                self.db.close()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Embedding Worker - Generate embeddings for enriched books"
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="Run mode: full load or incremental update"
    )
    
    parser.add_argument(
        "--tags-version",
        default="v2",
        help="Tags version to process (default: v2)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding (default: 128)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to process (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="models/data/enriched_v2",
        help="Output directory for embeddings"
    )
    
    parser.add_argument(
        "--ontology-dir",
        default="ontology",
        help="Directory containing ontology CSV files"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for embedding model"
    )
    
    args = parser.parse_args()
    
    # Initialize worker
    worker = EmbeddingWorker(
        tags_version=args.tags_version,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        ontology_dir=args.ontology_dir,
        device=args.device
    )
    
    # Run based on mode
    if args.mode == "full":
        worker.run_full_load(limit=args.limit)
    else:
        worker.run_incremental(limit=args.limit)


if __name__ == "__main__":
    main()
