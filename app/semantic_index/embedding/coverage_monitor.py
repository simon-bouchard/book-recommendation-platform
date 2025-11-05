# Location: app/semantic_index/embedding/coverage_monitor.py
# Tracks embedding progress and coverage statistics

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class CoverageMonitor:
    """
    Tracks embedding coverage and progress in metadata.json.
    
    Metadata includes:
    - Total items to embed
    - Items embedded so far
    - Coverage percentage
    - Last run timestamp
    - Run statistics (items/hour, errors, etc.)
    """
    
    def __init__(self, output_dir: str = "models/data/enriched_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.json"
        
        self.metadata = self._load_or_init()
    
    def _load_or_init(self) -> Dict:
        """Load existing metadata or initialize new"""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        else:
            return {
                "tags_version": "v2",
                "embed_version": "emb_v1",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "total_items": 0,
                "embedded_items": 0,
                "coverage_percent": 0.0,
                "created_at": datetime.now().isoformat(),
                "last_updated": None,
                "runs": [],
                "errors": []
            }
    
    def _save(self):
        """Persist metadata to disk"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def initialize_run(self, total_items: int, tags_version: str = "v2"):
        """Start tracking a new embedding run"""
        self.metadata["total_items"] = total_items
        self.metadata["tags_version"] = tags_version
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save()
    
    def update_progress(
        self,
        items_embedded: int,
        run_id: Optional[str] = None,
        elapsed_seconds: Optional[float] = None
    ):
        """Update progress after embedding a batch"""
        self.metadata["embedded_items"] += items_embedded
        
        if self.metadata["total_items"] > 0:
            self.metadata["coverage_percent"] = (
                self.metadata["embedded_items"] / self.metadata["total_items"]
            ) * 100
        
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Record run stats
        if run_id and elapsed_seconds:
            items_per_hour = (items_embedded / elapsed_seconds) * 3600 if elapsed_seconds > 0 else 0
            
            run_record = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "items_embedded": items_embedded,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "items_per_hour": round(items_per_hour, 2)
            }
            
            self.metadata["runs"].append(run_record)
            
            # Keep only last 100 runs
            if len(self.metadata["runs"]) > 100:
                self.metadata["runs"] = self.metadata["runs"][-100:]
        
        self._save()
    
    def record_error(self, error_msg: str, item_idx: Optional[int] = None):
        """Log an error during embedding"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "item_idx": item_idx,
            "error": error_msg
        }
        
        self.metadata["errors"].append(error_record)
        
        # Keep only last 50 errors
        if len(self.metadata["errors"]) > 50:
            self.metadata["errors"] = self.metadata["errors"][-50:]
        
        self._save()
    
    def get_stats(self) -> Dict:
        """Get current coverage statistics"""
        return {
            "total_items": self.metadata["total_items"],
            "embedded_items": self.metadata["embedded_items"],
            "remaining_items": self.metadata["total_items"] - self.metadata["embedded_items"],
            "coverage_percent": round(self.metadata["coverage_percent"], 2),
            "last_updated": self.metadata["last_updated"],
            "total_runs": len(self.metadata["runs"]),
            "total_errors": len(self.metadata["errors"])
        }
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("EMBEDDING COVERAGE STATISTICS")
        print("="*50)
        print(f"Total Items:      {stats['total_items']:,}")
        print(f"Embedded:         {stats['embedded_items']:,}")
        print(f"Remaining:        {stats['remaining_items']:,}")
        print(f"Coverage:         {stats['coverage_percent']:.2f}%")
        print(f"Last Updated:     {stats['last_updated']}")
        print(f"Total Runs:       {stats['total_runs']}")
        print(f"Errors:           {stats['total_errors']}")
        print("="*50 + "\n")
    
    def is_complete(self, threshold: float = 95.0) -> bool:
        """Check if embedding is complete (>= threshold% coverage)"""
        return self.metadata["coverage_percent"] >= threshold
