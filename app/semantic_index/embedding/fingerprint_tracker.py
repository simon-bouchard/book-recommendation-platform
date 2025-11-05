# Location: app/semantic_index/embedding/fingerprint_tracker.py
# SQLite-based fingerprint tracking to avoid re-embedding unchanged items

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Optional, Set


class FingerprintTracker:
    """
    Tracks fingerprints of embedded items to avoid redundant work.
    
    Fingerprint = hash of (item_idx, title, author, subjects, tone_ids, genre, vibe, tags_version)
    Only re-embed if fingerprint changed or missing.
    """
    
    def __init__(self, db_path: str = "models/data/enriched_v2/fingerprints.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create fingerprints table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                item_idx INTEGER PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                tags_version TEXT NOT NULL,
                embedded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index separately (SQLite syntax)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_version 
            ON fingerprints(tags_version)
        """)
        
        conn.commit()
        conn.close()
    
    def compute_fingerprint(self, item_data: dict) -> str:
        """
        Compute deterministic hash from enrichment data.
        Changes in any field will trigger re-embedding.
        """
        # Normalize data for consistent hashing
        normalized = {
            "item_idx": item_data["item_idx"],
            "title": (item_data.get("title") or "").strip(),
            "author": (item_data.get("author") or "").strip(),
            "subjects": sorted(item_data.get("subjects", [])),  # sort for consistency
            "tone_ids": sorted(item_data.get("tone_ids", [])),
            "genre": item_data.get("genre", ""),
            "vibe": (item_data.get("vibe") or "").strip(),
            "tags_version": item_data.get("tags_version", "v2")
        }
        
        # Hash the normalized JSON
        content = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_existing_fingerprints(self, item_indices: list[int], tags_version: str) -> dict[int, str]:
        """
        Fetch existing fingerprints for given items.
        Returns: {item_idx: fingerprint}
        """
        if not item_indices:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ",".join("?" * len(item_indices))
        cursor.execute(f"""
            SELECT item_idx, fingerprint
            FROM fingerprints
            WHERE item_idx IN ({placeholders})
            AND tags_version = ?
        """, (*item_indices, tags_version))
        
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result
    
    def update_fingerprints(self, items: list[dict], tags_version: str):
        """
        Upsert fingerprints for newly embedded items.
        Called after successful embedding batch.
        """
        if not items:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        records = [
            (item["item_idx"], self.compute_fingerprint(item), tags_version)
            for item in items
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO fingerprints (item_idx, fingerprint, tags_version)
            VALUES (?, ?, ?)
        """, records)
        
        conn.commit()
        conn.close()
    
    def get_stats(self, tags_version: str) -> dict:
        """Get coverage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM fingerprints WHERE tags_version = ?
        """, (tags_version,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return {"total_embedded": count, "tags_version": tags_version}
