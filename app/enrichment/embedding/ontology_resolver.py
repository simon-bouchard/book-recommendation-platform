# Location: app/semantic_index/embedding/ontology_resolver.py
# Loads tone and genre ontology mappings from CSV files

import csv
from pathlib import Path
from typing import Dict


class OntologyResolver:
    """
    Loads ontology CSVs and provides ID<->name mappings.
    Needed to resolve tone_ids and genre_slugs to human-readable names for embedding text.
    """
    
    def __init__(self, ontology_dir: str = "ontology"):
        self.ontology_dir = Path(ontology_dir)
        self.tone_id_to_name: Dict[int, str] = {}
        self.genre_slug_to_name: Dict[str, str] = {}
        
        self._load_tones()
        self._load_genres()
    
    def _load_tones(self):
        """Load tones_v2.csv: tone_id,slug,name"""
        tones_path = self.ontology_dir / "tones_v2.csv"
        
        if not tones_path.exists():
            raise FileNotFoundError(f"Tones CSV not found: {tones_path}")
        
        with open(tones_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tone_id = int(row["tone_id"])
                name = row.get("name") or row.get("slug")  # fallback to slug if name missing
                self.tone_id_to_name[tone_id] = name
        
        print(f"Loaded {len(self.tone_id_to_name)} tones from {tones_path}")
    
    def _load_genres(self):
        """Load genres_v1.csv: slug,name"""
        genres_path = self.ontology_dir / "genres_v1.csv"
        
        if not genres_path.exists():
            raise FileNotFoundError(f"Genres CSV not found: {genres_path}")
        
        with open(genres_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slug = row["slug"]
                name = row.get("name") or slug  # fallback to slug if name missing
                self.genre_slug_to_name[slug] = name
        
        print(f"Loaded {len(self.genre_slug_to_name)} genres from {genres_path}")
    
    def resolve_tones(self, tone_ids: list[int]) -> list[str]:
        """Convert tone_ids to human-readable names"""
        return [self.tone_id_to_name.get(tid, f"tone_{tid}") for tid in tone_ids]
    
    def resolve_genre(self, genre_slug: str) -> str:
        """Convert genre slug to human-readable name"""
        return self.genre_slug_to_name.get(genre_slug, genre_slug)
