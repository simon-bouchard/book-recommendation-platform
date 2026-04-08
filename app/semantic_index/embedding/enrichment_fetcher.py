# Location: app/semantic_index/embedding/enrichment_fetcher.py
# Fetches enriched books from MySQL with proper joins

from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


class EnrichmentFetcher:
    """
    Queries MySQL to fetch enriched books with all metadata.

    Performs multi-table joins to get:
    - Book title, author
    - LLM subjects
    - Tone IDs
    - Genre slug
    - Vibe text

    All filtered by tags_version (e.g., 'v2').
    """

    def __init__(self, db_session: Session, tags_version: str = "v2"):
        self.db = db_session
        self.tags_version = tags_version

    def fetch_enriched_items(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        quality_tiers: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch enriched books with all metadata.

        Args:
            limit: Max items to fetch (None = all)
            offset: Skip first N items
            quality_tiers: NOT USED (kept for API compatibility, enrichment_quality not in DB)

        Returns:
            List of dicts with: item_idx, title, author, subjects, tone_ids, genre, vibe, tags_version
        """

        # Note: quality_tiers parameter kept for API compatibility but not used
        # enrichment_quality is in Kafka metadata, not stored as DB column
        params = {"tags_version": self.tags_version}

        # Complex query with multiple joins
        query = text("""
            WITH enriched_books AS (
                -- Get books that have enrichment data
                SELECT DISTINCT b.item_idx
                FROM books b
                WHERE EXISTS (
                    SELECT 1 FROM book_genres bg
                    WHERE bg.item_idx = b.item_idx
                    AND bg.tags_version = :tags_version
                )
            ),
            book_metadata AS (
                -- Get title and author
                SELECT
                    eb.item_idx,
                    b.title,
                    COALESCE(a.name, '') AS author
                FROM enriched_books eb
                JOIN books b ON eb.item_idx = b.item_idx
                LEFT JOIN authors a ON b.author_idx = a.author_idx
            ),
            subjects_agg AS (
                -- Aggregate LLM subjects
                SELECT
                    bls.item_idx,
                    GROUP_CONCAT(ls.subject ORDER BY ls.subject SEPARATOR '|') AS subjects
                FROM book_llm_subjects bls
                JOIN llm_subjects ls ON bls.llm_subject_idx = ls.llm_subject_idx
                WHERE bls.tags_version = :tags_version
                GROUP BY bls.item_idx
            ),
            tones_agg AS (
                -- Aggregate tone IDs
                SELECT
                    bt.item_idx,
                    GROUP_CONCAT(bt.tone_id ORDER BY bt.tone_id SEPARATOR ',') AS tone_ids
                FROM book_tones bt
                WHERE bt.tags_version = :tags_version
                GROUP BY bt.item_idx
            ),
            genres_data AS (
                -- Get genre slug
                SELECT
                    bg.item_idx,
                    bg.genre_slug
                FROM book_genres bg
                WHERE bg.tags_version = :tags_version
            ),
            vibes_data AS (
                -- Get vibe text
                SELECT
                    bv.item_idx,
                    v.text AS vibe
                FROM book_vibes bv
                JOIN vibes v ON bv.vibe_id = v.vibe_id
                WHERE bv.tags_version = :tags_version
            )
            -- Combine all data
            SELECT
                bm.item_idx,
                bm.title,
                bm.author,
                COALESCE(sa.subjects, '') AS subjects,
                COALESCE(ta.tone_ids, '') AS tone_ids,
                gd.genre_slug,
                COALESCE(vd.vibe, '') AS vibe,
                :tags_version AS tags_version
            FROM book_metadata bm
            LEFT JOIN subjects_agg sa ON bm.item_idx = sa.item_idx
            LEFT JOIN tones_agg ta ON bm.item_idx = ta.item_idx
            LEFT JOIN genres_data gd ON bm.item_idx = gd.item_idx
            LEFT JOIN vibes_data vd ON bm.item_idx = vd.item_idx
            WHERE gd.genre_slug IS NOT NULL  -- Ensure has genre (required field)
            ORDER BY bm.item_idx
            LIMIT :limit OFFSET :offset
        """)

        params["limit"] = limit if limit is not None else 999999999
        params["offset"] = offset

        result = self.db.execute(query, params)

        # Parse results into structured dicts
        items = []
        for row in result:
            item = {
                "item_idx": row.item_idx,
                "title": row.title or "",
                "author": row.author or "",
                "subjects": [s.strip() for s in row.subjects.split("|") if s.strip()]
                if row.subjects
                else [],
                "tone_ids": [int(t) for t in row.tone_ids.split(",") if t.strip()]
                if row.tone_ids
                else [],
                "genre": row.genre_slug or "",
                "vibe": row.vibe or "",
                "tags_version": row.tags_version,
            }
            items.append(item)

        return items

    def count_enriched_items(self, quality_tiers: Optional[List[str]] = None) -> int:
        """
        Count total enriched items for tags_version.

        Args:
            quality_tiers: NOT USED (kept for API compatibility)
        """

        # Note: quality_tiers not used - enrichment_quality not stored in DB
        params = {"tags_version": self.tags_version}

        query = text("""
            SELECT COUNT(DISTINCT bg.item_idx)
            FROM book_genres bg
            WHERE bg.tags_version = :tags_version
        """)

        result = self.db.execute(query, params)
        return result.scalar()  # Location: app/semantic_index/embedding/enrichment_fetcher.py


# Fetches enriched books from MySQL with proper joins

from typing import Dict, List, Optional

from sqlalchemy.orm import Session


class EnrichmentFetcher:
    """
    Queries MySQL to fetch enriched books with all metadata.

    Performs multi-table joins to get:
    - Book title, author
    - LLM subjects
    - Tone IDs
    - Genre slug
    - Vibe text

    All filtered by tags_version (e.g., 'v2').
    """

    def __init__(self, db_session: Session, tags_version: str = "v2"):
        self.db = db_session
        self.tags_version = tags_version

    def fetch_enriched_items(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        quality_tiers: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch enriched books with all metadata.

        Args:
            limit: Max items to fetch (None = all)
            offset: Skip first N items
            quality_tiers: Filter by enrichment quality (e.g., ['RICH', 'SPARSE'])

        Returns:
            List of dicts with: item_idx, title, author, subjects, tone_ids, genre, vibe, tags_version
        """

        # Build quality tier filter
        quality_filter = ""
        params = {"tags_version": self.tags_version}

        if quality_tiers:
            placeholders = ", ".join([f":tier{i}" for i in range(len(quality_tiers))])
            quality_filter = f"AND enrichment_quality IN ({placeholders})"
            for i, tier in enumerate(quality_tiers):
                params[f"tier{i}"] = tier

        # Complex query with multiple joins
        query = text(f"""
            WITH enriched_books AS (
                -- Get books that have enrichment data
                SELECT DISTINCT b.item_idx
                FROM books b
                WHERE EXISTS (
                    SELECT 1 FROM book_genres bg
                    WHERE bg.item_idx = b.item_idx
                    AND bg.tags_version = :tags_version
                )
            ),
            book_metadata AS (
                -- Get title and author
                SELECT
                    eb.item_idx,
                    b.title,
                    COALESCE(a.name, '') AS author
                FROM enriched_books eb
                JOIN books b ON eb.item_idx = b.item_idx
                LEFT JOIN authors a ON b.author_idx = a.author_idx
            ),
            subjects_agg AS (
                -- Aggregate LLM subjects
                SELECT
                    bls.item_idx,
                    GROUP_CONCAT(ls.subject ORDER BY ls.subject SEPARATOR '|') AS subjects
                FROM book_llm_subjects bls
                JOIN llm_subjects ls ON bls.llm_subject_idx = ls.llm_subject_idx
                WHERE bls.tags_version = :tags_version
                GROUP BY bls.item_idx
            ),
            tones_agg AS (
                -- Aggregate tone IDs
                SELECT
                    bt.item_idx,
                    GROUP_CONCAT(bt.tone_id ORDER BY bt.tone_id SEPARATOR ',') AS tone_ids
                FROM book_tones bt
                WHERE bt.tags_version = :tags_version
                GROUP BY bt.item_idx
            ),
            genres_data AS (
                -- Get genre slug
                SELECT
                    bg.item_idx,
                    bg.genre_slug
                FROM book_genres bg
                WHERE bg.tags_version = :tags_version
            ),
            vibes_data AS (
                -- Get vibe text
                SELECT
                    bv.item_idx,
                    v.text AS vibe
                FROM book_vibes bv
                JOIN vibes v ON bv.vibe_id = v.vibe_id
                WHERE bv.tags_version = :tags_version
            )
            -- Combine all data
            SELECT
                bm.item_idx,
                bm.title,
                bm.author,
                COALESCE(sa.subjects, '') AS subjects,
                COALESCE(ta.tone_ids, '') AS tone_ids,
                gd.genre_slug,
                COALESCE(vd.vibe, '') AS vibe,
                :tags_version AS tags_version
            FROM book_metadata bm
            LEFT JOIN subjects_agg sa ON bm.item_idx = sa.item_idx
            LEFT JOIN tones_agg ta ON bm.item_idx = ta.item_idx
            LEFT JOIN genres_data gd ON bm.item_idx = gd.item_idx
            LEFT JOIN vibes_data vd ON bm.item_idx = vd.item_idx
            WHERE gd.genre_slug IS NOT NULL  -- Ensure has genre (required field)
            {quality_filter}
            ORDER BY bm.item_idx
            LIMIT :limit OFFSET :offset
        """)

        params["limit"] = limit if limit is not None else 999999999
        params["offset"] = offset

        result = self.db.execute(query, params)

        # Parse results into structured dicts
        items = []
        for row in result:
            item = {
                "item_idx": row.item_idx,
                "title": row.title or "",
                "author": row.author or "",
                "subjects": [s.strip() for s in row.subjects.split("|") if s.strip()]
                if row.subjects
                else [],
                "tone_ids": [int(t) for t in row.tone_ids.split(",") if t.strip()]
                if row.tone_ids
                else [],
                "genre": row.genre_slug or "",
                "vibe": row.vibe or "",
                "tags_version": row.tags_version,
            }
            items.append(item)

        return items

    def count_enriched_items(self, quality_tiers: Optional[List[str]] = None) -> int:
        """Count total enriched items for tags_version"""

        quality_filter = ""
        params = {"tags_version": self.tags_version}

        if quality_tiers:
            placeholders = ", ".join([f":tier{i}" for i in range(len(quality_tiers))])
            quality_filter = f"AND enrichment_quality IN ({placeholders})"
            for i, tier in enumerate(quality_tiers):
                params[f"tier{i}"] = tier

        query = text(f"""
            SELECT COUNT(DISTINCT bg.item_idx)
            FROM book_genres bg
            WHERE bg.tags_version = :tags_version
            {quality_filter}
        """)

        result = self.db.execute(query, params)
        return result.scalar()
