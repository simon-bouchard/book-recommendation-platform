# models/domain/recommendation.py
"""
Domain models for recommendation results - candidates and enriched book recommendations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Candidate:
    """
    Candidate book from a recommendation generator.

    Represents a book that has been identified as potentially relevant,
    before filtering and final ranking.

    Attributes:
        item_idx: Book item identifier
        score: Relevance score from the generator
        source: Name of the generator that produced this candidate
                (e.g., "als", "subject_similarity", "popularity")
    """

    item_idx: int
    score: float
    source: str

    def __post_init__(self):
        """Validate candidate data."""
        if self.score < 0:
            raise ValueError(f"Score must be non-negative, got {self.score}")
        if not self.source:
            raise ValueError("Source must be non-empty")


@dataclass
class RecommendedBook:
    """
    Final recommendation with complete book metadata.

    Represents a book that has been filtered, ranked, and enriched
    with metadata for presentation to the user.

    Attributes:
        item_idx: Book item identifier
        title: Book title
        score: Final recommendation score
        num_ratings: Number of user ratings
        author: Author name (optional)
        year: Publication year (optional)
        isbn: ISBN identifier (optional)
        cover_id: OpenLibrary cover ID (optional)
        avg_rating: Average user rating (optional)
    """

    item_idx: int
    title: str
    score: float
    num_ratings: int
    author: Optional[str] = None
    year: Optional[int] = None
    isbn: Optional[str] = None
    cover_id: Optional[str] = None
    avg_rating: Optional[float] = None

    def __post_init__(self):
        """Validate recommended book data."""
        if not self.title:
            raise ValueError("Title must be non-empty")
        if self.score < 0:
            raise ValueError(f"Score must be non-negative, got {self.score}")
        if self.num_ratings < 0:
            raise ValueError(f"num_ratings must be non-negative, got {self.num_ratings}")
        if self.avg_rating is not None and not (0 <= self.avg_rating <= 10):
            raise ValueError(f"avg_rating must be in [0, 10], got {self.avg_rating}")

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, omitting None values for optional fields.
        """
        result = {
            "item_idx": self.item_idx,
            "title": self.title,
            "score": self.score,
            "num_ratings": self.num_ratings,
        }

        if self.author is not None:
            result["author"] = self.author
        if self.year is not None:
            result["year"] = self.year
        if self.isbn is not None:
            result["isbn"] = self.isbn
        if self.cover_id is not None:
            result["cover_id"] = self.cover_id
        if self.avg_rating is not None:
            result["avg_rating"] = self.avg_rating

        return result
