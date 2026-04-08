# app/agents/domain/parsers.py
"""
Parsers for extracting and validating structured data from agent responses.
Used for backend validation only - does not add fields to frontend responses.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from .entities import BookRecommendation


@dataclass
class BookTagMatch:
    """A matched book tag found in response text."""

    item_idx: int
    tag_content: str  # Text inside the tag (e.g., "The Hobbit")
    full_tag: str  # Complete tag string for debugging
    start_pos: int
    end_pos: int


class InlineReferenceParser:
    """
    Extracts and validates inline book references from agent response text.

    This parser is used for backend validation only. It does NOT add fields
    to the AgentResponse - the frontend parses HTML tags directly from text.

    Pattern: <book id="12345">Book Title</book>
    """

    # Match <book id="12345">Title</book> or <book id='12345'>Title</book>
    BOOK_TAG_PATTERN = r'<book\s+id=["\'](\d+)["\']>([^<]+)</book>'

    @classmethod
    def extract_book_tags(cls, text: str) -> List[BookTagMatch]:
        """
        Extract all book tags from response text.

        Args:
            text: Response text potentially containing <book id="X">Title</book> tags

        Returns:
            List of BookTagMatch objects with extracted metadata
        """
        matches = []

        for match in re.finditer(cls.BOOK_TAG_PATTERN, text, re.IGNORECASE):
            item_idx = int(match.group(1))
            tag_content = match.group(2).strip()
            full_tag = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            matches.append(
                BookTagMatch(
                    item_idx=item_idx,
                    tag_content=tag_content,
                    full_tag=full_tag,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

        return matches

    @classmethod
    def validate_references(
        cls, text: str, book_recommendations: List[BookRecommendation]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that inline book references match book_recommendations.

        This is for backend quality control - ensures LLM didn't hallucinate
        book IDs or tag books not in the recommendation list.

        Args:
            text: Response text with book tags
            book_recommendations: List of recommended books from curation

        Returns:
            Tuple of (errors, warnings):
            - errors: Critical issues (orphaned references)
            - warnings: Non-critical issues (duplicate tags, etc.)
        """
        errors = []
        warnings = []

        # Extract all book tags
        tag_matches = cls.extract_book_tags(text)

        if not tag_matches:
            # No inline references found - this is fine (fallback to cards only)
            return errors, []

        # Build lookup for valid book IDs
        valid_book_ids = {book.item_idx for book in book_recommendations}

        # Track which books were referenced
        referenced_ids = []

        for match in tag_matches:
            referenced_ids.append(match.item_idx)

            # Check for orphaned references (critical error)
            if match.item_idx not in valid_book_ids:
                errors.append(
                    f'Orphaned book tag: <book id="{match.item_idx}">{match.tag_content}</book> '
                    f"- ID {match.item_idx} not in book_recommendations"
                )

        # Check for duplicate references (warning only)
        id_counts = {}
        for book_id in referenced_ids:
            id_counts[book_id] = id_counts.get(book_id, 0) + 1

        duplicates = {book_id: count for book_id, count in id_counts.items() if count > 1}
        if duplicates:
            warnings.append(
                f"Duplicate book tags found: {duplicates} (each book should be tagged at most once)"
            )

        # Check if too many books were tagged inline (guideline is 8-12)
        unique_referenced = len(set(referenced_ids))
        if unique_referenced > 12:
            warnings.append(
                f"Found {unique_referenced} inline book tags - "
                "recommend limiting to 8-12 for readability"
            )

        return errors, warnings

    @classmethod
    def strip_book_tags(cls, text: str) -> str:
        """
        Remove all book tags from text, leaving just the content.

        Used for plain text rendering or fallback displays.

        Example: "<book id='123'>The Hobbit</book>" → "The Hobbit"
        """
        return re.sub(cls.BOOK_TAG_PATTERN, r"\2", text, flags=re.IGNORECASE)

    @classmethod
    def get_inline_book_ids(cls, text: str) -> List[int]:
        """
        Extract just the book IDs from inline tags (for logging/analytics).

        Returns IDs in order of appearance in text.
        """
        matches = cls.extract_book_tags(text)
        return [match.item_idx for match in matches]
