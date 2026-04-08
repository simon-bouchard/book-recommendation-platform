# app/semantic_index/embedding/text_templates.py
"""
Centralized text template functions for all embedding strategies.
Each function constructs the text string that will be embedded and indexed.
"""

from typing import List


def build_baseline_old_text(
    title: str, author: str, description: str, ol_subjects: List[str]
) -> str:
    """
    Baseline-Old format (existing baseline index).

    Format: {title} — {author} | {description} | subjects: {subjects}

    Args:
        title: Book title
        author: Author name
        description: Raw book description (will be truncated to 500 chars)
        ol_subjects: List of raw OpenLibrary subject strings

    Returns:
        Formatted text string for embedding

    Notes:
        - Includes description (unlike baseline-clean)
        - Description truncated to 500 chars to avoid excessive length
        - Subjects limited to first 10
        - This matches the existing baseline index format
    """
    text_parts = []

    # Title and author
    if title and author:
        text_parts.append(f"{title} — {author}")
    elif title:
        text_parts.append(title)

    # Description (truncated)
    if description:
        desc = description[:500] if len(description) > 500 else description
        text_parts.append(desc)

    # Subjects
    if ol_subjects:
        subjects_str = ", ".join(ol_subjects[:10])
        text_parts.append(f"subjects: {subjects_str}")

    return " | ".join(text_parts)


def build_baseline_clean_text(title: str, author: str, ol_subjects: List[str]) -> str:
    """
    Baseline-Clean format (new variant without description).

    Format: {title} — {author} | subjects: {subjects}

    Args:
        title: Book title
        author: Author name
        ol_subjects: List of raw OpenLibrary subject strings

    Returns:
        Formatted text string for embedding

    Notes:
        - No description (testing if description adds noise)
        - Only books with OL subjects should be included
        - Subjects are NOT limited (unlike baseline-old)
    """
    text_parts = []

    # Title and author
    if title and author:
        text_parts.append(f"{title} — {author}")
    elif title:
        text_parts.append(title)

    # Subjects (no limit for clean version)
    if ol_subjects:
        subjects_str = ", ".join(ol_subjects)
        text_parts.append(f"subjects: {subjects_str}")

    return " | ".join(text_parts)


def build_v1_full_text(
    title: str, author: str, genre_name: str, subjects: List[str], tone_names: List[str], vibe: str
) -> str:
    """
    V1-Full format (corrected version with genre and tone names).

    Format: {title} — {author} | genre: {genre_name} | subjects: {subjects} | tones: {tones} | vibe: {vibe}

    Args:
        title: Book title
        author: Author name
        genre_name: Genre display name (e.g., "Science Fiction")
        subjects: List of LLM-curated subject strings
        tone_names: List of tone slugs/names (e.g., ["dark", "fast-paced"])
        vibe: Short vibe description string

    Returns:
        Formatted text string for embedding

    Notes:
        - Uses tone NAMES not IDs (fixes bug in original v1 index)
        - Includes genre (fixes missing field in original v1 index)
        - Empty fields (no tones, no vibe) are omitted per v2 design rules
        - This format needs to be used to REBUILD v1 index before comparison
    """
    text_parts = []

    # Title and author (always included)
    if title and author:
        text_parts.append(f"{title} — {author}")
    elif title:
        text_parts.append(title)

    # Genre (required for full format)
    if genre_name:
        text_parts.append(f"genre: {genre_name}")

    # Subjects
    if subjects:
        subjects_str = ", ".join(subjects)
        text_parts.append(f"subjects: {subjects_str}")

    # Tones (optional - omit if empty)
    if tone_names:
        tones_str = ", ".join(tone_names)
        text_parts.append(f"tones: {tones_str}")

    # Vibe (optional - omit if empty)
    if vibe:
        text_parts.append(f"vibe: {vibe}")

    return " | ".join(text_parts)


def build_v1_subjects_text(title: str, author: str, subjects: List[str]) -> str:
    """
    V1-Subjects format (subjects-only variant of v1).

    Format: {title} — {author} | subjects: {subjects}

    Args:
        title: Book title
        author: Author name
        subjects: List of V1 LLM-curated subject strings

    Returns:
        Formatted text string for embedding

    Notes:
        - No genre, tones, or vibe
        - Tests whether V1 LLM subjects alone are better than full metadata
        - Same structure as baseline-clean but with LLM-curated subjects
    """
    text_parts = []

    # Title and author
    if title and author:
        text_parts.append(f"{title} — {author}")
    elif title:
        text_parts.append(title)

    # Subjects
    if subjects:
        subjects_str = ", ".join(subjects)
        text_parts.append(f"subjects: {subjects_str}")

    return " | ".join(text_parts)


def build_v2_full_text(
    title: str, author: str, genre_name: str, subjects: List[str], tone_names: List[str], vibe: str
) -> str:
    """
    V2-Full format (complete v2 enrichment with all metadata).

    Format: {title} — {author} | genre: {genre_name} | subjects: {subjects} | tones: {tones} | vibe: {vibe}

    Args:
        title: Book title
        author: Author name
        genre_name: Genre display name (e.g., "Historical Fiction")
        subjects: List of V2 LLM-curated subject strings
        tone_names: List of tone slugs/names (e.g., ["lyrical", "epic", "atmospheric"])
        vibe: Short vibe description string

    Returns:
        Formatted text string for embedding

    Notes:
        - Identical structure to v1_full (but with v2 data)
        - Uses v2 ontology (36 tones in 6 buckets)
        - Empty fields are omitted following v2 design rules
        - Genre always required for full format
    """
    text_parts = []

    # Title and author (always included)
    if title and author:
        text_parts.append(f"{title} — {author}")
    elif title:
        text_parts.append(title)

    # Genre (required for full format)
    if genre_name:
        text_parts.append(f"genre: {genre_name}")

    # Subjects
    if subjects:
        subjects_str = ", ".join(subjects)
        text_parts.append(f"subjects: {subjects_str}")

    # Tones (optional - omit if empty)
    if tone_names:
        tones_str = ", ".join(tone_names)
        text_parts.append(f"tones: {tones_str}")

    # Vibe (optional - omit if empty)
    if vibe:
        text_parts.append(f"vibe: {vibe}")

    return " | ".join(text_parts)


def build_v2_subjects_text(title: str, author: str, subjects: List[str]) -> str:
    """
    V2-Subjects format (subjects-only variant of v2).

    Format: {title} — {author} | subjects: {subjects}

    Args:
        title: Book title
        author: Author name
        subjects: List of V2 LLM-curated subject strings

    Returns:
        Formatted text string for embedding

    Notes:
        - No genre, tones, or vibe
        - Tests whether V2 LLM subjects alone are better than full metadata
        - Key comparison: v2_subjects vs v2_full to see if metadata adds noise
        - Same structure as v1_subjects but with v2 data
    """
    text_parts = []

    # Title and author
    if title and author:
        text_parts.append(f"{title} — {author}")
    elif title:
        text_parts.append(title)

    # Subjects
    if subjects:
        subjects_str = ", ".join(subjects)
        text_parts.append(f"subjects: {subjects_str}")

    return " | ".join(text_parts)


# Utility function for validation
def validate_text_not_empty(text: str, context: str) -> None:
    """
    Validate that generated text is not empty.

    Args:
        text: Generated embedding text
        context: Context string for error message (e.g., "baseline-clean for book 123")

    Raises:
        ValueError: If text is empty or whitespace-only
    """
    if not text or not text.strip():
        raise ValueError(f"Generated empty embedding text for {context}")
