# app/enrichment/quality_classifier.py
"""
Quality classifier for book metadata.
Assigns quality tiers based on combined description + OL subjects signals.

Tier System:
- RICH (score >= 60): Full enrichment (5-8 subjects, 2-3 tones, 8-12w vibe)
- SPARSE (30-59): Focused enrichment (3-5 subjects, 0-2 tones, optional vibe)
- MINIMAL (10-29): Basic enrichment (1-3 subjects, 0-1 tone, no vibe)
- BASIC (< 10): Genre only (0-1 subject, no tones, no vibe)
- INSUFFICIENT: Missing title/author (skip enrichment)
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re


# Placeholder text patterns to filter out
PLACEHOLDER_PATTERNS = [
    r'^no description available\.?$',
    r'^description not available\.?$',
    r'^n/?a\.?$',
    r'^none\.?$',
    r'^$',
]

COMPILED_PLACEHOLDERS = [re.compile(pattern, re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS]


@dataclass
class QualitySignals:
    """
    Diagnostic signals about book metadata quality.
    Used for logging, debugging, and threshold tuning.
    """
    desc_words: int
    desc_valid: bool
    ol_subject_count: int
    desc_points: int
    ol_points: int
    has_suspicious_encoding: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            'desc_words': self.desc_words,
            'desc_valid': self.desc_valid,
            'ol_subject_count': self.ol_subject_count,
            'desc_points': self.desc_points,
            'ol_points': self.ol_points,
            'has_suspicious_encoding': self.has_suspicious_encoding,
        }


@dataclass
class QualityAssessment:
    """
    Complete quality assessment for a book.
    Includes tier assignment, score, and diagnostic signals.
    """
    tier: str  # "RICH" | "SPARSE" | "MINIMAL" | "BASIC" | "INSUFFICIENT"
    score: int  # 0-100
    signals: QualitySignals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            'tier': self.tier,
            'score': self.score,
            'signals': self.signals.to_dict(),
        }
    
    @property
    def is_enrichable(self) -> bool:
        """Check if book should be enriched (not INSUFFICIENT)"""
        return self.tier != 'INSUFFICIENT'
    
    @property
    def requires_conservative_approach(self) -> bool:
        """Check if tier requires conservative enrichment (MINIMAL or BASIC)"""
        return self.tier in ('MINIMAL', 'BASIC')


def is_valid_description(description: Optional[str]) -> bool:
    """
    Check if description is valid (not null, empty, or placeholder).
    
    Args:
        description: Raw description text from database
        
    Returns:
        True if description is usable for enrichment
    """
    if not description:
        return False
    
    desc_stripped = description.strip().lower()
    
    # Check against placeholder patterns
    for pattern in COMPILED_PLACEHOLDERS:
        if pattern.match(desc_stripped):
            return False
    
    return True


def has_encoding_issues(text: str) -> bool:
    """
    Detect potential encoding/corruption issues in text.
    
    Args:
        text: Text to check
        
    Returns:
        True if text shows signs of corruption
    """
    if not text:
        return False
    
    # Check for excessive special characters
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_char_ratio > 0.3:
        return True
    
    # Check for replacement characters
    if 'ï¿½' in text or '\ufffd' in text:
        return True
    
    # Check for all caps (often indicates OCR/import issues)
    if len(text) > 50 and text.isupper():
        return True
    
    return False


def count_description_words(description: Optional[str]) -> int:
    """
    Count words in description, handling edge cases.
    
    Args:
        description: Description text
        
    Returns:
        Word count (0 if invalid)
    """
    if not description or not is_valid_description(description):
        return 0
    
    # Simple word count (split on whitespace)
    words = description.split()
    
    # Filter out very short "words" (likely punctuation artifacts)
    meaningful_words = [w for w in words if len(w) > 1 or w.isalnum()]
    
    return len(meaningful_words)


def calculate_description_points(desc_words: int) -> int:
    """
    Calculate points contribution from description length.
    
    Scoring:
    - >= 100 words: 60 points (excellent - detailed description)
    - 50-99 words: 40 points (good - substantial description)
    - 20-49 words: 20 points (fair - brief description)
    - 10-19 words: 10 points (minimal - very brief)
    - < 10 words: 0 points (insufficient)
    
    Args:
        desc_words: Number of words in description
        
    Returns:
        Points (0-60)
    """
    if desc_words >= 100:
        return 60
    elif desc_words >= 50:
        return 40
    elif desc_words >= 20:
        return 20
    elif desc_words >= 10:
        return 10
    else:
        return 0


def calculate_ol_subject_points(ol_subject_count: int) -> int:
    """
    Calculate points contribution from OL subjects count.
    
    Scoring:
    - >= 10 subjects: 40 points (excellent - rich categorization)
    - 7-9 subjects: 30 points (good - solid categorization)
    - 5-6 subjects: 20 points (fair - basic categorization)
    - 3-4 subjects: 10 points (minimal - sparse categorization)
    - < 3 subjects: 0 points (insufficient)
    
    Args:
        ol_subject_count: Number of OL subjects
        
    Returns:
        Points (0-40)
    """
    if ol_subject_count >= 10:
        return 40
    elif ol_subject_count >= 7:
        return 30
    elif ol_subject_count >= 5:
        return 20
    elif ol_subject_count >= 3:
        return 10
    else:
        return 0


def assign_tier(score: int) -> str:
    """
    Assign quality tier based on combined score.
    
    Thresholds:
    - >= 60: RICH (full enrichment)
    - 30-59: SPARSE (focused enrichment)
    - 10-29: MINIMAL (basic enrichment)
    - < 10: BASIC (genre only)
    
    Args:
        score: Combined score (0-100)
        
    Returns:
        Tier name
    """
    if score >= 60:
        return "RICH"
    elif score >= 30:
        return "SPARSE"
    elif score >= 10:
        return "MINIMAL"
    else:
        return "BASIC"


def assess_book_quality(
    title: Optional[str],
    author: Optional[str],
    description: Optional[str],
    ol_subjects: Optional[List[str]]
) -> QualityAssessment:
    """
    Assess book metadata quality and assign tier.
    
    This is the main entry point for quality classification.
    
    Args:
        title: Book title
        author: Author name
        description: Book description text
        ol_subjects: List of OpenLibrary subject strings
        
    Returns:
        QualityAssessment with tier, score, and signals
        
    Examples:
        >>> assess = assess_book_quality(
        ...     title="The Great Gatsby",
        ...     author="F. Scott Fitzgerald",
        ...     description="A classic novel about the American Dream in the 1920s...",
        ...     ol_subjects=["Fiction", "Classic literature", "American literature"]
        ... )
        >>> assess.tier
        'SPARSE'
        >>> assess.score
        30
    """
    # Check critical metadata first
    if not title or not title.strip():
        return QualityAssessment(
            tier="INSUFFICIENT",
            score=0,
            signals=QualitySignals(
                desc_words=0,
                desc_valid=False,
                ol_subject_count=0,
                desc_points=0,
                ol_points=0,
                has_suspicious_encoding=False,
            )
        )
    
    if not author or not author.strip():
        return QualityAssessment(
            tier="INSUFFICIENT",
            score=0,
            signals=QualitySignals(
                desc_words=0,
                desc_valid=False,
                ol_subject_count=0,
                desc_points=0,
                ol_points=0,
                has_suspicious_encoding=False,
            )
        )
    
    # Analyze description
    desc_valid = is_valid_description(description)
    desc_words = count_description_words(description) if desc_valid else 0
    desc_points = calculate_description_points(desc_words)
    
    # Check for encoding issues
    suspicious_encoding = False
    if description:
        suspicious_encoding = has_encoding_issues(description)
    
    # Analyze OL subjects
    ol_subject_count = len(ol_subjects) if ol_subjects else 0
    
    # Remove duplicates and filter empty strings
    if ol_subjects:
        unique_subjects = {s.strip() for s in ol_subjects if s and s.strip()}
        ol_subject_count = len(unique_subjects)
    
    ol_points = calculate_ol_subject_points(ol_subject_count)
    
    # Calculate combined score
    score = desc_points + ol_points
    
    # Assign tier
    tier = assign_tier(score)
    
    # Build signals
    signals = QualitySignals(
        desc_words=desc_words,
        desc_valid=desc_valid,
        ol_subject_count=ol_subject_count,
        desc_points=desc_points,
        ol_points=ol_points,
        has_suspicious_encoding=suspicious_encoding,
    )
    
    return QualityAssessment(
        tier=tier,
        score=score,
        signals=signals,
    )


def get_tier_requirements(tier: str) -> Dict[str, Any]:
    """
    Get enrichment requirements for a given tier.
    
    Used by prompt builder and validator to enforce tier-specific rules.
    
    Args:
        tier: Tier name ("RICH" | "SPARSE" | "MINIMAL" | "BASIC")
        
    Returns:
        Dict with requirements for subjects, tones, vibe, genre
        
    Examples:
        >>> reqs = get_tier_requirements("SPARSE")
        >>> reqs['subjects']
        {'min': 3, 'max': 5, 'required': True}
        >>> reqs['vibe']
        {'min_words': 4, 'max_words': 8, 'required': False}
    """
    requirements = {
        "RICH": {
            "subjects": {"min": 5, "max": 8, "required": True},
            "tones": {"min": 2, "max": 3, "required": True},
            "vibe": {"min_words": 8, "max_words": 20, "required": True},
            "genre": {"required": True},
        },
        "SPARSE": {
            "subjects": {"min": 3, "max": 5, "required": True},
            "tones": {"min": 0, "max": 2, "required": False},
            "vibe": {"min_words": 4, "max_words": 10, "required": False},
            "genre": {"required": True},
        },
        "MINIMAL": {
            "subjects": {"min": 1, "max": 3, "required": True},
            "tones": {"min": 0, "max": 1, "required": False},
            "vibe": {"min_words": 0, "max_words": 0, "required": False},  # Must be empty
            "genre": {"required": True},
        },
        "BASIC": {
            "subjects": {"min": 0, "max": 1, "required": False},
            "tones": {"min": 0, "max": 0, "required": False},  # Must be empty
            "vibe": {"min_words": 0, "max_words": 0, "required": False},  # Must be empty
            "genre": {"required": True},
        },
    }
    
    return requirements.get(tier, requirements["BASIC"])


# Convenience function for quick tier checking
def is_tier_sufficient_for_semantic_search(tier: str) -> bool:
    """
    Check if tier produces enough metadata for semantic search.
    
    Args:
        tier: Tier name
        
    Returns:
        True if tier produces subjects + genre (minimum for search)
    """
    return tier in ("RICH", "SPARSE", "MINIMAL")
