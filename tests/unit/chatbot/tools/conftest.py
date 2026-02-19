# tests/unit/chatbot/tools/conftest.py
"""
Shared fixtures for all chatbot tool tests.
Provides mocked dependencies used across recsys, web, and docs tools.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from sqlalchemy.orm import Session


@pytest.fixture
def mock_book_meta():
    """
    Mock book metadata DataFrame.

    Matches the real book_meta schema used by _standardize_tool_output():
    index is item_idx, columns include book_num_ratings and cover_id.
    """
    df = pd.DataFrame(
        {
            "title": ["The Great Gatsby", "1984", "To Kill a Mockingbird"],
            "author": ["F. Scott Fitzgerald", "George Orwell", "Harper Lee"],
            "year": [1925, 1949, 1960],
            "book_num_ratings": [100, 200, 150],
            "cover_id": ["cov_1", "cov_2", None],
            "bayes": [4.5, 4.3, 4.1],
        }
    )
    df.index = pd.Index([1, 2, 3], name="item_idx")
    return df


@pytest.fixture
def mock_tone_map():
    """
    Mock tone ID to name mapping.

    Returns a dictionary mapping tone_id integers to tone names.
    """
    return {
        1: "Fast-paced",
        2: "Dark",
        3: "Humorous",
        4: "Reflective",
        5: "Tense",
    }


@pytest.fixture
def mock_db_session(mock_tone_map):
    """
    Mock database session with tone query support.

    Provides a mocked SQLAlchemy session that returns tone mappings
    when querying the Tone table.
    """
    session = Mock(spec=Session)

    tone_query = MagicMock()
    tone_results = [(tid, name) for tid, name in mock_tone_map.items()]
    tone_query.all.return_value = tone_results

    session.query.return_value = tone_query

    return session


@pytest.fixture
def mock_user():
    """
    Mock authenticated user object.

    Returns a user with basic attributes used by tools.
    """
    user = Mock()
    user.user_idx = 42
    user.username = "test_user"
    user.num_ratings = 25
    return user


@pytest.fixture
def sample_enriched_books():
    """
    Sample books with full enrichment metadata.

    Used to test semantic search results and standardization logic.
    """
    return [
        {
            "item_idx": 1,
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "year": 1925,
            "score": 0.95,
            "subjects": ["American Dream", "Jazz Age", "Tragedy"],
            "tone_ids": [2, 4],
            "genre": "Literary Fiction",
            "vibe": "Melancholic exploration of wealth and desire",
        },
        {
            "item_idx": 2,
            "title": "1984",
            "author": "George Orwell",
            "year": 1949,
            "score": 0.92,
            "subjects": ["Dystopia", "Surveillance", "Totalitarianism"],
            "tone_ids": [2, 5],
            "genre": "Science Fiction",
            "vibe": "Chilling portrait of authoritarian future",
        },
        {
            "item_idx": 3,
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "year": 1960,
            "score": 0.89,
            "subjects": ["Racism", "Justice", "Coming of Age"],
            "tone_ids": [4],
            "genre": "Literary Fiction",
            "vibe": "Poignant tale of morality and prejudice",
        },
    ]


@pytest.fixture
def sample_basic_books():
    """
    Sample books with only basic metadata (no enrichment).

    Used to test ALS and subject_hybrid results.
    """
    return [
        {
            "item_idx": 1,
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "year": 1925,
            "score": 0.95,
        },
        {
            "item_idx": 2,
            "title": "1984",
            "author": "George Orwell",
            "year": 1949,
            "score": 0.92,
        },
    ]


@pytest.fixture
def sample_semantic_search_raw():
    """
    Sample raw results from SemanticSearcher.search().

    Mimics the structure returned by the searcher with nested meta field.
    """
    return [
        {
            "item_idx": 1,
            "score": 0.95,
            "meta": {
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "year": 1925,
                "subjects": ["American Dream", "Jazz Age", "Tragedy"],
                "tone_ids": [2, 4],
                "genre": "Literary Fiction",
                "vibe": "Melancholic exploration of wealth and desire",
            },
        },
        {
            "item_idx": 2,
            "score": 0.92,
            "meta": {
                "title": "1984",
                "author": "George Orwell",
                "year": 1949,
                "subjects": ["Dystopia", "Surveillance", "Totalitarianism"],
                "tone_ids": [2, 5],
                "genre": "Science Fiction",
                "vibe": "Chilling portrait of authoritarian future",
            },
        },
    ]
