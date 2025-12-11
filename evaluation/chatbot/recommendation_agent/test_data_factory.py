# evaluation/chatbot/recommendation_agent/test_data_factory.py
"""
Test data factory for recommendation agent evaluation.
Provides mock strategies, candidate queries, and execution contexts for isolated stage testing.
"""

import random
from typing import Dict, List, Optional, Any
from app.agents.domain.recsys_schemas import PlannerStrategy, ExecutionContext


# ============================================================================
# RETRIEVAL TEST DATA - Mock PlannerStrategy Objects
# ============================================================================


def get_mock_strategy(scenario: str, **kwargs) -> PlannerStrategy:
    """
    Get mock planner strategy for retrieval agent testing.

    Args:
        scenario: Strategy type - 'semantic', 'als', 'subject', 'profile',
                  'negative', or 'fallback'
        **kwargs: Optional overrides (profile_data, negative_constraints, etc.)

    Returns:
        PlannerStrategy object for testing
    """
    strategies = {
        "semantic": PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            fallback_tools=["popular_books"],
            reasoning="Descriptive query with atmosphere - semantic search ideal",
            profile_data=None,
            negative_constraints=None,
        ),
        "als": PlannerStrategy(
            recommended_tools=["als_recs"],
            fallback_tools=["popular_books"],
            reasoning="Vague query with warm user - personalized recommendations best",
            profile_data=None,
            negative_constraints=None,
        ),
        "subject": PlannerStrategy(
            recommended_tools=["subject_id_search", "subject_hybrid_pool"],
            fallback_tools=["book_semantic_search"],
            reasoning="Genre query - need explicit subject IDs for accurate results",
            profile_data=None,
            negative_constraints=None,
        ),
        "profile": PlannerStrategy(
            recommended_tools=["subject_hybrid_pool"],
            fallback_tools=["popular_books"],
            reasoning="Cold user with profile - use favorite subjects for personalization",
            profile_data={
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],
                    "favorite_genres": ["mystery", "thriller", "crime"],
                }
            },
            negative_constraints=None,
        ),
        "negative": PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            fallback_tools=["popular_books"],
            reasoning="Query with negative constraints - semantic search then filter",
            profile_data=None,
            negative_constraints=kwargs.get("negative_constraints", ["cozy"]),
        ),
        "fallback": PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            fallback_tools=["popular_books"],
            reasoning="Primary tool may underperform - have fallback ready",
            profile_data=None,
            negative_constraints=None,
        ),
    }

    if scenario not in strategies:
        raise ValueError(f"Unknown strategy scenario: {scenario}")

    strategy = strategies[scenario]

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)

    return strategy


# ============================================================================
# CURATION TEST DATA - Candidate Queries
# ============================================================================


def get_candidates(scenario: str, db, **kwargs) -> Dict[str, Any]:
    """
    Get candidate books for curation agent testing.

    Args:
        scenario: Candidate type - 'basic', 'negative_cozy', 'negative_serial_killer',
                  'genre_fantasy', 'genre_historical', 'als', 'subject'
        db: Database session
        **kwargs: Additional params (user_id, subject_ids, query, etc.)

    Returns:
        Dict with 'books' list and metadata
    """
    from app.agents.tools.native_tools import book_semantic_search, subject_hybrid_pool, als_recs

    # Basic scenarios - simple semantic search
    if scenario == "basic":
        query = kwargs.get("query", "mystery novels")
        result = book_semantic_search(query=query, limit=60, db=db)
        return {
            "books": result.get("books", []),
            "tools_used": ["book_semantic_search"],
            "query_used": query,
        }

    # Negative constraint scenarios - mix base + constraint books, shuffle
    elif scenario == "negative_cozy":
        # Get regular mysteries (should pass through)
        base = book_semantic_search(
            query="mystery detective crime investigation thriller", limit=40, db=db
        )

        # Get cozy mysteries (should be filtered out)
        cozy = book_semantic_search(
            query="cozy mystery amateur sleuth small town cats bakery tea shop inn", limit=20, db=db
        )

        # Combine and shuffle to mix constraint books throughout
        all_books = base.get("books", []) + cozy.get("books", [])
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["book_semantic_search"],
            "query_used": "mystery novels",
        }

    elif scenario == "negative_serial_killer":
        # Get regular thrillers (should pass through)
        base = book_semantic_search(
            query="thriller suspense action espionage political international", limit=40, db=db
        )

        # Get serial killer books (should be filtered out)
        serial = book_semantic_search(
            query="serial killer FBI profiler forensics criminal minds detective", limit=20, db=db
        )

        # Combine and shuffle
        all_books = base.get("books", []) + serial.get("books", [])
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["book_semantic_search"],
            "query_used": "thriller books",
        }

    # Genre matching scenarios - mix correct + wrong genre, shuffle
    elif scenario == "genre_fantasy":
        # Get fantasy books (should pass through)
        fantasy = subject_hybrid_pool(
            subject_ids=[1378],  # Fantasy subject ID
            limit=40,
            db=db,
        )

        # Get wrong-genre books (should be filtered out)
        wrong = book_semantic_search(
            query="mystery detective crime thriller suspense investigation", limit=20, db=db
        )

        # Combine and shuffle
        all_books = fantasy.get("books", []) + wrong.get("books", [])
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["subject_hybrid_pool"],
            "query_used": "fantasy books",
        }

    elif scenario == "genre_historical":
        # Get historical fiction (should pass through)
        historical = subject_hybrid_pool(
            subject_ids=[1501],  # Historical Fiction subject ID
            limit=40,
            db=db,
        )

        # Get wrong-genre books (should be filtered out)
        wrong = book_semantic_search(
            query="science fiction space fantasy magic futuristic aliens", limit=20, db=db
        )

        # Combine and shuffle
        all_books = historical.get("books", []) + wrong.get("books", [])
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["subject_hybrid_pool"],
            "query_used": "historical fiction",
        }

    # Personalization scenarios
    elif scenario == "als":
        user_id = kwargs.get("user_id", 278859)
        result = als_recs(user_id=user_id, limit=60, db=db)
        return {
            "books": result.get("books", []),
            "tools_used": ["als_recs"],
            "query_used": "personalized recommendations",
        }

    elif scenario == "subject":
        subject_ids = kwargs.get("subject_ids", [978, 1066])
        result = subject_hybrid_pool(subject_ids=subject_ids, limit=60, db=db)
        return {
            "books": result.get("books", []),
            "tools_used": ["subject_hybrid_pool"],
            "query_used": "subject-based recommendations",
        }

    else:
        raise ValueError(f"Unknown candidate scenario: {scenario}")


# ============================================================================
# CURATION TEST DATA - Mock ExecutionContext Objects
# ============================================================================


def get_execution_context(scenario: str, **kwargs) -> ExecutionContext:
    """
    Get mock execution context for curation agent testing.

    Args:
        scenario: Context type - 'semantic', 'als', 'subject', 'profile',
                  'negative', 'fallback', 'no_personalization'
        **kwargs: Optional overrides (profile_data, tools_used, etc.)

    Returns:
        ExecutionContext object for testing
    """
    contexts = {
        "semantic": ExecutionContext(
            planner_reasoning="Descriptive query with atmosphere - semantic search ideal",
            tools_used=["book_semantic_search"],
            profile_data=None,
        ),
        "als": ExecutionContext(
            planner_reasoning="Vague query with warm user - personalized recommendations",
            tools_used=["als_recs"],
            profile_data=None,
        ),
        "subject": ExecutionContext(
            planner_reasoning="Genre query - using subject search for accurate results",
            tools_used=["subject_id_search", "subject_hybrid_pool"],
            profile_data=None,
        ),
        "profile": ExecutionContext(
            planner_reasoning="Cold user but profile shows favorite subjects",
            tools_used=["subject_hybrid_pool"],
            profile_data={
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],
                    "favorite_genres": ["mystery", "thriller", "crime"],
                }
            },
        ),
        "profile_recent": ExecutionContext(
            planner_reasoning="User with recent activity - considering recent reads",
            tools_used=["subject_hybrid_pool"],
            profile_data={
                "recent_interactions": [
                    {"title": "Gone Girl", "rating": 5, "item_idx": 12345},
                    {"title": "The Girl with the Dragon Tattoo", "rating": 5, "item_idx": 67890},
                    {"title": "Big Little Lies", "rating": 4, "item_idx": 11111},
                ]
            },
        ),
        "negative": ExecutionContext(
            planner_reasoning="Query with negative constraints - semantic search then filter",
            tools_used=["book_semantic_search"],
            profile_data=None,
        ),
        "fallback": ExecutionContext(
            planner_reasoning="Primary tool underperformed - using fallback",
            tools_used=["book_semantic_search", "popular_books"],
            profile_data=None,
        ),
        "no_personalization": ExecutionContext(
            planner_reasoning="Descriptive query - no personalization needed",
            tools_used=["book_semantic_search"],
            profile_data=None,
        ),
    }

    if scenario not in contexts:
        raise ValueError(f"Unknown context scenario: {scenario}")

    context = contexts[scenario]

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)

    return context
