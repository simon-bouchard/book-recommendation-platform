# evaluation/chatbot/recommendation_agent/test_data_factory.py
"""
Test data factory for isolated recommendation agent testing.
Provides mock strategies, real candidate books, and execution contexts.
"""

from typing import Dict, Any, Optional, List
import random

from app.agents.domain.recsys_schemas import PlannerStrategy, ExecutionContext
from app.agents.tools.registry import ToolRegistry, InternalToolGates


# ============================================================================
# MOCK PLANNER STRATEGIES
# ============================================================================


def get_mock_strategy(scenario: str, **kwargs) -> PlannerStrategy:
    """
    Get mock planner strategy for retrieval testing.

    Args:
        scenario: Strategy type - 'semantic', 'als', 'als_with_profile',
                  'als_no_profile', 'subject', 'profile', 'negative', 'fallback'
        **kwargs: Optional profile_data for profile scenarios

    Returns:
        PlannerStrategy object ready for retrieval agent
    """

    if scenario == "semantic":
        return PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            reasoning="Descriptive query - use semantic search",
            fallback_tools=[
                "subject_hybrid_pool"
            ],  # Updated: descriptive queries often mention genres
            profile_data=None,
        )

    elif scenario == "als":
        # Generic ALS - defaults to no profile fallback
        return PlannerStrategy(
            recommended_tools=["als_recs"],
            reasoning="Vague query from warm user - use collaborative filtering",
            fallback_tools=["popular_books"],
            profile_data=None,
        )

    elif scenario == "als_with_profile":
        # ALS with profile data - uses subject fallback
        profile_data = kwargs.get(
            "profile_data",
            {
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],  # Mystery, Detective, Crime
                    "favorite_genres": ["mystery", "thriller"],
                }
            },
        )
        return PlannerStrategy(
            recommended_tools=["als_recs"],
            reasoning="Vague query from warm user with profile - use collaborative filtering",
            fallback_tools=["subject_hybrid_pool"],  # Updated: better fallback when profile exists
            profile_data=profile_data,
        )

    elif scenario == "als_no_profile":
        # ALS without profile - uses popular books fallback
        return PlannerStrategy(
            recommended_tools=["als_recs"],
            reasoning="Vague query from warm user without profile - use collaborative filtering",
            fallback_tools=["popular_books"],
            profile_data=None,
        )

    elif scenario == "subject":
        return PlannerStrategy(
            recommended_tools=["subject_id_search", "subject_hybrid_pool"],
            reasoning="Simple genre query - resolve subject and search",
            fallback_tools=[
                "book_semantic_search"
            ],  # Updated: semantic as fallback for genre queries
            profile_data=None,
        )

    elif scenario == "profile":
        # Cold user with profile data
        profile_data = kwargs.get(
            "profile_data",
            {
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],  # Mystery, Detective, Crime
                    "favorite_genres": ["mystery", "thriller"],
                }
            },
        )
        return PlannerStrategy(
            recommended_tools=["subject_hybrid_pool"],
            reasoning="Vague query from cold user with profile - use favorite subjects",
            fallback_tools=["popular_books"],
            profile_data=profile_data,
        )

    elif scenario == "negative":
        return PlannerStrategy(
            recommended_tools=["book_semantic_search"],
            reasoning="Query with negative constraint - semantic search then filter",
            fallback_tools=[
                "subject_hybrid_pool"
            ],  # Updated: negative constraints usually on genre queries
            profile_data=None,
        )

    elif scenario == "fallback":
        return PlannerStrategy(
            recommended_tools=["popular_books"],
            reasoning="New user, no profile - use popular books",
            fallback_tools=["book_semantic_search"],  # Updated: semantic as last resort
            profile_data=None,
        )

    else:
        raise ValueError(f"Unknown strategy scenario: {scenario}")


# ============================================================================
# REAL CANDIDATE BOOKS (via ToolRegistry)
# ============================================================================


def get_candidates(scenario: str, db, **kwargs) -> Dict[str, Any]:
    """
    Get candidate books for curation agent testing using ToolRegistry.

    Args:
        scenario: Candidate type - 'basic', 'negative_cozy', 'negative_serial_killer',
                  'genre_fantasy', 'genre_historical', 'als', 'subject'
        db: Database session
        **kwargs: Additional params (user_id, subject_ids, query, user_num_ratings)

    Returns:
        Dict with 'books' list and metadata
    """
    # Get user if provided
    user_id = kwargs.get("user_id")
    current_user = None
    if user_id and db:
        from app.table_models import User

        current_user = db.query(User).filter(User.user_id == user_id).first()

    # Create tool gates
    user_num_ratings = kwargs.get("user_num_ratings", 12)
    gates = InternalToolGates(
        user_num_ratings=user_num_ratings,
        warm_threshold=10,
        profile_allowed=True,  # Allow profile access for testing
    )

    # Create registry with retrieval tools
    registry = ToolRegistry.for_retrieval(gates=gates, ctx_user=current_user, ctx_db=db)

    # Basic scenarios - simple semantic search
    if scenario == "basic":
        query = kwargs.get("query", "mystery novels")
        semantic_tool = registry.get_tool("book_semantic_search")
        if not semantic_tool:
            raise RuntimeError("book_semantic_search tool not available")

        books = semantic_tool.execute(query=query, top_k=60)
        return {"books": books, "tools_used": ["book_semantic_search"], "query_used": query}

    # Negative constraint scenarios - mix base + constraint books
    elif scenario == "negative_cozy":
        # Get 40 mystery books
        semantic_tool = registry.get_tool("book_semantic_search")
        base_books = semantic_tool.execute(
            query="mystery detective crime thriller suspense", top_k=40
        )

        # Get 20 cozy mystery books
        cozy_books = semantic_tool.execute(
            query="cozy mystery amateur sleuth small town cats bakery tea shop inn charming",
            top_k=20,
        )

        # Mix and shuffle
        all_books = base_books + cozy_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["book_semantic_search"],
            "query_used": "mystery NOT cozy",
            "base_count": len(base_books),
            "constraint_count": len(cozy_books),
        }

    elif scenario == "negative_serial_killer":
        semantic_tool = registry.get_tool("book_semantic_search")

        # Get 40 thriller books
        base_books = semantic_tool.execute(query="thriller suspense crime psychological", top_k=40)

        # Get 20 serial killer books
        serial_books = semantic_tool.execute(
            query="serial killer psychopath killer profiler FBI forensics murder investigation",
            top_k=20,
        )

        # Mix and shuffle
        all_books = base_books + serial_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["book_semantic_search"],
            "query_used": "thriller NOT serial killer",
            "base_count": len(base_books),
            "constraint_count": len(serial_books),
        }

    # Genre matching scenarios - mix correct + wrong genre
    elif scenario == "genre_fantasy":
        # Get 40 fantasy books using subject_hybrid_pool
        subject_tool = registry.get_tool("subject_hybrid_pool")
        fantasy_books = subject_tool.execute(
            top_k=40,
            fav_subjects_idxs=[1378],  # Fantasy subject
            weight=0.7,
        )

        # Get 20 mystery/thriller books
        semantic_tool = registry.get_tool("book_semantic_search")
        wrong_books = semantic_tool.execute(
            query="mystery detective crime thriller suspense", top_k=20
        )

        # Mix and shuffle
        all_books = fantasy_books + wrong_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["subject_hybrid_pool", "book_semantic_search"],
            "query_used": "fantasy",
            "correct_genre_count": len(fantasy_books),
            "wrong_genre_count": len(wrong_books),
        }

    elif scenario == "genre_historical":
        # Get 40 historical fiction books
        subject_tool = registry.get_tool("subject_hybrid_pool")
        historical_books = subject_tool.execute(
            top_k=40,
            fav_subjects_idxs=[1669, 1415],  # Historical Fiction subject
            weight=0.7,
        )

        # Get 20 sci-fi/fantasy books
        semantic_tool = registry.get_tool("book_semantic_search")
        wrong_books = semantic_tool.execute(
            query="science fiction fantasy space alien dragon magic", top_k=20
        )

        # Mix and shuffle
        all_books = historical_books + wrong_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["subject_hybrid_pool", "book_semantic_search"],
            "query_used": "historical fiction",
            "correct_genre_count": len(historical_books),
            "wrong_genre_count": len(wrong_books),
        }

    # ALS personalization scenario
    elif scenario == "als":
        if not current_user:
            raise ValueError("ALS scenario requires user_id")

        als_tool = registry.get_tool("als_recs")
        if not als_tool:
            # Fall back to semantic if ALS not available (cold user)
            semantic_tool = registry.get_tool("book_semantic_search")
            books = semantic_tool.execute(query="popular books", top_k=60)
            return {
                "books": books,
                "tools_used": ["book_semantic_search"],
                "query_used": "fallback due to cold user",
            }

        books = als_tool.execute(top_k=60)
        return {
            "books": books,
            "tools_used": ["als_recs"],
            "query_used": "personalized recommendations",
        }

    # Subject-based scenario
    elif scenario == "subject":
        subject_ids = kwargs.get("subject_ids", [978, 1066, 2317])  # Mystery, Detective, Crime
        subject_tool = registry.get_tool("subject_hybrid_pool")
        if not subject_tool:
            raise RuntimeError("subject_hybrid_pool tool not available")

        books = subject_tool.execute(top_k=60, fav_subjects_idxs=subject_ids, weight=0.6)
        return {
            "books": books,
            "tools_used": ["subject_hybrid_pool"],
            "query_used": f"subjects {subject_ids}",
        }

    else:
        raise ValueError(f"Unknown candidate scenario: {scenario}")


# ============================================================================
# MOCK EXECUTION CONTEXTS
# ============================================================================


def get_execution_context(scenario: str, **kwargs) -> ExecutionContext:
    """
    Get mock execution context for curation testing.

    Args:
        scenario: Context type - 'semantic', 'als', 'als_no_profile', 'als_with_profile',
                  'subject', 'profile', 'profile_recent', 'negative', 'fallback',
                  'no_personalization'
        **kwargs: Optional profile_data for profile scenarios

    Returns:
        ExecutionContext object ready for curation agent
    """

    if scenario == "semantic":
        return ExecutionContext(
            planner_reasoning="Descriptive query about dark mysteries - used semantic search",
            tools_used=["book_semantic_search"],
            profile_data=None,
        )

    elif scenario == "als" or scenario == "als_no_profile":
        # ALS without profile (als_no_profile is explicit variant)
        return ExecutionContext(
            planner_reasoning="Vague query from warm user - used collaborative filtering",
            tools_used=["als_recs"],
            profile_data=None,
        )

    elif scenario == "als_with_profile":
        # ALS with profile context
        profile_data = kwargs.get(
            "profile_data",
            {
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],
                    "favorite_genres": ["mystery", "thriller"],
                }
            },
        )
        return ExecutionContext(
            planner_reasoning="Vague query from warm user with profile - used collaborative filtering",
            tools_used=["als_recs"],
            profile_data=profile_data,
        )

    elif scenario == "subject":
        return ExecutionContext(
            planner_reasoning="Simple genre query - resolved to subjects and searched",
            tools_used=["subject_id_search", "subject_hybrid_pool"],
            profile_data=None,
        )

    elif scenario == "profile":
        profile_data = kwargs.get(
            "profile_data",
            {
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],
                    "favorite_genres": ["mystery", "thriller"],
                }
            },
        )
        return ExecutionContext(
            planner_reasoning="Vague query from cold user with profile - used favorite subjects",
            tools_used=["subject_hybrid_pool"],
            profile_data=profile_data,
        )

    elif scenario == "profile_recent":
        profile_data = kwargs.get(
            "profile_data",
            {
                "user_profile": {"favorite_subjects": [978, 1066], "favorite_genres": ["mystery"]},
                "recent_interactions": [
                    {"book_title": "The Silent Patient", "rating": 5},
                    {"book_title": "Gone Girl", "rating": 5},
                ],
            },
        )
        return ExecutionContext(
            planner_reasoning="Vague query with recent interaction context - used favorite subjects",
            tools_used=["subject_hybrid_pool"],
            profile_data=profile_data,
        )

    elif scenario == "negative":
        return ExecutionContext(
            planner_reasoning="Query with negative constraint - will filter in curation",
            tools_used=["book_semantic_search"],
            profile_data=None,
        )

    elif scenario == "fallback":
        return ExecutionContext(
            planner_reasoning="Vague query from new user - used popular books",
            tools_used=["popular_books"],
            profile_data=None,
        )

    elif scenario == "no_personalization":
        return ExecutionContext(
            planner_reasoning="Descriptive query, no personalization - query-based search only",
            tools_used=["book_semantic_search"],
            profile_data=None,
        )

    else:
        raise ValueError(f"Unknown context scenario: {scenario}")
