# evaluation/chatbot/recommendation_agent/test_data_factory.py
"""
Test data factory for isolated recommendation agent testing.
Provides mock strategies, real candidate books, and execution contexts.
"""

import random
from typing import Any, Dict

from app.agents.domain.recsys_schemas import ExecutionContext, PlannerStrategy
from app.agents.tools.registry import InternalToolGates, ToolRegistry


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
            fallback_tools=["subject_hybrid_pool"],
            profile_data=None,
        )

    elif scenario == "als":
        return PlannerStrategy(
            recommended_tools=["als_recs"],
            reasoning="Vague query from warm user - use collaborative filtering",
            fallback_tools=["popular_books"],
            profile_data=None,
        )

    elif scenario == "als_with_profile":
        profile_data = kwargs.get(
            "profile_data",
            {
                "user_profile": {
                    "favorite_subjects": [978, 1066, 2317],
                    "favorite_genres": ["mystery", "thriller"],
                }
            },
        )
        return PlannerStrategy(
            recommended_tools=["als_recs"],
            reasoning="Vague query from warm user with profile - use collaborative filtering",
            fallback_tools=["subject_hybrid_pool"],
            profile_data=profile_data,
        )

    elif scenario == "als_no_profile":
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
            fallback_tools=["book_semantic_search"],
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
            fallback_tools=["subject_hybrid_pool"],
            profile_data=None,
        )

    elif scenario == "fallback":
        return PlannerStrategy(
            recommended_tools=["popular_books"],
            reasoning="New user, no profile - use popular books",
            fallback_tools=["book_semantic_search"],
            profile_data=None,
        )

    else:
        raise ValueError(f"Unknown strategy scenario: {scenario}")


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
    user_id = kwargs.get("user_id")
    current_user = None
    if user_id and db:
        from app.table_models import User

        current_user = db.query(User).filter(User.user_id == user_id).first()

    user_num_ratings = kwargs.get("user_num_ratings", 12)
    gates = InternalToolGates(
        user_num_ratings=user_num_ratings,
        warm_threshold=10,
        profile_allowed=True,
    )

    registry = ToolRegistry.for_retrieval(gates=gates, ctx_user=current_user, ctx_db=db)

    if scenario == "basic":
        query = kwargs.get("query", "mystery novels")
        semantic_tool = registry.get_tool("book_semantic_search")
        if not semantic_tool:
            raise RuntimeError("book_semantic_search tool not available")

        books = semantic_tool.invoke({"query": query, "top_k": 60})
        return {"books": books, "tools_used": ["book_semantic_search"], "query_used": query}

    elif scenario == "negative_cozy":
        semantic_tool = registry.get_tool("book_semantic_search")
        base_books = semantic_tool.invoke(
            {"query": "mystery detective crime thriller suspense", "top_k": 40}
        )

        cozy_books = semantic_tool.invoke(
            {
                "query": "cozy mystery amateur sleuth small town cats bakery tea shop inn charming",
                "top_k": 20,
            }
        )

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

        base_books = semantic_tool.invoke(
            {"query": "thriller suspense crime psychological", "top_k": 40}
        )

        serial_books = semantic_tool.invoke(
            {
                "query": "serial killer psychopath killer profiler FBI forensics murder investigation",
                "top_k": 20,
            }
        )

        all_books = base_books + serial_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["book_semantic_search"],
            "query_used": "thriller NOT serial killer",
            "base_count": len(base_books),
            "constraint_count": len(serial_books),
        }

    elif scenario == "genre_fantasy":
        subject_tool = registry.get_tool("subject_hybrid_pool")
        fantasy_books = subject_tool.invoke(
            {
                "top_k": 40,
                "fav_subjects_idxs": [1378],
                "weight": 0.7,
            }
        )

        semantic_tool = registry.get_tool("book_semantic_search")
        wrong_books = semantic_tool.invoke(
            {"query": "mystery detective crime thriller suspense", "top_k": 20}
        )

        all_books = fantasy_books + wrong_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["subject_hybrid_pool", "book_semantic_search"],
            "query_used": "fantasy",
            "correct_genre_count": len(fantasy_books),
            "wrong_genre_count": len(wrong_books),
        }

    elif scenario == "genre_science_fiction":
        subject_tool = registry.get_tool("subject_hybrid_pool")
        scifi_books = subject_tool.invoke(
            {
                "top_k": 40,
                "fav_subjects_idxs": [2922],
                "weight": 0.7,
            }
        )

        semantic_tool = registry.get_tool("book_semantic_search")
        wrong_books = semantic_tool.invoke(
            {"query": "cozy mystery romance historical drama family saga", "top_k": 20}
        )

        all_books = scifi_books + wrong_books
        random.shuffle(all_books)

        return {
            "books": all_books,
            "tools_used": ["subject_hybrid_pool", "book_semantic_search"],
            "query_used": "science fiction",
            "correct_genre_count": len(scifi_books),
            "wrong_genre_count": len(wrong_books),
        }

    elif scenario == "als":
        if not current_user:
            raise ValueError("ALS scenario requires user_id")

        als_tool = registry.get_tool("als_recs")
        if not als_tool:
            semantic_tool = registry.get_tool("book_semantic_search")
            books = semantic_tool.invoke({"query": "popular books", "top_k": 60})
            return {
                "books": books,
                "tools_used": ["book_semantic_search"],
                "query_used": "fallback due to cold user",
            }

        books = als_tool.invoke({"top_k": 60})
        return {
            "books": books,
            "tools_used": ["als_recs"],
            "query_used": "personalized recommendations",
        }

    elif scenario == "subject":
        subject_ids = kwargs.get("subject_ids", [978, 1066, 2317])
        subject_tool = registry.get_tool("subject_hybrid_pool")
        if not subject_tool:
            raise RuntimeError("subject_hybrid_pool tool not available")

        books = subject_tool.invoke({"top_k": 60, "fav_subjects_idxs": subject_ids, "weight": 0.6})
        return {
            "books": books,
            "tools_used": ["subject_hybrid_pool"],
            "query_used": f"subjects {subject_ids}",
        }

    else:
        raise ValueError(f"Unknown candidate scenario: {scenario}")


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
        return ExecutionContext(
            planner_reasoning="Vague query from warm user - used collaborative filtering",
            tools_used=["als_recs"],
            profile_data=None,
        )

    elif scenario == "als_with_profile":
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
