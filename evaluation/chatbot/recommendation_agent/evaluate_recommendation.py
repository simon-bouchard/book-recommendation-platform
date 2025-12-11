# evaluation/chatbot/recommendation_agent/evaluate_recommendation.py
"""
Comprehensive evaluation suite for the 3-stage recommendation agent pipeline.
Tests Planner → Retrieval → Curation stages individually and end-to-end integration.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import sys
import os
import argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent
from app.agents.infrastructure.recsys.planner_agent import PlannerAgent
from app.agents.infrastructure.recsys.retrieval_agent import RetrievalAgent
from app.agents.infrastructure.recsys.curation_agent import CurationAgent
from app.agents.domain.recsys_schemas import (
    PlannerInput,
    RetrievalInput,
    ExecutionContext,
    CurationInput,
)
from app.agents.domain.entities import AgentRequest, BookRecommendation
from app.agents.domain.parsers import InlineReferenceParser
from app.database import SessionLocal
from app.table_models import User, Interaction, Book
from sqlalchemy import func

# Import test data factory and new judges
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))
from test_data_factory import get_candidates, get_execution_context
from llm_judges import llm_judge_genre_match, llm_judge_personalization_prose


# ============================================================================
# DATABASE HELPERS
# ============================================================================


def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate that query is not empty or whitespace-only.

    Returns:
        (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, f"Empty or whitespace-only query: '{query}'"
    return True, ""


def get_user_by_id(db, user_id: int) -> Tuple[User, int]:
    """
    Get user by ID and their rating count.

    Returns:
        Tuple of (User object, rating count)
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise RuntimeError(f"User with ID {user_id} not found in database")

    rating_count = (
        db.query(func.count(Interaction.id)).filter(Interaction.user_id == user_id).scalar()
    )

    return user, rating_count


def load_test_cases(json_path: Path) -> Dict[str, List[Dict]]:
    """Load test cases from JSON file."""
    with open(json_path) as f:
        return json.load(f)


# ============================================================================


# ============================================================================
# LLM-AS-JUDGE FOR NEGATIVE CONSTRAINTS
# ============================================================================


def llm_judge_negative_constraints(
    books: List[BookRecommendation], negative_constraints: List[str], db, judge_llm
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to determine if recommended books match negative constraints.

    Args:
        books: List of recommended books
        negative_constraints: List of phrases that should NOT appear (e.g., ["cozy mystery", "serial killer"])
        db: Database session to fetch book details
        judge_llm: LLM instance from agent (reuse same model as agent uses)

    Returns:
        Dict with verdict, passed status, and reasoning
    """
    # Get book details from database
    book_ids = [book.item_idx for book in books]
    items = db.query(Book).filter(Book.item_idx.in_(book_ids)).all()

    book_details = []
    for item in items:
        author_name = item.author.name if item.author else "Unknown"
        book_details.append(
            f"- {item.title} by {author_name}: {item.description[:200] if item.description else 'No description'}..."
        )

    if not book_details:
        return {
            "passed": False,
            "verdict": "ERROR",
            "reasoning": "Could not fetch book details from database",
        }

    # Create LLM judge prompt
    prompt = f"""You are evaluating book recommendations to ensure they don't contain unwanted content.

**Negative constraints (books should NOT match these):**
{", ".join(f'"{c}"' for c in negative_constraints)}

**Recommended books:**
{chr(10).join(book_details)}

**Task:**
Determine if ANY of the recommended books match the negative constraints. A book matches if:
- Its title, author, or description mentions the constrained topic
- It's clearly about the constrained theme

**Response format (JSON only):**
{{
    "verdict": "PASS" or "FAIL",
    "reasoning": "Brief explanation",
    "violating_books": ["book title 1", ...]  // Only if verdict is FAIL
}}"""

    try:
        response = judge_llm.invoke([{"role": "user", "content": prompt}])

        # Parse JSON response (handle markdown code blocks)
        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        judge_result = json.loads(content)

        passed = judge_result.get("verdict") == "PASS"
        return {
            "passed": passed,
            "verdict": judge_result.get("verdict", "UNKNOWN"),
            "reasoning": judge_result.get("reasoning", ""),
            "violating_books": judge_result.get("violating_books", []),
        }

    except Exception as e:
        # Judge failure should FAIL the test, not skip it
        return {
            "passed": False,
            "verdict": "ERROR",
            "reasoning": f"LLM judge failed: {str(e)}",
        }


# ============================================================================
# NEGATIVE CONSTRAINT EVALUATION
# ============================================================================


def evaluate_negative_constraints(
    retrieval_output, final_response, test_case: Dict, db, judge_llm
) -> Dict[str, Any]:
    """
    Evaluate handling of negative constraints across retrieval and curation.

    Checks:
    1. Retrieval doesn't include negative terms in search queries
    2. Curation filters out books matching negative constraints (LLM judge)
    """
    results = {"checks": {}, "all_passed": True}

    negative_terms = test_case.get("negative_constraints", [])
    if not negative_terms:
        return results

    # Check 1: Retrieval should NOT include negative terms in queries
    tool_executions = retrieval_output.tool_executions
    for exec in tool_executions:
        if exec.tool_name in ["book_semantic_search", "subject_id_search"]:
            query_used = exec.arguments.get("query", "")
            phrases_used = exec.arguments.get("phrases", [])

            # Check if negative terms appear in query or phrases
            query_text = query_used if isinstance(query_used, str) else " ".join(phrases_used)
            contains_negative = any(term.lower() in query_text.lower() for term in negative_terms)

            results["checks"]["retrieval_excludes_constraints"] = {
                "expected": "negative terms NOT in query",
                "actual": f"query='{query_text[:100]}'",
                "passed": not contains_negative,
                "details": {
                    "tool": exec.tool_name,
                    "negative_terms": negative_terms,
                    "contains_negative": contains_negative,
                },
            }

            if contains_negative:
                results["all_passed"] = False

    # Check 2: Curation should filter out books matching constraints (LLM judge)
    expected_curation = test_case.get("expected_curation", {})
    if expected_curation.get("llm_judge_needed"):
        final_books = final_response.book_recommendations

        if len(final_books) > 0:
            judge_result = llm_judge_negative_constraints(
                final_books, negative_terms, db, judge_llm
            )

            results["checks"]["curation_filters_constraints"] = {
                "expected": "no books matching negative constraints",
                "actual": judge_result["verdict"],
                "passed": judge_result["passed"],
                "details": {
                    "reasoning": judge_result["reasoning"],
                    "violating_books": judge_result.get("violating_books", []),
                },
            }

            if not judge_result["passed"]:
                results["all_passed"] = False
        else:
            # No books returned - can't judge, but mark as passed (no violations)
            results["checks"]["curation_filters_constraints"] = {
                "expected": "no books matching negative constraints",
                "actual": "no books returned",
                "passed": True,
            }

    return results


def run_negative_constraint_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single negative constraint test case.

    Tests both retrieval and curation handling of negative constraints.
    Uses test_data_factory for shuffled mixed candidates.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "negative_constraints",
        "agent_success": False,
        "overall_pass": False,
    }

    # Validate query is not empty
    if not query or not query.strip():
        result["error"] = "Empty or whitespace-only query"
        result["evaluation"] = {
            "checks": {
                "query_validation": {
                    "expected": "non-empty query",
                    "actual": f"'{query}'",
                    "passed": False,
                }
            },
            "all_passed": False,
        }
        return result

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # Step 1: Run planner
        planner = PlannerAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            has_als_recs_available=user_state["is_warm"],
            allow_profile=user_state["allow_profile"],
        )

        available_tools = [
            "book_semantic_search",
            "subject_hybrid_pool",
            "subject_id_search",
            "popular_books",
        ]
        if user_state["is_warm"]:
            available_tools.insert(0, "als_recs")

        planner_input = PlannerInput(
            query=query,
            has_als_recs_available=user_state["is_warm"],
            allow_profile=user_state["allow_profile"],
            available_retrieval_tools=available_tools,
        )

        strategy = planner.execute(planner_input)

        # Get judge LLM (reuse planner's LLM for judging)
        judge_llm = planner.llm

        # Step 2: Run retrieval
        retrieval = RetrievalAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            has_als_recs_available=user_state["is_warm"],
        )

        retrieval_input = RetrievalInput(
            query=query, strategy=strategy, profile_data=strategy.profile_data
        )

        retrieval_output = retrieval.execute(retrieval_input)

        # Step 3: Run full pipeline for curation
        orchestrator = RecommendationAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            warm_threshold=10,
            allow_profile=user_state["allow_profile"],
        )

        request = AgentRequest(user_text=query, conversation_history=[])
        final_response = orchestrator.execute(request)

        result["agent_success"] = final_response.success
        result["pipeline"] = {
            "candidate_count": len(retrieval_output.candidates),
            "final_book_count": len(final_response.book_recommendations),
            "tools_used": retrieval_output.execution_context.tools_used,
        }

        # Evaluate negative constraints (pass judge_llm)
        eval_result = evaluate_negative_constraints(
            retrieval_output, final_response, test_case, db, judge_llm
        )
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"] and final_response.success

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# GENRE MATCHING TESTS
# ============================================================================


def run_genre_matching_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single genre matching test case using isolated curation testing.

    Tests that curation filters out wrong-genre books from mixed candidate pool.
    Uses test data factory for candidates and LLM-as-judge for validation.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    test_scenario = test_case["test_scenario"]
    expected_genre = test_case["expected_genre"]
    expected_curation = test_case["expected_curation"]

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "curation_genre",
        "agent_success": False,
        "overall_pass": False,
    }

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # Get mixed candidates (correct genre + wrong genre, shuffled)
        candidates_data = get_candidates(test_scenario, db)

        # Get execution context (genre query uses subject search)
        context = get_execution_context("subject")

        # Run curation agent with mixed candidates
        curation = CurationAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
        )

        curation_input = CurationInput(
            query=query,
            candidates=candidates_data["books"],
            execution_context=context,
        )

        curation_output = curation.execute(curation_input)

        # Parse the response to get final books
        from app.agents.domain.entities import BookRecommendation

        final_books = (
            curation_output.book_recommendations
            if hasattr(curation_output, "book_recommendations")
            else []
        )

        result["agent_success"] = len(final_books) >= expected_curation.get("min_final_books", 3)
        result["pipeline"] = {
            "candidate_count": len(candidates_data["books"]),
            "final_book_count": len(final_books),
        }

        # Validate genre matching with LLM-as-judge
        if expected_curation.get("llm_judge_needed") and len(final_books) > 0:
            judge_result = llm_judge_genre_match(
                books=final_books, expected_genre=expected_genre, db=db, judge_llm=curation.llm
            )

            result["evaluation"] = {
                "checks": {
                    "genre_match": {
                        "expected": f"books match genre: {expected_genre}",
                        "actual": judge_result["verdict"],
                        "passed": judge_result["passed"],
                        "details": {
                            "reasoning": judge_result["reasoning"],
                            "violating_books": judge_result.get("violating_books", []),
                            "match_rate": judge_result.get("match_count", 0)
                            / judge_result.get("total_count", 1)
                            if judge_result.get("total_count", 0) > 0
                            else 0,
                        },
                    }
                },
                "all_passed": judge_result["passed"],
            }
            result["overall_pass"] = judge_result["passed"] and result["agent_success"]
        else:
            result["evaluation"] = {
                "checks": {
                    "min_books": {
                        "expected": f">= {expected_curation.get('min_final_books', 3)} books",
                        "actual": len(final_books),
                        "passed": len(final_books) >= expected_curation.get("min_final_books", 3),
                    }
                },
                "all_passed": result["agent_success"],
            }
            result["overall_pass"] = result["agent_success"]

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


def run_personalization_prose_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single personalization prose test case using isolated curation testing.

    Tests that curation prose correctly reflects personalization context.
    Uses test data factory for candidates/context and LLM-as-judge for validation.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    test_scenario = test_case["test_scenario"]
    context_scenario = test_case["context_scenario"]
    expected_prose = test_case["expected_prose"]

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "curation_prose",
        "agent_success": False,
        "overall_pass": False,
    }

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # Get candidates based on test scenario
        if test_scenario == "als":
            candidates_data = get_candidates("als", db, user_id=user.user_id)
        elif test_scenario == "subject":
            # Use user's actual favorite subjects if available
            candidates_data = get_candidates("subject", db, subject_ids=[978, 1066, 2317])
        else:
            candidates_data = get_candidates("basic", db, query=query)

        # Get execution context (tells curation what personalization was used)
        context = get_execution_context(context_scenario)

        # Run curation agent
        curation = CurationAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
        )

        curation_input = CurationInput(
            query=query,
            candidates=candidates_data["books"],
            execution_context=context,
        )

        curation_output = curation.execute(curation_input)

        # Get response text and books
        response_text = (
            curation_output.response_text if hasattr(curation_output, "response_text") else ""
        )
        final_books = (
            curation_output.book_recommendations
            if hasattr(curation_output, "book_recommendations")
            else []
        )

        result["agent_success"] = len(response_text) > 50 and len(final_books) >= 3
        result["pipeline"] = {
            "candidate_count": len(candidates_data["books"]),
            "final_book_count": len(final_books),
            "response_length": len(response_text),
        }

        # Validate prose with LLM-as-judge
        if expected_prose.get("llm_judge_needed") and len(response_text) > 0:
            # Check if this is a negative test (should NOT claim personalization)
            expect_no_personalization = expected_prose.get(
                "should_NOT_claim_personalization", False
            )

            judge_result = llm_judge_personalization_prose(
                response_text=response_text,
                execution_context=context,
                judge_llm=curation.llm,
                expect_no_personalization=expect_no_personalization,
            )

            result["evaluation"] = {
                "checks": {
                    "prose_validation": {
                        "expected": "prose correctly reflects personalization context",
                        "actual": judge_result["verdict"],
                        "passed": judge_result["passed"],
                        "details": {
                            "reasoning": judge_result["reasoning"],
                            "issues": judge_result.get("issues", []),
                        },
                    }
                },
                "all_passed": judge_result["passed"],
            }
            result["overall_pass"] = judge_result["passed"] and result["agent_success"]
        else:
            result["evaluation"] = {
                "checks": {
                    "basic_structure": {
                        "expected": "prose and books present",
                        "actual": f"{len(response_text)} chars, {len(final_books)} books",
                        "passed": result["agent_success"],
                    }
                },
                "all_passed": result["agent_success"],
            }
            result["overall_pass"] = result["agent_success"]

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# PART 1: PLANNER EVALUATION (18 tests)
# ============================================================================


def evaluate_planner_strategy(
    strategy, expected_tools: Dict[str, Any], query_type: str
) -> Dict[str, Any]:
    """
    Evaluate planner's tool selection and reasoning.

    Checks:
    - Recommended tools match query type
    - Fallback tools are appropriate
    - Reasoning is coherent
    """
    results = {"checks": {}, "all_passed": True}

    recommended = strategy.recommended_tools
    fallback = strategy.fallback_tools

    # Check for vague queries
    if query_type == "vague":
        # Should recommend personalization tools for warm users
        if expected_tools.get("user_is_warm"):
            if "als_recs" not in recommended:
                results["checks"]["warm_user_strategy"] = {
                    "expected": "als_recs in recommended",
                    "actual": recommended,
                    "passed": False,
                }
                results["all_passed"] = False
            else:
                results["checks"]["warm_user_strategy"] = {
                    "expected": "als_recs in recommended",
                    "actual": recommended,
                    "passed": True,
                }

        # Cold users should use subject_hybrid_pool or popular_books
        elif expected_tools.get("has_profile"):
            uses_subject_or_popular = (
                "subject_hybrid_pool" in recommended or "popular_books" in recommended
            )
            results["checks"]["cold_user_strategy"] = {
                "expected": "subject_hybrid_pool or popular_books",
                "actual": recommended,
                "passed": uses_subject_or_popular,
            }
            if not uses_subject_or_popular:
                results["all_passed"] = False

    # Check for descriptive queries
    elif query_type == "descriptive":
        if "book_semantic_search" not in recommended:
            results["checks"]["descriptive_strategy"] = {
                "expected": "book_semantic_search in recommended",
                "actual": recommended,
                "passed": False,
            }
            results["all_passed"] = False
        else:
            results["checks"]["descriptive_strategy"] = {
                "expected": "book_semantic_search in recommended",
                "actual": recommended,
                "passed": True,
            }

    # Check for genre queries
    elif query_type == "genre":
        uses_subject_tools = (
            "subject_id_search" in recommended or "subject_hybrid_pool" in recommended
        )
        results["checks"]["genre_strategy"] = {
            "expected": "subject tools (id_search or hybrid_pool)",
            "actual": recommended,
            "passed": uses_subject_tools,
        }
        if not uses_subject_tools:
            results["all_passed"] = False

    # Check fallback tools exist
    if not fallback or len(fallback) == 0:
        results["checks"]["has_fallback"] = {
            "expected": "at least 1 fallback tool",
            "actual": len(fallback),
            "passed": False,
        }
        results["all_passed"] = False
    else:
        results["checks"]["has_fallback"] = {
            "expected": "at least 1 fallback tool",
            "actual": len(fallback),
            "passed": True,
        }

    # Check reasoning exists and is non-empty
    if not strategy.reasoning or len(strategy.reasoning.strip()) < 10:
        results["checks"]["has_reasoning"] = {
            "expected": "reasoning string > 10 chars",
            "actual": len(strategy.reasoning) if strategy.reasoning else 0,
            "passed": False,
        }
        results["all_passed"] = False
    else:
        results["checks"]["has_reasoning"] = {
            "expected": "reasoning string > 10 chars",
            "actual": len(strategy.reasoning),
            "passed": True,
        }

    return results


def run_planner_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single planner test case.

    Creates planner agent and validates strategy output.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    expected_tools = test_case.get("expected_tools", {})

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "planner",
        "agent_success": False,
        "overall_pass": False,
    }

    # Validate query is not empty
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        result["error"] = error_msg
        result["evaluation"] = {
            "checks": {
                "query_validation": {
                    "expected": "non-empty query",
                    "actual": f"'{query}'",
                    "passed": False,
                }
            },
            "all_passed": False,
        }
        return result

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # Create planner
        planner = PlannerAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            has_als_recs_available=user_state["is_warm"],
            allow_profile=user_state["allow_profile"],
        )

        # Build input
        available_tools = [
            "book_semantic_search",
            "subject_hybrid_pool",
            "subject_id_search",
            "popular_books",
        ]
        if user_state["is_warm"]:
            available_tools.insert(0, "als_recs")

        planner_input = PlannerInput(
            query=query,
            has_als_recs_available=user_state["is_warm"],
            allow_profile=user_state["allow_profile"],
            available_retrieval_tools=available_tools,
        )

        # Execute planner
        strategy = planner.execute(planner_input)

        result["agent_success"] = True
        result["strategy"] = {
            "recommended_tools": strategy.recommended_tools,
            "fallback_tools": strategy.fallback_tools,
            "reasoning": strategy.reasoning[:150] + "..."
            if len(strategy.reasoning) > 150
            else strategy.reasoning,
            "has_profile_data": strategy.profile_data is not None,
        }

        # Determine query type for validation
        query_lower = query.lower()
        if any(
            word in query_lower
            for word in ["dark", "cozy", "atmospheric", "heartwarming", "gothic"]
        ):
            query_type = "descriptive"
        elif any(
            word in query_lower for word in ["fiction", "mystery", "fantasy", "romance", "thriller"]
        ):
            query_type = "genre"
        else:
            query_type = "vague"

        # Add query type expectation
        expected_tools["user_is_warm"] = user_state["is_warm"]
        expected_tools["has_profile"] = user_state.get("allow_profile", False)

        # Evaluate strategy
        eval_result = evaluate_planner_strategy(strategy, expected_tools, query_type)
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"]

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# PART 2: RETRIEVAL EVALUATION (15 tests)
# ============================================================================


def evaluate_retrieval_execution(
    retrieval_output, strategy, expected_checks: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate retrieval agent's execution.

    Checks:
    - Followed planner's recommended tools
    - Called tools with correct arguments (CRITICAL for subject_hybrid_pool)
    - Gathered sufficient candidates
    - Used fallback appropriately
    """
    results = {"checks": {}, "all_passed": True}

    tools_used = retrieval_output.execution_context.tools_used
    candidates = retrieval_output.candidates

    # Build tool execution lookup for argument validation
    tool_exec_map = {}
    if retrieval_output.tool_executions:
        for exec in retrieval_output.tool_executions:
            tool_exec_map[exec.tool_name] = exec

    # Check: Used recommended tools from strategy
    if expected_checks.get("should_follow_strategy"):
        used_recommended = any(tool in tools_used for tool in strategy.recommended_tools)
        results["checks"]["followed_strategy"] = {
            "expected": f"use one of {strategy.recommended_tools}",
            "actual": tools_used,
            "passed": used_recommended,
        }
        if not used_recommended:
            results["all_passed"] = False

    # Check: Gathered sufficient candidates
    min_candidates = expected_checks.get("min_candidates", 20)
    has_enough = len(candidates) >= min_candidates
    results["checks"]["candidate_count"] = {
        "expected": f">= {min_candidates}",
        "actual": len(candidates),
        "passed": has_enough,
    }
    if not has_enough:
        results["all_passed"] = False

    # Check: Used appropriate tools for query type
    if expected_checks.get("should_use_semantic_search"):
        used_semantic = "book_semantic_search" in tools_used
        results["checks"]["semantic_search_used"] = {
            "expected": True,
            "actual": used_semantic,
            "passed": used_semantic,
        }
        if not used_semantic:
            results["all_passed"] = False

    # Check: No ALS for cold users
    if expected_checks.get("should_not_use_als"):
        used_als = "als_recs" in tools_used
        results["checks"]["als_not_used"] = {
            "expected": False,
            "actual": used_als,
            "passed": not used_als,
        }
        if used_als:
            results["all_passed"] = False

    # Check: Subject tools for genre queries
    if expected_checks.get("should_use_subject_tools"):
        used_subject = any(
            tool in tools_used for tool in ["subject_id_search", "subject_hybrid_pool"]
        )
        results["checks"]["subject_tools_used"] = {
            "expected": True,
            "actual": used_subject,
            "passed": used_subject,
        }
        if not used_subject:
            results["all_passed"] = False

    # ========================================================================
    # CRITICAL BUG TEST: subject_hybrid_pool argument validation
    # ========================================================================

    # Get query type and user subjects from expected_checks
    query_type = expected_checks.get("query_type", "unknown")
    user_favorite_subjects = expected_checks.get("user_favorite_subjects", [])
    has_favorite_subjects = len(user_favorite_subjects) > 0

    # Check if subject_id_search was used (indicates genre query with explicit subject resolution)
    used_subject_id_search = "subject_id_search" in tools_used

    # BUG TEST: subject_hybrid_pool usage and argument validation
    if "subject_hybrid_pool" in tools_used:
        # Get the actual tool execution with arguments
        subject_hybrid_exec = tool_exec_map.get("subject_hybrid_pool")

        if subject_hybrid_exec:
            args = subject_hybrid_exec.arguments
            fav_subjects_arg = args.get("fav_subjects_idxs")

            # Case 1: Genre query - MUST use subject_id_search first
            if query_type == "genre":
                if not used_subject_id_search:
                    results["checks"]["genre_query_missing_subject_resolution"] = {
                        "expected": "subject_id_search before subject_hybrid_pool",
                        "actual": "subject_hybrid_pool without subject_id_search",
                        "passed": False,
                        "severity": "CRITICAL BUG",
                        "details": {
                            "query_type": "genre",
                            "fav_subjects_arg": fav_subjects_arg,
                            "issue": (
                                f"BUG: Genre query called subject_hybrid_pool(fav_subjects_idxs={fav_subjects_arg}) "
                                f"without subject_id_search. Likely relying on auto-fetch which won't match "
                                f"the requested genre."
                            ),
                        },
                    }
                    results["all_passed"] = False

                # Additional check: If no subject_id_search, fav_subjects_idxs should be None (bug)
                # This catches the specific bug where LLM doesn't pass subjects for genre queries
                if not used_subject_id_search and fav_subjects_arg is None:
                    results["checks"]["genre_query_no_explicit_subjects"] = {
                        "expected": "fav_subjects_idxs=[resolved_ids] from subject_id_search",
                        "actual": "fav_subjects_idxs=None (auto-fetch)",
                        "passed": False,
                        "severity": "CRITICAL BUG",
                        "details": {
                            "issue": (
                                "BUG: LLM called subject_hybrid_pool() with NO fav_subjects_idxs for genre query. "
                                "Tool will auto-fetch user's profile subjects instead of using genre subjects."
                            )
                        },
                    }
                    results["all_passed"] = False

            # Case 2: Vague query + NO user subjects - should use popular_books instead
            elif query_type == "vague" and not has_favorite_subjects:
                results["checks"]["vague_query_no_subjects_bug"] = {
                    "expected": "popular_books (not subject_hybrid_pool)",
                    "actual": f"subject_hybrid_pool(fav_subjects_idxs={fav_subjects_arg})",
                    "passed": False,
                    "severity": "CRITICAL BUG",
                    "details": {
                        "query_type": "vague",
                        "user_has_subjects": False,
                        "fav_subjects_arg": fav_subjects_arg,
                        "issue": (
                            "BUG: Used subject_hybrid_pool for vague query when user has no subjects. "
                            "Tool will auto-fetch nothing and fall back to popular_books internally. "
                            "Should call popular_books directly."
                        ),
                    },
                }
                results["all_passed"] = False

            # Case 3: Vague query + user HAS subjects - OK (auto-fetch works)
            # No validation needed - both with and without args are correct

        else:
            # Tool was used but we don't have execution details (shouldn't happen)
            results["checks"]["tool_execution_missing"] = {
                "expected": "tool execution details available",
                "actual": "tool_executions not populated",
                "passed": False,
                "details": "Cannot validate arguments - tool_executions not available",
            }
            results["all_passed"] = False

    return results


def run_retrieval_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single retrieval test case using mock strategy (no planner call).

    Tests retrieval agent's tool execution and argument validation in isolation.
    """
    from test_data_factory import get_mock_strategy

    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    expected_output = test_case.get("expected_output", {})

    # Known test user subject indexes
    USER_SUBJECTS = {
        278859: [978, 1066, 2317, 3248],  # Crime, Detective, Mystery, Thriller
        278857: [115, 1378],  # Adventure, Fantasy
        278867: [],  # No subjects
    }

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "retrieval",
        "agent_success": False,
        "overall_pass": False,
    }

    # Validate query is not empty
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        result["error"] = error_msg
        result["evaluation"] = {
            "checks": {
                "query_validation": {
                    "expected": "non-empty query",
                    "actual": f"'{query}'",
                    "passed": False,
                }
            },
            "all_passed": False,
        }
        return result

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # Get mock strategy based on query type (NO planner call)
        query_lower = query.lower()
        genre_keywords = [
            "fantasy",
            "mystery",
            "thriller",
            "romance",
            "fiction",
            "crime",
            "detective",
            "adventure",
            "science fiction",
            "historical",
        ]
        is_genre_query = any(keyword in query_lower for keyword in genre_keywords)
        is_vague_query = any(
            word in query_lower for word in ["recommend", "suggest", "what should", "good book"]
        )

        # Determine appropriate mock strategy
        if is_genre_query and not is_vague_query:
            # Genre query - use subject search strategy
            strategy = get_mock_strategy("subject")
        elif "dark" in query_lower or "atmospheric" in query_lower or "cozy" in query_lower:
            # Descriptive query - use semantic search
            strategy = get_mock_strategy("semantic")
        elif user_state["is_warm"] and is_vague_query:
            # Warm user vague query - use ALS
            strategy = get_mock_strategy("als")
        elif not user_state["is_warm"] and user_state["allow_profile"]:
            # Cold user with profile - use profile strategy
            strategy = get_mock_strategy(
                "profile",
                profile_data={
                    "user_profile": {
                        "favorite_subjects": USER_SUBJECTS.get(user.user_id, []),
                        "favorite_genres": ["mystery", "thriller"],
                    }
                },
            )
        else:
            # Default to semantic
            strategy = get_mock_strategy("semantic")

        # Run retrieval with mock strategy
        retrieval = RetrievalAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            has_als_recs_available=user_state["is_warm"],
        )

        retrieval_input = RetrievalInput(
            query=query, strategy=strategy, profile_data=strategy.profile_data
        )

        retrieval_output = retrieval.execute(retrieval_input)

        result["agent_success"] = True
        result["retrieval"] = {
            "candidate_count": len(retrieval_output.candidates),
            "tools_used": retrieval_output.execution_context.tools_used,
            "reasoning": retrieval_output.reasoning[:100] + "...",
        }

        # Classify query type for argument validation
        user_id = test_case["user_id"]
        user_favorite_subjects = USER_SUBJECTS.get(user_id, [])

        # Build expected checks
        expected_checks = {
            "should_follow_strategy": True,
            "min_candidates": expected_output.get("min_candidates", 20),
            "query_type": "genre" if is_genre_query and not is_vague_query else "vague",
            "user_favorite_subjects": user_favorite_subjects,
        }

        # Add specific checks based on test expectations
        if test_case.get("expected_tools", {}).get("should_use_semantic_search"):
            expected_checks["should_use_semantic_search"] = True
        if test_case.get("expected_tools", {}).get("should_not_use_als"):
            expected_checks["should_not_use_als"] = True
        if test_case.get("expected_tools", {}).get("should_use_subject_search"):
            expected_checks["should_use_subject_tools"] = True

        # Evaluate retrieval
        eval_result = evaluate_retrieval_execution(retrieval_output, strategy, expected_checks)
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"]

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# PART 3: CURATION EVALUATION# ============================================================================
# PART 3: CURATION EVALUATION (10 tests)
# ============================================================================


def evaluate_curation_output(
    final_response, candidates: List[BookRecommendation]
) -> Dict[str, Any]:
    """
    Evaluate curation agent's output quality.

    Checks:
    - Valid JSON structure
    - Required fields present
    - Inline book references valid
    - Book ordering preserved
    """
    results = {"checks": {}, "all_passed": True}

    # Check: Has response text
    has_text = bool(final_response.text and len(final_response.text.strip()) > 50)
    results["checks"]["has_response_text"] = {
        "expected": "> 50 chars",
        "actual": len(final_response.text) if final_response.text else 0,
        "passed": has_text,
    }
    if not has_text:
        results["all_passed"] = False

    # Check: Has book recommendations
    has_books = len(final_response.book_recommendations) >= 3
    results["checks"]["has_books"] = {
        "expected": ">= 3 books",
        "actual": len(final_response.book_recommendations),
        "passed": has_books,
    }
    if not has_books:
        results["all_passed"] = False

    # Check: Book IDs are valid integers
    all_valid_ids = all(
        hasattr(book, "item_idx") and isinstance(book.item_idx, int)
        for book in final_response.book_recommendations
    )
    results["checks"]["valid_book_ids"] = {
        "expected": "all integers",
        "actual": "valid" if all_valid_ids else "invalid",
        "passed": all_valid_ids,
    }
    if not all_valid_ids:
        results["all_passed"] = False

    # Check: Inline references are valid
    if final_response.text:
        inline_errors, inline_warnings = InlineReferenceParser.validate_references(
            text=final_response.text, book_recommendations=final_response.book_recommendations
        )

        has_inline_errors = len(inline_errors) > 0
        results["checks"]["inline_references_valid"] = {
            "expected": "no errors",
            "actual": f"{len(inline_errors)} errors, {len(inline_warnings)} warnings",
            "passed": not has_inline_errors,
            "details": {"errors": inline_errors, "warnings": inline_warnings},
        }
        if has_inline_errors:
            results["all_passed"] = False

    return results


def run_curation_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single curation test case using isolated testing (no full pipeline).

    Tests curation agent's JSON structure and inline reference validation.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "curation",
        "agent_success": False,
        "overall_pass": False,
    }

    # Validate query is not empty
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        result["error"] = error_msg
        result["evaluation"] = {
            "checks": {
                "query_validation": {
                    "expected": "non-empty query",
                    "actual": f"'{query}'",
                    "passed": False,
                }
            },
            "all_passed": False,
        }
        return result

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # Get candidates and context from factory (NO full pipeline)
        candidates_data = get_candidates("basic", db, query=query)
        context = get_execution_context("semantic")

        # Run curation agent in isolation
        curation = CurationAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
        )

        curation_input = CurationInput(
            query=query,
            candidates=candidates_data["books"],
            execution_context=context,
        )

        curation_output = curation.execute(curation_input)

        # Get final response
        final_books = (
            curation_output.book_recommendations
            if hasattr(curation_output, "book_recommendations")
            else []
        )
        response_text = (
            curation_output.response_text if hasattr(curation_output, "response_text") else ""
        )

        result["agent_success"] = len(final_books) >= 3 and len(response_text) > 50
        result["curation"] = {
            "candidate_count": len(candidates_data["books"]),
            "book_count": len(final_books),
            "text_length": len(response_text),
            "has_inline_refs": "<book id=" in response_text,
        }

        # Create a mock final_response object for evaluation
        from dataclasses import dataclass

        @dataclass
        class MockResponse:
            success: bool
            text: str
            book_recommendations: list

        mock_response = MockResponse(
            success=True, text=response_text, book_recommendations=final_books
        )

        # Evaluate curation output
        eval_result = evaluate_curation_output(mock_response, final_books)
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"] and result["agent_success"]

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# PART 4: INTEGRATION EVALUATION (8 tests)
# ============================================================================


def evaluate_full_pipeline(
    final_response, expected_behavior: Dict[str, Any], execution_context=None
) -> Dict[str, Any]:
    """
    Evaluate full 3-stage pipeline integration.

    Checks:
    - All stages completed successfully
    - Data flowed correctly between stages
    - Final output meets quality standards
    - Tool usage matches expectations (for high-impact tests)
    """
    results = {"checks": {}, "all_passed": True}

    # Check: Agent succeeded
    if not final_response.success:
        results["checks"]["agent_success"] = {"expected": True, "actual": False, "passed": False}
        results["all_passed"] = False
        return results

    # Check: Has prose
    if expected_behavior.get("final_response_should_have_prose"):
        has_prose = bool(final_response.text and len(final_response.text.strip()) > 20)
        results["checks"]["has_prose"] = {
            "expected": True,
            "actual": has_prose,
            "text_length": len(final_response.text) if final_response.text else 0,
            "passed": has_prose,
        }
        if not has_prose:
            results["all_passed"] = False

    # Check: Has books
    if expected_behavior.get("final_response_should_have_books"):
        has_books = len(final_response.book_recommendations) >= 3
        results["checks"]["has_books"] = {
            "expected": ">= 3 books",
            "actual": len(final_response.book_recommendations),
            "passed": has_books,
        }
        if not has_books:
            results["all_passed"] = False

    # Check: Graceful error handling for edge cases
    if expected_behavior.get("should_handle_gracefully"):
        # If we got here without exception, it handled gracefully
        results["checks"]["graceful_handling"] = {
            "expected": "no crash",
            "actual": "handled",
            "passed": True,
        }

    # Check: Fallback handling for no candidates
    if expected_behavior.get("should_return_error_or_fallback"):
        # Should either have books or have an informative error message
        has_books = len(final_response.book_recommendations) > 0
        has_error_msg = final_response.text and len(final_response.text) > 20
        handled = has_books or has_error_msg

        results["checks"]["fallback_handling"] = {
            "expected": "books or error message",
            "actual": f"books={has_books}, msg={has_error_msg}",
            "passed": handled,
        }
        if not handled:
            results["all_passed"] = False

    # High-impact integration checks (require execution_context)
    if execution_context:
        tools_used = (
            execution_context.tools_used if hasattr(execution_context, "tools_used") else []
        )

        # Check: Genre queries should use subject search
        if expected_behavior.get("should_use_subject_search_for_fantasy"):
            uses_subject = any(
                tool in tools_used for tool in ["subject_id_search", "subject_hybrid_pool"]
            )
            results["checks"]["genre_uses_subject_search"] = {
                "expected": "subject_id_search or subject_hybrid_pool",
                "actual": tools_used,
                "passed": uses_subject,
            }
            if not uses_subject:
                results["all_passed"] = False

        # Check: Genre queries should NOT use personalization
        if expected_behavior.get("should_NOT_use_als_or_profile"):
            uses_personalization = "als_recs" in tools_used
            results["checks"]["no_personalization_for_genre"] = {
                "expected": "NOT als_recs",
                "actual": tools_used,
                "passed": not uses_personalization,
            }
            if uses_personalization:
                results["all_passed"] = False

        # Check: Must use subject_id_search
        if expected_behavior.get("must_use_subject_id_search"):
            uses_subject_id = "subject_id_search" in tools_used
            results["checks"]["uses_subject_id_search"] = {
                "expected": "subject_id_search",
                "actual": tools_used,
                "passed": uses_subject_id,
            }
            if not uses_subject_id:
                results["all_passed"] = False

        # Check: Should NOT use popular_books
        if expected_behavior.get("should_NOT_use_popular_books"):
            uses_popular = "popular_books" in tools_used
            results["checks"]["no_popular_books"] = {
                "expected": "NOT popular_books",
                "actual": tools_used,
                "passed": not uses_popular,
            }
            if uses_popular:
                results["all_passed"] = False

    return results


def run_integration_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single integration test case.

    Tests the full 3-stage pipeline end-to-end.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    expected_behavior = test_case.get("expected_behavior", {})

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "integration",
        "agent_success": False,
        "overall_pass": False,
    }

    # Validate query is not empty (unless it's explicitly testing empty query)
    if name != "empty_query_e2e":
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            result["error"] = error_msg
            result["evaluation"] = {
                "checks": {
                    "query_validation": {
                        "expected": "non-empty query",
                        "actual": f"'{query}'",
                        "passed": False,
                    }
                },
                "all_passed": False,
            }
            return result

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        # For high-impact tests, run planner + retrieval separately to get execution_context
        execution_context = None
        if any(
            expected_behavior.get(key)
            for key in [
                "should_use_subject_search_for_fantasy",
                "should_NOT_use_als_or_profile",
                "must_use_subject_id_search",
                "should_NOT_use_popular_books",
            ]
        ):
            # Run planner
            planner = PlannerAgent(
                current_user=user,
                db=db,
                user_num_ratings=rating_count,
                has_als_recs_available=user_state["is_warm"],
                allow_profile=user_state["allow_profile"],
            )

            available_tools = [
                "book_semantic_search",
                "subject_hybrid_pool",
                "subject_id_search",
                "popular_books",
            ]
            if user_state["is_warm"]:
                available_tools.insert(0, "als_recs")

            planner_input = PlannerInput(
                query=query,
                has_als_recs_available=user_state["is_warm"],
                allow_profile=user_state["allow_profile"],
                available_retrieval_tools=available_tools,
            )

            strategy = planner.execute(planner_input)

            # Run retrieval
            retrieval = RetrievalAgent(
                current_user=user,
                db=db,
                user_num_ratings=rating_count,
                has_als_recs_available=user_state["is_warm"],
            )

            retrieval_input = RetrievalInput(
                query=query, strategy=strategy, profile_data=strategy.profile_data
            )

            retrieval_output = retrieval.execute(retrieval_input)
            execution_context = retrieval_output.execution_context

        # Run full pipeline
        orchestrator = RecommendationAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            warm_threshold=10,
            allow_profile=user_state["allow_profile"],
        )

        request = AgentRequest(user_text=query, conversation_history=[])

        final_response = orchestrator.execute(request)

        result["agent_success"] = final_response.success
        result["pipeline"] = {
            "success": final_response.success,
            "book_count": len(final_response.book_recommendations),
            "text_length": len(final_response.text) if final_response.text else 0,
        }

        if execution_context:
            result["pipeline"]["tools_used"] = execution_context.tools_used

        # Evaluate integration
        eval_result = evaluate_full_pipeline(final_response, expected_behavior, execution_context)
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"]

    except Exception as e:
        # For edge cases, not crashing is a pass
        if expected_behavior.get("should_handle_gracefully"):
            result["evaluation"] = {
                "checks": {
                    "no_crash": {
                        "expected": "should not crash",
                        "actual": f"caught exception: {str(e)[:50]}",
                        "passed": False,
                    }
                },
                "all_passed": False,
            }
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================


def evaluate_all(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Run all evaluation types across all test cases.

    Returns:
        Aggregated results with statistics
    """
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")

    db = SessionLocal()

    try:
        all_results = []
        category_stats = {}

        # Map test categories to evaluation functions
        category_handlers = {
            "tool_selection_warm_user": ("planner", run_planner_test),
            "tool_selection_cold_user": ("planner", run_planner_test),
            "two_stage_integration": ("integration", run_integration_test),
            "integration_high_impact": ("integration", run_integration_test),
            "edge_cases": ("integration", run_integration_test),
            "negative_constraints": ("negative_constraints", run_negative_constraint_test),
            "curation_critical": ("curation", run_curation_test),
            "curation_genre_matching": ("curation_genre", run_genre_matching_test),
            "curation_personalization_prose": ("curation_prose", run_personalization_prose_test),
            "curation_false_personalization": ("curation_prose", run_personalization_prose_test),
        }

        for category, cases in test_cases.items():
            print(f"\n{'=' * 70}")
            print(f"Running {category}: {len(cases)} test cases")
            print("=" * 70)

            # Determine handler
            if category in category_handlers:
                eval_type, handler = category_handlers[category]
            else:
                # Default to retrieval tests
                eval_type, handler = ("retrieval", run_retrieval_test)

            category_results = []
            for i, test_case in enumerate(cases, 1):
                name = test_case.get("name", f"test_{i}")
                print(f"\n[{i}/{len(cases)}] {name}...", end=" ")

                result = handler(test_case, db)
                category_results.append(result)

                status = "✅ PASS" if result["overall_pass"] else "❌ FAIL"
                print(status)

            all_results.extend(category_results)

            # Category stats
            passed = sum(1 for r in category_results if r["overall_pass"])
            category_stats[category] = {
                "passed": passed,
                "total": len(category_results),
                "pass_rate": passed / len(category_results) if category_results else 0,
                "eval_type": eval_type,
            }

        # Overall stats by eval type
        eval_type_stats = {}
        for eval_type in [
            "planner",
            "retrieval",
            "curation",
            "curation_genre",
            "curation_prose",
            "negative_constraints",
            "integration",
        ]:
            type_results = [r for r in all_results if r.get("test_type") == eval_type]
            if type_results:
                passed = sum(1 for r in type_results if r["overall_pass"])
                eval_type_stats[eval_type] = {
                    "passed": passed,
                    "total": len(type_results),
                    "pass_rate": passed / len(type_results),
                }

        # Overall stats
        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)

        return {
            "results": all_results,
            "category_stats": category_stats,
            "eval_type_stats": eval_type_stats,
            "overall": {
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        db.close()


def print_results(eval_results: Dict[str, Any]):
    """Print detailed evaluation results."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION AGENT EVALUATION RESULTS")
    print("=" * 70)

    # Overall stats
    overall = eval_results["overall"]
    print(
        f"\nOverall Pass Rate: {overall['pass_rate']:.1%} ({overall['passed']}/{overall['total']})"
    )

    # Stats by evaluation type
    if "eval_type_stats" in eval_results:
        print("\nResults by Evaluation Type:")
        for eval_type, stats in eval_results["eval_type_stats"].items():
            print(
                f"  {eval_type:15s} {stats['pass_rate']:>6.1%}  ({stats['passed']}/{stats['total']})"
            )

    # Category breakdown
    print("\nResults by Category:")
    for category, stats in eval_results["category_stats"].items():
        eval_type = stats.get("eval_type", "unknown")
        print(
            f"  {category:35s} [{eval_type:12s}] {stats['pass_rate']:>6.1%}  ({stats['passed']}/{stats['total']})"
        )

    # Failed cases detail
    failures = [r for r in eval_results["results"] if not r["overall_pass"]]
    if failures:
        print(f"\n❌ Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"\n  [{f['name']}] ({f['test_type']})")
            print(f"  Query: {f['query'][:80]}...")

            if "error" in f:
                print(f"  Error: {f['error']}")

            # Show check failures
            if "evaluation" in f and "checks" in f["evaluation"]:
                for check_name, check in f["evaluation"]["checks"].items():
                    if not check.get("passed", False):
                        print(
                            f"    ❌ {check_name}: expected {check.get('expected')}, got {check.get('actual')}"
                        )
    else:
        print("\n✅ All test cases passed!")

    print("\n" + "=" * 70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"recommendation_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    """Main evaluation function with CLI argument support."""
    parser = argparse.ArgumentParser(
        description="Recommendation Agent Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests (default)
  python evaluate_recommendation.py

  # Run only planner tests
  python evaluate_recommendation.py --categories planner

  # Run planner and retrieval tests
  python evaluate_recommendation.py --categories planner retrieval

  # Run only negative constraint tests
  python evaluate_recommendation.py --categories negative_constraints

  # List available categories
  python evaluate_recommendation.py --list-categories
        """,
    )

    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Test categories to run (space-separated). Run all if not specified.",
    )

    parser.add_argument(
        "--list-categories",
        "-l",
        action="store_true",
        help="List available test categories and exit",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("RECOMMENDATION AGENT EVALUATION")
    print("3-Stage Architecture: Planner → Retrieval → Curation")
    print("=" * 70)
    print("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    # List categories if requested
    if args.list_categories:
        print("\nAvailable test categories:")
        for category, cases in all_test_cases.items():
            print(f"  - {category}: {len(cases)} tests")
        print(
            f"\nTotal: {sum(len(c) for c in all_test_cases.values())} tests across {len(all_test_cases)} categories"
        )
        return

    # Filter categories if specified
    if args.categories:
        test_cases = {}
        invalid_categories = []

        for cat in args.categories:
            if cat in all_test_cases:
                test_cases[cat] = all_test_cases[cat]
            else:
                invalid_categories.append(cat)

        if invalid_categories:
            print(f"\n❌ ERROR: Invalid categories: {', '.join(invalid_categories)}")
            print("\nAvailable categories:")
            for category in all_test_cases.keys():
                print(f"  - {category}")
            sys.exit(1)

        total_cases = sum(len(cases) for cases in test_cases.values())
        print(f"Running {total_cases} tests from {len(test_cases)} selected categories:")
        for cat, cases in test_cases.items():
            print(f"  - {cat}: {len(cases)} tests")
    else:
        test_cases = all_test_cases
        total_cases = sum(len(cases) for cases in test_cases.values())
        print(f"Loaded {total_cases} test cases across {len(test_cases)} categories")

    print("\n📊 Testing with REAL database connection and users")
    print("   Tools will execute against actual data\n")

    print("Running evaluation...")
    eval_results = evaluate_all(test_cases)

    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
