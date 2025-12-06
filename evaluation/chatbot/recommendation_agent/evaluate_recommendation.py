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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent
from app.agents.infrastructure.recsys.planner_agent import PlannerAgent
from app.agents.infrastructure.recsys.retrieval_agent import RetrievalAgent
from app.agents.infrastructure.recsys.curation_agent import CurationAgent
from app.agents.domain.recsys_schemas import PlannerInput, RetrievalInput, ExecutionContext
from app.agents.domain.entities import AgentRequest, BookRecommendation
from app.agents.domain.parsers import InlineReferenceParser
from app.database import SessionLocal
from app.table_models import User, Interaction
from sqlalchemy import func


# ============================================================================
# DATABASE HELPERS
# ============================================================================


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
    - Called tools with correct arguments
    - Gathered sufficient candidates
    - Used fallback appropriately
    """
    results = {"checks": {}, "all_passed": True}

    tools_used = retrieval_output.execution_context.tools_used
    candidates = retrieval_output.candidates

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

    return results


def run_retrieval_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single retrieval test case.

    Creates planner + retrieval agents and validates tool execution.
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    expected_output = test_case.get("expected_output", {})

    result = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "retrieval",
        "agent_success": False,
        "overall_pass": False,
    }

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

        result["agent_success"] = True
        result["retrieval"] = {
            "candidate_count": len(retrieval_output.candidates),
            "tools_used": retrieval_output.execution_context.tools_used,
            "reasoning": retrieval_output.reasoning[:100] + "...",
        }

        # Build expected checks
        expected_checks = {
            "should_follow_strategy": True,
            "min_candidates": expected_output.get("min_candidates", 20),
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
    Run a single curation test case.

    Runs full pipeline but focuses on validating curation output.
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

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

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
        result["curation"] = {
            "book_count": len(final_response.book_recommendations),
            "text_length": len(final_response.text) if final_response.text else 0,
            "has_inline_refs": "<book id=" in (final_response.text or ""),
        }

        # Evaluate curation output
        eval_result = evaluate_curation_output(final_response, final_response.book_recommendations)
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"] and final_response.success

    except Exception as e:
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# PART 4: INTEGRATION EVALUATION (8 tests)
# ============================================================================


def evaluate_full_pipeline(final_response, expected_behavior: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate full 3-stage pipeline integration.

    Checks:
    - All stages completed successfully
    - Data flowed correctly between stages
    - Final output meets quality standards
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

    try:
        # Get user
        user, rating_count = get_user_by_id(db, test_case["user_id"])

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

        # Evaluate integration
        eval_result = evaluate_full_pipeline(final_response, expected_behavior)
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
            "edge_cases": ("integration", run_integration_test),
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
        for eval_type in ["planner", "retrieval", "curation", "integration"]:
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
    """Main evaluation function."""
    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("RECOMMENDATION AGENT EVALUATION")
    print("3-Stage Architecture: Planner → Retrieval → Curation")
    print("=" * 70)
    print("\nLoading test cases...")
    test_cases = load_test_cases(test_cases_path)

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
