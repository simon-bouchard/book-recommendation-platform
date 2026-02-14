# evaluation/chatbot/recommendation_agent/evaluate_retrieval.py
"""
Retrieval agent evaluation - validates tool execution and candidate gathering.
Tests the second stage of the recommendation pipeline.
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import json

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.agents.infrastructure.recsys.retrieval_agent import RetrievalAgent
from app.agents.domain.recsys_schemas import RetrievalInput, PlannerStrategy
from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()

# Add eval directory to path for shared helpers
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from shared_helpers import (
    load_test_cases,
    get_user_by_id,
    validate_query,
    print_results,
    save_results,
)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _validate_semantic_query_format(search_query: str, original_user_query: str) -> bool:
    """
    Validate that semantic search query uses structured format matching vector DB.

    The vector database is structured with the following fields:
        Title: ...
        Author: ...
        genre: ... (single genre per book)
        subjects: ... (can be multiple)
        tone: ... (can be multiple)
        vibe: ... (short sentence describing vibe)

    The agent should construct queries that match this structure, NOT just pass
    the raw user query. This ensures better semantic matching.

    Args:
        search_query: The query passed to book_semantic_search
        original_user_query: Original user query for comparison

    Returns:
        True if query appears to be structured, False otherwise
    """
    if not search_query or not search_query.strip():
        return False

    # If query is exactly the raw user input, it's not structured
    if search_query.strip() == original_user_query.strip():
        return False

    # Check for presence of expected field markers
    # At minimum, should have some of: Title, genre, subjects, tone, vibe
    # (Author is optional since user queries rarely mention authors)
    field_markers = ["Title:", "genre:", "subjects:", "tone:", "vibe:"]

    # Query should have at least 2 of these field markers to be considered structured
    markers_found = sum(1 for marker in field_markers if marker in search_query)

    return markers_found >= 2


# ============================================================================
# EVALUATION LOGIC
# ============================================================================


async def evaluate_retrieval_case(test: Dict[str, Any], db) -> Dict[str, Any]:
    """
    Evaluate a single retrieval test case.

    Args:
        test: Test case dictionary with query, user_id, expected checks
        db: Database session

    Returns:
        Result dictionary with evaluation details
    """
    test_name = test["name"]
    query = test["query"]
    user_id = test.get("user_id", 12)  # Default to warm user
    user_state = test.get("user_state", {})
    expected_tools = test.get("expected_tools", {})
    expected_output = test.get("expected_output", {})
    strategy_scenario = test.get("strategy_scenario", "basic")

    # Build result structure
    result = {
        "name": test_name,
        "query": query,
        "test_type": "retrieval",
        "category": test.get("category", "retrieval_strategy_adherence"),
        "overall_pass": False,
        "evaluation": {
            "checks": {},
        },
    }

    try:
        # Validate query
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            result["error"] = error_msg
            return result

        # Get user and rating count
        user, rating_count = get_user_by_id(db, user_id)
        has_als = rating_count >= 10

        # Create agent
        agent = RetrievalAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            has_als_recs_available=has_als,
        )

        # Build strategy based on strategy_scenario or infer from expected_tools
        # Map scenario to appropriate tools
        if strategy_scenario == "als_no_profile":
            recommended_tools = ["als_recs"]
            fallback_tools = ["popular_books"]
        elif strategy_scenario == "semantic":
            recommended_tools = ["book_semantic_search"]
            fallback_tools = ["popular_books"]
        elif strategy_scenario == "subject":
            recommended_tools = ["subject_hybrid_pool", "subject_id_search"]
            fallback_tools = ["popular_books"]
        else:
            # Infer strategy from expected_tools
            recommended_tools = []

            if expected_tools.get("should_use_semantic_search"):
                recommended_tools.append("book_semantic_search")

            if expected_tools.get("should_use_subject_search"):
                recommended_tools.extend(["subject_hybrid_pool", "subject_id_search"])

            # For warm users with vague queries and no explicit tool expectations
            if has_als and not recommended_tools and not expected_tools.get("should_not_use_als"):
                recommended_tools.append("als_recs")

            # For new/cold users with vague queries
            if expected_tools.get("should_recommend_popular_books"):
                recommended_tools.append("popular_books")

            # Default fallback
            fallback_tools = ["popular_books"]

            # If no tools inferred, use a reasonable default
            if not recommended_tools:
                if has_als:
                    recommended_tools = ["als_recs", "book_semantic_search"]
                else:
                    recommended_tools = ["book_semantic_search"]

        strategy = PlannerStrategy(
            recommended_tools=recommended_tools,
            fallback_tools=fallback_tools,
            reasoning=f"Test strategy for {test_name} ({strategy_scenario})",
            profile_data=None,
        )

        # Build retrieval input
        retrieval_input = RetrievalInput(
            query=query,
            strategy=strategy,
            profile_data=None,
        )

        # Execute agent (MUST AWAIT!)
        output = await agent.execute(retrieval_input)

        # Store output for checks
        result["output"] = {
            "candidate_count": len(output.candidates),
            "tools_used": [exec.tool_name for exec in output.tool_executions],
            "reasoning": output.reasoning,
        }

        # Run validation checks
        checks = {}

        # Check 1: Minimum candidate count
        if "min_candidates" in expected_output:
            min_candidates = expected_output["min_candidates"]
            actual_count = len(output.candidates)
            checks["min_candidates"] = {
                "passed": actual_count >= min_candidates,
                "expected": f">= {min_candidates}",
                "actual": actual_count,
            }

        # Check 2: Should use semantic search
        if expected_tools.get("should_use_semantic_search"):
            actual_tools = set(output.execution_context.tools_used)
            used_semantic = "book_semantic_search" in actual_tools
            checks["should_use_semantic_search"] = {
                "passed": used_semantic,
                "expected": True,
                "actual": used_semantic,
            }

        # Check 3: Should use subject search
        if expected_tools.get("should_use_subject_search"):
            actual_tools = set(output.execution_context.tools_used)
            used_subject = any(
                tool in actual_tools for tool in ["subject_hybrid_pool", "subject_id_search"]
            )
            checks["should_use_subject_search"] = {
                "passed": used_subject,
                "expected": True,
                "actual": used_subject,
            }

        # Check 4: Should use any retrieval tool
        if expected_tools.get("should_use_any_retrieval_tool"):
            actual_tools = set(output.execution_context.tools_used)
            retrieval_tools = {
                "als_recs",
                "book_semantic_search",
                "subject_hybrid_pool",
                "subject_id_search",
                "popular_books",
            }
            used_any = bool(actual_tools & retrieval_tools)
            checks["should_use_any_retrieval_tool"] = {
                "passed": used_any,
                "expected": True,
                "actual": used_any,
                "tools_used": list(actual_tools),
            }

        # Check 5: Should NOT use ALS (for cold users)
        if expected_tools.get("should_not_use_als"):
            actual_tools = set(output.execution_context.tools_used)
            used_als = "als_recs" in actual_tools
            checks["should_not_use_als"] = {
                "passed": not used_als,
                "expected": False,
                "actual": used_als,
            }

        # Check 6: Should recommend popular books
        if expected_tools.get("should_recommend_popular_books"):
            actual_tools = set(output.execution_context.tools_used)
            used_popular = "popular_books" in actual_tools
            checks["should_recommend_popular_books"] = {
                "passed": used_popular,
                "expected": True,
                "actual": used_popular,
            }

        # Check 7: Should NOT use certain tools
        if expected_tools.get("should_NOT_recommend_subject_hybrid_pool"):
            actual_tools = set(output.execution_context.tools_used)
            used_subject_hybrid = "subject_hybrid_pool" in actual_tools
            checks["should_NOT_recommend_subject_hybrid_pool"] = {
                "passed": not used_subject_hybrid,
                "expected": False,
                "actual": used_subject_hybrid,
            }

        # Check 8: Semantic search query format validation
        # ALWAYS verify that semantic search uses structured format matching vector DB
        # This is a critical best practice - queries should match the vector DB structure
        if "book_semantic_search" in output.execution_context.tools_used:
            semantic_exec = next(
                (
                    exec
                    for exec in output.tool_executions
                    if exec.tool_name == "book_semantic_search"
                ),
                None,
            )

            if semantic_exec and semantic_exec.arguments:
                search_query = semantic_exec.arguments.get("query", "")

                # Validate query is NOT just raw user input
                # Should be structured with fields: Title:, genre:, subjects:, tone:, vibe:
                query_is_structured = _validate_semantic_query_format(search_query, query)

                checks["semantic_query_format"] = {
                    "passed": query_is_structured,
                    "expected": "structured format with fields (Title, genre, subjects, tone, vibe)",
                    "actual": search_query[:150] if search_query else "no query",
                    "is_raw_user_query": search_query.strip() == query.strip(),
                }
            else:
                # Tool was used but we couldn't validate format
                checks["semantic_query_format"] = {
                    "passed": False,
                    "expected": "structured format",
                    "actual": "no arguments captured",
                }

        # Store checks
        result["evaluation"]["checks"] = checks

        # Overall pass: all checks must pass
        result["overall_pass"] = all(check.get("passed", False) for check in checks.values())

    except Exception as e:
        import traceback

        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["overall_pass"] = False

    return result


async def evaluate_retrieval_tests(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Run all retrieval evaluation tests.

    Args:
        test_cases: Dictionary mapping categories to test case lists

    Returns:
        Evaluation results with statistics
    """
    # Categories that map to retrieval stage testing
    # These all test that the retrieval agent properly gathers candidates
    RETRIEVAL_CATEGORIES = {
        "tool_selection_warm_user",  # 6 tests - tool execution for warm users
        "tool_selection_cold_user",  # 5 tests - tool execution for cold users
        "retrieval_strategy_adherence",  # 8 tests - core retrieval behavior
        "edge_cases",  # 3 tests - edge case retrieval scenarios
    }

    db = SessionLocal()

    try:
        all_results = []
        category_stats = {}

        # Process only retrieval-related categories
        for category, cases in test_cases.items():
            if category not in RETRIEVAL_CATEGORIES:
                continue

            print(f"\nEvaluating category: {category} ({len(cases)} tests)")

            category_results = []
            for test in cases:
                # Skip tests without retrieval expectations (e.g., error handling tests)
                if category == "edge_cases":
                    if "expected_tools" not in test and "expected_output" not in test:
                        print(f"  Skipping: {test['name']} (not a retrieval test)")
                        continue

                print(f"  Testing: {test['name']}...", end=" ")

                # Pass category to evaluation function
                test["category"] = category
                result = await evaluate_retrieval_case(test, db)
                category_results.append(result)
                all_results.append(result)

                if result["overall_pass"]:
                    print("✅")
                else:
                    print("❌")
                    if "error" in result:
                        print(f"    Error: {result['error']}")
                    else:
                        # Show which checks failed
                        failed_checks = [
                            check_name
                            for check_name, check in result.get("evaluation", {})
                            .get("checks", {})
                            .items()
                            if not check.get("passed", False)
                        ]
                        if failed_checks:
                            print(f"    Failed checks: {', '.join(failed_checks)}")

                            # Special handling for semantic query format failures
                            if "semantic_query_format" in failed_checks:
                                sem_check = result["evaluation"]["checks"]["semantic_query_format"]
                                if sem_check.get("is_raw_user_query"):
                                    print(
                                        f"      → Used raw user query instead of structured format"
                                    )
                                else:
                                    print(
                                        f"      → Query: {sem_check.get('actual', 'N/A')[:80]}..."
                                    )

            # Category stats
            passed = sum(1 for r in category_results if r["overall_pass"])
            total = len(category_results)
            category_stats[category] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0,
                "eval_type": "retrieval",
            }

        # Overall stats
        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)

        return {
            "results": all_results,
            "category_stats": category_stats,
            "eval_type_stats": {
                "retrieval": {
                    "passed": total_passed,
                    "total": total_tests,
                    "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
                }
            },
            "overall": {
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        db.close()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


async def main_async():
    """Async main entry point."""
    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("RETRIEVAL AGENT EVALUATION")
    print("Tests candidate gathering and tool execution")
    print("=" * 70)

    # Load test cases
    print("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    # Count retrieval tests - tests with expected_tools or expected_output
    RETRIEVAL_CATEGORIES = {
        "tool_selection_warm_user",
        "tool_selection_cold_user",
        "retrieval_strategy_adherence",
        "edge_cases",
    }

    retrieval_count = 0
    for cat in RETRIEVAL_CATEGORIES:
        for test in all_test_cases.get(cat, []):
            # Count only tests with retrieval expectations
            if "expected_tools" in test or "expected_output" in test:
                retrieval_count += 1

    print(f"Found {retrieval_count} retrieval test cases")

    print("\n🔧 Testing with REAL database connection and users")
    print("   Agent will use actual retrieval tools\n")

    # Run evaluation
    eval_results = await evaluate_retrieval_tests(all_test_cases)

    # Print results
    print_results(eval_results)

    # Save results
    save_results(eval_results, results_dir / "stages")

    # Return exit code
    return 0 if eval_results["overall"]["passed"] == eval_results["overall"]["total"] else 1


def main():
    """Synchronous wrapper for async main."""
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
