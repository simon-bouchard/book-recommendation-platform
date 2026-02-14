# evaluation/chatbot/recommendation_agent/evaluate_planner.py
"""
Evaluation suite for PlannerAgent.
Tests strategy selection, tool recommendations, and profile data handling.
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Suppress noisy HTTP client logs (httpcore, primp, rquest, etc)
# Also suppress OpenAI client DEBUG logs (raw JSON payloads)
import logging
from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()

# Additionally suppress OpenAI and httpx at DEBUG level
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from app.agents.infrastructure.recsys.planner_agent import PlannerAgent
from app.agents.domain.recsys_schemas import PlannerInput
from app.database import SessionLocal

# Import shared helpers
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))
from shared_helpers import (
    get_user_by_id,
    validate_query,
    load_test_cases,
    print_results,
    save_results,
)


# ============================================================================
# PLANNER EVALUATION LOGIC
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

    Args:
        strategy: PlannerStrategy object returned by planner
        expected_tools: Dict with expectations (user_is_warm, has_profile)
        query_type: Query classification ('vague', 'descriptive', 'genre')

    Returns:
        Dict with checks and all_passed status
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

        # Cold users with profile should use subject_hybrid_pool or popular_books
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

        # Unconnected/cold users with NO profile should use popular_books
        else:
            # This is the unconnected user case (no ratings, no profile)
            # Should recommend popular_books, NOT subject_hybrid_pool
            uses_popular = "popular_books" in recommended
            uses_subject_hybrid = "subject_hybrid_pool" in recommended

            results["checks"]["unconnected_user_strategy"] = {
                "expected": "popular_books (NOT subject_hybrid_pool)",
                "actual": recommended,
                "passed": uses_popular and not uses_subject_hybrid,
            }

            if not uses_popular:
                results["all_passed"] = False

            if uses_subject_hybrid:
                # Critical bug: Using subject_hybrid_pool for user with no subjects
                results["checks"]["bug_subject_hybrid_for_unconnected"] = {
                    "expected": "NOT subject_hybrid_pool (user has no subjects)",
                    "actual": recommended,
                    "passed": False,
                    "severity": "CRITICAL BUG",
                }
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


# ============================================================================
# PLANNER TEST RUNNER
# ============================================================================


async def run_planner_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Run a single planner test case.

    Creates planner agent and validates strategy output.

    Args:
        test_case: Test case dict from test_cases.json
        db: Database session

    Returns:
        Result dict with test outcome and evaluation details
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

        # Execute planner (async)
        strategy = await planner.execute(planner_input)

        result["agent_success"] = True
        result["strategy"] = {
            "recommended_tools": strategy.recommended_tools,
            "fallback_tools": strategy.fallback_tools,
            "reasoning": strategy.reasoning[:150] + "..."
            if len(strategy.reasoning) > 150
            else strategy.reasoning,
            "has_profile_data": strategy.profile_data is not None,
        }

        # Get expected query type from test case (explicit field)
        query_type = test_case.get("expected_query_type")

        if query_type is None:
            # Fallback for tests without explicit type (legacy support)
            query_lower = query.lower()
            if len(query.split()) <= 3 and any(
                g in query_lower for g in ["fantasy", "mystery", "romance"]
            ):
                query_type = "genre"  # Simple genre queries
            elif any(word in query_lower for word in ["recommend", "suggest", "what should"]):
                query_type = "vague"  # Vague queries
            else:
                query_type = "descriptive"  # Default: descriptive

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
# EVALUATION ORCHESTRATION
# ============================================================================


async def evaluate_planner_tests(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Run all planner test cases and aggregate results.

    Args:
        test_cases: Dict mapping category names to test case lists

    Returns:
        Aggregated results with statistics
    """
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")

    db = SessionLocal()

    try:
        all_results = []
        category_stats = {}

        # Planner test categories
        planner_categories = ["tool_selection_warm_user", "tool_selection_cold_user"]

        for category in planner_categories:
            if category not in test_cases:
                continue

            cases = test_cases[category]
            print(f"\n{'=' * 70}")
            print(f"Running {category}: {len(cases)} test cases")
            print("=" * 70)

            category_results = []
            for i, test_case in enumerate(cases, 1):
                name = test_case.get("name", f"test_{i}")
                query = test_case.get("query", "")

                print(f"\n{'─' * 70}")
                print(f"[{i}/{len(cases)}] {name}")
                print(f"Query: {query}")
                print(f"{'─' * 70}")

                result = await run_planner_test(test_case, db)
                category_results.append(result)

                # Show strategy if test succeeded
                if result["agent_success"] and "strategy" in result:
                    strategy = result["strategy"]
                    print(f"\n  Recommended: {strategy['recommended_tools']}")
                    print(f"  Fallback: {strategy['fallback_tools']}")
                    print(f"  Reasoning: {strategy['reasoning']}")
                    if strategy.get("has_profile_data"):
                        print(f"  Profile: Available")

                # Show pass/fail and any failures
                if result["overall_pass"]:
                    print(f"\n  ✅ PASS")
                else:
                    print(f"\n  ❌ FAIL")
                    if "evaluation" in result and "checks" in result["evaluation"]:
                        for check_name, check in result["evaluation"]["checks"].items():
                            if not check.get("passed", False):
                                print(
                                    f"     • {check_name}: expected {check.get('expected')}, got {check.get('actual')}"
                                )
                    if "error" in result:
                        print(f"     • Error: {result['error']}")

                print()  # Blank line between tests

            all_results.extend(category_results)

            # Category stats
            passed = sum(1 for r in category_results if r["overall_pass"])
            category_stats[category] = {
                "passed": passed,
                "total": len(category_results),
                "pass_rate": passed / len(category_results) if category_results else 0,
                "eval_type": "planner",
            }

        # Overall stats
        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)

        from datetime import datetime

        return {
            "results": all_results,
            "category_stats": category_stats,
            "eval_type_stats": {
                "planner": {
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


def main():
    """Run planner evaluation with CLI support."""
    parser = argparse.ArgumentParser(
        description="PlannerAgent Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all planner tests
  python evaluate_planner.py

  # Run with specific test categories
  python evaluate_planner.py --categories tool_selection_warm_user
        """,
    )

    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Test categories to run (space-separated). Run all planner tests if not specified.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("PLANNER AGENT EVALUATION")
    print("Query Analysis & Strategy Planning")
    print("=" * 70)
    print("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    # Filter to planner categories
    planner_test_categories = ["tool_selection_warm_user", "tool_selection_cold_user"]

    if args.categories:
        # Use only specified categories
        test_cases = {}
        for cat in args.categories:
            if cat in all_test_cases and cat in planner_test_categories:
                test_cases[cat] = all_test_cases[cat]
            else:
                print(f"\n❌ ERROR: '{cat}' is not a valid planner test category")
                print(f"Valid categories: {', '.join(planner_test_categories)}")
                sys.exit(1)
    else:
        # Use all planner categories
        test_cases = {
            cat: all_test_cases[cat] for cat in planner_test_categories if cat in all_test_cases
        }

    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"Loaded {total_cases} planner test cases across {len(test_cases)} categories")

    print("\n📊 Testing with REAL database connection and users")
    print("   PlannerAgent will fetch actual profile data\n")

    print("Running evaluation...")

    # Run async evaluation
    eval_results = asyncio.run(evaluate_planner_tests(test_cases))

    print_results(eval_results)

    # Save to stages subdirectory for standalone execution
    stages_dir = results_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = stages_dir / f"planner_eval_{timestamp}.json"

    import json

    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
