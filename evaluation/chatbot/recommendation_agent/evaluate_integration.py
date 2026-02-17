# evaluation/chatbot/recommendation_agent/evaluate_integration.py
"""
Integration evaluation for full recommendation pipeline quality.
Evaluates Planner → Retrieval → Curation end-to-end behavior and output quality.

EVAL FOCUS (not testing):
- Strategic quality: Does the pipeline make good decisions?
- Cross-stage coherence: Does final output reflect the strategy?
- Output quality: Are recommendations relevant and well-explained?
- Graceful degradation: Does it handle edge cases well?

This is about evaluating QUALITY to improve the agent, not testing correctness.
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from app.agents.settings import get_llm

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Suppress noisy logs
import logging
from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent
from app.agents.domain.entities import AgentRequest
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
from llm_judges import (
    llm_judge_prose_reasoning,
    llm_judge_query_relevance,
    llm_judge_personalization_prose,
)

# NOTE: These judges are SHARED with curation eval for consistency.
# Integration tests reuse the curation agent's LLM instance to ensure
# the same judge configuration is used across both evaluation stages.
# This prevents false positives/negatives from judge variation.


# ============================================================================
# INTEGRATION QUALITY EVALUATION
# ============================================================================


def evaluate_full_pipeline_quality(
    response,
    test_case: Dict[str, Any],
    execution_context=None,
    judge_llm=None,
) -> Dict[str, Any]:
    """
    Evaluate quality of full pipeline execution.

    Quality dimensions:
    1. Strategic coherence - Do tool choices match high-impact requirements?
    2. Output quality - Book count, prose presence
    3. Semantic quality (LLM judges) - Prose reasoning, query relevance, personalization
    4. Graceful degradation - Handles edge cases without crashing

    Args:
        response: Final AgentResponse from orchestrator
        test_case: Test case with expected_behavior
        execution_context: Optional ExecutionContext from retrieval stage
        judge_llm: LLM instance for semantic validation

    Returns:
        Dict with checks and all_passed status
    """
    results = {"checks": {}, "all_passed": True}

    expected_behavior = test_case.get("expected_behavior", {})
    query = test_case.get("query", "")

    # ========================================================================
    # QUALITY CHECK 1: BASIC OUTPUT QUALITY
    # ========================================================================

    # Check: Final response should have books (6-30 range)
    if expected_behavior.get("final_response_should_have_books"):
        has_books = len(response.book_recommendations) > 0
        book_count = len(response.book_recommendations)
        in_range = 6 <= book_count <= 30

        results["checks"]["has_books"] = {
            "expected": "6-30 books",
            "actual": book_count,
            "passed": has_books and in_range,
        }

        if not (has_books and in_range):
            results["all_passed"] = False

    # Check: Should have explanatory prose
    has_prose = response.text and len(response.text) > 50
    results["checks"]["has_prose"] = {
        "expected": "prose explanation > 50 chars",
        "actual": len(response.text) if response.text else 0,
        "passed": has_prose,
    }

    if not has_prose:
        results["all_passed"] = False

    # ========================================================================
    # QUALITY CHECK 2: STRATEGIC COHERENCE (HIGH-IMPACT BEHAVIORS)
    # ========================================================================

    # These checks validate that the pipeline made good strategic decisions
    # based on query type and user state

    if execution_context and hasattr(execution_context, "tools_used"):
        tools_used = execution_context.tools_used

        # Check: Should use ALS for warm users with vague queries
        if expected_behavior.get("should_use_als"):
            uses_als = "als_recs" in tools_used
            results["checks"]["uses_als_for_vague_warm"] = {
                "expected": "als_recs",
                "actual": tools_used,
                "passed": uses_als,
            }
            if not uses_als:
                results["all_passed"] = False

        # Check: Should use subject_hybrid_pool for cold users with profile
        if expected_behavior.get("should_use_subject_hybrid_pool"):
            uses_subject_hybrid = "subject_hybrid_pool" in tools_used
            results["checks"]["uses_subject_hybrid_pool"] = {
                "expected": "subject_hybrid_pool",
                "actual": tools_used,
                "passed": uses_subject_hybrid,
            }
            if not uses_subject_hybrid:
                results["all_passed"] = False

        # Check: Should use popular_books for new users
        if expected_behavior.get("should_use_popular_books"):
            uses_popular = "popular_books" in tools_used
            results["checks"]["uses_popular_books"] = {
                "expected": "popular_books",
                "actual": tools_used,
                "passed": uses_popular,
            }
            if not uses_popular:
                results["all_passed"] = False

        # Check: Should use semantic search for descriptive queries
        if expected_behavior.get("should_use_semantic_search"):
            uses_semantic = "book_semantic_search" in tools_used
            results["checks"]["uses_semantic_search"] = {
                "expected": "book_semantic_search",
                "actual": tools_used,
                "passed": uses_semantic,
            }
            if not uses_semantic:
                results["all_passed"] = False

        # Check: Genre queries should use subject search (not personalization)
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

        # Check: Should NOT use ALS (for cold users)
        if expected_behavior.get("should_NOT_use_als"):
            uses_als = "als_recs" in tools_used
            results["checks"]["no_als_for_cold_user"] = {
                "expected": "NOT als_recs",
                "actual": tools_used,
                "passed": not uses_als,
            }
            if uses_als:
                results["all_passed"] = False

        # Check: Should NOT use subject_hybrid_pool (for new users with no subjects)
        if expected_behavior.get("should_NOT_use_subject_hybrid_pool"):
            uses_subject_hybrid = "subject_hybrid_pool" in tools_used
            results["checks"]["no_subject_hybrid_for_new_user"] = {
                "expected": "NOT subject_hybrid_pool",
                "actual": tools_used,
                "passed": not uses_subject_hybrid,
            }
            if uses_subject_hybrid:
                results["all_passed"] = False

        # Check: Should NOT use personalization for descriptive queries
        if expected_behavior.get("should_NOT_use_personalization"):
            uses_personalization = "als_recs" in tools_used or (
                execution_context.profile_data is not None
            )
            results["checks"]["no_personalization_for_descriptive"] = {
                "expected": "NOT als_recs or profile_data",
                "actual": f"als={('als_recs' in tools_used)}, profile={execution_context.profile_data is not None}",
                "passed": not uses_personalization,
            }
            if uses_personalization:
                results["all_passed"] = False

        # Check: Cold user genre queries MUST use subject_id_search
        if expected_behavior.get("must_use_subject_id_search"):
            uses_subject_id = "subject_id_search" in tools_used
            results["checks"]["uses_subject_id_search"] = {
                "expected": "subject_id_search",
                "actual": tools_used,
                "passed": uses_subject_id,
            }
            if not uses_subject_id:
                results["all_passed"] = False

        # Check: Should NOT use popular_books (when query is specific)
        if expected_behavior.get("should_NOT_use_popular_books"):
            uses_popular = "popular_books" in tools_used
            results["checks"]["no_popular_books"] = {
                "expected": "NOT popular_books",
                "actual": tools_used,
                "passed": not uses_popular,
            }
            if uses_popular:
                results["all_passed"] = False

    # ========================================================================
    # QUALITY CHECK 3: SEMANTIC QUALITY (LLM JUDGES)
    # ========================================================================

    # Only run LLM judges if we have prose and a judge LLM
    if judge_llm and has_prose:
        # Judge 1: Prose Reasoning Quality
        # Does prose explain WHY books are recommended?
        prose_judge = llm_judge_prose_reasoning(response.text, judge_llm)
        results["checks"]["prose_explains_reasoning"] = {
            "expected": "prose explains why books are recommended",
            "actual": prose_judge.get("verdict", "Unknown"),
            "passed": prose_judge.get("passed", False),
            "judge_reasoning": prose_judge.get("reasoning", ""),
            "issues": prose_judge.get("issues", []),
        }
        if not prose_judge.get("passed", False):
            results["all_passed"] = False

        # Judge 2: Query Relevance
        # Does prose address the user's query?
        tools_used_list = execution_context.tools_used if execution_context else ["unknown"]
        relevance_judge = llm_judge_query_relevance(
            response.text, query, judge_llm, tools_used=tools_used_list
        )
        results["checks"]["prose_addresses_query"] = {
            "expected": "prose relates to query and acknowledges method",
            "actual": relevance_judge.get("verdict", "Unknown"),
            "passed": relevance_judge.get("passed", False),
            "judge_reasoning": relevance_judge.get("reasoning", ""),
            "issues": relevance_judge.get("issues", []),
        }
        if not relevance_judge.get("passed", False):
            results["all_passed"] = False

        # Judge 3: Personalization Appropriateness
        # Does prose correctly reflect whether personalization was used?
        if execution_context:
            # Determine if personalization should NOT be mentioned
            expect_no_personalization = (
                "book_semantic_search" in tools_used_list
                and "als_recs" not in tools_used_list
                and not execution_context.profile_data
            )

            personalization_judge = llm_judge_personalization_prose(
                response.text,
                execution_context,
                judge_llm,
                expect_no_personalization=expect_no_personalization,
            )
            results["checks"]["personalization_appropriateness"] = {
                "expected": (
                    "no false personalization claims"
                    if expect_no_personalization
                    else "mentions personalization when used"
                ),
                "actual": personalization_judge.get("verdict", "Unknown"),
                "passed": personalization_judge.get("passed", False),
                "judge_reasoning": personalization_judge.get("reasoning", ""),
                "issues": personalization_judge.get("issues", []),
            }
            if not personalization_judge.get("passed", False):
                results["all_passed"] = False

    # ========================================================================
    # QUALITY CHECK 4: GRACEFUL DEGRADATION
    # ========================================================================

    # Check: Edge cases should be handled gracefully
    if expected_behavior.get("should_handle_gracefully"):
        # Success means it didn't crash and returned SOMETHING
        has_books = len(response.book_recommendations) > 0
        has_error_msg = response.text and len(response.text) > 20
        handled_gracefully = has_books or has_error_msg

        results["checks"]["graceful_handling"] = {
            "expected": "books or informative error message",
            "actual": f"books={has_books}, msg_len={len(response.text) if response.text else 0}",
            "passed": handled_gracefully,
        }
        if not handled_gracefully:
            results["all_passed"] = False

    return results


# ============================================================================
# TEST EXECUTION
# ============================================================================


async def run_integration_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Execute a single integration test case.

    Runs full pipeline and evaluates end-to-end quality.

    Args:
        test_case: Test case dictionary with query, user_id, expected_behavior
        db: Database session

    Returns:
        Result dictionary with evaluation details
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

    # Skip validation for edge case tests that test graceful handling
    # These tests intentionally use problematic queries (empty, malformed, etc.)
    if not expected_behavior.get("should_handle_gracefully"):
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

        # For high-impact tests, we need to track tools used
        # We'll get this from the orchestrator's execution
        execution_context = None

        # Execute full pipeline
        orchestrator = RecommendationAgent(
            current_user=user,
            db=db,
            user_num_ratings=rating_count,
            warm_threshold=10,
            allow_profile=user_state.get("allow_profile", False),
        )

        request = AgentRequest(user_text=query, conversation_history=[])

        # Execute using STREAMING path (matches production)
        chunks = []
        async for chunk in orchestrator.execute_stream(request):
            chunks.append(chunk)

        # Extract completion chunk
        completion_chunks = [c for c in chunks if c.type == "complete"]
        if not completion_chunks:
            raise RuntimeError("No completion chunk received from streaming")

        completion_data = completion_chunks[-1].data

        # Reconstruct response text from token chunks
        text_tokens = [c.content for c in chunks if c.type == "token"]
        response_text = "".join(text_tokens)

        # Get book_ids from completion data (extracted from citations by curation)
        book_ids = completion_data.get("book_ids", [])

        # Create minimal BookRecommendation objects for evaluation
        # The evaluation code expects book_recommendations, not just IDs
        from app.agents.domain.entities import AgentResponse, BookRecommendation

        book_recommendations = [
            BookRecommendation(item_idx=book_id, title="", author="") for book_id in book_ids
        ]

        final_response = AgentResponse(
            text=response_text,
            target_category=completion_data.get("target", "recsys"),
            success=completion_data.get("success", False),
            book_recommendations=book_recommendations,
            policy_version="recsys.streaming",
        )

        result["agent_success"] = final_response.success
        result["pipeline"] = {
            "success": final_response.success,
            "book_count": len(book_recommendations),
            "text_length": len(response_text),
        }

        # Extract execution context from completion data
        # Orchestrator now includes tools_used in completion chunk
        tools_used = completion_data.get("tools_used", [])

        if tools_used:
            from app.agents.domain.recsys_schemas import ExecutionContext

            execution_context = ExecutionContext(
                planner_reasoning="",  # Not available in streaming
                tools_used=tools_used,
                profile_data=None,  # Not available in streaming
            )
            result["pipeline"]["tools_used"] = tools_used
        else:
            execution_context = None
            # Log warning if strategic checks are expected
            if expected_behavior.get("should_use_als") or expected_behavior.get(
                "should_use_subject_hybrid_pool"
            ):
                import warnings

                warnings.warn(
                    f"Test '{name}' expects tool validation but completion data has no tools_used. "
                    f"Strategic coherence checks will be skipped.",
                    stacklevel=2,
                )

        # REUSE curation agent's LLM for judging (same config as curation eval)
        # This ensures consistent judge behavior across curation and integration evals
        judge_llm = get_llm(
            tier="medium",
            json_mode=True,
            temperature=0.0,
        )

        # Evaluate pipeline quality
        eval_result = evaluate_full_pipeline_quality(
            final_response, test_case, execution_context, judge_llm
        )
        result["evaluation"] = eval_result
        result["overall_pass"] = eval_result["all_passed"]

    except Exception as e:
        # For edge cases, graceful handling means not crashing
        if expected_behavior.get("should_handle_gracefully"):
            # If it crashed, that's a failure
            result["evaluation"] = {
                "checks": {
                    "no_crash": {
                        "expected": "graceful handling",
                        "actual": f"exception: {str(e)[:100]}",
                        "passed": False,
                    }
                },
                "all_passed": False,
            }
        result["error"] = str(e)
        result["overall_pass"] = False

    return result


# ============================================================================
# EVALUATION ORCHESTRATION
# ============================================================================


async def evaluate_integration_tests(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Run all integration test cases and aggregate results.

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

        # Integration test categories
        integration_categories = [
            "integration_classic_scenarios",  # NEW: Representative scenarios covering all quality dimensions
            "integration_high_impact",
            "edge_cases",
        ]

        for category in integration_categories:
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
                print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
                print(f"{'─' * 70}")

                result = await run_integration_test(test_case, db)
                category_results.append(result)

                # Show pipeline results if test succeeded
                if result["agent_success"] and "pipeline" in result:
                    pipeline = result["pipeline"]
                    print(f"\n  Pipeline executed successfully:")
                    print(f"  - Books: {pipeline['book_count']}")
                    print(f"  - Prose length: {pipeline['text_length']} chars")
                    if "tools_used" in pipeline:
                        print(f"  - Tools: {', '.join(pipeline['tools_used'])}")

                # Show pass/fail and any failures
                if result["overall_pass"]:
                    print(f"\n  ✅ PASS")
                else:
                    print(f"\n  ❌ FAIL")
                    if "evaluation" in result and "checks" in result["evaluation"]:
                        for check_name, check in result["evaluation"]["checks"].items():
                            if not check.get("passed", False):
                                print(f"     • {check_name}:")
                                print(f"       Expected: {check.get('expected')}")
                                print(f"       Actual: {check.get('actual')}")

                                # Show judge reasoning if available
                                if "judge_reasoning" in check:
                                    reasoning = check["judge_reasoning"][:150]
                                    print(f"       Judge: {reasoning}...")

                                # Show issues if available
                                if check.get("issues"):
                                    print(f"       Issues: {', '.join(check['issues'][:3])}")

                    if "error" in result:
                        print(f"     • Error: {result['error'][:100]}...")

                print()  # Blank line between tests

            all_results.extend(category_results)

            # Category stats
            passed = sum(1 for r in category_results if r["overall_pass"])
            category_stats[category] = {
                "passed": passed,
                "total": len(category_results),
                "pass_rate": passed / len(category_results) if category_results else 0,
                "eval_type": "integration",
            }

        # Overall stats
        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)

        # ====================================================================
        # CHECK-BASED STATS: Which quality dimensions are failing?
        # ====================================================================
        check_stats = {}

        # Collect all unique check names
        all_check_names = set()
        for result in all_results:
            if "evaluation" in result and "checks" in result["evaluation"]:
                all_check_names.update(result["evaluation"]["checks"].keys())

        # For each check, compute pass rate across all queries
        for check_name in all_check_names:
            passed_count = 0
            total_count = 0
            failed_queries = []

            for result in all_results:
                if "evaluation" in result and "checks" in result["evaluation"]:
                    checks = result["evaluation"]["checks"]
                    if check_name in checks:
                        total_count += 1
                        if checks[check_name].get("passed", False):
                            passed_count += 1
                        else:
                            failed_queries.append(result["name"])

            check_stats[check_name] = {
                "passed": passed_count,
                "total": total_count,
                "pass_rate": passed_count / total_count if total_count > 0 else 0,
                "failed_queries": failed_queries,
            }

        # ====================================================================
        # QUERY-BASED STATS: How many queries pass ALL checks?
        # ====================================================================
        passed_all_checks = sum(1 for r in all_results if r.get("overall_pass", False))
        failed_one_or_more = sum(1 for r in all_results if not r.get("overall_pass", False))

        query_stats = {
            "passed_all_checks": passed_all_checks,
            "failed_one_or_more": failed_one_or_more,
            "total": total_tests,
            "perfect_rate": passed_all_checks / total_tests if total_tests > 0 else 0,
        }

        return {
            "results": all_results,
            "check_stats": check_stats,  # NEW: Check-based perspective
            "query_stats": query_stats,  # NEW: Query-based perspective
            "category_stats": category_stats,
            "eval_type_stats": {
                "integration": {
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
    """Run integration evaluation with CLI support."""
    parser = argparse.ArgumentParser(
        description="Integration Evaluation Suite - Full Pipeline Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all integration tests (including classic scenarios)
  python evaluate_integration.py

  # Run only classic scenarios
  python evaluate_integration.py --categories integration_classic_scenarios

  # Run only high-impact tests
  python evaluate_integration.py --categories integration_high_impact

  # Run classic scenarios and high-impact tests
  python evaluate_integration.py --categories integration_classic_scenarios integration_high_impact
        """,
    )

    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=["integration_classic_scenarios", "integration_high_impact", "edge_cases"],
        help="Test categories to run (default: all)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    classic_scenarios_path = script_dir / "integration_classic_scenarios.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("INTEGRATION EVALUATION - FULL PIPELINE QUALITY")
    print("Evaluates: Strategic Coherence + Output Quality + Graceful Degradation")
    print("=" * 70)
    print("\nLoading test cases...")

    # Load both test case files
    all_test_cases = load_test_cases(test_cases_path)

    # Load classic scenarios if file exists
    if classic_scenarios_path.exists():
        classic_scenarios = load_test_cases(classic_scenarios_path)
        all_test_cases.update(classic_scenarios)
        print(f"Loaded classic scenarios from {classic_scenarios_path.name}")
    else:
        print(f"⚠️  Classic scenarios file not found: {classic_scenarios_path.name}")
        print("   Only using scenarios from test_cases.json")

    # Filter categories if specified
    if args.categories:
        test_cases = {cat: all_test_cases[cat] for cat in args.categories if cat in all_test_cases}
        if not test_cases:
            print(f"\n❌ ERROR: None of the specified categories found")
            print(f"Available categories: {', '.join(all_test_cases.keys())}")
            sys.exit(1)
    else:
        # Only integration categories
        test_cases = {
            k: v
            for k, v in all_test_cases.items()
            if k in ["integration_classic_scenarios", "integration_high_impact", "edge_cases"]
        }

    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"Loaded {total_cases} test cases across {len(test_cases)} categories")

    print("\n📊 Testing with REAL database connection and LLM judges")
    print("   Full pipeline: Planner → Retrieval → Curation")
    print("   Quality focus: Strategic decisions + Output quality + Semantic validation\n")

    print("Running evaluation...")

    # Run async evaluation
    eval_results = asyncio.run(evaluate_integration_tests(test_cases))

    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
