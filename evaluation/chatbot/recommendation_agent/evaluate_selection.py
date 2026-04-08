# evaluation/chatbot/recommendation_agent/evaluate_selection.py
"""
Selection agent evaluation suite.
Tests ID validity, structural constraints, genre filtering, and negative constraint
filtering using isolated selection testing with mock candidates from the test data factory.

Stage 3a of the recommendation pipeline: Planner -> Retrieval -> Selection -> Curation.
SelectionAgent receives 60-120 candidates and returns 6-30 validated, ranked IDs.
"""

import argparse
import asyncio
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()

from app.agents.domain.entities import AgentRequest, BookRecommendation
from app.agents.infrastructure.recsys.selection_agent import SelectionAgent
from app.database import SessionLocal

eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from llm_judges import (
    llm_judge_genre_match,
    llm_judge_negative_constraint_filtering,
)
from shared_helpers import (
    get_user_by_id,
    load_test_cases,
    print_results,
    save_results,
    validate_query,
)
from test_data_factory import get_candidates, get_execution_context

# ============================================================================
# STRUCTURAL CHECKS (deterministic, always run)
# ============================================================================


def _check_id_validity(
    selected: List[BookRecommendation], candidate_pool: List[BookRecommendation]
) -> Dict[str, Any]:
    """
    Verify every selected ID exists in the input candidate pool.

    This is the core correctness guarantee of SelectionAgent — it must never
    invent IDs that weren't in the candidates it was given.

    Args:
        selected: Books returned by SelectionAgent
        candidate_pool: Full pool fed into SelectionAgent

    Returns:
        Check result dict with passed status and invalid ID details
    """
    pool_ids = {c.item_idx for c in candidate_pool}
    invalid = [b.item_idx for b in selected if b.item_idx not in pool_ids]

    return {
        "passed": len(invalid) == 0,
        "expected": "all selected IDs exist in candidate pool",
        "actual": f"{len(selected)} selected, {len(invalid)} invalid",
        "details": {"invalid_ids": invalid},
    }


def _check_no_duplicates(selected: List[BookRecommendation]) -> Dict[str, Any]:
    """
    Verify no book ID appears more than once in the selection.

    Args:
        selected: Books returned by SelectionAgent

    Returns:
        Check result dict with passed status and duplicate ID details
    """
    seen: set = set()
    duplicates = []
    for book in selected:
        if book.item_idx in seen:
            duplicates.append(book.item_idx)
        seen.add(book.item_idx)

    return {
        "passed": len(duplicates) == 0,
        "expected": "all selected books are unique",
        "actual": f"{len(selected)} books, {len(duplicates)} duplicates",
        "details": {"duplicate_ids": duplicates},
    }


def _check_count_in_range(
    selected: List[BookRecommendation], min_count: int = 6, max_count: int = 30
) -> Dict[str, Any]:
    """
    Verify the selection count falls within the required 6-30 range.

    Args:
        selected: Books returned by SelectionAgent
        min_count: Minimum acceptable count (default 6)
        max_count: Maximum acceptable count (default 30)

    Returns:
        Check result dict with passed status and count details
    """
    count = len(selected)
    return {
        "passed": min_count <= count <= max_count,
        "expected": f"between {min_count} and {max_count} books",
        "actual": count,
    }


def _check_diversity(selected: List[BookRecommendation]) -> Dict[str, Any]:
    """
    Verify no single author dominates the selection (max 2 books per author).

    Mirrors the SelectionAgent's own stated diversity constraint. If author
    data is unavailable on the candidates, the check is skipped (passed=True).

    Args:
        selected: Books returned by SelectionAgent

    Returns:
        Check result dict with passed status and over-represented author details
    """
    authors = [b.author for b in selected if b.author and b.author.strip()]

    if not authors:
        return {
            "passed": True,
            "expected": "max 2 books per author",
            "actual": "author data unavailable, check skipped",
        }

    counts = Counter(authors)
    violating = {author: count for author, count in counts.items() if count > 2}

    return {
        "passed": len(violating) == 0,
        "expected": "max 2 books per author",
        "actual": f"{len(selected)} books, {len(violating)} authors with >2 books",
        "details": {"over_represented_authors": violating},
    }


# ============================================================================
# TEST RUNNERS
# ============================================================================


async def run_genre_filtering_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Execute a genre filtering test against SelectionAgent.

    Feeds a mixed candidate pool (correct genre + wrong genre books) into
    SelectionAgent and verifies it selects primarily on-genre books.

    Args:
        test_case: Test case dict with query, test_scenario, expected_genre,
                   expected_selection, and user_state fields
        db: Database session

    Returns:
        Test result dict with evaluation checks and pass/fail status
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    test_scenario = test_case["test_scenario"]
    context_scenario = test_case.get("context_scenario", "subject")
    expected_genre = test_case["expected_genre"]
    expected_selection = test_case["expected_selection"]

    result: Dict[str, Any] = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "selection_genre",
        "agent_success": False,
        "overall_pass": False,
    }

    query_valid, error_msg = validate_query(query)
    if not query_valid:
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
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        candidates_data = get_candidates(
            test_scenario, db, user_id=user.user_id, user_num_ratings=rating_count
        )
        execution_context = get_execution_context(context_scenario)

        candidates = [
            BookRecommendation(
                item_idx=book["item_idx"],
                title=book.get("title"),
                author=book.get("author"),
                year=book.get("year"),
                num_ratings=book.get("num_ratings"),
                subjects=book.get("subjects"),
                genre=book.get("genre"),
            )
            for book in candidates_data["books"]
        ]

        selection = SelectionAgent()
        request = AgentRequest(user_text=query, conversation_history=[])

        selected = await selection.execute(
            request=request,
            candidates=candidates,
            execution_context=execution_context,
        )

        min_books = expected_selection.get("min_selected_books", 6)
        result["agent_success"] = len(selected) >= min_books
        result["pipeline"] = {
            "candidate_count": len(candidates),
            "correct_genre_count": candidates_data.get("correct_genre_count"),
            "wrong_genre_count": candidates_data.get("wrong_genre_count"),
            "selected_count": len(selected),
        }

        # Always-run structural checks
        checks = {
            "id_validity": _check_id_validity(selected, candidates),
            "no_duplicates": _check_no_duplicates(selected),
            "count_in_range": _check_count_in_range(selected),
            "diversity": _check_diversity(selected),
        }

        # LLM judge for genre accuracy (only when agent returned enough books)
        if expected_selection.get("llm_judge_needed") and len(selected) > 0:
            judge_result = llm_judge_genre_match(
                books=selected,
                expected_genre=expected_genre,
                db=db,
                judge_llm=selection.llm,
            )
            checks["genre_matching"] = {
                "expected": f"≥80% of selected books match genre '{expected_genre}'",
                "actual": judge_result["verdict"],
                "passed": judge_result["passed"],
                "details": {
                    "reasoning": judge_result["reasoning"],
                    "violating_books": judge_result.get("violating_books", []),
                    "match_count": judge_result.get("match_count", 0),
                    "total_count": judge_result.get("total_count", len(selected)),
                },
            }

        all_passed = all(check["passed"] for check in checks.values())
        result["evaluation"] = {"checks": checks, "all_passed": all_passed}
        result["overall_pass"] = all_passed and result["agent_success"]

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        result["error"] = str(e)
        result["error_detail"] = error_detail
        result["overall_pass"] = False
        result["evaluation"] = {
            "checks": {
                "execution_error": {
                    "expected": "successful execution",
                    "actual": f"Exception: {str(e)}",
                    "passed": False,
                }
            },
            "all_passed": False,
        }
        print(f"\n  ERROR DETAILS:\n{error_detail}")

    return result


async def run_negative_constraint_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Execute a negative constraint filtering test against SelectionAgent.

    Feeds a candidate pool that deliberately includes books matching the
    excluded constraint (e.g., cozy mysteries when user said "NOT cozy")
    and verifies SelectionAgent filters them out.

    Args:
        test_case: Test case dict with query, test_scenario, negative_constraints,
                   expected_selection, and user_state fields
        db: Database session

    Returns:
        Test result dict with evaluation checks and pass/fail status
    """
    name = test_case["name"]
    query = test_case["query"]
    user_state = test_case["user_state"]
    test_scenario = test_case["test_scenario"]
    context_scenario = test_case.get("context_scenario", "negative")
    negative_constraints = test_case["negative_constraints"]
    expected_selection = test_case["expected_selection"]

    result: Dict[str, Any] = {
        "name": name,
        "query": query,
        "user_state": user_state,
        "test_type": "selection_negative",
        "agent_success": False,
        "overall_pass": False,
    }

    query_valid, error_msg = validate_query(query)
    if not query_valid:
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
        user, rating_count = get_user_by_id(db, test_case["user_id"])

        candidates_data = get_candidates(
            test_scenario, db, user_id=user.user_id, user_num_ratings=rating_count
        )
        execution_context = get_execution_context(context_scenario)

        candidates = [
            BookRecommendation(
                item_idx=book["item_idx"],
                title=book.get("title"),
                author=book.get("author"),
                year=book.get("year"),
                num_ratings=book.get("num_ratings"),
                subjects=book.get("subjects"),
                genre=book.get("genre"),
            )
            for book in candidates_data["books"]
        ]

        selection = SelectionAgent()
        request = AgentRequest(user_text=query, conversation_history=[])

        selected = await selection.execute(
            request=request,
            candidates=candidates,
            execution_context=execution_context,
        )

        min_books = expected_selection.get("min_selected_books", 6)
        result["agent_success"] = len(selected) >= min_books
        result["pipeline"] = {
            "candidate_count": len(candidates),
            "base_count": candidates_data.get("base_count"),
            "constraint_count": candidates_data.get("constraint_count"),
            "selected_count": len(selected),
        }

        # Always-run structural checks
        checks = {
            "id_validity": _check_id_validity(selected, candidates),
            "no_duplicates": _check_no_duplicates(selected),
            "count_in_range": _check_count_in_range(selected),
            "diversity": _check_diversity(selected),
        }

        # LLM judge for constraint filtering (only when agent returned enough books)
        if expected_selection.get("llm_judge_needed") and len(selected) > 0:
            judge_result = llm_judge_negative_constraint_filtering(
                books=selected,
                negative_constraints=negative_constraints,
                db=db,
                judge_llm=selection.llm,
            )
            checks["negative_constraint_filtering"] = {
                "expected": f"no selected books match excluded terms: {negative_constraints}",
                "actual": judge_result["verdict"],
                "passed": judge_result["passed"],
                "details": {
                    "reasoning": judge_result["reasoning"],
                    "violating_books": judge_result.get("violating_books", []),
                    "violation_count": judge_result.get("violation_count", 0),
                },
            }

        all_passed = all(check["passed"] for check in checks.values())
        result["evaluation"] = {"checks": checks, "all_passed": all_passed}
        result["overall_pass"] = all_passed and result["agent_success"]

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        result["error"] = str(e)
        result["error_detail"] = error_detail
        result["overall_pass"] = False
        result["evaluation"] = {
            "checks": {
                "execution_error": {
                    "expected": "successful execution",
                    "actual": f"Exception: {str(e)}",
                    "passed": False,
                }
            },
            "all_passed": False,
        }
        print(f"\n  ERROR DETAILS:\n{error_detail}")

    return result


# ============================================================================
# STATS AGGREGATION
# ============================================================================


def _build_check_stats(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate per-check pass rates across all test results.

    Produces a check_stats dict compatible with print_results(), including
    the failed_queries list needed for diagnostic output.

    Args:
        all_results: All test result dicts from the evaluation run

    Returns:
        Dict mapping check names to pass rate stats and failing query names
    """
    check_names = [
        "id_validity",
        "no_duplicates",
        "count_in_range",
        "diversity",
        "genre_matching",
        "negative_constraint_filtering",
    ]

    stats: Dict[str, Any] = {}

    for check_name in check_names:
        results_with_check = [
            r for r in all_results if check_name in r.get("evaluation", {}).get("checks", {})
        ]

        if not results_with_check:
            continue

        passed_results = [
            r
            for r in results_with_check
            if r["evaluation"]["checks"][check_name].get("passed", False)
        ]
        failed_results = [r for r in results_with_check if r not in passed_results]

        stats[check_name] = {
            "passed": len(passed_results),
            "total": len(results_with_check),
            "pass_rate": len(passed_results) / len(results_with_check),
            "failed_queries": [r["name"] for r in failed_results],
        }

    return stats


def _build_query_stats(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Compute what percentage of queries pass ALL their checks.

    Args:
        all_results: All test result dicts from the evaluation run

    Returns:
        Dict with perfect_rate, passed_all_checks, failed_one_or_more, and total counts
    """
    total = len(all_results)
    passed_all = sum(1 for r in all_results if r["overall_pass"])
    failed_any = total - passed_all

    return {
        "total": total,
        "passed_all_checks": passed_all,
        "failed_one_or_more": failed_any,
        "perfect_rate": passed_all / total if total > 0 else 0.0,
    }


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================


async def evaluate_selection_tests(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Execute all selection agent evaluation tests.

    Routes test cases to the appropriate test runner based on category.
    Collects per-category, per-check, and per-query statistics.

    Args:
        test_cases: Dict mapping category names to test case lists

    Returns:
        Aggregated evaluation results compatible with print_results() and save_results()
    """
    SELECTION_CATEGORIES = {
        "selection_genre_matching": "selection_genre",
        "selection_negative_constraints": "selection_negative",
    }

    db = SessionLocal()

    try:
        all_results: List[Dict] = []
        category_stats: Dict[str, Any] = {}

        for category, test_type in SELECTION_CATEGORIES.items():
            cases = test_cases.get(category, [])
            if not cases:
                continue

            print(f"\nCategory: {category} ({len(cases)} tests)")
            category_results: List[Dict] = []

            for i, test_case in enumerate(cases, 1):
                print(f"  Test {i}/{len(cases)}: {test_case['name']}... ", end="", flush=True)

                if test_type == "selection_genre":
                    result = await run_genre_filtering_test(test_case, db)
                else:
                    result = await run_negative_constraint_test(test_case, db)

                status = "PASS" if result["overall_pass"] else "FAIL"
                print(status)

                if not result["overall_pass"] and "evaluation" in result:
                    failed_checks = [
                        name
                        for name, check in result["evaluation"]["checks"].items()
                        if not check.get("passed", False)
                    ]
                    if failed_checks:
                        print(f"    Failed checks: {', '.join(failed_checks)}")

                category_results.append(result)

            all_results.extend(category_results)

            if category_results:
                passed = sum(1 for r in category_results if r["overall_pass"])
                category_stats[category] = {
                    "passed": passed,
                    "total": len(category_results),
                    "pass_rate": passed / len(category_results),
                    "eval_type": test_type,
                }

        # Eval type rollup
        eval_type_stats: Dict[str, Any] = {}
        for eval_type in ["selection_genre", "selection_negative"]:
            type_results = [r for r in all_results if r.get("test_type") == eval_type]
            if type_results:
                passed = sum(1 for r in type_results if r["overall_pass"])
                eval_type_stats[eval_type] = {
                    "passed": passed,
                    "total": len(type_results),
                    "pass_rate": passed / len(type_results),
                }

        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)

        return {
            "results": all_results,
            "category_stats": category_stats,
            "eval_type_stats": eval_type_stats,
            "check_stats": _build_check_stats(all_results),
            "query_stats": _build_query_stats(all_results),
            "overall": {
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        db.close()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """CLI entry point for selection agent evaluation."""
    parser = argparse.ArgumentParser(
        description="Selection Agent Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all selection tests
  python evaluate_selection.py

  # Run only genre filtering tests
  python evaluate_selection.py --categories selection_genre_matching

  # Run only negative constraint tests
  python evaluate_selection.py --categories selection_negative_constraints
        """,
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Test categories to run (default: all selection categories)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("SELECTION AGENT EVALUATION")
    print("Stage 3a: Filtering, Ranking, and ID Validation")
    print("=" * 70)

    all_test_cases = load_test_cases(test_cases_path)

    valid_selection_categories = {"selection_genre_matching", "selection_negative_constraints"}
    selection_cases = {k: v for k, v in all_test_cases.items() if k in valid_selection_categories}

    if not selection_cases:
        print(
            "\nERROR: No selection test categories found in test_cases.json.\n"
            "Add 'selection_genre_matching' and/or 'selection_negative_constraints' sections."
        )
        sys.exit(1)

    if args.categories:
        test_cases = {
            cat: selection_cases[cat] for cat in args.categories if cat in selection_cases
        }
        if not test_cases:
            print("\nERROR: None of the specified categories found.")
            print(f"Available selection categories: {', '.join(selection_cases.keys())}")
            sys.exit(1)
    else:
        test_cases = selection_cases

    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"\nLoaded {total_cases} test cases across {len(test_cases)} categories")
    print("Testing with REAL database connection")
    print("All tests use mock candidates from the test data factory\n")

    eval_results = asyncio.run(evaluate_selection_tests(test_cases))

    print_results(eval_results)
    save_results(eval_results, results_dir / "selection", stage_name="selection")


if __name__ == "__main__":
    main()
