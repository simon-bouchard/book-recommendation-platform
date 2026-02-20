# evaluation/chatbot/recommendation_agent/evaluate_curation.py
"""
Curation agent evaluation suite.
Tests prose quality, personalization accuracy, and hallucination prevention using a small
pre-selected candidate pool (8-12 books), simulating the output of SelectionAgent.

Genre filtering and negative constraint filtering are tested in evaluate_selection.py.
"""

import re
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()

from app.agents.infrastructure.recsys.curation_agent import CurationAgent
from app.agents.domain.entities import AgentRequest, BookRecommendation
from app.database import SessionLocal

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
    llm_judge_personalization_prose,
    llm_judge_prose_reasoning,
    llm_judge_query_relevance,
)
from test_data_factory import get_candidates, get_execution_context
from evaluation.chatbot.eval_utils import execute_with_streaming

# Regex matching the agent's citation format: [Book Title](item_idx)
_CITATION_PATTERN = re.compile(r"\[([^\]]+)\]\((\d+)\)")


def _extract_cited_ids(text: str) -> List[int]:
    """
    Extract unique book IDs from citation markup in response text.

    Parses the same [Title](item_idx) format written by the curation agent,
    preserving first-appearance order and deduplicating.

    Args:
        text: Raw response text containing inline citations

    Returns:
        Ordered list of unique integer book IDs
    """
    return list(
        dict.fromkeys(  # deduplicate while preserving order
            int(m.group(2)) for m in _CITATION_PATTERN.finditer(text)
        )
    )


async def run_personalization_prose_test(test_case: Dict, db) -> Dict[str, Any]:
    """
    Execute personalization prose test using isolated curation with a small pre-selected pool.

    Simulates CurationAgent's production input by feeding it 10 real books from the
    test data factory — the same pool SelectionAgent would have already filtered down.
    Tests that prose correctly reflects personalization context and query intent.

    Args:
        test_case: Test case dict with query, context_scenario, expected_prose
        db: Database session

    Returns:
        Test result dict with evaluation checks and pass/fail status
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

        if test_scenario == "als":
            candidates_data = get_candidates(
                "als", db, user_id=user.user_id, user_num_ratings=rating_count
            )
        elif test_scenario == "subject":
            candidates_data = get_candidates(
                "subject",
                db,
                user_id=user.user_id,
                user_num_ratings=rating_count,
                subject_ids=[978, 1066, 2317],
            )
        else:
            candidates_data = get_candidates(
                "basic", db, query=query, user_id=user.user_id, user_num_ratings=rating_count
            )

        context = get_execution_context(context_scenario)

        # Slice to 10 books — simulates the pre-filtered pool CurationAgent receives
        # from SelectionAgent in production rather than the full 60-120 candidate pool.
        selected_books = candidates_data["books"][:10]

        candidates = [
            BookRecommendation(
                item_idx=book["item_idx"],
                title=book.get("title"),
                author=book.get("author"),
                year=book.get("year"),
                num_ratings=book.get("num_ratings"),
            )
            for book in selected_books
        ]

        curation = CurationAgent()
        request = AgentRequest(user_text=query, conversation_history=[])

        curation_output = await execute_with_streaming(
            agent=curation,
            request=request,
            candidates=candidates,
            execution_context=context,
        )

        response_text = curation_output.text
        final_books = curation_output.book_recommendations

        result["agent_success"] = len(response_text) > 50 and len(final_books) >= 3
        result["pipeline"] = {
            "candidate_count": len(candidates),
            "final_book_count": len(final_books),
            "response_length": len(response_text),
        }

        # Validate citations against candidates (detect hallucinations).
        # Parse IDs directly from the response text using the same regex the agent uses,
        # so the check is independent of any internal state the agent may or may not write.
        cited_book_ids = _extract_cited_ids(response_text)
        candidate_ids_set = {c.item_idx for c in candidates}
        invalid_citations = [bid for bid in cited_book_ids if bid not in candidate_ids_set]

        has_hallucinations = len(invalid_citations) > 0

        # Check for duplicate recommendations
        has_duplicates = len(final_books) != len(set(b.item_idx for b in final_books))
        duplicate_ids = []
        if has_duplicates:
            seen = set()
            for book in final_books:
                if book.item_idx in seen:
                    duplicate_ids.append(book.item_idx)
                seen.add(book.item_idx)

        # Run common LLM judges (prose reasoning and query relevance)
        prose_reasoning_result = llm_judge_prose_reasoning(
            response_text=response_text,
            judge_llm=curation.llm,
        )

        query_relevance_result = llm_judge_query_relevance(
            response_text=response_text,
            query=query,
            judge_llm=curation.llm,
            tools_used=context.tools_used,
        )

        # Build evaluation checks
        checks = {
            "no_hallucinations": {
                "expected": "all citations from candidates",
                "actual": f"{len(cited_book_ids)} cited, {len(invalid_citations)} invalid",
                "passed": not has_hallucinations,
                "details": {
                    "invalid_book_ids": invalid_citations,
                },
            },
            "no_duplicates": {
                "expected": "all books unique",
                "actual": f"{len(final_books)} books, {len(duplicate_ids)} duplicates",
                "passed": not has_duplicates,
                "details": {
                    "duplicate_book_ids": duplicate_ids,
                },
            },
            "prose_reasoning": {
                "expected": "prose explains why books recommended",
                "actual": prose_reasoning_result["verdict"],
                "passed": prose_reasoning_result["passed"],
                "details": {
                    "reasoning": prose_reasoning_result["reasoning"],
                    "issues": prose_reasoning_result.get("issues", []),
                },
            },
            "query_relevance": {
                "expected": "prose relates to query",
                "actual": query_relevance_result["verdict"],
                "passed": query_relevance_result["passed"],
                "details": {
                    "reasoning": query_relevance_result["reasoning"],
                    "issues": query_relevance_result.get("issues", []),
                },
            },
        }

        # Add personalization-specific check if needed
        if expected_prose.get("llm_judge_needed") and len(response_text) > 0:
            expect_no_personalization = expected_prose.get(
                "should_NOT_claim_personalization", False
            )

            judge_result = llm_judge_personalization_prose(
                response_text=response_text,
                execution_context=context,
                judge_llm=curation.llm,
                expect_no_personalization=expect_no_personalization,
            )

            checks["personalization_prose"] = {
                "expected": "prose correctly reflects personalization context",
                "actual": judge_result["verdict"],
                "passed": judge_result["passed"],
                "details": {
                    "reasoning": judge_result["reasoning"],
                    "issues": judge_result.get("issues", []),
                },
            }

        # Compute overall pass (all checks must pass)
        all_passed = all(check["passed"] for check in checks.values())

        result["evaluation"] = {
            "checks": checks,
            "all_passed": all_passed,
        }
        result["overall_pass"] = all_passed and result["agent_success"]

        # If no personalization judge needed, just use the common checks
        if not expected_prose.get("llm_judge_needed"):
            checks["basic_structure"] = {
                "expected": "prose and books present",
                "actual": f"{len(response_text)} chars, {len(final_books)} books",
                "passed": result["agent_success"],
            }
            all_passed = all(check["passed"] for check in checks.values())
            result["evaluation"] = {
                "checks": checks,
                "all_passed": all_passed,
            }
            result["overall_pass"] = all_passed

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


async def evaluate_curation_tests(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Execute all curation evaluation tests.

    Routes test cases to appropriate test functions based on category.
    Supports both genre matching and personalization prose tests.

    Args:
        test_cases: Dictionary mapping category names to test case lists

    Returns:
        Aggregated evaluation results with per-category and overall stats
    """
    db = SessionLocal()

    try:
        all_results = []
        category_stats = {}

        # Curation test categories — prose quality only.
        # Genre filtering is tested in evaluate_selection.py.
        CURATION_CATEGORIES = {
            "curation_personalization_prose",
            "curation_false_personalization",
        }

        for category, cases in test_cases.items():
            # Skip non-curation categories
            if category not in CURATION_CATEGORIES:
                continue

            print(f"\nCategory: {category} ({len(cases)} tests)")

            category_results = []

            for i, test_case in enumerate(cases, 1):
                print(f"  Test {i}/{len(cases)}: {test_case['name']}... ", end="", flush=True)

                test_type = "curation_prose"
                result = await run_personalization_prose_test(test_case, db)

                # Add test_type to result for stats
                result["test_type"] = test_type
                category_results.append(result)

                status = "PASS" if result["overall_pass"] else "FAIL"
                print(status)

            all_results.extend(category_results)

            if category_results:
                passed = sum(1 for r in category_results if r["overall_pass"])
                eval_type = category_results[0].get("test_type", "unknown")
                category_stats[category] = {
                    "passed": passed,
                    "total": len(category_results),
                    "pass_rate": passed / len(category_results),
                    "eval_type": eval_type,
                }

        eval_type_stats = {}
        for eval_type in ["curation_prose"]:
            type_results = [r for r in all_results if r.get("test_type") == eval_type]
            if type_results:
                passed = sum(1 for r in type_results if r["overall_pass"])
                eval_type_stats[eval_type] = {
                    "passed": passed,
                    "total": len(type_results),
                    "pass_rate": passed / len(type_results),
                }

        # Aggregate results by check type
        check_type_stats = {}
        check_types = [
            "no_hallucinations",
            "no_duplicates",
            "prose_reasoning",
            "query_relevance",
            "personalization_prose",
        ]

        for check_type in check_types:
            # Find all results that have this check
            results_with_check = [
                r
                for r in all_results
                if r.get("evaluation", {}).get("checks", {}).get(check_type) is not None
            ]

            if results_with_check:
                passed = sum(
                    1 for r in results_with_check if r["evaluation"]["checks"][check_type]["passed"]
                )
                check_type_stats[check_type] = {
                    "passed": passed,
                    "total": len(results_with_check),
                    "pass_rate": passed / len(results_with_check),
                }

        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)

        return {
            "results": all_results,
            "category_stats": category_stats,
            "eval_type_stats": eval_type_stats,
            "check_type_stats": check_type_stats,
            "overall": {
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        db.close()


def main():
    """CLI entry point for curation agent evaluation."""
    parser = argparse.ArgumentParser(
        description="Curation Agent Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all curation tests
  python evaluate_curation.py

  # Run only personalization prose tests
  python evaluate_curation.py --categories curation_personalization_prose

  # Run only false personalization tests
  python evaluate_curation.py --categories curation_false_personalization
        """,
    )

    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Test categories to run (default: all curation categories)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("CURATION AGENT EVALUATION")
    print("Stage 4: Prose Generation (candidates pre-filtered by SelectionAgent)")
    print("=" * 70)
    print("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    curation_categories = {
        k: v
        for k, v in all_test_cases.items()
        if k in ["curation_personalization_prose", "curation_false_personalization"]
    }

    if args.categories:
        test_cases = {
            cat: curation_categories[cat] for cat in args.categories if cat in curation_categories
        }
        if not test_cases:
            print(f"\nERROR: None of the specified categories found")
            print(f"Available curation categories: {', '.join(curation_categories.keys())}")
            sys.exit(1)
    else:
        test_cases = curation_categories

    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"Loaded {total_cases} test cases across {len(test_cases)} categories")

    print("\nTesting with REAL database connection and users")
    print("All tests use mock candidates from test data factory\n")

    print("Running evaluation...")
    eval_results = asyncio.run(evaluate_curation_tests(test_cases))

    print_results(eval_results)

    # Print check-type aggregation (common + specific checks across all tests)
    if "check_type_stats" in eval_results and eval_results["check_type_stats"]:
        print("\n" + "=" * 70)
        print("RESULTS BY CHECK TYPE (Across All Tests)")
        print("=" * 70)
        for check_type, stats in eval_results["check_type_stats"].items():
            check_label = check_type.replace("_", " ").title()
            print(
                f"  {check_label:30s} {stats['pass_rate']:>6.1%}  ({stats['passed']}/{stats['total']})"
            )
        print("=" * 70)

    # Save to results/curation/ for standalone execution
    save_results(eval_results, results_dir / "curation", stage_name="curation")


if __name__ == "__main__":
    main()
