# evaluation/chatbot/recommendation_agent/run_all_recommendation_tests.py
"""
Unified runner for all recommendation agent evaluation stages.
Orchestrates Planner → Retrieval → Curation → Integration tests and merges results.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Suppress noisy HTTP client logs (httpcore, primp, rquest, etc)
from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()

# Import individual evaluation modules
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from shared_helpers import load_test_cases, print_results, save_results

# Individual eval modules will be imported dynamically to avoid import issues


# ============================================================================
# RESULT MERGING
# ============================================================================


def merge_results(
    planner_results, retrieval_results, curation_results, integration_results
) -> Dict[str, Any]:
    """
    Merge results from all 4 evaluation stages into unified result structure.

    Aggregates:
    - Individual test results
    - Category statistics
    - Evaluation type statistics
    - Check-level statistics (which checks are failing)
    - Query-level statistics (how many queries pass all checks)

    Args:
        planner_results: Results from evaluate_planner_tests()
        retrieval_results: Results from evaluate_retrieval_tests()
        curation_results: Results from evaluate_curation_tests()
        integration_results: Results from evaluate_integration_tests()

    Returns:
        Merged results dict matching dashboard expectations with check-level stats
    """
    # Merge all test results
    all_results = []
    all_results.extend(planner_results.get("results", []))
    all_results.extend(retrieval_results.get("results", []))
    all_results.extend(curation_results.get("results", []))
    all_results.extend(integration_results.get("results", []))

    # Merge category stats
    category_stats = {}
    category_stats.update(planner_results.get("category_stats", {}))
    category_stats.update(retrieval_results.get("category_stats", {}))
    category_stats.update(curation_results.get("category_stats", {}))
    category_stats.update(integration_results.get("category_stats", {}))

    # Merge eval type stats
    eval_type_stats = {}
    eval_type_stats.update(planner_results.get("eval_type_stats", {}))
    eval_type_stats.update(retrieval_results.get("eval_type_stats", {}))
    eval_type_stats.update(curation_results.get("eval_type_stats", {}))
    eval_type_stats.update(integration_results.get("eval_type_stats", {}))

    # Calculate overall stats
    total_passed = sum(1 for r in all_results if r.get("overall_pass", False))
    total_tests = len(all_results)

    # Aggregate check-level statistics
    check_stats = _aggregate_check_stats(all_results)
    query_stats = _aggregate_query_stats(all_results)

    return {
        "results": all_results,
        "category_stats": category_stats,
        "eval_type_stats": eval_type_stats,
        "check_stats": check_stats,
        "query_stats": query_stats,
        "overall": {
            "passed": total_passed,
            "total": total_tests,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }


def _aggregate_check_stats(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate statistics for each individual check across all tests.

    Shows which specific quality checks are systematically failing.

    Returns:
        Dict mapping check names to {passed, total, pass_rate, failed_queries}
    """
    check_stats = {}

    for result in all_results:
        if "evaluation" not in result or "checks" not in result["evaluation"]:
            continue

        query = result.get("query", result.get("name", "unknown"))

        for check_name, check_data in result["evaluation"]["checks"].items():
            if check_name not in check_stats:
                check_stats[check_name] = {
                    "passed": 0,
                    "total": 0,
                    "failed_queries": [],
                }

            check_stats[check_name]["total"] += 1

            if check_data.get("passed", False):
                check_stats[check_name]["passed"] += 1
            else:
                # Track which queries failed this check (limit to first 5)
                if len(check_stats[check_name]["failed_queries"]) < 5:
                    check_stats[check_name]["failed_queries"].append(query[:50])

    # Calculate pass rates
    for check_name, stats in check_stats.items():
        stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0

    return check_stats


def _aggregate_query_stats(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate query-level statistics.

    Shows what percentage of queries pass ALL checks (perfect output).

    Returns:
        Dict with passed_all_checks, failed_one_or_more, total, perfect_rate
    """
    passed_all = 0
    total = len(all_results)

    for result in all_results:
        if result.get("overall_pass", False):
            passed_all += 1

    return {
        "passed_all_checks": passed_all,
        "failed_one_or_more": total - passed_all,
        "total": total,
        "perfect_rate": passed_all / total if total > 0 else 0,
    }


# ============================================================================
# ORCHESTRATION
# ============================================================================


async def run_all_tests(
    test_cases: Dict[str, List[Dict]],
    selected_stages: List[str] = None,
    results_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run all recommendation agent evaluation stages.

    Saves both individual stage results and merged results with single timestamp:
    - Individual: results/planner/planner_eval_{timestamp}.json
    - Individual: results/retrieval/retrieval_eval_{timestamp}.json
    - Individual: results/curation/curation_eval_{timestamp}.json
    - Individual: results/integration/integration_eval_{timestamp}.json
    - Merged: results/recommendation_eval_{timestamp}.json

    Args:
        test_cases: All test cases loaded from test_cases.json
        selected_stages: Optional list of stages to run ['planner', 'retrieval', 'curation', 'integration']
                        If None, runs all stages
        results_dir: Directory to save results. If provided, saves stage results as they complete.

    Returns:
        Merged results from all stages
    """
    stages = (
        selected_stages if selected_stages else ["planner", "retrieval", "curation", "integration"]
    )

    # Generate single timestamp for all files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}

    # Stage 1: Planner
    if "planner" in stages:
        print("\n" + "=" * 70)
        print("STAGE 1: PLANNER EVALUATION")
        print("=" * 70)
        from evaluate_planner import evaluate_planner_tests

        results["planner"] = await evaluate_planner_tests(test_cases)

        # Save individual stage result
        if results_dir:
            save_results(
                results["planner"],
                results_dir / "planner",
                stage_name="planner",
                timestamp=timestamp,
            )
    else:
        results["planner"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }

    # Stage 2: Retrieval
    if "retrieval" in stages:
        print("\n" + "=" * 70)
        print("STAGE 2: RETRIEVAL EVALUATION")
        print("=" * 70)
        from evaluate_retrieval import evaluate_retrieval_tests

        results["retrieval"] = await evaluate_retrieval_tests(test_cases)

        # Save individual stage result
        if results_dir:
            save_results(
                results["retrieval"],
                results_dir / "retrieval",
                stage_name="retrieval",
                timestamp=timestamp,
            )
    else:
        results["retrieval"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }

    # Stage 3: Curation
    if "curation" in stages:
        print("\n" + "=" * 70)
        print("STAGE 3: CURATION EVALUATION")
        print("=" * 70)
        from evaluate_curation import evaluate_curation_tests

        results["curation"] = await evaluate_curation_tests(test_cases)

        # Save individual stage result
        if results_dir:
            save_results(
                results["curation"],
                results_dir / "curation",
                stage_name="curation",
                timestamp=timestamp,
            )
    else:
        results["curation"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }

    # Stage 4: Integration
    if "integration" in stages:
        print("\n" + "=" * 70)
        print("STAGE 4: INTEGRATION EVALUATION")
        print("=" * 70)
        from evaluate_integration import evaluate_integration_tests

        results["integration"] = await evaluate_integration_tests(test_cases)

        # Save individual stage result
        if results_dir:
            save_results(
                results["integration"],
                results_dir / "integration",
                stage_name="integration",
                timestamp=timestamp,
            )
    else:
        results["integration"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }

    # Merge all results
    merged = merge_results(
        results["planner"], results["retrieval"], results["curation"], results["integration"]
    )

    # Save merged result (for dashboard)
    if results_dir:
        save_results(merged, results_dir, timestamp=timestamp)

    return merged


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """Run all recommendation tests with CLI support."""
    parser = argparse.ArgumentParser(
        description="Recommendation Agent Full Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 4 stages
  python run_all_recommendation_tests.py

  # Run only planner and retrieval
  python run_all_recommendation_tests.py --stages planner retrieval

  # Run specific categories across all stages
  python run_all_recommendation_tests.py --categories tool_selection_warm_user retrieval_strategy_adherence
        """,
    )

    parser.add_argument(
        "--stages",
        "-s",
        nargs="+",
        choices=["planner", "retrieval", "curation", "integration"],
        help="Test stages to run (default: all)",
    )

    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Specific test categories to run (default: all categories in selected stages)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    classic_scenarios_path = script_dir / "integration_classic_scenarios.json"
    results_dir = script_dir / "results"

    print("=" * 70)
    print("RECOMMENDATION AGENT FULL EVALUATION SUITE")
    print("Multi-Agent Architecture: Planner → Retrieval → Curation")
    print("=" * 70)
    print("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    # Load classic scenarios if file exists
    if classic_scenarios_path.exists():
        classic_scenarios = load_test_cases(classic_scenarios_path)
        all_test_cases.update(classic_scenarios)
        print(f"Loaded {len(classic_scenarios)} classic scenario categories")

    # Filter categories if specified
    if args.categories:
        test_cases = {cat: all_test_cases[cat] for cat in args.categories if cat in all_test_cases}
        if not test_cases:
            print("\n❌ ERROR: None of the specified categories found")
            print(f"Available categories: {', '.join(all_test_cases.keys())}")
            sys.exit(1)
    else:
        test_cases = all_test_cases

    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"Loaded {total_cases} test cases across {len(test_cases)} categories")

    stages_to_run = (
        args.stages if args.stages else ["planner", "retrieval", "curation", "integration"]
    )
    print(f"\nStages to run: {', '.join(stages_to_run)}")

    print("\n📊 Testing with REAL database connection and users")
    print("   All agents will use actual data and tools\n")

    print("Running evaluation...")

    # Run async evaluation (saves individual + merged results internally)
    eval_results = asyncio.run(run_all_tests(test_cases, stages_to_run, results_dir))

    print_results(eval_results)


if __name__ == "__main__":
    main()
