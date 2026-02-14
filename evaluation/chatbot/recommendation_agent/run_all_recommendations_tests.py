# evaluation/chatbot/recommendation_agent/run_all_recommendation_tests.py
"""
Unified runner for all recommendation agent evaluation stages.
Orchestrates Planner → Retrieval → Curation → Integration tests and merges results.
"""

import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Suppress noisy HTTP client logs (httpcore, primp, rquest, etc)
import logging
from app.agents.logging import suppress_noisy_loggers

suppress_noisy_loggers()

# Import individual evaluation modules
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from shared_helpers import load_test_cases, print_results
# Individual eval modules will be imported dynamically to avoid import issues


# ============================================================================
# RESULT MERGING
# ============================================================================


def merge_results(
    planner_results, retrieval_results, curation_results, integration_results
) -> Dict[str, Any]:
    """
    Merge results from all 4 evaluation stages into unified result structure.

    Args:
        planner_results: Results from evaluate_planner_tests()
        retrieval_results: Results from evaluate_retrieval_tests()
        curation_results: Results from evaluate_curation_tests()
        integration_results: Results from evaluate_integration_tests()

    Returns:
        Merged results dict matching dashboard expectations
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


# ============================================================================
# ORCHESTRATION
# ============================================================================


async def run_all_tests(
    test_cases: Dict[str, List[Dict]], selected_stages: List[str] = None
) -> Dict[str, Any]:
    """
    Run all recommendation agent evaluation stages.

    Args:
        test_cases: All test cases loaded from test_cases.json
        selected_stages: Optional list of stages to run ['planner', 'retrieval', 'curation', 'integration']
                        If None, runs all stages

    Returns:
        Merged results from all stages
    """
    stages = (
        selected_stages if selected_stages else ["planner", "retrieval", "curation", "integration"]
    )

    results = {}

    # Stage 1: Planner
    if "planner" in stages:
        print("\n" + "=" * 70)
        print("STAGE 1: PLANNER EVALUATION")
        print("=" * 70)
        from evaluate_planner import evaluate_planner_tests

        results["planner"] = await evaluate_planner_tests(test_cases)
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
        # TODO: Import and run when evaluate_retrieval.py is created
        # from evaluate_retrieval import evaluate_retrieval_tests
        # results["retrieval"] = await evaluate_retrieval_tests(test_cases)
        print("⚠️  Retrieval evaluation not yet implemented")
        results["retrieval"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }
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
        # TODO: Import and run when evaluate_curation.py is created
        # from evaluate_curation import evaluate_curation_tests
        # results["curation"] = await evaluate_curation_tests(test_cases)
        print("⚠️  Curation evaluation not yet implemented")
        results["curation"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }
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
        # TODO: Import and run when evaluate_integration.py is created
        # from evaluate_integration import evaluate_integration_tests
        # results["integration"] = await evaluate_integration_tests(test_cases)
        print("⚠️  Integration evaluation not yet implemented")
        results["integration"] = {
            "results": [],
            "category_stats": {},
            "eval_type_stats": {},
            "overall": {"passed": 0, "total": 0, "pass_rate": 0},
        }
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
    results_dir = script_dir / "results"

    print("=" * 70)
    print("RECOMMENDATION AGENT FULL EVALUATION SUITE")
    print("Multi-Agent Architecture: Planner → Retrieval → Curation")
    print("=" * 70)
    print("\nLoading test cases...")
    all_test_cases = load_test_cases(test_cases_path)

    # Filter categories if specified
    if args.categories:
        test_cases = {cat: all_test_cases[cat] for cat in args.categories if cat in all_test_cases}
        if not test_cases:
            print(f"\n❌ ERROR: None of the specified categories found")
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

    # Run async evaluation
    eval_results = asyncio.run(run_all_tests(test_cases, stages_to_run))

    print_results(eval_results)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = results_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual stage results to stages/ subdirectory (for debugging)
    if "planner" in stages_to_run and results["planner"]["overall"]["total"] > 0:
        with open(stages_dir / f"planner_eval_{timestamp}.json", "w") as f:
            json.dump(results["planner"], f, indent=2)

    if "retrieval" in stages_to_run and results["retrieval"]["overall"]["total"] > 0:
        with open(stages_dir / f"retrieval_eval_{timestamp}.json", "w") as f:
            json.dump(results["retrieval"], f, indent=2)

    if "curation" in stages_to_run and results["curation"]["overall"]["total"] > 0:
        with open(stages_dir / f"curation_eval_{timestamp}.json", "w") as f:
            json.dump(results["curation"], f, indent=2)

    if "integration" in stages_to_run and results["integration"]["overall"]["total"] > 0:
        with open(stages_dir / f"integration_eval_{timestamp}.json", "w") as f:
            json.dump(results["integration"], f, indent=2)

    # Save combined results to main results/ directory (for dashboard)
    output_file = results_dir / f"recommendation_eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nCombined results saved to: {output_file}")
    print(f"Individual stage results saved to: {stages_dir}/")


if __name__ == "__main__":
    main()
