# evaluation/chatbot/recommendation_agent/shared_helpers.py
"""
Shared helper functions for recommendation agent evaluation.
Database utilities, query validation, and result formatting used across all test files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy import func

from app.table_models import Interaction, User

# ============================================================================
# DATABASE HELPERS
# ============================================================================


def get_user_by_id(db, user_id: int) -> Tuple[User, int]:
    """
    Get user by ID and their rating count.

    Args:
        db: Database session
        user_id: User ID to fetch

    Returns:
        Tuple of (User object, rating count)

    Raises:
        RuntimeError: If user not found in database
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise RuntimeError(f"User with ID {user_id} not found in database")

    rating_count = (
        db.query(func.count(Interaction.id)).filter(Interaction.user_id == user_id).scalar()
    )

    return user, rating_count


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate that query is not empty or whitespace-only.

    Args:
        query: Query string to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not query or not query.strip():
        return False, f"Empty or whitespace-only query: '{query}'"
    return True, ""


# ============================================================================
# TEST CASE LOADING
# ============================================================================


def load_test_cases(json_path: Path) -> Dict[str, List[Dict]]:
    """
    Load test cases from JSON file.

    Args:
        json_path: Path to test_cases.json

    Returns:
        Dictionary mapping category names to lists of test case dicts
    """
    with open(json_path) as f:
        return json.load(f)


# ============================================================================
# RESULT FORMATTING AND OUTPUT
# ============================================================================


def print_results(eval_results: Dict[str, Any]):
    """
    Print detailed evaluation results to console.

    Displays:
    - Overall pass rate
    - Results by evaluation type (planner, retrieval, curation, integration)
    - Results by category
    - Failed case details with specific check failures

    Args:
        eval_results: Results dictionary from evaluate_all()
    """
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

    # ========================================================================
    # CHECK-BASED STATS: Which quality dimensions are failing?
    # ========================================================================
    if "check_stats" in eval_results:
        print("\n" + "=" * 70)
        print("CHECK PASS RATES (Quality Dimensions)")
        print("=" * 70)
        print("Shows which quality checks are systematically failing\n")

        # Sort checks by pass rate (worst first) to highlight problems
        sorted_checks = sorted(eval_results["check_stats"].items(), key=lambda x: x[1]["pass_rate"])

        for check_name, stats in sorted_checks:
            pass_rate = stats["pass_rate"]
            passed = stats["passed"]
            total = stats["total"]

            # Choose emoji based on pass rate
            if pass_rate >= 0.8:
                emoji = "✅"
            elif pass_rate >= 0.5:
                emoji = "⚠️ "
            else:
                emoji = "❌"

            print(f"  {emoji} {check_name:40s} {pass_rate:>6.1%}  ({passed}/{total})")

            # Show which queries failed this check (if any)
            if stats["failed_queries"] and len(stats["failed_queries"]) <= 3:
                failed_list = ", ".join(stats["failed_queries"])
                print(f"     Failed on: {failed_list}")
            elif stats["failed_queries"]:
                print(f"     Failed on: {len(stats['failed_queries'])} queries")

    # ========================================================================
    # QUERY-BASED STATS: How many queries pass ALL checks?
    # ========================================================================
    if "query_stats" in eval_results:
        print("\n" + "=" * 70)
        print("QUERY PASS RATES (Overall Pipeline Health)")
        print("=" * 70)
        print("Shows what % of queries produce perfect output\n")

        qstats = eval_results["query_stats"]
        perfect_rate = qstats["perfect_rate"]

        # Choose emoji based on perfect rate
        if perfect_rate >= 0.8:
            emoji = "✅"
        elif perfect_rate >= 0.5:
            emoji = "⚠️ "
        else:
            emoji = "❌"

        print(
            f"  {emoji} Passed all checks:    {qstats['passed_all_checks']:>3}/{qstats['total']}  ({perfect_rate:.1%})"
        )
        print(
            f"  ❌ Failed 1+ checks:     {qstats['failed_one_or_more']:>3}/{qstats['total']}  ({1 - perfect_rate:.1%})"
        )
        print()

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


def save_results(
    eval_results: Dict[str, Any],
    output_dir: Path,
    stage_name: str = None,
    timestamp: str = None,
):
    """
    Save evaluation results to timestamped JSON file.

    Args:
        eval_results: Results dictionary from evaluation
        output_dir: Directory to save results (will be created if needed)
        stage_name: Optional stage name (e.g., 'planner', 'retrieval').
                   If provided, saves as {stage_name}_eval_{timestamp}.json
                   If None, saves as recommendation_eval_{timestamp}.json (merged)
        timestamp: Optional timestamp string. If not provided, generates current timestamp.
                  Allows run_all to use same timestamp for all files.

    Examples:
        # Individual stage (standalone run)
        save_results(results, results_dir / "planner", stage_name="planner")
        # → results/planner/planner_eval_20240216_143022.json

        # Merged results (run_all)
        save_results(merged, results_dir, timestamp="20240216_143022")
        # → results/recommendation_eval_20240216_143022.json

        # Individual stage with shared timestamp (run_all)
        save_results(planner_results, results_dir / "planner", "planner", "20240216_143022")
        # → results/planner/planner_eval_20240216_143022.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate or use provided timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine filename based on stage
    if stage_name:
        filename = f"{stage_name}_eval_{timestamp}.json"
    else:
        filename = f"recommendation_eval_{timestamp}.json"

    output_file = output_dir / filename

    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
