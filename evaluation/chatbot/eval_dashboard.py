# evaluation/chatbot/eval_dashboard.py
"""
Evaluation dashboard that displays the latest test results for all chatbot agents.
Scans result directories and provides a comprehensive summary with pass rates and failed tests.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class AgentResult:
    """Container for agent evaluation results."""

    agent_name: str
    file_path: Path
    timestamp: datetime
    overall_stats: Dict
    category_stats: Dict
    failed_tests: List[Dict]
    total_tests: int
    passed_tests: int
    check_stats: Optional[Dict] = None  # Check-level statistics (recommendation agent)
    query_stats: Optional[Dict] = None  # Query-level statistics (recommendation agent)


class EvalDashboard:
    """Dashboard for displaying chatbot agent evaluation results."""

    AGENT_DIRS = {
        "router": "router/results",
        "router_history": "router/results",
        "docs": "docs_agent/results",
        "web": "web_agent/results",
        "response": "response_agent/results",
        "recommendation": "recommendation_agent/results",
    }

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def find_latest_results(self) -> Dict[str, List[Path]]:
        """Find the latest and previous result files for each agent."""
        results = defaultdict(list)

        # First try standard directory structure
        for agent_name, rel_path in self.AGENT_DIRS.items():
            results_dir = self.base_path / rel_path
            if results_dir.exists():
                # Find all eval files for this agent
                if agent_name == "router":
                    pattern = "router_eval_*.json"
                elif agent_name == "router_history":
                    pattern = "router_history_eval_*.json"
                else:
                    pattern = f"{agent_name.split('_')[0]}_*eval_*.json"

                files = sorted(results_dir.glob(pattern), reverse=True)
                results[agent_name] = files[:2]  # Latest 2 files for comparison

        # If no results found, try flat directory structure (all files in base_path)
        if not any(results.values()):
            patterns = {
                "router": "router_eval_*.json",
                "router_history": "router_history_eval_*.json",
                "docs": "docs_*eval_*.json",
                "web": "web_*eval_*.json",
                "response": "response_*eval_*.json",
                "recommendation": "recommendation_*eval_*.json",
            }

            for agent_name, pattern in patterns.items():
                files = sorted(self.base_path.glob(pattern), reverse=True)
                if files:
                    results[agent_name] = files[:2]

        return results

    def parse_result_file(self, agent_name: str, file_path: Path) -> Optional[AgentResult]:
        """Parse a result JSON file and extract key metrics."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract timestamp
            timestamp_str = data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)

            # Extract overall stats - handle two formats
            overall = data.get("overall", {})
            if not overall:
                # Stats might be at the top level instead
                overall = {
                    k: v
                    for k, v in data.items()
                    if k
                    in [
                        "passed",
                        "total",
                        "correct",
                        "pass_rate",
                        "accuracy",
                        "retrieval_pass_rate",
                        "quality_pass_rate",
                        "avg_quality_score",
                        "tool_pass_rate",
                    ]
                }

            # Determine pass/total metrics based on structure
            if "passed" in overall:
                passed = overall["passed"]
                total = overall["total"]
            elif "correct" in overall:
                passed = overall["correct"]
                total = overall["total"]
            else:
                passed = 0
                total = 0

            # Extract category stats
            category_stats = data.get("category_stats", {})

            # Extract check-level and query-level stats (recommendation agent)
            check_stats = data.get("check_stats")
            query_stats = data.get("query_stats")

            # Extract failed tests
            failed_tests = []
            if "results" in data:
                for test in data["results"]:
                    # Check various pass/fail fields
                    is_passed = test.get("passed", test.get("pass", test.get("overall_pass", True)))
                    if not is_passed:
                        failed_tests.append(test)

            return AgentResult(
                agent_name=agent_name,
                file_path=file_path,
                timestamp=timestamp,
                overall_stats=overall,
                category_stats=category_stats,
                failed_tests=failed_tests,
                total_tests=total,
                passed_tests=passed,
                check_stats=check_stats,
                query_stats=query_stats,
            )
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def format_percentage(self, value: float) -> str:
        """Format a percentage value with color coding."""
        pct = value * 100
        if pct == 100:
            symbol = "✅"
        elif pct >= 80:
            symbol = "⚠️ "
        else:
            symbol = "❌"
        return f"{symbol} {pct:5.1f}%"

    def format_change(self, current: float, previous: float) -> str:
        """Format the change between two values."""
        if previous == 0:
            return ""

        diff = current - previous
        if abs(diff) < 0.001:
            return ""

        pct_change = diff * 100
        if diff > 0:
            return f" (↑{pct_change:+.1f}%)"
        else:
            return f" (↓{pct_change:+.1f}%)"

    def print_agent_summary(self, result: AgentResult, previous: Optional[AgentResult] = None):
        """Print summary for a single agent."""
        print(f"\n{'=' * 80}")
        print(f"  {result.agent_name.upper().replace('_', ' ')} AGENT")
        print(f"{'=' * 80}")

        # Overall stats
        pass_rate = result.passed_tests / result.total_tests if result.total_tests > 0 else 0
        change_str = ""
        if previous:
            prev_rate = (
                previous.passed_tests / previous.total_tests if previous.total_tests > 0 else 0
            )
            change_str = self.format_change(pass_rate, prev_rate)

        print(f"\n📊 Overall Performance: {result.passed_tests}/{result.total_tests} tests passed")
        print(f"   Pass Rate: {self.format_percentage(pass_rate)}{change_str}")
        print(f"   Evaluated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Category breakdown
        if result.category_stats:
            print("\n📁 Category Breakdown:")
            for category, stats in sorted(result.category_stats.items()):
                if isinstance(stats, dict):
                    # Handle different stat formats
                    if "passed" in stats:
                        cat_passed = stats["passed"]
                        cat_total = stats["total"]
                    elif "correct" in stats:
                        cat_passed = stats["correct"]
                        cat_total = stats["total"]
                    else:
                        continue

                    cat_rate = cat_passed / cat_total if cat_total > 0 else 0

                    # Compare with previous if available
                    prev_change = ""
                    if previous and category in previous.category_stats:
                        prev_cat = previous.category_stats[category]
                        if isinstance(prev_cat, dict):
                            prev_total = prev_cat.get("total", 0)
                            prev_passed = prev_cat.get("passed", prev_cat.get("correct", 0))
                            prev_rate = prev_passed / prev_total if prev_total > 0 else 0
                            prev_change = self.format_change(cat_rate, prev_rate)

                    print(
                        f"   {category:30s} {cat_passed:2d}/{cat_total:2d} {self.format_percentage(cat_rate)}{prev_change}"
                    )

        # Check-level statistics (recommendation agent)
        if result.check_stats:
            print("\n🔍 Check-Level Statistics (Quality Dimensions):")
            print("   Shows which specific quality checks are systematically failing\n")

            # Sort by pass rate (worst first) to highlight problems
            sorted_checks = sorted(result.check_stats.items(), key=lambda x: x[1]["pass_rate"])

            # Show top 10 worst-performing checks
            for check_name, stats in sorted_checks[:10]:
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

                print(
                    f"   {emoji} {check_name:35s} {passed:2d}/{total:2d} {self.format_percentage(pass_rate)}"
                )

                # Show failed queries if available and not too many
                failed_queries = stats.get("failed_queries", [])
                if failed_queries and len(failed_queries) <= 3:
                    queries_str = ", ".join(q[:40] for q in failed_queries)
                    print(f"      Failed on: {queries_str}")

        # Query-level statistics (recommendation agent)
        if result.query_stats:
            print("\n📊 Query-Level Statistics (Pipeline Health):")
            qstats = result.query_stats
            perfect_rate = qstats["perfect_rate"]

            if perfect_rate >= 0.8:
                emoji = "✅"
            elif perfect_rate >= 0.5:
                emoji = "⚠️ "
            else:
                emoji = "❌"

            print(
                f"   {emoji} Passed all checks:  {qstats['passed_all_checks']:3d}/{qstats['total']} {self.format_percentage(perfect_rate)}"
            )
            print(f"   ❌ Failed 1+ checks:   {qstats['failed_one_or_more']:3d}/{qstats['total']}")

        # Additional metrics (for specific agents)
        if "retrieval_pass_rate" in result.overall_stats:
            print("\n📚 Retrieval Metrics:")
            print(
                f"   Retrieval Pass Rate: {self.format_percentage(result.overall_stats['retrieval_pass_rate'])}"
            )
            print(
                f"   Quality Pass Rate:   {self.format_percentage(result.overall_stats['quality_pass_rate'])}"
            )
            print(
                f"   Avg Quality Score:   {result.overall_stats.get('avg_quality_score', 0):.2f}/2"
            )

        if "tool_pass_rate" in result.overall_stats:
            print("\n🔧 Tool Usage Metrics:")
            print(
                f"   Tool Pass Rate:    {self.format_percentage(result.overall_stats['tool_pass_rate'])}"
            )
            print(
                f"   Quality Pass Rate: {self.format_percentage(result.overall_stats['quality_pass_rate'])}"
            )

        # Failed tests
        if result.failed_tests:
            print(f"\n❌ Failed Tests ({len(result.failed_tests)}):")
            for i, test in enumerate(result.failed_tests[:10], 1):  # Show first 10
                query = test.get("query", test.get("name", "Unknown"))
                category = test.get("category", "N/A")

                # Extract reason for failure
                reasons = []

                # Check retrieval failure
                if "retrieval_pass" in test and not test.get("retrieval_pass"):
                    reason = test.get("retrieval_reason", "Failed")
                    reasons.append(f"Retrieval: {reason}")

                # Check quality failure
                if "quality_pass" in test and not test.get("quality_pass"):
                    reason = test.get("quality_reason", "Failed")
                    reasons.append(f"Quality: {reason}")

                # Check tool selection failures
                if "tool_selection" in test:
                    tool_sel = test["tool_selection"]
                    if isinstance(tool_sel, dict) and not tool_sel.get("all_passed", True):
                        failed_checks = []
                        for check_name, check_val in tool_sel.get("checks", {}).items():
                            if isinstance(check_val, dict) and not check_val.get("passed", True):
                                expected = check_val.get("expected", "N/A")
                                actual = check_val.get("actual", "N/A")
                                failed_checks.append(
                                    f"{check_name}: expected {expected}, got {actual}"
                                )
                        if failed_checks:
                            reasons.append(f"Tool Selection: {'; '.join(failed_checks)}")

                # Check output structure failures
                if "output_structure" in test:
                    output = test["output_structure"]
                    if isinstance(output, dict) and not output.get("all_passed", True):
                        failed_checks = []
                        for check_name, check_val in output.get("checks", {}).items():
                            if isinstance(check_val, dict) and not check_val.get("passed", True):
                                failed_checks.append(check_name)
                        if failed_checks:
                            reasons.append(f"Output: {', '.join(failed_checks)}")

                # Check integration failures
                if "integration" in test:
                    integration = test["integration"]
                    if isinstance(integration, dict) and not integration.get("all_passed", True):
                        reasons.append("Integration failed")

                # Check tool usage for web agent
                if "tool_used" in test and "recency_expected" in test:
                    tool_used = test.get("tool_used")
                    recency_expected = test.get("recency_expected")

                    if recency_expected and not tool_used:
                        reasons.append("Tool: Should have used web_search but didn't")
                    elif not recency_expected and tool_used:
                        reasons.append("Tool: Used web_search unnecessarily (stable topic)")

                # Check quality for web agent
                if "quality_pass" in test and not test.get("quality_pass"):
                    quality_reason = test.get("quality_reason", "Failed")
                    reasons.append(f"Quality: {quality_reason}")

                # Check relevance/recency for web agent
                if "relevance" in test and test["relevance"] == 0:
                    reasons.append("Relevance: Response not relevant to query")
                if "recency" in test and test.get("recency") == 0:
                    reasons.append("Recency: Response lacks current information")

                # Fallback to generic reasoning
                if not reasons:
                    if "reasoning" in test:
                        reasons.append(test["reasoning"])
                    elif "reason" in test:
                        reasons.append(test["reason"])
                    elif "tool_reason" in test:
                        reasons.append(f"Tool: {test['tool_reason']}")
                    else:
                        # Check if all known checks passed but overall failed
                        all_passed = all(
                            [
                                test.get("quality_pass", True),
                                test.get("retrieval_pass", True),
                                test.get("relevance", 1) == 1,
                                test.get("recency", 1) == 1,
                            ]
                        )
                        if all_passed:
                            reasons.append(
                                "Test configuration issue or undocumented failure reason"
                            )
                        else:
                            reasons.append("Unknown failure reason")

                reason_str = "; ".join(reasons) if reasons else "Unknown failure"

                print(f"\n   {i}. [{category}] {query[:70]}")
                print(f"      → {reason_str[:120]}")

            if len(result.failed_tests) > 10:
                print(f"\n      ... and {len(result.failed_tests) - 10} more failed tests")
        else:
            print("\n✅ All tests passed!")

    def print_summary_table(
        self, results: Dict[str, Tuple[Optional[AgentResult], Optional[AgentResult]]]
    ):
        """Print a summary table of all agents."""
        print("\n" + "=" * 80)
        print("  🎯 EVALUATION DASHBOARD SUMMARY")
        print("=" * 80)
        print()
        print(f"{'Agent':<20} {'Pass Rate':<15} {'Tests':<12} {'Status':<15}")
        print("-" * 80)

        # Sort by pass rate (lowest first to highlight issues)
        agent_list = []
        for agent_name, (current, _) in results.items():
            if current:
                pass_rate = (
                    current.passed_tests / current.total_tests if current.total_tests > 0 else 0
                )
                agent_list.append((agent_name, current, pass_rate))

        agent_list.sort(key=lambda x: x[2])

        for agent_name, result, pass_rate in agent_list:
            pct_str = f"{pass_rate * 100:.1f}%"
            tests_str = f"{result.passed_tests}/{result.total_tests}"

            if pass_rate == 1.0:
                status = "✅ Perfect"
            elif pass_rate >= 0.8:
                status = "⚠️  Good"
            elif pass_rate >= 0.5:
                status = "⚠️  Needs Work"
            else:
                status = "❌ Critical"

            print(f"{agent_name:<20} {pct_str:<15} {tests_str:<12} {status:<15}")

        print("-" * 80)

        # Overall recommendation
        print("\n💡 Priority Recommendations:")
        critical = [a for a, r, p in agent_list if p < 0.5]
        needs_work = [a for a, r, p in agent_list if 0.5 <= p < 0.8]

        if critical:
            print(f"   🔴 Critical: {', '.join(critical)}")
        if needs_work:
            print(f"   🟡 Needs Improvement: {', '.join(needs_work)}")
        if not critical and not needs_work:
            print("   🟢 All agents performing well!")

        print()

    def run(self, show_comparison: bool = True, verbose: bool = False):
        """Run the dashboard and display results."""
        print("\n" + "🔍 Scanning evaluation results...\n")

        latest_files = self.find_latest_results()

        if not latest_files:
            print("No evaluation results found!")
            return

        results = {}
        for agent_name, files in latest_files.items():
            if not files:
                continue

            current = self.parse_result_file(agent_name, files[0])
            previous = (
                self.parse_result_file(agent_name, files[1])
                if len(files) > 1 and show_comparison
                else None
            )

            if current:
                results[agent_name] = (current, previous)

        # Print summary table first
        self.print_summary_table(results)

        # Print detailed results for each agent
        if verbose:
            for agent_name, (current, previous) in sorted(results.items()):
                self.print_agent_summary(current, previous)

        print("\n" + "=" * 80)
        print("  Dashboard complete!")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chatbot Agent Evaluation Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Show summary table
  %(prog)s -v                       # Show detailed breakdown
  %(prog)s --no-comparison          # Don't compare with previous runs
  %(prog)s -p /path/to/evaluation   # Use custom base path
        """,
    )

    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=Path(__file__).parent,
        help="Base path to evaluation directory (default: current directory)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed breakdown for each agent"
    )

    parser.add_argument(
        "--no-comparison", action="store_true", help="Do not compare with previous evaluation runs"
    )

    args = parser.parse_args()

    dashboard = EvalDashboard(args.path)
    dashboard.run(show_comparison=not args.no_comparison, verbose=args.verbose)


if __name__ == "__main__":
    main()
