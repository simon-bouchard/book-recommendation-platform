# evaluation/chatbot/run_all_evals.py
"""
Unified entry point for running all chatbot agent evaluations.
Executes router, docs, web, response, and recommendation agent tests sequentially.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json


class EvaluationRunner:
    """Orchestrates execution of all agent evaluation scripts."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results = []

        # Map agent names to their evaluation scripts
        self.agents = {
            "router": {"path": "router/evaluate_router.py", "name": "Router Agent"},
            "router_history": {
                "path": "router/eveluate_router_with_history.py",
                "name": "Router with History",
            },
            "docs": {"path": "docs_agent/evaluate_docs.py", "name": "Docs Agent"},
            "web": {"path": "web_agent/evaluate_web.py", "name": "Web Agent"},
            "response": {"path": "response_agent/evaluate_response.py", "name": "Response Agent"},
            "recommendation": {
                "path": "recommendation_agent/run_all_recommendation_tests.py",
                "name": "Recommendation Agent",
            },
        }

    def run_evaluation(self, agent_key: str) -> Dict[str, Any]:
        """
        Run evaluation for a single agent.

        Args:
            agent_key: Key identifying the agent

        Returns:
            Dictionary with execution results
        """
        agent_info = self.agents[agent_key]
        script_path = self.base_dir / agent_info["path"]

        if not script_path.exists():
            return {
                "agent": agent_key,
                "name": agent_info["name"],
                "status": "SKIPPED",
                "error": f"Script not found: {script_path}",
            }

        print(f"\n{'=' * 70}")
        print(f"Running {agent_info['name']}...")
        print(f"Script: {script_path}")
        print("=" * 70)

        try:
            start_time = datetime.now()

            # Run the evaluation script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per agent
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Check if successful
            success = result.returncode == 0

            # Try to extract pass rate from output
            pass_rate = self._extract_pass_rate(result.stdout)

            eval_result = {
                "agent": agent_key,
                "name": agent_info["name"],
                "status": "SUCCESS" if success else "FAILED",
                "duration_seconds": duration,
                "pass_rate": pass_rate,
                "stdout_preview": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr if result.stderr else None,
            }

            # Print summary
            if success:
                print(f"✅ {agent_info['name']}: COMPLETED")
                if pass_rate is not None:
                    print(f"   Pass Rate: {pass_rate:.1%}")
                print(f"   Duration: {duration:.1f}s")
            else:
                print(f"❌ {agent_info['name']}: FAILED")
                print(f"   Error: {result.stderr[:200] if result.stderr else 'Unknown error'}")

            return eval_result

        except subprocess.TimeoutExpired:
            return {
                "agent": agent_key,
                "name": agent_info["name"],
                "status": "TIMEOUT",
                "error": "Evaluation exceeded 10 minute timeout",
            }
        except Exception as e:
            return {
                "agent": agent_key,
                "name": agent_info["name"],
                "status": "ERROR",
                "error": str(e),
            }

    def _extract_pass_rate(self, output: str) -> float:
        """
        Extract pass rate from evaluation output.

        Looks for patterns like "Pass Rate: 85.0% (17/20)"
        """
        if not output:
            return None

        # Look for pass rate pattern
        import re

        match = re.search(r"Pass Rate:\s*(\d+\.?\d*)%", output)
        if match:
            return float(match.group(1)) / 100.0

        # Alternative pattern: "Overall: 85.0%"
        match = re.search(r"Overall:\s*(\d+\.?\d*)%", output)
        if match:
            return float(match.group(1)) / 100.0

        return None

    def run_all(self, selected_agents: List[str] = None) -> Dict[str, Any]:
        """
        Run all agent evaluations sequentially.

        Args:
            selected_agents: Optional list of agent keys to run. If None, runs all.

        Returns:
            Aggregated results
        """
        agents_to_run = selected_agents if selected_agents else list(self.agents.keys())

        print("\n" + "=" * 70)
        print("RUNNING ALL AGENT EVALUATIONS")
        print("=" * 70)
        print(f"\nAgents to evaluate: {len(agents_to_run)}")
        for agent_key in agents_to_run:
            print(f"  - {self.agents[agent_key]['name']}")
        print()

        start_time = datetime.now()

        # Run each agent evaluation
        for agent_key in agents_to_run:
            result = self.run_evaluation(agent_key)
            self.results.append(result)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Aggregate results
        return {
            "results": self.results,
            "summary": self._build_summary(),
            "total_duration_seconds": total_duration,
            "timestamp": datetime.now().isoformat(),
        }

    def _build_summary(self) -> Dict[str, Any]:
        """Build summary statistics from all results."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r["status"] == "SUCCESS")
        failed = sum(1 for r in self.results if r["status"] == "FAILED")
        errors = sum(1 for r in self.results if r["status"] in ["ERROR", "TIMEOUT"])
        skipped = sum(1 for r in self.results if r["status"] == "SKIPPED")

        # Average pass rate (only for successful runs)
        pass_rates = [r["pass_rate"] for r in self.results if r.get("pass_rate") is not None]
        avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else None

        return {
            "total_agents": total,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "average_pass_rate": avg_pass_rate,
        }

    def print_summary(self, run_results: Dict[str, Any]):
        """Print final summary of all evaluations."""
        print("\n" + "=" * 70)
        print("ALL EVALUATIONS COMPLETE")
        print("=" * 70)

        summary = run_results["summary"]

        print(f"\nTotal Agents: {summary['total_agents']}")
        print(f"  ✅ Successful: {summary['successful']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ⚠️  Errors: {summary['errors']}")
        print(f"  ⏭️  Skipped: {summary['skipped']}")

        if summary["average_pass_rate"] is not None:
            print(f"\nAverage Pass Rate: {summary['average_pass_rate']:.1%}")

        print(f"\nTotal Duration: {run_results['total_duration_seconds']:.1f}s")

        # Individual results
        print("\nIndividual Results:")
        for result in run_results["results"]:
            status_icon = {
                "SUCCESS": "✅",
                "FAILED": "❌",
                "ERROR": "⚠️",
                "TIMEOUT": "⏱️",
                "SKIPPED": "⏭️",
            }.get(result["status"], "❓")

            name = result["name"]
            status = result["status"]

            line = f"  {status_icon} {name:30s} {status:10s}"

            if result.get("pass_rate") is not None:
                line += f" {result['pass_rate']:>6.1%}"

            if result.get("duration_seconds") is not None:
                line += f" ({result['duration_seconds']:.1f}s)"

            print(line)

        print("\n" + "=" * 70)
        print("Run eval_dashboard.py to view detailed results")
        print("=" * 70)

    def save_summary(self, run_results: Dict[str, Any], output_file: Path):
        """Save summary to JSON file."""
        with open(output_file, "w") as f:
            json.dump(run_results, f, indent=2)
        print(f"\nSummary saved to: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run all chatbot agent evaluations")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["router", "router_history", "docs", "web", "response", "recommendation"],
        help="Specific agents to evaluate (default: all)",
    )
    parser.add_argument(
        "--save-summary", action="store_true", help="Save execution summary to JSON"
    )

    args = parser.parse_args()

    # Determine base directory (should be evaluation/chatbot/)
    script_dir = Path(__file__).parent

    # Create runner
    runner = EvaluationRunner(script_dir)

    # Run evaluations
    run_results = runner.run_all(selected_agents=args.agents)

    # Print summary
    runner.print_summary(run_results)

    # Save summary if requested
    if args.save_summary:
        summary_file = script_dir / "eval_summary.json"
        runner.save_summary(run_results, summary_file)

    # Exit with appropriate code
    summary = run_results["summary"]
    if summary["failed"] > 0 or summary["errors"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
