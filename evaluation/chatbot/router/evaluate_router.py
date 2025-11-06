"""
Router evaluation script.
Tests RouterLLM classification accuracy with real LLM.
"""
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.agents.orchestrator.router import RouterLLM
from app.agents.schemas import TurnInput


def load_test_queries(json_path: Path) -> Dict[str, List[Dict]]:
    """Load test queries from JSON file."""
    with open(json_path) as f:
        return json.load(f)


async def evaluate_router(
    test_queries: Dict[str, List[Dict]], 
    router: RouterLLM
) -> Dict[str, Any]:
    """
    Evaluate router on test queries.
    
    Returns:
        Dict with results and statistics
    """
    results = []
    category_stats = {}
    
    for category, queries in test_queries.items():
        category_correct = 0
        category_total = len(queries)
        
        for test_case in queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            # Create TurnInput
            turn_input = TurnInput(
                user_text=query,
                full_history=[],
                profile_allowed=False,
                user_num_ratings=0
            )
            
            # Classify
            plan = router.classify(turn_input)
            actual = plan.target
            passed = (actual == expected)
            
            if passed:
                category_correct += 1
            
            results.append({
                "category": category,
                "query": query,
                "expected": expected,
                "actual": actual,
                "passed": passed,
                "reasoning": plan.reason
            })
        
        category_stats[category] = {
            "correct": category_correct,
            "total": category_total,
            "accuracy": category_correct / category_total if category_total > 0 else 0
        }
    
    # Overall stats
    total_correct = sum(r["passed"] for r in results)
    total_queries = len(results)
    overall_accuracy = total_correct / total_queries if total_queries > 0 else 0
    
    return {
        "results": results,
        "category_stats": category_stats,
        "overall": {
            "correct": total_correct,
            "total": total_queries,
            "accuracy": overall_accuracy
        },
        "timestamp": datetime.now().isoformat()
    }


def print_results(eval_results: Dict[str, Any]):
    """Print evaluation results to console."""
    print("\n" + "="*70)
    print("ROUTER EVALUATION RESULTS")
    print("="*70)
    
    # Overall stats
    overall = eval_results["overall"]
    print(f"\nOverall Accuracy: {overall['accuracy']:.1%} ({overall['correct']}/{overall['total']})")
    
    # Category breakdown
    print("\nAccuracy by Category:")
    for category, stats in eval_results["category_stats"].items():
        print(f"  {category:25s} {stats['accuracy']:>6.1%}  ({stats['correct']}/{stats['total']})")
    
    # Failed cases
    failures = [r for r in eval_results["results"] if not r["passed"]]
    if failures:
        print(f"\n❌ Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"\n  Query: '{f['query']}'")
            print(f"  Expected: {f['expected']}, Got: {f['actual']}")
            print(f"  Reasoning: {f['reasoning']}")
    else:
        print("\n✅ All test cases passed!")
    
    print("\n" + "="*70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"router_eval_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


async def main():
    """Main evaluation function."""
    # Setup paths
    script_dir = Path(__file__).parent
    test_queries_path = script_dir / "router_test_queries.json"
    results_dir = script_dir / "results"
    
    # Load test queries
    print("Loading test queries...")
    test_queries = load_test_queries(test_queries_path)
    
    total_queries = sum(len(queries) for queries in test_queries.values())
    print(f"Loaded {total_queries} test queries across {len(test_queries)} categories")
    
    # Create router
    print("Initializing router...")
    router = RouterLLM()  # Uses real LLM
    
    # Run evaluation
    print("Running evaluation...")
    eval_results = await evaluate_router(test_queries, router)
    
    # Print results
    print_results(eval_results)
    
    # Save results
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    asyncio.run(main())
