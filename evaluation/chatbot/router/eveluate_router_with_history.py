"""
Router evaluation with conversation history.
Tests router's ability to handle context and intent switches.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.orchestrator.router import RouterLLM
from app.agents.context_builder import make_router_input


def load_conversation_tests(json_path: Path) -> Dict[str, List[Dict]]:
    """Load conversation test cases."""
    with open(json_path) as f:
        return json.load(f)


def evaluate_with_history(
    test_data: Dict[str, List[Dict]], 
    router: RouterLLM
) -> Dict[str, Any]:
    """
    Evaluate router on conversations (tests only final turn).
    
    Uses make_router_input() to preprocess history exactly as Conductor does.
    """
    results = []
    category_stats = {}
    
    for category, test_cases in test_data.items():
        category_correct = 0
        category_total = len(test_cases)
        
        for test_case in test_cases:
            history = test_case["history"]
            final_query = test_case["final_query"]
            expected = test_case["expected"]
            name = test_case.get("name", "unnamed")
            
            # Use actual router input preprocessing (k_user=2 default)
            router_input = make_router_input(
                history=history,
                user_text=final_query,
                k_user=2
            )
            
            # Classify
            plan = router.classify(router_input)
            actual = plan.target
            passed = (actual == expected)
            
            if passed:
                category_correct += 1
            
            results.append({
                "name": name,
                "category": category,
                "history": history,
                "final_query": final_query,
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
    
    # Overall
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
    """Print evaluation results."""
    print("\n" + "="*70)
    print("ROUTER EVALUATION WITH HISTORY")
    print("="*70)
    
    # Overall
    overall = eval_results["overall"]
    print(f"\nOverall Accuracy: {overall['accuracy']:.1%} ({overall['correct']}/{overall['total']})")
    
    # By difficulty
    print("\nBy Difficulty:")
    category_labels = {
        "continuation": "Continuation (easy)",
        "easy_switches": "Easy switches (respond/docs → task)",
        "hard_switches": "Hard switches (recsys ↔ web/docs)"
    }
    
    for category, stats in eval_results["category_stats"].items():
        label = category_labels.get(category, category)
        print(f"  {label:40s} {stats['accuracy']:>6.1%}  ({stats['correct']}/{stats['total']})")
    
    # Failed cases
    failures = [r for r in eval_results["results"] if not r["passed"]]
    if failures:
        print(f"\n❌ Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"\n  [{f['name']}]")
            print(f"  History: {f['history'][-1]['u']}")
            print(f"  Query: '{f['final_query']}'")
            print(f"  Expected: {f['expected']}, Got: {f['actual']}")
            print(f"  Reasoning: {f['reasoning']}")
    else:
        print("\n✅ All test cases passed!")
    
    print("\n" + "="*70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"router_history_eval_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Main evaluation."""
    script_dir = Path(__file__).parent
    test_data_path = script_dir / "router_conversation_tests.json"
    results_dir = script_dir / "results"
    
    print("Loading conversation tests...")
    test_data = load_conversation_tests(test_data_path)
    
    total = sum(len(cases) for cases in test_data.values())
    print(f"Loaded {total} conversations across {len(test_data)} categories")
    
    print("Initializing router...")
    router = RouterLLM()
    
    print("Running evaluation...")
    eval_results = evaluate_with_history(test_data, router)
    
    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
