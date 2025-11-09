# evaluation/chatbot/recommendation_agent/evaluate_recommendation.py
"""
Recommendation Agent evaluation with deterministic tool selection checks and integration tests.
Tests two-stage pipeline: Retrieval → Curation.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.infrastructure.recsys.orchestrator import RecommendationAgent
from app.agents.infrastructure.recsys.retrieval_agent import RetrievalAgent
from app.agents.domain.entities import AgentRequest, ExecutionContext
from app.database import SessionLocal
from app.table_models import User, Interaction
from sqlalchemy import func


def get_user_by_id(db, user_id: int) -> User:
    """Get user by ID from database."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise RuntimeError(f"User with ID {user_id} not found in database")
    
    # Get rating count for logging
    rating_count = db.query(func.count(Interaction.id)).filter(
        Interaction.user_id == user_id
    ).scalar()
    
    print(f"  Using user {user_id} with {rating_count} ratings")
    return user


def load_test_cases(json_path: Path) -> Dict[str, List[Dict]]:
    """Load test cases from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def check_tool_selection(
    query: str,
    tool_executions: List,
    expected_tools: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Layer 1: Deterministic tool selection validation.
    
    Checks if agent called appropriate tools based on query type and user state.
    No LLM-as-judge needed - just inspect tool execution list.
    """
    tool_names = [exec.tool_name for exec in tool_executions]
    retrieval_tools = ["als_recs", "subject_hybrid_pool", "book_semantic_search"]
    
    results = {
        "tool_names": tool_names,
        "checks": {},
        "all_passed": True
    }
    
    # Check: should_use_user_profile
    if "should_use_user_profile" in expected_tools:
        should_use = expected_tools["should_use_user_profile"]
        actually_used = "user_profile" in tool_names
        
        passed = should_use == actually_used
        results["checks"]["user_profile_usage"] = {
            "expected": should_use,
            "actual": actually_used,
            "passed": passed
        }
        if not passed:
            results["all_passed"] = False
    
    # Check: profile_before_retrieval
    if expected_tools.get("profile_before_retrieval"):
        if "user_profile" in tool_names:
            profile_idx = tool_names.index("user_profile")
            retrieval_indices = [
                i for i, name in enumerate(tool_names) 
                if name in retrieval_tools
            ]
            
            if retrieval_indices:
                profile_first = all(profile_idx < idx for idx in retrieval_indices)
                results["checks"]["profile_ordering"] = {
                    "expected": "profile before retrieval",
                    "actual": "correct" if profile_first else "profile called after retrieval",
                    "passed": profile_first
                }
                if not profile_first:
                    results["all_passed"] = False
        else:
            results["checks"]["profile_ordering"] = {
                "expected": "profile before retrieval",
                "actual": "profile not called",
                "passed": False
            }
            results["all_passed"] = False
    
    # Check: should_not_use_als
    if expected_tools.get("should_not_use_als"):
        actually_used = "als_recs" in tool_names
        passed = not actually_used
        
        results["checks"]["als_not_used"] = {
            "expected": "should NOT use als_recs",
            "actual": "used" if actually_used else "not used",
            "passed": passed
        }
        if not passed:
            results["all_passed"] = False
    
    # Check: should_use_any_retrieval_tool (for vague queries after profile)
    if expected_tools.get("should_use_any_retrieval_tool"):
        used_retrieval = any(tool in tool_names for tool in retrieval_tools)
        passed = used_retrieval
        
        actual_retrieval = [tool for tool in retrieval_tools if tool in tool_names]
        results["checks"]["any_retrieval_tool_usage"] = {
            "expected": "at least one retrieval tool",
            "actual": actual_retrieval if actual_retrieval else "none",
            "passed": passed
        }
        if not passed:
            results["all_passed"] = False
    
    # Check: should_use_subject_hybrid (removed - covered by any_retrieval_tool)
    
    # Check: should_use_semantic_search
    if expected_tools.get("should_use_semantic_search"):
        actually_used = "book_semantic_search" in tool_names
        passed = actually_used
        
        results["checks"]["semantic_search_usage"] = {
            "expected": True,
            "actual": actually_used,
            "passed": passed
        }
        if not passed:
            results["all_passed"] = False
    
    # Check: should_not_use_profile
    if expected_tools.get("should_not_use_profile"):
        actually_used = "user_profile" in tool_names
        passed = not actually_used
        
        results["checks"]["profile_not_used"] = {
            "expected": "should NOT use user_profile",
            "actual": "used" if actually_used else "not used",
            "passed": passed
        }
        if not passed:
            results["all_passed"] = False
    
    # Check: should_use_subject_search
    if expected_tools.get("should_use_subject_search"):
        # Either subject_id_search or subject_hybrid_pool
        used_subject_id = "subject_id_search" in tool_names
        used_subject_hybrid = "subject_hybrid_pool" in tool_names
        actually_used = used_subject_id or used_subject_hybrid
        
        passed = actually_used
        results["checks"]["subject_search_usage"] = {
            "expected": True,
            "actual": f"subject_id: {used_subject_id}, subject_hybrid: {used_subject_hybrid}",
            "passed": passed
        }
        if not passed:
            results["all_passed"] = False
    
    return results


def check_two_stage_integration(
    retrieval_response,
    final_response,
    expected_behavior: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Layer 2: Two-stage integration validation.
    
    Checks that retrieval → curation pipeline works correctly.
    """
    results = {
        "checks": {},
        "all_passed": True
    }
    
    # Check: retrieval gathered candidates
    if expected_behavior.get("retrieval_should_gather_candidates"):
        has_candidates = len(retrieval_response.book_recommendations) > 0
        results["checks"]["retrieval_has_candidates"] = {
            "expected": True,
            "actual": has_candidates,
            "count": len(retrieval_response.book_recommendations),
            "passed": has_candidates
        }
        if not has_candidates:
            results["all_passed"] = False
    
    # Check: retrieval created summary
    if expected_behavior.get("retrieval_should_create_summary"):
        has_summary = False
        if retrieval_response.execution_state:
            summary = retrieval_response.execution_state.intermediate_outputs.get(
                "retrieval_summary", []
            )
            has_summary = len(summary) > 0
        
        results["checks"]["retrieval_has_summary"] = {
            "expected": True,
            "actual": has_summary,
            "passed": has_summary
        }
        if not has_summary:
            results["all_passed"] = False
    
    # Check: final response has prose
    if expected_behavior.get("final_response_should_have_prose"):
        has_prose = bool(final_response.text and len(final_response.text.strip()) > 20)
        results["checks"]["final_has_prose"] = {
            "expected": True,
            "actual": has_prose,
            "text_length": len(final_response.text) if final_response.text else 0,
            "passed": has_prose
        }
        if not has_prose:
            results["all_passed"] = False
    
    # Check: final response has books
    if expected_behavior.get("final_response_should_have_books"):
        has_books = len(final_response.book_recommendations) >= 3
        results["checks"]["final_has_books"] = {
            "expected": "at least 3 books",
            "actual": len(final_response.book_recommendations),
            "passed": has_books
        }
        if not has_books:
            results["all_passed"] = False
    
    return results


def check_output_structure(
    response,
    expected_output: Dict[str, Any],
    is_retrieval_only: bool = False
) -> Dict[str, Any]:
    """
    Check output structure (candidate counts, final book counts).
    
    Args:
        response: Agent response
        expected_output: Expected output constraints
        is_retrieval_only: True if testing retrieval agent only (candidates, not final books)
    """
    results = {
        "checks": {},
        "all_passed": True
    }
    
    num_books = len(response.book_recommendations)
    
    if is_retrieval_only:
        # For retrieval-only: Check candidates only
        # Candidate counts can be high (80-300)
        if "min_candidates" in expected_output:
            min_required = expected_output["min_candidates"]
            passed = num_books >= min_required
            
            results["checks"]["min_candidates"] = {
                "expected": f">= {min_required}",
                "actual": num_books,
                "passed": passed
            }
            if not passed:
                results["all_passed"] = False
        
        # Don't check final_book_count for retrieval-only tests
        # (that's for curation output)
        
    else:
        # For full pipeline: Check final book count after curation
        if "min_final_books" in expected_output:
            min_books = expected_output["min_final_books"]
            max_books = expected_output.get("max_final_books", 30)
            
            passed = min_books <= num_books <= max_books
            results["checks"]["final_book_count"] = {
                "expected": f"{min_books}-{max_books}",
                "actual": num_books,
                "passed": passed
            }
            if not passed:
                results["all_passed"] = False
    
    # Check prose exists (only for full pipeline, not retrieval-only)
    if not is_retrieval_only:
        has_prose = bool(response.text and len(response.text.strip()) > 10)
        results["checks"]["has_prose"] = {
            "expected": True,
            "actual": has_prose,
            "passed": has_prose
        }
        if not has_prose:
            results["all_passed"] = False
    else:
        # For retrieval-only, prose is not expected
        results["checks"]["has_prose"] = {
            "expected": "N/A (retrieval only)",
            "actual": "N/A",
            "passed": True
        }
    
    return results


def run_test_case(test_case: Dict, category: str, db) -> Dict[str, Any]:
    """
    Run a single test case with real database connection.
    
    For tool selection tests: Tests retrieval agent directly (to see tool calls)
    For integration tests: Tests full orchestrator (retrieval + curation)
    
    Returns test results with pass/fail status.
    """
    name = test_case.get("name", "unnamed")
    query = test_case["query"]
    user_id = test_case["user_id"]
    user_state_dict = test_case["user_state"]
    
    num_ratings = user_state_dict["num_ratings"]
    allow_profile = user_state_dict["allow_profile"]
    is_warm = user_state_dict.get("is_warm", num_ratings >= 10)
    
    # Get the specific user from database
    try:
        current_user = get_user_by_id(db, user_id)
    except RuntimeError as e:
        return {
            "name": name,
            "category": category,
            "query": query,
            "user_state": user_state_dict,
            "agent_success": False,
            "error": str(e),
            "overall_pass": False
        }
    
    # Decide which agent to test based on category
    is_tool_selection_test = "tool_selection" in category
    
    if is_tool_selection_test:
        # For tool selection tests: Use retrieval agent directly
        # This way we can see tool executions
        agent = RetrievalAgent(
            current_user=current_user,
            db=db,
            user_num_ratings=num_ratings,
            allow_profile=allow_profile
        )
    else:
        # For integration/edge case tests: Use full orchestrator
        agent = RecommendationAgent(
            current_user=current_user,
            db=db,
            user_num_ratings=num_ratings,
            warm_threshold=10,
            allow_profile=allow_profile
        )
    
    # Execute
    request = AgentRequest(
        user_text=query,
        conversation_history=[],
        context=ExecutionContext()
    )
    
    try:
        response = agent.execute(request)
        agent_success = response.success
        
        # Get retrieval response if available (for integration tests)
        retrieval_response = None
        if hasattr(agent, 'retrieval_agent'):
            # Try to get retrieval response from orchestrator
            # Note: This is a simplification - real implementation would need
            # to capture retrieval response during execution
            pass
        
    except Exception as e:
        return {
            "name": name,
            "category": category,
            "query": query,
            "user_state": user_state_dict,
            "agent_success": False,
            "error": str(e),
            "overall_pass": False
        }
    
    # Build result
    result = {
        "name": name,
        "category": category,
        "query": query,
        "user_state": user_state_dict,
        "agent_success": agent_success,
        "response_text_preview": response.text[:100] if response.text else "",
        "num_books": len(response.book_recommendations),
        "tool_calls": response.tool_calls_count,
        "test_type": "retrieval_only" if is_tool_selection_test else "full_pipeline"
    }
    
    # Get tool executions
    tool_executions = []
    if response.execution_state:
        tool_executions = response.execution_state.tool_executions
    
    # Part 1: Tool selection checks (only for tool_selection tests)
    if "expected_tools" in test_case:
        tool_check = check_tool_selection(
            query, tool_executions, test_case["expected_tools"]
        )
        result["tool_selection"] = tool_check
    
    # Part 2: Output structure checks
    if "expected_output" in test_case:
        if is_tool_selection_test:
            # For retrieval-only tests: check candidates (not final books)
            output_check = check_output_structure(
                response, 
                test_case["expected_output"],
                is_retrieval_only=True
            )
        else:
            # For full pipeline tests: check final books
            output_check = check_output_structure(
                response, 
                test_case["expected_output"],
                is_retrieval_only=False
            )
        result["output_structure"] = output_check
    
    # Part 2: Integration checks (for two-stage tests)
    if "expected_behavior" in test_case:
        # For integration tests, we'd need retrieval response
        # Simplified for now - just check final response
        integration_check = {
            "checks": {},
            "all_passed": True
        }
        
        # Check final response has prose
        if test_case["expected_behavior"].get("final_response_should_have_prose"):
            has_prose = bool(response.text and len(response.text.strip()) > 20)
            integration_check["checks"]["has_prose"] = {
                "passed": has_prose,
                "actual": len(response.text) if response.text else 0
            }
            if not has_prose:
                integration_check["all_passed"] = False
        
        # Check final response has books
        if test_case["expected_behavior"].get("final_response_should_have_books"):
            has_books = len(response.book_recommendations) >= 3
            integration_check["checks"]["has_books"] = {
                "passed": has_books,
                "actual": len(response.book_recommendations)
            }
            if not has_books:
                integration_check["all_passed"] = False
        
        result["integration"] = integration_check
    
    # Determine overall pass
    checks_to_validate = []
    if "tool_selection" in result:
        checks_to_validate.append(result["tool_selection"]["all_passed"])
    if "output_structure" in result:
        checks_to_validate.append(result["output_structure"]["all_passed"])
    if "integration" in result:
        checks_to_validate.append(result["integration"]["all_passed"])
    
    result["overall_pass"] = agent_success and all(checks_to_validate)
    
    return result


def evaluate(test_cases: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Run evaluation on all test cases with real database."""
    # Create database session
    if SessionLocal is None:
        raise RuntimeError(
            "Database not configured. Set DATABASE_URL environment variable."
        )
    
    db = SessionLocal()
    
    try:
        all_results = []
        category_stats = {}
        
        for category, cases in test_cases.items():
            print(f"\n{'='*70}")
            print(f"Running {category}: {len(cases)} test cases")
            print('='*70)
            
            category_results = []
            for i, test_case in enumerate(cases, 1):
                name = test_case.get("name", f"test_{i}")
                print(f"\n[{i}/{len(cases)}] {name}...", end=" ")
                
                result = run_test_case(test_case, category, db)
                category_results.append(result)
                
                status = "✅ PASS" if result["overall_pass"] else "❌ FAIL"
                print(status)
            
            all_results.extend(category_results)
            
            # Category stats
            passed = sum(1 for r in category_results if r["overall_pass"])
            category_stats[category] = {
                "passed": passed,
                "total": len(category_results),
                "pass_rate": passed / len(category_results) if category_results else 0
            }
        
        # Overall stats
        total_passed = sum(1 for r in all_results if r["overall_pass"])
        total_tests = len(all_results)
        
        return {
            "results": all_results,
            "category_stats": category_stats,
            "overall": {
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    
    finally:
        db.close()


def print_results(eval_results: Dict[str, Any]):
    """Print detailed evaluation results."""
    print("\n" + "="*70)
    print("RECOMMENDATION AGENT EVALUATION RESULTS")
    print("="*70)
    
    # Overall stats
    overall = eval_results["overall"]
    print(f"\nOverall Pass Rate: {overall['pass_rate']:.1%} ({overall['passed']}/{overall['total']})")
    
    # Category breakdown
    print("\nResults by Category:")
    for category, stats in eval_results["category_stats"].items():
        print(f"  {category:35s} {stats['pass_rate']:>6.1%}  ({stats['passed']}/{stats['total']})")
    
    # Failed cases detail
    failures = [r for r in eval_results["results"] if not r["overall_pass"]]
    if failures:
        print(f"\n❌ Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"\n  [{f['name']}]")
            print(f"  Query: {f['query']}")
            print(f"  User: warm={f['user_state']['is_warm']}, profile={f['user_state']['allow_profile']}")
            
            if "error" in f:
                print(f"  Error: {f['error']}")
            
            # Tool selection failures
            if "tool_selection" in f and not f["tool_selection"]["all_passed"]:
                print("  Tool Selection Issues:")
                for check_name, check in f["tool_selection"]["checks"].items():
                    if not check["passed"]:
                        print(f"    - {check_name}: expected {check['expected']}, got {check['actual']}")
            
            # Output structure failures
            if "output_structure" in f and not f["output_structure"]["all_passed"]:
                print("  Output Structure Issues:")
                for check_name, check in f["output_structure"]["checks"].items():
                    if not check["passed"]:
                        print(f"    - {check_name}: expected {check['expected']}, got {check['actual']}")
            
            # Integration failures
            if "integration" in f and not f["integration"]["all_passed"]:
                print("  Integration Issues:")
                for check_name, check in f["integration"]["checks"].items():
                    if not check["passed"]:
                        print(f"    - {check_name}: {check}")
    else:
        print("\n✅ All test cases passed!")
    
    print("\n" + "="*70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"recommendation_eval_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Main evaluation function."""
    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"
    
    print("="*70)
    print("RECOMMENDATION AGENT EVALUATION")
    print("="*70)
    print("\nLoading test cases...")
    test_cases = load_test_cases(test_cases_path)
    
    total_cases = sum(len(cases) for cases in test_cases.values())
    print(f"Loaded {total_cases} test cases across {len(test_cases)} categories")
    
    print("\n📊 Testing with REAL database connection and users")
    print("   Tools will execute against actual data\n")
    
    print("Running evaluation...")
    eval_results = evaluate(test_cases)
    
    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
